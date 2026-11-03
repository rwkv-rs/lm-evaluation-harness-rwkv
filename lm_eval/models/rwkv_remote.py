import hashlib
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import replace
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from jinja2 import TemplateError

from lm_eval.api.checkpoint import (
    LOGLIKELIHOOD_PROTOCOL_VERSION,
    ItemKey,
    compute_idempotency_key,
    compute_item_id,
    deserialize_response,
    serialize_response,
)
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import (
    LocalCompletionsAPI,
    _RWKVCompletion,
    _rwkv_sampling_config,
)
from lm_eval.models.rwkv_choice import (
    RWKVLoglikelihoodResponse,
    build_choice_prompt,
    choice_letters,
    parse_choice_index,
    rollout_encodings,
    rollout_evidence,
)


eval_logger = logging.getLogger(__name__)

# Mirrors lighteval-rwkv's `http_model.MAX_NEW_TOKENS`: open_think/fake_think
# generations think before answering, so the generic API-model default
# (`self._max_gen_toks`, 256) truncates the chain before a choice letter is
# ever produced. Avg@k choice resolution always uses this instead.
MAX_NEW_TOKENS = 8192


class _RWKVPrompt(str):
    """A rendered prompt retaining its original role boundaries locally."""

    def __new__(cls, text, messages=None, add_generation_prompt=True):
        value = super().__new__(cls, text)
        value.messages = messages
        value.add_generation_prompt = add_generation_prompt
        return value


@register_model("rwkv7-remote")
class RWKV7Remote(LocalCompletionsAPI):
    """RWKV generation through vLLM's completions API.

    NoCoT multiple-choice tasks use native sequence loglikelihoods. Thinking
    modes retain their separate generative choice scoring. The official
    local tokenizer renders generation prompts and encodes likelihood inputs.
    Sampling parameters are sent explicitly as vLLM request fields.

    ``rwkv_generation_prompt`` (``no_think``/``fake_think``/``open_think``) always
    selects one of three fixed sampling presets (``SAMPLING_PARAMETERS``), whether
    or not a tokenizer is configured. These are not read from a per-checkpoint
    ``generation_config.json`` and cannot be overridden via ``gen_kwargs``.
    """

    # Signals to the evaluator's per-task override loop (lm_eval/evaluator.py)
    # that this backend implements avg@k: multiple_choice tasks are scored by
    # sampling `Instance.repeats` independent generations (through the real
    # no_think/open_think/fake_think sampling profile) and averaging correctness over
    # them, rather than the echo/logprob loglikelihood path used by default.
    SUPPORTS_AVG_AT_K = True
    INDEPENDENT_ROLLOUTS = True
    PRESERVE_INPUT_CONTEXT = True
    EVALUATION_PROTOCOL_VERSION = "rwkv-generative-v2"

    DEFAULT_BASE_URL = "https://api.rwkv.rs/v1/completions"
    PROMPT_TEMPLATES = {"assistant", "bot", "function_calling"}
    PROMPT_STOPS = {
        "assistant": "\nUser:",
        "bot": "✿",
        "function_calling": "\n### User",
    }
    GENERATION_PROMPTS = {"no_think", "open_think", "fake_think"}
    # Fixed sampling presets for the three RWKV7 generation modes, mirroring the RWKV
    # weight family's own defaults (and lighteval-rwkv's SAMPLING_PARAMETERS). These
    # are shipped in source rather than read from a per-checkpoint generation_config.json
    # so the reported sampling_config is a stable, auditable constant independent of
    # which tokenizer directory (if any) a given run happens to point at.
    SAMPLING_PARAMETERS = {
        "no_think": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 1.0,
            "do_sample": False,
            "stop_strings": ["✿", "\nUser:", "\n### User"],
        },
        "open_think": {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
            "do_sample": True,
            "stop_strings": ["✿", "\nUser:", "\n### User"],
        },
        "fake_think": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "do_sample": True,
            "stop_strings": ["✿", "\nUser:", "\n### User"],
        },
    }

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        tokenizer=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        rwkv_prompt_template="assistant",
        rwkv_generation_prompt="fake_think",
        cot_mode=None,
        **kwargs,
    ):
        if tokenizer is not None and tokenizer_backend is None:
            tokenizer_backend = "huggingface"
        if tokenized_requests is None:
            tokenized_requests = tokenizer is not None
        if cot_mode is not None:
            rwkv_generation_prompt = cot_mode
        if rwkv_prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError("rwkv_prompt_template must be assistant, bot, or function_calling.")
        if rwkv_generation_prompt not in self.GENERATION_PROMPTS:
            raise ValueError(
                "rwkv_generation_prompt must be no_think, open_think, or fake_think."
            )

        self.rwkv_prompt_template = rwkv_prompt_template
        self.rwkv_generation_prompt = rwkv_generation_prompt
        self.SUPPORTS_LOGLIKELIHOOD = rwkv_generation_prompt == "no_think"
        if self.SUPPORTS_LOGLIKELIHOOD:
            self.EVALUATION_PROTOCOL_VERSION = LOGLIKELIHOOD_PROTOCOL_VERSION
        kwargs.setdefault(
            "max_gen_toks", 256 if rwkv_generation_prompt == "no_think" else MAX_NEW_TOKENS
        )
        super().__init__(
            base_url=base_url,
            tokenizer=tokenizer,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

        if self._batch_size != 1:
            raise ValueError("rwkv7-remote requires batch_size=1; use num_concurrent for concurrency")
        self.max_length = int(kwargs.get("max_length", 2048))
        self._chat_template_source: Optional[str] = None
        self._chat_template_sha256: Optional[str] = None
        self._generation_profile: dict = dict(
            self.SAMPLING_PARAMETERS[rwkv_generation_prompt]
        )
        # Per-doc list of k independently-sampled per-choice (ll, is_greedy)
        # encodings, populated by loglikelihood() and consumed by the avg@k
        # process_results patch installed in lm_eval/evaluator.py. Cleared as
        # each doc is consumed so it never leaks across evaluation runs.
        self._avg_k_samples: Dict[Any, List[List[Tuple[float, bool]]]] = {}
        # Parallel to `_avg_k_samples[doc_id]`: whether each of the k samples
        # actually had a choice letter parsed out of it, so the avg@k patch
        # can score unparseable generations at zero instead of
        # whatever placeholder index they were encoded with.
        self._avg_k_parsed: Dict[Any, List[bool]] = {}
        if tokenizer is not None:
            self._chat_template_source = self.tokenizer.chat_template
            if not isinstance(self._chat_template_source, str) or not self._chat_template_source:
                raise RuntimeError("The RWKV tokenizer must provide the official chat template.")
            self._chat_template_sha256 = hashlib.sha256(
                self._chat_template_source.encode("utf-8")
            ).hexdigest()

    @cached_property
    def api_key(self) -> str:
        key = os.environ.get("RWKV_API_KEY")
        if not key:
            raise ValueError(
                "API key not found. Please set the `RWKV_API_KEY` environment variable."
            )
        return key

    @cached_property
    def _generation_prompt_cue(self) -> Optional[str]:
        """The literal suffix ``apply_chat_template()`` appends when asked for a
        generation prompt -- e.g. ``"\\n\\nAssistant: <think"`` for open_think --
        derived empirically rather than hardcoded, since the real cue text
        depends on ``rwkv_prompt_template``/``rwkv_generation_prompt`` and lives
        in the tokenizer's own Jinja template, not in this file.

        When the harness applies the chat template itself (``apply_chat_template
        = true`` in the eval config), ``Instance.args[0]`` already IS the output
        of ``apply_chat_template()`` -- a complete prompt ending in this cue,
        ready for the model to continue directly from (e.g. from ``>`` in
        open_think mode). That cue marks the boundary between "the real user
        turn" and "the model's own completion": nothing may be appended after
        it, only before it. ``loglikelihood()`` uses this to find that boundary
        in an already-templated context so choice content can be spliced in
        ahead of the cue instead of tacked on after it.
        """
        if self._chat_template_source is None:
            return None
        sentinel = "\x00RWKV_CUE_PROBE\x00"
        probe = self.apply_chat_template(
            [{"role": "user", "content": sentinel}], add_generation_prompt=True
        )
        if sentinel not in probe:
            return None
        return probe.split(sentinel, 1)[1]

    @property
    def tokenizer_name(self) -> str:
        if self._chat_template_sha256 is None:
            return super().tokenizer_name
        return (
            f"{self.model}:{self._chat_template_sha256}:"
            f"{self.rwkv_prompt_template}:{self.rwkv_generation_prompt}"
        )

    def get_model_info(self) -> dict:
        return {"sampling_config": _rwkv_sampling_config({**self._generation_profile, "stop_strings": [self.PROMPT_STOPS[self.rwkv_prompt_template]]})}

    def loglikelihood(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
        item_keys: Optional[List[Optional[ItemKey]]] = None,
    ) -> List[Tuple[float, bool]]:
        """Use native likelihoods for NoCoT, and generation for thinking modes.

        Unlike the echo-based loglikelihood this class inherits from
        ``LocalCompletionsAPI``/``TemplateAPI`` (which never issues a
        ``generate=True`` request, so the open_think/fake_think sampling
        profile in ``_create_payload`` never applies to multiple_choice
        tasks), this asks the model to generate a lettered answer
        ``Instance.repeats`` times per doc -- exercising the real sampling
        profile and protocol -- and records the k per-sample outcomes for
        the avg@k ``process_results`` patch (installed in
        ``lm_eval/evaluator.py``) to average against gold. The single
        ``(loglikelihood, is_greedy)`` list returned here still encodes a
        majority-vote winner per doc, so ``acc``/``acc_norm`` keep working
        unmodified if no avg@k patch is installed.

        ``item_keys`` (checkpoint identity, parallel to ``requests``) is
        accepted because the evaluator's checkpoint-aware dispatch always
        passes it to backends with ``_supports_checkpoint_items = True``
        (which this class inherits); since scoring here bypasses the base
        class's echo-based ``_loglikelihood_tokens`` entirely, completion is
        marked directly against ``self.checkpoint_store`` instead.
        """
        if self.SUPPORTS_LOGLIKELIHOOD:
            if urlsplit(self.base_url).path.rstrip("/") != "/v1/completions":
                raise ValueError("NoCoT loglikelihood requires /v1/completions")
            if self.tokenizer is None or not self.tokenized_requests:
                raise ValueError(
                    "NoCoT loglikelihood requires the server's matching tokenizer "
                    "and tokenized_requests=True"
                )
            requests = [
                replace(request, arguments=(
                    context if not context or re.search(r"\nAnswer:\s*\Z", context)
                    else context + "\nAnswer: ",
                    continuation,
                ))
                for request in requests
                for context, continuation in [request.args]
            ]
            return super().loglikelihood(
                requests, disable_tqdm=disable_tqdm, item_keys=item_keys
            )
        results: List[Optional[Tuple[float, bool]]] = [None] * len(requests)
        groups: Dict[Any, List[int]] = defaultdict(list)
        for position, request in enumerate(requests):
            groups[(request.task_name, request.doc_id)].append(position)

        # Every rollout instance gets its own checkpoint item (distinct from
        # the final per-choice `item_keys` this method receives), so that if
        # `generate_until()` below raises partway through, whichever docs'
        # rollouts *did* finish are durably on disk -- via the same per-batch
        # `mark_items_completed`/`mark_items_completed_async` writes
        # `generate_until`/`amodel_call` already perform for every other
        # backend -- instead of the exception losing every doc's work in this
        # dispatch, not just the one that failed. `repeat_ord` here must carry
        # the rollout's ordinal within its doc (not a constant), because
        # `generate_until`'s `INDEPENDENT_ROLLOUTS` seeding reads
        # `key.repeat_ord` as the per-sample seed offset when `item_keys` are
        # supplied; collapsing it to one value would make every rollout in a
        # doc sample with the same seed.
        checkpoint_active = bool(item_keys) and self.checkpoint_store is not None
        rollout_item_keys: List[ItemKey] = []

        # Build every doc's rollout prompts up front and dispatch them as a
        # single generate_until() call instead of one call per doc: all these
        # instances share the same gen_kwargs, so the Collator's
        # group_by="gen_kwargs" batches them together and get_batched_requests's
        # asyncio.Semaphore(num_concurrent) actually parallelizes across docs
        # (previously each doc's k-sample call ran in its own asyncio.run(),
        # serializing the whole benchmark whenever k was small, e.g. k=1
        # without avg@k).
        doc_plans: List[dict] = []
        pending_instances: List[Instance] = []
        for positions in groups.values():
            positions = sorted(positions, key=lambda p: requests[p].idx)
            group = [requests[p] for p in positions]
            contexts = {req.args[0] for req in group}
            if len(contexts) != 1:
                raise NotImplementedError("Generative choice scoring requires one shared context and no unconditional likelihood requests")

            # `Instance.repeats` (avg@k's k) duplicates each *choice* Instance
            # k times consecutively (see evaluator.py's `cloned_reqs`), so this
            # group is [choice_0 x k, choice_1 x k, ...] rather than one entry
            # per real choice -- cluster consecutive same-`idx` positions back
            # into the real, distinct choices before building the prompt.
            choice_positions: List[List[int]] = []
            choice_texts: List[str] = []
            last_idx = None
            for position, request in zip(positions, group):
                if last_idx is None or request.idx != last_idx:
                    choice_positions.append([])
                    choice_texts.append(request.args[1].strip())
                    last_idx = request.idx
                choice_positions[-1].append(position)

            context = group[0].args[0]
            choices = choice_texts
            if len({len(cg) for cg in choice_positions}) != 1 or len(choice_positions) < 2:
                raise ValueError("Generative choice scoring requires complete, equally repeated option groups")
            k = min(len(cg) for cg in choice_positions)
            letters = choice_letters(choices)
            # `context` may already be the harness's own rendering of the
            # chat template (whenever the eval config sets
            # apply_chat_template=true, `fewshot_context()` calls this same
            # class's `apply_chat_template()` before an Instance is ever
            # built -- see `_generation_prompt_cue`'s docstring). In that
            # case it already ends in the model's turn-start cue (e.g.
            # "Assistant: <think"), and the choice content must be spliced
            # in *before* that cue -- appending after it, or re-templating
            # the whole thing again, both land content inside what is
            # supposed to be the model's own completion.
            cue = self._generation_prompt_cue
            if isinstance(context, _RWKVPrompt) and context.messages is not None:
                from copy import deepcopy
                messages = deepcopy(context.messages)
                for message in reversed(messages):
                    if message["role"] == "user":
                        message["content"] = build_choice_prompt(message["content"], choices)
                        break
                else:
                    raise ValueError("Choice scoring requires a user turn")
                send_prompt = self.apply_chat_template(messages, add_generation_prompt=context.add_generation_prompt)
            elif cue and context.endswith(cue):
                body = context[: -len(cue)]
                send_prompt = build_choice_prompt(body, choices) + cue
            else:
                # `context` is bare, untemplated text (apply_chat_template
                # was false at the harness level, or no tokenizer is
                # configured at all): build the menu directly and, if a
                # chat template is available, render it exactly once so the
                # model still gets a real turn-start cue to answer from.
                menu_prompt = build_choice_prompt(context, choices)
                send_prompt = menu_prompt
                if self._chat_template_source is not None:
                    send_prompt = self.apply_chat_template(
                        [{"role": "user", "content": menu_prompt}]
                    )

            start = len(pending_instances)
            pending_instances.extend(self._rollout_instances(send_prompt, k))
            if checkpoint_active:
                doc_task_name = group[0].task_name
                for rollout_ord in range(k):
                    r_item_id = compute_item_id(
                        doc_task_name, group[0].doc_id, 0, "rwkv_rollout", rollout_ord
                    )
                    rollout_item_keys.append(
                        ItemKey(
                            item_id=r_item_id,
                            request_fingerprint=r_item_id,
                            idempotency_key=compute_idempotency_key(r_item_id, r_item_id),
                            task_name=doc_task_name,
                            doc_id=group[0].doc_id,
                            instance_idx=0,
                            request_type="rwkv_rollout",
                            repeat_ord=rollout_ord,
                        )
                    )
            doc_plans.append(
                {
                    "doc_id": group[0].doc_id,
                    "task_name": group[0].task_name,
                    "letters": letters,
                    "choices": choices,
                    "choice_positions": choice_positions,
                    "k": k,
                    "start": start,
                    "prompt": send_prompt,
                }
            )

        if checkpoint_active and rollout_item_keys:
            self.checkpoint_store.register_pending_items(rollout_item_keys)

        dispatch_error: Optional[BaseException] = None
        try:
            if not pending_instances:
                generations = []
            elif checkpoint_active:
                generations = self.generate_until(pending_instances, item_keys=rollout_item_keys)
            else:
                generations = self.generate_until(pending_instances)
        except BaseException as exc:  # noqa: BLE001
            if not checkpoint_active:
                raise
            # Recover whichever rollouts' per-batch checkpoint writes (the
            # same mechanism `generate_until`/`amodel_call` use for every
            # other backend) landed before the failure, so docs whose full
            # k-sample set already finished are still scored and persisted
            # below instead of the whole dispatch's progress being discarded.
            eval_logger.warning(
                "rwkv7-remote: generate_until failed mid-dispatch (%r); "
                "recovering already-completed rollouts from the checkpoint "
                "store and scoring only docs whose rollouts fully finished "
                "before the failure. The original exception is re-raised "
                "after persisting that partial progress.",
                exc,
            )
            dispatch_error = exc
            task_names = {plan["task_name"] for plan in doc_plans}
            items_by_id: Dict[str, dict] = {}
            for task_name_ in task_names:
                items_by_id.update(self.checkpoint_store.get_items(task_name_))
            generations = [None] * len(pending_instances)
            for position, key in enumerate(rollout_item_keys):
                record = items_by_id.get(key.item_id)
                if (
                    record is not None
                    and record.get("status") == "completed"
                    and record.get("result_json") is not None
                ):
                    generations[position] = deserialize_response(record["result_json"])

        for plan in doc_plans:
            doc_id = plan["doc_id"]
            letters = plan["letters"]
            choices = plan["choices"]
            k = plan["k"]
            doc_generations = generations[plan["start"] : plan["start"] + k]
            if any(generation is None for generation in doc_generations):
                # Only reachable when `generate_until` above failed partway
                # through dispatch: this doc's rollouts never finished (or
                # weren't recovered from the checkpoint store), so its
                # `results` positions stay `None` and it is left `pending` --
                # the next resume redispatches only this doc, not the whole
                # task.
                eval_logger.warning(
                    "rwkv7-remote: doc %r missing %d/%d rollouts after a "
                    "dispatch failure; leaving it pending for the next resume.",
                    doc_id,
                    sum(1 for generation in doc_generations if generation is None),
                    k,
                )
                continue
            raw_indices = [
                parse_choice_index(str(generated), letters, choices, require_think_close=self.rwkv_generation_prompt == "open_think")
                for generated in doc_generations
            ]
            parsed_flags = [index >= 0 for index in raw_indices]
            sample_encodings = rollout_encodings(raw_indices, len(choices))
            # Per-rollout record carried on every response for this doc, so
            # avg@k (and its truncation rate) can be recomputed from the
            # responses themselves -- including after a checkpoint resume,
            # where this doc's requests are restored rather than
            # re-generated and `_avg_k_samples` therefore has no entry.
            rollouts = rollout_evidence(doc_generations, raw_indices)
            self._avg_k_samples[doc_id] = sample_encodings
            self._avg_k_parsed[doc_id] = parsed_flags

            valid_indices = [index for index in raw_indices if index >= 0]
            doc_unparsed = not valid_indices
            if valid_indices:
                majority_index = Counter(valid_indices).most_common(1)[0][0]
            else:
                # No sample parsed means "no answer given", not "the model
                # chose option A" --
                # `majority_index` stays 0 only as a placeholder index, and
                # every outcome below is tagged `unparsed=True` so
                # task-specific `process_results` can reject it instead of
                # reading the placeholder as a genuine answer.
                majority_index = 0
                eval_logger.warning(
                    f"RWKV7Remote could not parse a choice letter from any of the "
                    f"{k} sampled answers for doc {doc_id!r}; marking this doc "
                    f"unparsed rather than defaulting to a choice."
                )
            # Attach the real generation the majority vote actually came from
            # (never a synthesized/averaged one) as evidence, so Scoreboard
            # publication reflects the genuine prompt/completion instead of
            # reconstructing a fake one from the score -- see
            # RWKVLoglikelihoodResponse.
            evidence_generation = next(
                (
                    generation
                    for generation, index in zip(doc_generations, raw_indices)
                    if index == majority_index
                ),
                doc_generations[0] if doc_generations else None,
            )
            parsed_answer = None if doc_unparsed else letters[majority_index]
            for choice_idx, cg in enumerate(plan["choice_positions"]):
                is_greedy = choice_idx == majority_index
                loglikelihood = 0.0 if is_greedy else -1e6
                if evidence_generation is not None:
                    # Prefer the prompt actually captured at dispatch time
                    # (`_annotate_generation`'s decode of the literal tokens/
                    # text sent) over `plan["prompt"]`, which is the
                    # pre-tokenization string built before `generate_until()`
                    # ever encoded and sent it -- the two should usually
                    # match, but only the dispatched-content decode is
                    # guaranteed to reflect what the model actually received.
                    outcome = RWKVLoglikelihoodResponse(
                        loglikelihood,
                        is_greedy,
                        prompt=getattr(evidence_generation, "prompt", None) or plan["prompt"],
                        text=str(evidence_generation),
                        raw_response=getattr(evidence_generation, "raw_response", None),
                        finish_reason=getattr(evidence_generation, "finish_reason", None),
                        output_token_ids=getattr(
                            evidence_generation, "output_token_ids", None
                        ),
                        truncated=bool(getattr(evidence_generation, "truncated", False)),
                        unparsed=doc_unparsed,
                        parsed_answer=parsed_answer,
                        rollouts=rollouts,
                        prompt_token_ids=getattr(evidence_generation, "prompt_token_ids", None),
                    )
                else:
                    outcome = RWKVLoglikelihoodResponse(
                        loglikelihood,
                        is_greedy,
                        unparsed=doc_unparsed,
                        parsed_answer=parsed_answer,
                        rollouts=rollouts,
                    )
                for position in cg:
                    results[position] = outcome

            # Persist this doc's outcome the moment it's scored, not after
            # every doc in the dispatch finishes: a later doc's generation
            # failure (handled above via `dispatch_error`) must not keep this
            # already-finished doc's result out of the checkpoint store.
            if item_keys and self.checkpoint_store is not None:
                doc_positions = [p for cg in plan["choice_positions"] for p in cg]
                doc_completed = [
                    (item_keys[p].item_id, serialize_response(results[p]))
                    for p in doc_positions
                    if item_keys[p] is not None
                ]
                if doc_completed:
                    self.checkpoint_store.mark_items_completed(doc_completed)

        if dispatch_error is not None:
            # Already-finished docs are persisted above; re-raise so the run
            # still surfaces the failure (crash-on-error, as before) instead
            # of silently returning partial/`None` results for the docs that
            # never finished.
            raise dispatch_error

        return results

    def parse_logprobs(self, outputs, tokens=None, ctxlens=None, **kwargs):
        """Sum the echoed continuation token logprobs, excluding generated tokens."""
        if not self.SUPPORTS_LOGLIKELIHOOD:
            return super().parse_logprobs(outputs, tokens=tokens, ctxlens=ctxlens, **kwargs)
        if tokens is None or ctxlens is None or len(tokens) != len(ctxlens):
            raise ValueError("NoCoT loglikelihood requires token boundaries")
        outputs = outputs if isinstance(outputs, list) else [outputs]
        choices = [
            (output, choice)
            for output in outputs
            for choice in sorted(output["choices"], key=lambda choice: choice["index"])
        ]
        if len(choices) != len(tokens):
            raise ValueError("NoCoT loglikelihood response count does not match requests")
        results = []
        for (output, choice), prompt_tokens, ctxlen in zip(choices, tokens, ctxlens, strict=True):
            if not 0 < ctxlen < len(prompt_tokens):
                raise ValueError("NoCoT loglikelihood requires a nonempty context and continuation")
            logprobs = choice.get("logprobs") or {}
            token_logprobs = logprobs.get("token_logprobs") or []
            top_logprobs = logprobs.get("top_logprobs") or []
            if len(token_logprobs) < len(prompt_tokens) or len(top_logprobs) < len(prompt_tokens):
                raise ValueError("NoCoT requires complete echo prompt logprobs; generation fallback is forbidden")
            continuation_logprobs = token_logprobs[ctxlen:len(prompt_tokens)]
            continuation_top = top_logprobs[ctxlen:len(prompt_tokens)]
            for value, top in zip(continuation_logprobs, continuation_top, strict=True):
                if (
                    type(value) not in (int, float) or not math.isfinite(value) or value > 0
                    or not isinstance(top, dict) or not top
                    or any(type(prob) not in (int, float) or not math.isfinite(prob) or prob > 0
                           for prob in top.values())
                    or value > max(top.values())
                ):
                    raise ValueError("NoCoT received invalid conditional token logprobs")
            prompt, continuation = self.tokenizer.batch_decode(
                [prompt_tokens, prompt_tokens[ctxlen:]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            results.append(RWKVLoglikelihoodResponse(
                sum(continuation_logprobs),
                all(value == max(top.values())
                    for value, top in zip(continuation_logprobs, continuation_top, strict=True)),
                prompt=prompt,
                text=continuation,
                raw_response=output,
                request_type="loglikelihood",
                prompt_token_ids=list(prompt_tokens),
            ))
        return results

    def _rollout_instances(self, prompt: str, k: int) -> List[Instance]:
        """Build ``k`` independent generate_until instances for one doc's prompt.

        Every doc's rollouts share this same ``gen_kwargs`` literal, so
        ``loglikelihood()`` can hand every doc's instances to a single
        ``generate_until`` call: the Collator's ``group_by="gen_kwargs"``
        batches them all together and the ``num_concurrent`` asyncio
        semaphore parallelizes across docs, not just within one doc's k
        samples.
        """
        gen_kwargs = {"until": [], "max_gen_toks": self._max_gen_toks}
        return [
            Instance(
                request_type="generate_until",
                doc={},
                arguments=(prompt, gen_kwargs),
                idx=i,
            )
            for i in range(k)
        ]

    def _resolve_choices(
        self, prompt: str, letters: List[str], choices: List[str], k: int
    ) -> List[int]:
        """Resolve ``k`` independent choice generations for one doc, standalone.

        Returns the raw parsed index per rollout, which may be ``-1`` if no
        choice letter could be extracted -- callers decide how to treat an
        unparseable rollout (e.g. excluding it from a majority vote) rather
        than this method silently guessing. ``loglikelihood()`` no longer
        calls this itself (it batches every doc's rollouts into one
        ``generate_until`` call instead); kept for any caller that needs a
        single doc resolved on its own.
        """
        generations = self.generate_until(self._rollout_instances(prompt, k))
        return [
            parse_choice_index(str(generated), letters, choices, require_think_close=self.rwkv_generation_prompt == "open_think") for generated in generations
        ]

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        if self._chat_template_source is None:
            raise ValueError("rwkv7-remote needs a tokenizer to render its chat template; set tokenizer_name")

        from lm_eval.utils import env

        template_generation_prompt = self.rwkv_generation_prompt
        if template_generation_prompt == "no_think":
            # Older official tokenizer templates reject ``no_think``. Render an
            # open-think cue and remove it, preserving the assistant turn prefix.
            template_generation_prompt = "open_think"
        template = env.from_string(self._chat_template_source)

        def raise_exception(message: str) -> None:
            raise TemplateError(message)

        rendered = template.render(
            messages=chat_history,
            add_generation_prompt=add_generation_prompt,
            rwkv_prompt_template=self.rwkv_prompt_template,
            rwkv_generation_prompt=template_generation_prompt,
            tools=kwargs.pop("tools", None),
            raise_exception=raise_exception,
            **kwargs,
        )
        if (
            add_generation_prompt
            and self.rwkv_generation_prompt == "no_think"
            and rendered.endswith("<think")
        ):
            rendered = rendered.removesuffix("<think")
        if (
            add_generation_prompt
            and self.rwkv_generation_prompt == "fake_think"
            and rendered.endswith("<think></think")
        ):
            rendered += ">\n"
        return _RWKVPrompt(rendered, chat_history, add_generation_prompt)

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        payload = super()._create_payload(
            messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
        )
        if not generate:
            return payload

        # vLLM reads sampling parameters from the request's top level.
        payload.update({key: self._generation_profile[key] for key in (
            "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay"
        )})
        payload["stop"] = list(dict.fromkeys([*(payload.get("stop") or []), self.PROMPT_STOPS[self.rwkv_prompt_template]]))

        return payload

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations = []
        for output in outputs:
            choices = [None] * len(output["choices"])
            for choice in output["choices"]:
                choices[choice["index"]] = _RWKVCompletion(
                    choice.get("text", ""),
                    choice.get("finish_reason"),
                    raw_response=output,
                )
            generations.extend(choices)
        return generations
