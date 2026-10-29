import json
import logging
import os
from collections import Counter, defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.checkpoint import ItemKey, serialize_response
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import (
    LocalChatCompletion,
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
from lm_eval.models.rwkv_remote import MAX_NEW_TOKENS, RWKV7Remote


eval_logger = logging.getLogger(__name__)


@register_model("rwkv7-remote-chat")
class RWKV7RemoteChat(LocalChatCompletion):
    """Adapter for RWKV's hosted ``/v1/chat/completions`` endpoint.

    Unlike ``rwkv7-remote`` (which targets ``/v1/completions`` and must render the
    RWKV chat template itself using a local tokenizer), this class sends unrendered
    ``messages`` plus ``chat_template_kwargs={"rwkv_prompt_template", "rwkv_generation_prompt"}``
    and lets the vLLM-backed server apply its own chat template -- mirroring
    lighteval-rwkv's ``RWKVHttpModel``. No local/HF-Hub tokenizer is required or
    accepted; the server has been confirmed (against api.rwkv.rs) to apply the
    matching per-template stop sequence (e.g. ``✿`` for ``bot``) on its own, so this
    adapter does not send an explicit ``stop`` for the RWKV template either.
    """

    SUPPORTS_AVG_AT_K = True
    INDEPENDENT_ROLLOUTS = True
    PRESERVE_INPUT_CONTEXT = True
    EVALUATION_PROTOCOL_VERSION = "rwkv-generative-v2"

    DEFAULT_BASE_URL = "https://api.rwkv.rs/v1/chat/completions"
    PROMPT_TEMPLATES = RWKV7Remote.PROMPT_TEMPLATES
    GENERATION_PROMPTS = RWKV7Remote.GENERATION_PROMPTS
    SAMPLING_PARAMETERS = RWKV7Remote.SAMPLING_PARAMETERS

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        rwkv_prompt_template="assistant",
        rwkv_generation_prompt="fake_think",
        cot_mode=None,
        **kwargs,
    ):
        if cot_mode is not None:
            rwkv_generation_prompt = cot_mode
        if rwkv_prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError("rwkv_prompt_template must be assistant, bot, or function_calling.")
        if rwkv_generation_prompt not in self.GENERATION_PROMPTS:
            raise ValueError("rwkv_generation_prompt must be open_think or fake_think.")

        kwargs.setdefault("max_gen_toks", MAX_NEW_TOKENS)
        self.rwkv_prompt_template = rwkv_prompt_template
        self.rwkv_generation_prompt = rwkv_generation_prompt
        super().__init__(
            base_url=base_url,
            tokenizer_backend=None,
            tokenized_requests=False,
            **kwargs,
        )

        self.max_length = int(kwargs.get("max_length", 2048))
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

    @cached_property
    def api_key(self) -> str:
        key = os.environ.get("RWKV_API_KEY")
        if not key:
            raise ValueError(
                "API key not found. Please set the `RWKV_API_KEY` environment variable."
            )
        return key

    @property
    def tokenizer_name(self) -> str:
        return f"{self.model}:{self.rwkv_prompt_template}:{self.rwkv_generation_prompt}"

    def get_model_info(self) -> dict:
        return {"sampling_config": _rwkv_sampling_config(self._generation_profile)}

    def loglikelihood(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
        item_keys: Optional[List[Optional[ItemKey]]] = None,
    ) -> List[Tuple[float, bool]]:
        """Score ``multiple_choice`` docs via avg@k generation (see ``RWKV7Remote.loglikelihood``).

        Identical strategy to ``RWKV7Remote``: this backend has no echo/logprob
        path to fall back on (chat-completions APIs don't expose prompt logprobs),
        so every loglikelihood request is resolved by generating a lettered answer
        ``Instance.repeats`` times per doc and averaging correctness over them.
        """
        results: List[Optional[Tuple[float, bool]]] = [None] * len(requests)
        groups: Dict[Any, List[int]] = defaultdict(list)
        for position, request in enumerate(requests):
            groups[(request.task_name, request.doc_id)].append(position)

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
            # Captured per-plan (never as an outer-scope loop variable) so
            # the second loop below can't leak one doc's prompt onto
            # another doc's evidence -- see RWKV7Remote.loglikelihood for
            # the same fix and its rationale.
            prompt = self._choice_prompt(context, choices)

            start = len(pending_instances)
            pending_instances.extend(self._rollout_instances(prompt, k))
            doc_plans.append(
                {
                    "doc_id": group[0].doc_id,
                    "letters": letters,
                    "choices": choices,
                    "choice_positions": choice_positions,
                    "k": k,
                    "start": start,
                    "prompt": prompt,
                }
            )

        generations = self.generate_until(pending_instances) if pending_instances else []

        for plan in doc_plans:
            doc_id = plan["doc_id"]
            letters = plan["letters"]
            choices = plan["choices"]
            k = plan["k"]
            doc_generations = generations[plan["start"] : plan["start"] + k]
            raw_indices = [
                parse_choice_index(str(generated), letters, choices, require_think_close=getattr(self, "rwkv_generation_prompt", None) == "open_think")
                for generated in doc_generations
            ]
            parsed_flags = [index >= 0 for index in raw_indices]
            sample_encodings = rollout_encodings(raw_indices, len(choices))
            # See RWKV7Remote.loglikelihood: this per-rollout record is what
            # keeps avg@k and its truncation rate complete across a
            # checkpoint resume.
            rollouts = rollout_evidence(doc_generations, raw_indices)
            self._avg_k_samples[doc_id] = sample_encodings
            self._avg_k_parsed[doc_id] = parsed_flags

            valid_indices = [index for index in raw_indices if index >= 0]
            doc_unparsed = not valid_indices
            if valid_indices:
                majority_index = Counter(valid_indices).most_common(1)[0][0]
            else:
                # None of the k samples could be parsed into a choice letter
                # -- this is "no answer given", not "the model chose option
                # A". Encoding it as a confident pick of the first choice
                # would silently corrupt downstream metrics (e.g.
                # discrim_eval's bias-diff aggregations) with a fake,
                # systematically-biased answer. `majority_index` below is
                # kept only as an arbitrary placeholder index to satisfy the
                # (loglikelihood, is_greedy)-per-choice shape the
                # `multiple_choice` pipeline expects; every outcome for this
                # doc is tagged `unparsed=True` so task-specific
                # `process_results` can recognize and exclude it instead of
                # reading the placeholder as a genuine answer.
                majority_index = 0
                eval_logger.warning(
                    f"RWKV7RemoteChat could not parse a choice letter from any of "
                    f"the {k} sampled answers for doc {doc_id!r}; marking this doc "
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
                    outcome = RWKVLoglikelihoodResponse(
                        loglikelihood,
                        is_greedy,
                        prompt=(
                            plan["prompt"].prompt
                            if isinstance(plan["prompt"], JsonChatStr)
                            else plan["prompt"]
                        ),
                        text=str(evidence_generation),
                        raw_response=getattr(evidence_generation, "raw_response", None),
                        finish_reason=getattr(
                            evidence_generation, "finish_reason", None
                        ),
                        output_token_ids=getattr(
                            evidence_generation, "output_token_ids", None
                        ),
                        truncated=bool(
                            getattr(evidence_generation, "truncated", False)
                        ),
                        unparsed=doc_unparsed,
                        parsed_answer=parsed_answer,
                        rollouts=rollouts,
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

        if item_keys and self.checkpoint_store is not None:
            completed = [
                (key.item_id, serialize_response(res))
                for key, res in zip(item_keys, results)
                if key is not None and res is not None
            ]
            if completed:
                self.checkpoint_store.mark_items_completed(completed)

        return results

    @staticmethod
    def _choice_prompt(context, choices):
        if not isinstance(context, JsonChatStr):
            return build_choice_prompt(context, choices)
        messages = json.loads(context.prompt)
        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] = build_choice_prompt(message["content"], choices)
                break
        else:
            raise ValueError("generative multiple choice requires a user message")
        return JsonChatStr(json.dumps(messages, ensure_ascii=False))

    def _rollout_instances(self, prompt: str | JsonChatStr, k: int) -> list[Instance]:
        """Build ``k`` independent generate_until instances for one doc's prompt.

        Every doc's rollouts share this same ``gen_kwargs`` literal, so
        ``loglikelihood()`` can hand every doc's instances to a single
        ``generate_until`` call: the Collator's ``group_by="gen_kwargs"``
        batches them all together and the ``num_concurrent`` asyncio
        semaphore parallelizes across docs, not just within one doc's k
        samples.
        """
        gen_kwargs = {"until": [], "max_gen_toks": self._max_gen_toks}
        # `generate_until` unwraps a `JsonChatStr` into the [{"role","content"}, ...]
        # list this class's `_create_payload` requires; a bare string (fine for the
        # raw-completions `rwkv7-remote`) would fail LocalChatCompletion's assertion.
        chat_prompt = (
            prompt
            if isinstance(prompt, JsonChatStr)
            else JsonChatStr(
                json.dumps([{"role": "user", "content": prompt}], ensure_ascii=False)
            )
        )
        return [
            Instance(
                request_type="generate_until",
                doc={},
                arguments=(chat_prompt, gen_kwargs),
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
        unparseable rollout rather than this method silently guessing.
        ``loglikelihood()`` no longer calls this itself (it batches every
        doc's rollouts into one ``generate_until`` call instead); kept for
        any caller that needs a single doc resolved on its own.
        """
        generations = self.generate_until(self._rollout_instances(prompt, k))
        return [
            parse_choice_index(str(generated), letters, choices, require_think_close=getattr(self, "rwkv_generation_prompt", None) == "open_think") for generated in generations
        ]

    def _create_payload(
        self,
        messages: Union[str, List[Dict]],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        # generate_until only produces a JsonChatStr (-> list[dict]) when the
        # caller renders one via apply_chat_template/_resolve_choice; plain
        # generate_until tasks with apply_chat_template=false hand create_message
        # a bare string instead (see TemplateAPI.create_message). The server-side
        # template needs a message list regardless, so wrap it as a single user
        # turn ourselves rather than requiring every caller to opt into chat
        # templating just to use this transport.
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        payload = super()._create_payload(
            messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
        )
        if not generate:
            return payload

        payload.update({key: self._generation_profile[key] for key in (
            "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay"
        )})
        payload["chat_template_kwargs"] = {
            "rwkv_prompt_template": self.rwkv_prompt_template,
            "rwkv_generation_prompt": self.rwkv_generation_prompt,
        }

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
                    choice.get("message", {}).get("content", ""),
                    choice.get("finish_reason"),
                    raw_response=output,
                )
            generations.extend(choices)
        return generations
