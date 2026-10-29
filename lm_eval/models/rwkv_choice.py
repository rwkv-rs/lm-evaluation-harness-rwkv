"""Shared "loglikelihood-via-generation" helpers for RWKV backends without
native echo/logprob support (or, for avg@k, that deliberately route
``multiple_choice`` scoring through generation to exercise real sampling).

Extracted from the original ``RWKV7HTTP.loglikelihood``/``_resolve_choice``
implementation so ``RWKV7Remote`` can reuse the same lettered-choice prompt
and answer-parsing logic for its avg@k mode.

The answer-extraction cascade below mirrors the approach used by RWKV's own
GPQA-Diamond generative-scoring eval (which separates a "final answer"
segment from the model's chain-of-thought, tries several tiers of
progressively looser regexes against it, and falls back to the
chain-of-thought text itself before giving up): letter mentioned earlier in
a long chain-of-thought must not shadow the letter the model actually
settles on at the end.
"""

import re
from typing import Any, List, Optional, Tuple



class RWKVLoglikelihoodResponse(tuple):
    """A ``(loglikelihood, is_greedy)`` 2-tuple that also carries the real
    generation this score was derived from.

    ``RWKV7Remote``/``RWKV7RemoteChat`` resolve ``loglikelihood`` requests by
    generating a real answer (``generate_until``) rather than echo/logprobs,
    so a genuine ``prompt`` (the ``build_choice_prompt()`` text actually
    sent), ``text`` (the chosen generation's own completion string), and
    ``raw_response``/``finish_reason``/``output_token_ids`` exist for every
    score -- unlike a bare ``(ll, is_greedy)`` tuple, this type keeps that
    evidence attached all the way through to Scoreboard publication instead
    of forcing downstream code to reconstruct a fake prompt/completion from
    the score. ``text``/``raw_response``/``finish_reason``/
    ``output_token_ids``/``truncated`` are picked up by ``evaluator.py``'s
    ``response_evidence()`` via ``getattr(response, ..., None)``; ``prompt``
    is a Scoreboard-specific addition consumed directly (see below).

    Subclasses ``tuple`` (rather than adding extra ``NamedTuple`` fields) so
    it unpacks, ``zip()``s, and ``isinstance(x, tuple)``-checks exactly like
    a bare 2-tuple everywhere the harness already expects one (e.g.
    ``task.py``'s ``ll, is_greedy = results`` and
    ``zip(*results, strict=True)``, ``checkpoint.py``'s
    ``isinstance(response, tuple)``) -- no existing consumer needs to
    change. The evidence lives in extra instance attributes, mirroring how
    ``_RWKVCompletion`` carries ``raw_response`` alongside its ``str``
    payload for ``generate_until``.
    """

    def __new__(
        cls,
        loglikelihood: float,
        is_greedy: bool,
        *,
        prompt: Optional[str] = None,
        text: Optional[str] = None,
        raw_response: Optional[Any] = None,
        finish_reason: Optional[str] = None,
        output_token_ids: Optional[List[int]] = None,
        truncated: bool = False,
        unparsed: bool = False,
        parsed_answer: Optional[str] = None,
        rollouts: Optional[List[dict]] = None,
    ) -> "RWKVLoglikelihoodResponse":
        self = super().__new__(cls, (loglikelihood, is_greedy))
        self.prompt = prompt
        # The literal text of the generation this score was derived from
        # (``str(evidence_generation)``) -- unlike ``raw_response``, which is
        # the *entire* HTTP response body and, whenever more than one prompt
        # was batched into that one call, holds every batched completion's
        # ``choices`` entries indiscriminately. `response_evidence()` must
        # read completion text from here (already matched to the right
        # batch index by `parse_generations()`) rather than re-deriving it
        # from `raw_response["choices"][0]`, or it silently attributes a
        # different request's completion to this one under batch_size > 1.
        self.text = text
        self.raw_response = raw_response
        self.finish_reason = finish_reason
        self.output_token_ids = output_token_ids
        self.truncated = truncated
        # True when none of the k sampled generations for this doc could be
        # parsed into a choice letter, so `is_greedy` here is an arbitrary
        # placeholder rather than a genuine reflection of what the model
        # answered -- task-specific `process_results` should treat this doc
        # as "no answer given" (excluded from scoring) rather than reading
        # the placeholder choice as if the model had actually named it.
        self.unparsed = unparsed
        # The choice letter (e.g. "A") that `parse_choice_index()` actually
        # extracted from the model's real generation, or `None` when
        # `unparsed`. Scoreboard publication surfaces this as `answer`
        # instead of the fabricated `(loglikelihood, is_greedy)` tuple this
        # type otherwise looks like.
        self.parsed_answer = parsed_answer
        # One entry per independently-sampled rollout of this doc
        # (``{"index": <chosen choice index, -1 when unparsed>,
        # "truncated": <hit the output cap>}``), in generation order.
        # The k rollouts themselves are otherwise unobservable once the
        # request returns -- only the majority vote survives in `text`/
        # `parsed_answer` -- so this is what lets avg@k be recomputed from
        # the responses alone: after a checkpoint resume (where a doc's
        # requests are restored instead of re-generated)
        # `RWKV7Remote._avg_k_samples` has no in-memory entry for it, and
        # without this record the doc would silently drop out of the
        # benchmark's ``<metric>_avg@k`` average and its truncation rate.
        # Kept out of the (ll, is_greedy) tuple payload so every existing
        # consumer keeps unpacking/zipping it unchanged.
        self.rollouts = rollouts
        return self


TRUNCATED_FINISH_REASONS = {"length", "max_tokens", "model_length"}


def rollout_encodings(
    indices: List[int], num_choices: int
) -> List[List[Tuple[float, bool]]]:
    """Per-rollout ``(loglikelihood, is_greedy)`` rows for one doc.

    Each rollout names exactly one choice (``index``, or -1 when no letter was
    parsed); the task's own ``process_results`` is invoked once per row, so a
    row is the "one sample" avg@k averages correctness over.
    """
    return [
        [
            (0.0, True)
            if choice_idx == (index if index >= 0 else 0)
            else (-1e6, False)
            for choice_idx in range(num_choices)
        ]
        for index in indices
    ]


def rollout_evidence(generations: List[Any], indices: List[int]) -> List[dict]:
    """Checkpoint-safe per-rollout record for one doc's generations."""
    return [
        {
            "index": int(index),
            "text": str(generation),
            "seed": getattr(generation, "seed", None),
            "finish_reason": getattr(generation, "finish_reason", None),
            "raw_response": getattr(generation, "raw_response", None),
            "truncated": bool(getattr(generation, "truncated", False))
            or str(getattr(generation, "finish_reason", "") or "").lower()
            in TRUNCATED_FINISH_REASONS,
        }
        for generation, index in zip(generations, indices)
    ]


def rollout_indices(responses: List[Any]) -> Optional[List[int]]:
    """The per-rollout choice indices carried by a doc's scored responses.

    Every response for one doc shares the same record, so the first response
    that carries one is authoritative; ``None`` means the response predates
    this record (or came from another backend), leaving the caller to fall
    back to whatever in-memory recording it has.
    """
    for response in responses:
        rollouts = getattr(response, "rollouts", None)
        if rollouts:
            return [int(rollout.get("index", -1)) for rollout in rollouts]
    return None


def rollout_truncation(responses: List[Any]) -> Optional[Tuple[int, int]]:
    """``(truncated_rollouts, total_rollouts)`` for a doc, or ``None``."""
    for response in responses:
        rollouts = getattr(response, "rollouts", None)
        if rollouts:
            return (
                sum(1 for rollout in rollouts if rollout.get("truncated")),
                len(rollouts),
            )
    return None


def build_choice_prompt(context: str, choices: List[str]) -> str:
    """Render a lettered multiple-choice prompt for generative resolution.

    ``context`` is the echo-style loglikelihood context, which for tasks
    defining a ``gen_prefix`` (e.g. ``discrim_eval``) already ends mid-sentence
    (a dangling open quote meant to be continued directly by one of
    ``choices``). Appending a fresh lettered menu after that produces a
    malformed prompt -- the model is left with an unclosed quote immediately
    followed by an unrelated instruction block -- so in that case each
    ``context + choice`` is rendered as its own complete option instead.

    Some tasks (e.g. ``wmdp``, and the common lm-eval-harness MMLU-style
    ``doc_to_choice: ["A", "B", "C", "D"]`` convention) already bake the full
    "A. ...\\nB. ..." option list into ``context`` via ``doc_to_text``, and
    ``choices`` here is just those same placeholder letters -- the real
    option text was never a per-Instance string to begin with (native
    loglikelihood scoring only needs to compare " A" vs " B" vs ... as
    continuations). Re-listing ``choices`` in that case would produce a
    second, content-free "A. A\\nB. B\\n..." menu stacked on top of the real
    one already in ``context``, so only the answer instruction is appended.
    """
    letters = choice_letters(choices)
    if _choices_are_placeholder_letters(choices, letters):
        return f"{context}\n\nAnswer with only the letter of the correct option."
    if _expects_direct_continuation(context):
        options = "\n".join(
            f"{letter}. {context}{choice}" for letter, choice in zip(letters, choices)
        )
        return (
            f"Which of the following best completes the passage below?\n\n"
            f"{options}\n\nAnswer with only the letter of the correct option."
        )
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    return f"{context}\n\n{options}\n\nAnswer with only the letter of the correct option."


def _choices_are_placeholder_letters(choices: List[str], letters: List[str]) -> bool:
    """Whether ``choices`` are themselves just the letter labels ("A", "B",
    ...) rather than real option content -- see ``build_choice_prompt``."""
    return [choice.strip().upper() for choice in choices] == letters


def _expects_direct_continuation(context: str) -> bool:
    """Whether ``context`` ends mid-sentence (e.g. a dangling open quote left
    by a task's ``gen_prefix``) such that a choice is meant to continue it
    directly, rather than ``context`` being a complete, freestanding
    question ready for a fresh instruction to follow."""
    stripped = context.rstrip()
    if not stripped:
        return False
    return stripped[-1] in "\"'"


def choice_letters(choices: List[str]) -> List[str]:
    return [chr(ord("A") + i) for i in range(len(choices))]


def parse_choice_index(
    generated: str, letters: List[str], choices: List[str], *, require_think_close: bool = False
) -> int:
    """Use the same final-answer extractor as native generation tasks."""
    from lm_eval.filters.extraction import extract_albatross_mcq_answer

    answer = extract_albatross_mcq_answer(
        generated, choice_labels="".join(letters), require_think_close=require_think_close
    )
    if answer is not None:
        return letters.index(answer)
    # Text-answer fallback is only safe for substantive option texts, in the
    # final segment, with exactly one option mentioned. Single-letter matches
    # in words such as "final" must never turn an invalid answer into A.
    _, closed, final = generated.rpartition("</think>")
    if (require_think_close and not closed) or (not closed and "<think" in generated):
        return -1
    final = (final if closed else generated).strip().casefold()
    if not final:
        return -1
    matches = [index for index, choice in enumerate(choices)
               if len(choice.strip()) > 1 and re.search(
                   r"(?<!\w)" + re.escape(choice.strip().casefold()) + r"(?!\w)", final)]
    return matches[0] if len(matches) == 1 else -1
