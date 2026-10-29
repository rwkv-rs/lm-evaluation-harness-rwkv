"""Publish one completed native lm-eval benchmark to Scoreboard."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lm_eval.api.benchmark import BENCHMARK_FIELDS


if TYPE_CHECKING:
    from collections.abc import Mapping


eval_logger = logging.getLogger(__name__)


SOURCE = "lm-eval-harness"
MAX_SAMPLES_PER_OUTCOME = 20
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
HTTP_STATUS_UNPROCESSABLE = 422
COT_MODES = {"NoCoT", "FakeCoT", "CoT"}
ARCH_VERSIONS = {"rwkv7", "rwkv7a", "rwkv7b"}
PARAM_SIZES = {"1.5b", "2.9b", "7.2b", "13.3b"}
# lm-eval-rwkv model names encode arch/data/param version as
# "<arch_version>-<data_version>-<param_size>[-...]", e.g.
# "rwkv7-g1j-1.5b" or "rwkv7-g1i-1.5b-20260805-ctx16384".
MODEL_NAME_PATTERN = re.compile(r"^([a-z0-9]+)-([a-z][0-9][a-z0-9]*)-([0-9.]+[a-z]+)(?:-.*)?$")
# Weight releases name their context window, e.g.
# "rwkv7-g1k-13.3b-20260930-ctx25600"; the run directory keeps that revision.
WEIGHT_CTX_PATTERN = re.compile(r"ctx(\d+)")
DEFAULT_MAX_GENERATED_TOKENS = 8192
DEFAULT_SEED = 1234
NEW_SAMPLING_FIELDS = {
    "temperature": "temp",
    "top_p": "top_p",
    "top_k": "top_k",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "penalty_decay": "penalty_decay",
    "seed": "seed",
}
REQUIRED_SAMPLING_FIELDS = (
    "temp",
    "top_k",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "penalty_decay",
)
# Run statistics written next to a task's metrics (the same four fields
# lighteval-rwkv's results JSON carries). They describe how much was sampled,
# not how well the model did, so publication must never pick one as the
# headline metric nor let one leak into the benchmark-level aggregation.
RESULT_BOOKKEEPING_FIELDS = {
    "sample_len",
    "n_samples",
    "n_completions",
    "n_truncated",
    "truncation_rate",
}
# avg@k surfaces its average under a k-suffixed copy of the task's own metric
# name (see `lm_eval.evaluator._install_avg_at_k_patch`).
AVG_AT_K_PATTERN = re.compile(r"^(?P<base>.+)_avg@(?P<k>\d+)$")


class ScoreboardError(RuntimeError):
    """Raised when Scoreboard publication fails."""


def _json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    return _json(value).decode("utf-8")


def _task_policy(
    config: PublicationConfig, task_name: str, benchmark: str
) -> dict[str, Any]:
    policy = config.tasks.get(task_name, config.tasks.get(benchmark, {}))
    return policy if isinstance(policy, dict) else {}


def _messages(
    arguments: Any, index: int = 0, evidence_prompt: Any = None
) -> list[dict[str, str]]:
    r"""The OpenAI-style chat messages actually sent for one request.

    ``evidence_prompt`` -- when it's a non-empty string -- is
    ``RWKVLoglikelihoodResponse.prompt`` surfaced via
    ``response_evidence[0]["prompt"]``: the genuine prompt captured at the
    moment the request was actually issued. It takes priority over anything
    reconstructed from ``arguments`` because RWKV's avg@k generative
    ``multiple_choice`` path (``rwkv_remote.py``/``rwkv_remote_chat.py``)
    fills in the lettered `A./B./C.` options -- and, for the completions
    backend, chat-template markup -- *after* the original loglikelihood
    request's ``arguments`` was already captured, so ``arguments`` alone is
    missing content the model genuinely saw (it's just the bare echo-style
    context). Only a plain generate_until doc (no avg@k choice resolution)
    leaves ``evidence_prompt`` unset, in which case this falls back to the
    old reconstruction below.

    A chat request's `arg_0` is a `JsonChatStr` NamedTuple (see
    `lm_eval.models.api_models.JsonChatStr`), which round-trips through
    `evaluation_tracker.save_results_samples` as a one-element list holding
    a JSON-encoded message list -- e.g.
    `["[{\\"role\\": \\"user\\", \\"content\\": \\"...\\"}]"]`. Anything
    that doesn't match that shape (a bare prompt string) falls back to a
    single synthesized user message.
    """
    if isinstance(evidence_prompt, str) and evidence_prompt.strip():
        try:
            parsed = json.loads(evidence_prompt)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list) and parsed and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in parsed
        ):
            return [
                {"role": str(item["role"]), "content": _text(item["content"])}
                for item in parsed
            ]
        return [{"role": "user", "content": evidence_prompt}]

    entry = arguments.get(f"gen_args_{index}") if isinstance(arguments, dict) else None
    arg0 = entry.get("arg_0") if isinstance(entry, dict) else None
    if isinstance(arg0, list) and len(arg0) == 1 and isinstance(arg0[0], str):
        try:
            parsed = json.loads(arg0[0])
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list) and parsed and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in parsed
        ):
            return [
                {"role": str(item["role"]), "content": _text(item["content"])}
                for item in parsed
            ]
    return [{"role": "user", "content": _text(arg0)}]


def _ground_truth(row: dict[str, Any], lettered: bool) -> str:
    """The gold answer, in the same representation as ``answer``.

    ``multiple_choice`` tasks store ``target`` as a raw numeric choice
    index (lm-eval-harness's native gold format) -- but when the doc went
    through RWKV's avg@k lettered-generation path, ``answer`` is a letter
    (``A``/``B``/...), not an index (see ``_detail``'s ``parsed_answer``
    handling). Showing "2" as ground truth next to "A" as the answer is
    two different representations of the same thing, so here that index
    is mapped into the identical letter space `choice_letters()` used to
    build the answer: choice count comes from ``arguments``'s
    ``gen_args_N`` entries, which are populated one per choice in the
    same index order the RWKV backends iterate to assign letters.
    """
    target = row.get("target")
    if not lettered:
        return _text(target)
    arguments = row.get("arguments") or {}
    n_choices = sum(
        1 for key in arguments if isinstance(key, str) and key.startswith("gen_args_")
    )
    try:
        index = int(target)
    except (TypeError, ValueError):
        return _text(target)
    if not (0 <= index < n_choices):
        return _text(target)
    return chr(ord("A") + index)


def _extracted_answer(row: dict[str, Any]) -> str | None:
    """The `filter_list` pipeline's extracted answer for this doc (e.g.
    the "yes"/"no"/letter a `yes_no`/`albatross_mcq` filter resolves a
    chain-of-thought completion down to), or ``None`` if the row carries
    no `filtered_resps` at all. Mirrors the `results[0][0]` shape task
    `process_results` functions read (one filtered string per repeat;
    `[0]` is the first/only repeat for `repeats: 1` tasks).
    """
    filtered = row.get("filtered_resps")
    if not isinstance(filtered, list) or not filtered:
        return None
    first = filtered[0]
    if not isinstance(first, list) or not first:
        return None
    return _text(first[0])


def _detail(row: dict[str, Any], metric_name: str) -> tuple[dict[str, Any], str, bool]:
    evidence = row.get("response_evidence") or []
    first_evidence = evidence[0] if evidence else {}
    raw_completion = _text(first_evidence.get("text") or row["resps"])
    score = row.get(metric_name)
    scalar_score = (
        float(score) if isinstance(score, (int, float)) else None
    )
    extracted = _extracted_answer(row)
    # `extracted is None` means the task has no `filter_list` at all (not
    # that filtering failed) -- the raw completion itself is already "the
    # answer" there (see the `answer` field logic below), so judge
    # usability from it directly rather than treating a legitimately
    # unfiltered task as if nothing could be extracted.
    has_usable_answer = (
        bool(raw_completion.strip()) if extracted is None else extracted != "[invalid]"
    )
    if not raw_completion.strip() or first_evidence.get("unparsed") is True or extracted == "[invalid]":
        outcome = "failed"
    elif scalar_score == 1:
        outcome = "passed"
    elif scalar_score is not None and 0 <= scalar_score < 1:
        outcome = "wrong"
    elif scalar_score is None:
        # Some tasks (e.g. discrim_eval's bias metrics) score each doc with
        # an aggregation input -- a tuple like `(demographics, bias_name,
        # chose_yes)` -- rather than a binary correct/incorrect value: there
        # is no "wrong answer" concept to apply (any yes/no is a legitimate
        # answer; the benchmark measures the *distribution* across
        # demographics, not per-doc correctness). Defaulting this case to
        # "failed" (the prior behavior) mislabeled every successfully
        # answered doc as a failure; the only real failure here is not
        # getting a usable answer out of the completion at all.
        outcome = "passed" if has_usable_answer else "failed"
    else:
        outcome = "failed"
    stop_reason = str(first_evidence.get("finish_reason") or "unknown")
    truncated = first_evidence.get("truncated") is True or stop_reason in {
        "length",
        "max_tokens",
        "model_length",
    }
    messages = _messages(
        row.get("arguments") or {}, evidence_prompt=first_evidence.get("prompt")
    )
    messages.append({"role": "assistant", "content": raw_completion})
    # For RWKV's avg@k generative multiple_choice path, `filtered_resps`/
    # `resps` hold a fabricated `(loglikelihood, is_greedy)` tuple rather
    # than real model text -- `parsed_answer` (the choice letter
    # `parse_choice_index()` actually extracted from the real generation,
    # see RWKVLoglikelihoodResponse) is the genuine answer to surface here.
    # A missing `parsed_answer` key means this doc wasn't scored through
    # that path at all.
    lettered = "parsed_answer" in first_evidence
    if lettered:
        answer = first_evidence.get("parsed_answer") or "-"
    else:
        # Plain `generate_until` tasks with a `filter_list` (e.g. the
        # `yes_no`/`albatross_mcq` filters) reduce a long chain-of-thought
        # completion down to the clean value actually used for scoring --
        # that distilled value belongs here, not the raw, unfiltered
        # completion the filter exists to cut through. Only fall back to
        # the raw completion when no filter extracted anything (e.g. a
        # task with no filter_list at all, where the full completion *is*
        # the answer).
        extracted = _extracted_answer(row)
        if extracted is None:
            answer = raw_completion
        elif extracted == "[invalid]":
            answer = "-"
        else:
            answer = extracted
    detail = {
        "messages": messages,
        "answer": answer,
        "ground_truth": _ground_truth(row, lettered),
        "is_passed": outcome == "passed",
    }
    return detail, outcome, truncated


@dataclass(frozen=True, slots=True)
class _SampleEvidence:
    """One leaf's per-sample evidence, as read from its samples jsonl."""

    buckets: dict[str, list[dict[str, Any]]]
    # Docs actually scored by this leaf -- the count the board reports as the
    # benchmark's `num_samples`, and the denominator of its avg_k.
    docs: int
    # Completions the leaf actually sampled: one per sample the run took per
    # doc, summed over the docs it scored. This is the numerator of the
    # published avg_k, and the only honest source for it when an artifact
    # records neither its own `n_completions` nor a `<metric>_avg@k` key.
    completions: int
    # Doc-level truncation, i.e. the fraction of scored docs whose reported
    # completion hit the output cap. Only a fallback: a run that counted its
    # own completions records a per-completion `truncation_rate` in the
    # results instead (see `_leaf_upload`).
    truncation_rate: float


def _completion_count(row: dict[str, Any]) -> int:
    """How many completions one doc was scored with.

    A sample row's ``resps`` is ``[filter][completion]``: the inner list holds
    one entry per sample the run took for that doc (see
    `EvaluationTracker.save_results_samples`), so its length is what the doc
    contributes to the leaf's completion count.
    """
    resps = row.get("resps")
    if isinstance(resps, list) and resps and isinstance(resps[0], list):
        return len(resps[0]) or 1
    return 1


def _details(
    path: Path,
    metric_name: str,
    uploaded_counts: dict[str, int],
    filter_key: str | None = None,
) -> _SampleEvidence:
    """Read one leaf's per-sample details, capped at `MAX_SAMPLES_PER_OUTCOME`.

    `uploaded_counts` tracks how many samples of each outcome have already
    been uploaded for the *benchmark* this leaf belongs to. Callers building
    a multi-leaf benchmark publication must share one `uploaded_counts` dict
    across all `_details` calls for that benchmark so the cap applies to the
    benchmark as a whole rather than being re-applied per leaf (which would
    let an N-leaf benchmark upload up to `N * MAX_SAMPLES_PER_OUTCOME`
    samples per outcome).

    One line is written per (doc, filter), so a task with more than one
    filter would otherwise be counted (and detailed) twice; docs are
    deduplicated by `doc_id`, which is also what makes the doc count here a
    trustworthy stand-in for artifacts that predate the recorded counts.
    """
    buckets: dict[str, list[dict[str, Any]]] = {"passed": [], "wrong": [], "failed": []}
    docs = truncated = completions = 0
    seen: set[Any] = set()
    with path.open(encoding="utf-8") as sample_file:
        for line in sample_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if filter_key is not None and row.get("filter", filter_key) != filter_key:
                continue
            doc_id = row.get("doc_id")
            if doc_id is not None:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
            detail, outcome, is_truncated = _detail(row, metric_name)
            docs += 1
            completions += _completion_count(row)
            truncated += is_truncated
            if uploaded_counts[outcome] >= MAX_SAMPLES_PER_OUTCOME:
                continue
            uploaded_counts[outcome] += 1
            buckets[outcome].append(detail)
    return _SampleEvidence(
        buckets=buckets,
        docs=docs,
        completions=completions,
        truncation_rate=(truncated / docs if docs else 0.0),
    )


def _leaf_dataset_identity(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """The dataset slice a leaf reads.

    Two leaves that draw on the same slice can be nested -- one may score a
    filtered subset of the other's documents (winogender's `winogender_all`
    task and its `winogender_female`/`_male`/`_neutral` filters). Leaves on
    different slices can never overlap, so they are never compared.
    """
    return (
        config.get("dataset_path"),
        config.get("dataset_name"),
        config.get("test_split"),
    )


def _sample_document_hashes(path: Path) -> frozenset[Any] | None:
    """The documents one leaf scored, identified by content hash.

    Read for nesting detection only, which is why it walks the whole samples
    file instead of stopping like `_details` does: a leaf's documents are a
    superset of a sibling's only if every one of the sibling's documents
    appears here.

    ``doc_hash`` is the document's content, so two leaves that read different
    datasets can never share it. ``doc_id`` is *not* usable as a fallback --
    it is a per-dataset index, and two unrelated tasks both numbering their
    docs from 0 would look nested. Rows written before `doc_hash` existed
    therefore yield ``None``: nesting is undecidable, not absent.
    """
    hashes: set[Any] = set()
    with path.open(encoding="utf-8") as sample_file:
        for line in sample_file:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("doc_hash")
            if value is None:
                return None
            hashes.add(value)
    return frozenset(hashes)


def _parent_leaves(
    leaf_docs: dict[str, dict[str, Any]],
    leaf_artifact_paths: dict[str, dict[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Leaves that only re-score documents a sibling leaf already covers.

    A group may list a task together with filtered versions of itself (the
    winogender group scores `winogender_all` and then re-scores its
    `female`/`male`/`neutral` subsets; the `gotcha` task and its `_female` /
    `_male` filters are another such pair). Publishing every leaf would count
    those documents twice -- once in the parent, again in each child -- and
    would weight the parent's own average by its full document count. The
    producer on the lighteval side skips exactly this case (it drops the
    `all` row before blending), so the parent is dropped here too and the
    children carry the benchmark.

    Returns a mapping of parent leaf -> the sibling leaves whose documents it
    contains.
    """
    groups: dict[tuple[Any, Any, Any], list[str]] = {}
    for leaf_name, leaf_doc in leaf_docs.items():
        identity = _leaf_dataset_identity(leaf_doc["configs"][leaf_name])
        groups.setdefault(identity, []).append(leaf_name)

    parents: dict[str, tuple[str, ...]] = {}
    undecidable: list[str] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        keys: dict[str, frozenset[Any]] = {}
        for leaf_name in members:
            samples_path = (leaf_artifact_paths.get(leaf_name) or {}).get("samples")
            if samples_path and Path(samples_path).is_file():
                leaf_hashes = _sample_document_hashes(Path(samples_path))
                if leaf_hashes is None:
                    undecidable.append(leaf_name)
                else:
                    keys[leaf_name] = leaf_hashes
        for leaf_name, own_keys in keys.items():
            children = tuple(
                sorted(
                    other
                    for other, other_keys in keys.items()
                    if other != leaf_name and own_keys > other_keys
                )
            )
            if children:
                parents[leaf_name] = children
    if undecidable:
        eval_logger.warning(
            "leaves %s have samples without doc_hash, so whether they merely "
            "re-score a sibling's documents could not be checked; they are "
            "published as separate tasks.",
            ", ".join(sorted(undecidable)),
        )
    return parents


def _weight_ctx_len(
    model_name: str,
    *artifact_paths: str | Path | None,
    weight_revision: str | None = None,
) -> int:
    """Return the context window baked into the served weight release.

    Scoreboard treats ``ctx_len`` as part of the model identity -- the window
    the weight was trained/released with, e.g.
    ``rwkv7-g1k-13.3b-20260930-ctx25600``. It is deliberately *not* the
    evaluation's ``max_length``: that is only the harness-side truncation
    budget, so evaluating one weight at a shorter ``max_length`` must not fork
    it into a second model on the board.

    The revision is read from ``weight_revision`` (the config's explicit
    full-release-name field) when given, then from ``model_name`` when it
    carries one -- ``model_name`` doubles as the literal API ``model=``
    value, so it often can't carry the revision stem -- otherwise from the
    run's artifact paths, which live under a run directory named after the
    weight that was served.
    """
    sources: list[tuple[str, str]] = []
    if weight_revision is not None:
        sources.append((str(weight_revision), "weight_revision"))
    sources.append((str(model_name), "model name"))
    sources.extend(
        (str(path), "artifact path")
        for path in artifact_paths
        if path is not None
    )
    found = [
        (kind, int(match.group(1)))
        for text, kind in sources
        if (match := WEIGHT_CTX_PATTERN.search(text)) is not None
    ]
    if not found:
        searched = ", ".join(f"{kind} {text!r}" for text, kind in sources)
        raise ScoreboardError(
            "cannot determine the served weight's context window (ctx_len): no "
            f"'ctx<number>' token in {searched}. Set the config's top-level (or "
            "[[models]] entry's) weight_revision to the full served weight name "
            "(e.g. 'rwkv7-g1k-13.3b-20260930-ctx25600'); model_name stays the "
            "short API alias. The evaluation's max_length is intentionally not "
            "used because ctx_len identifies the weight."
        )
    ctx_len = found[0][1]
    if any(value != ctx_len for _, value in found[1:]):
        eval_logger.warning(
            "weight context window disagrees across sources %s; using %d from the %s",
            found,
            ctx_len,
            found[0][0],
        )
    return ctx_len


def _rwkv_model(model_name: str, ctx_len: int) -> dict[str, Any]:
    """Build the board's model identity from the served weight."""
    match = MODEL_NAME_PATTERN.match(model_name)
    if match is None:
        raise ScoreboardError(
            f"cannot derive arch_version/data_version/param_size from model name "
            f"{model_name!r}; expected '<arch_version>-<data_version>-<param_size>'"
        )
    arch_version, data_token, param_size = match.groups()
    if arch_version not in ARCH_VERSIONS:
        raise ScoreboardError(
            f"model name {model_name!r} has unknown arch_version {arch_version!r}; "
            f"expected one of {sorted(ARCH_VERSIONS)}"
        )
    if re.fullmatch(r"g1[a-z0-9]+", data_token) is None:
        raise ScoreboardError(
            f"model name {model_name!r} has unknown data_version {data_token!r}"
        )
    data_version = "G1" + data_token[2:]
    if param_size not in PARAM_SIZES:
        raise ScoreboardError(
            f"model name {model_name!r} has unknown param_size {param_size!r}; "
            f"expected one of {sorted(PARAM_SIZES)}"
        )
    return {
        "arch_version": arch_version,
        "data_version": data_version,
        "param_size": param_size,
        "ctx_len": int(ctx_len),
    }




def _sampling_config(
    results: dict[str, Any],
    task_config: dict[str, Any],
    model_args: dict[str, Any],
    task_metadata: dict[str, Any],
) -> dict[str, Any]:
    run_config = results["config"]
    effective = {
        **model_args,
        **task_metadata,
        **(run_config.get("gen_kwargs") or {}),
        **(task_config.get("generation_kwargs") or {}),
        **{
            key: value
            for key, value in (results.get("sampling_config") or {}).items()
            if key != "max_gen_toks"
        },
        **{
            key: value
            for key, value in (run_config.get("sampling_config") or {}).items()
            if key != "max_gen_toks"
        },
    }
    sampling: dict[str, Any] = {
        target: effective[source]
        for source, target in NEW_SAMPLING_FIELDS.items()
        if source in effective
    }
    sampling.setdefault(
        "max_generated_tokens",
        effective.get("max_gen_toks")
        or effective.get("max_tokens")
        or (run_config.get("sampling_config") or {}).get("max_gen_toks")
        or DEFAULT_MAX_GENERATED_TOKENS,
    )
    if run_config.get("model") == "rwkv7-lightning-cuda":
        sampling.setdefault("seed", -1)
        sampling["seed_strategy"] = "unsupported"
    else:
        sampling.setdefault("seed", DEFAULT_SEED)
    if (
        run_config.get("evaluation_protocol")
        and run_config.get("model") != "rwkv7-lightning-cuda"
    ):
        sampling["seed_strategy"] = "base_plus_repeat_ord"
        sampling["evaluation_protocol"] = run_config["evaluation_protocol"]
    missing = [field for field in REQUIRED_SAMPLING_FIELDS if field not in sampling]
    if missing:
        raise ScoreboardError(
            "publication sampling_config is missing required fields: "
            + ", ".join(missing)
        )
    return sampling


def _primary_metric(
    metrics: dict[str, float], task_config: dict[str, Any], policy: dict[str, Any]
) -> str:
    name = policy.get("primary_metric")
    if name is None:
        configured = task_config["metric_list"][0]["metric"]
        # `!function`-defined metrics (e.g. cruxeval/humaneval_infilling's
        # pass_at_k) serialize to their full source instead of a bare name,
        # so a non-identifier value can never match an actual metric key --
        # there is no configured name to check against, so fall straight
        # through to the alphabetical fallback below.
        name = configured if re.fullmatch(r"[A-Za-z0-9_]+", configured) else None
    if name is not None:
        for metric in metrics:
            if metric == name or metric.split(",", 1)[0] == name:
                return metric
        raise ScoreboardError(f"primary metric {name!r} is not present in results")
    non_stderr = sorted(
        metric for metric in metrics if not metric.split(",", 1)[0].endswith("_stderr")
    )
    if non_stderr:
        return non_stderr[0]
    if metrics:
        return min(metrics)
    raise ScoreboardError(f"primary metric {name!r} is not present in results")


def _match_metric_key(metrics: dict[str, float], name: str | None) -> str | None:
    if name is None:
        return None
    for metric in metrics:
        if metric == name or metric.split(",", 1)[0] == name:
            return metric
    return None


def _metrics(results: dict[str, Any], task_name: str) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in results["results"][task_name].items()
        if name not in RESULT_BOOKKEEPING_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


def _common(values: list[Any]) -> Any:
    """The shared value across `values`, or `None` if any two disagree."""
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _benchmark_info(
    benchmark_name: str,
    field: str,
    num_samples: int,
    avg_k: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Register the benchmark together with how much was sampled for it.

    `num_samples` and `avg_k` are *derived* from what the run actually
    produced -- docs scored, and completions per doc -- exactly as
    lighteval-rwkv's producer measures them (`n_completions / num_samples`
    there, weighted across a benchmark's tasks). A hand-written
    ``publication.tasks.<name>.avg_k`` is therefore only accepted when it
    agrees with the results; it can no longer disagree with the score that
    gets uploaded next to it.

    The value it *publishes* is an integer: the board's ``avg_k`` is a label
    (``avg@4``), so a benchmark whose leaves were sampled at different rates
    publishes that rate rounded to whole samples per doc.
    """
    if field not in BENCHMARK_FIELDS:
        raise ScoreboardError(
            f"benchmark.field for {benchmark_name!r} must be one of "
            f"{sorted(BENCHMARK_FIELDS)}, got {field!r}"
        )
    declared_field = policy.get("field")
    if declared_field is not None and declared_field != field:
        raise ScoreboardError(
            f"publication.tasks.{benchmark_name}.field is {declared_field!r}, "
            f"but the saved benchmark.field is {field!r}; update the central "
            "benchmark config and re-run instead of overriding a field at "
            "upload time"
        )
    declared = policy.get("avg_k")
    if declared is not None:
        if not isinstance(declared, (int, float)) or isinstance(declared, bool):
            raise ScoreboardError(
                f"publication.tasks.{benchmark_name}.avg_k must be numeric when set"
            )
        if not math.isclose(float(declared), avg_k, rel_tol=0.0, abs_tol=1e-6):
            raise ScoreboardError(
                f"publication.tasks.{benchmark_name}.avg_k is {declared}, but the "
                f"results were evaluated at avg_k={avg_k:g}; avg_k is derived from "
                "the results (completions per sample), so remove the field or set "
                "it to the value the run actually used"
            )
    # The board's avg_k is an integer label ("avg@4") and part of a cell's
    # identity, so a benchmark whose leaves were sampled at different k cannot
    # publish its fractional rate: wmdp's leaves run 4 / 8 / 2 samples per doc
    # (the avg@k budget is planned per task), which averages to 3.36 and would
    # otherwise render as "avg@3.361504907306434" in the board's Metric column.
    # Publish the rate rounded to whole samples per doc instead. A benchmark
    # whose leaves share one k -- every other benchmark in the campaign --
    # publishes exactly that k, unchanged.
    published_avg_k = float(math.floor(avg_k + 0.5))
    if published_avg_k != float(avg_k):
        eval_logger.warning(
            "benchmark %s: its leaves were sampled at different k, averaging %g "
            "completions per doc; publishing the integer label avg@%d, which is "
            "the rate rounded to whole samples per doc.",
            benchmark_name,
            avg_k,
            int(published_avg_k),
        )
    return {
        "name": benchmark_name,
        "field": field,
        "num_samples": num_samples,
        "avg_k": published_avg_k,
    }


@dataclass(frozen=True, slots=True)
class _LeafUpload:
    """One leaf's contribution to a benchmark upload."""

    docs: int
    # Completions actually sampled for this leaf = docs x the k of the average
    # that is available in its results. Benchmark-wide `avg_k` is the
    # completion-weighted mean of this, so a benchmark whose leaves ran at
    # different k reports the sampling it really did (lighteval-rwkv's
    # `total_completions / total_samples`) instead of one hand-typed number.
    completions: float
    score: float
    truncation_rate: float
    # The task's own primary metric (e.g. `acc,none`) and the key the average
    # actually came from (e.g. `acc_avg@8,none`). Leaves share a headline
    # metric when their `metric_name`s agree -- differing only in the avg@k
    # suffix is expected, since each leaf sizes its own k.
    metric_name: str
    metric_key: str
    k: int


def _recorded_count(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _leaf_metric_key(
    metrics: dict[str, float],
    primary: str,
    recorded_k: float | None,
    planned_k: int,
    leaf_name: str,
    observed_k: int | None = None,
) -> tuple[str, int]:
    """``(metric_key, k)`` for the average actually available for one leaf.

    `observed_k` is how many completions per doc the leaf's samples actually
    hold. It is the last resort when the artifact carries neither a recorded
    completion count nor a ``<metric>_avg@k`` key: the samples are still
    evidence of the sampling that happened, and reporting k=1 there would
    understate it (`race`/`wmdp` artifacts from before the avg@k coverage
    fix, whose leaves were sampled 2-8 times per doc).
    """
    base, _, filter_key = primary.partition(",")
    averaged = {
        int(match.group("k")): name
        for name in metrics
        if (match := AVG_AT_K_PATTERN.match(name.partition(",")[0]))
        and match.group("base") == base
        and name.partition(",")[2] == filter_key
    }
    if not averaged:
        return primary, observed_k or 1
    if recorded_k is not None and int(round(recorded_k)) in averaged:
        k = int(round(recorded_k))
        return averaged[k], k
    if planned_k in averaged:
        return averaged[planned_k], planned_k
    if len(averaged) == 1:
        k = next(iter(averaged))
        return averaged[k], k
    k = max(averaged)
    eval_logger.warning(
        "leaf %s: several avg@k averages are present (%s) and none matches the "
        "run's planned k=%d; using the largest.",
        leaf_name,
        sorted(averaged),
        planned_k,
    )
    return averaged[k], k


@dataclass(frozen=True, slots=True)
class PublicationConfig:
    base_url: str
    token_env: str
    cot_mode: str
    tasks: dict[str, dict[str, Any]]
    timeout: float
    control_timeout: float
    retries: int
    retry_delay: float
    rerun_reason: str | None
    dry_run: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PublicationConfig:
        value = dict(raw)
        cot_mode = value.get("cot_mode")
        if cot_mode not in COT_MODES:
            raise ScoreboardError(
                f"publication.cot_mode must be one of {sorted(COT_MODES)}, got {cot_mode!r}"
            )
        return cls(
            base_url=str(value["base_url"]),
            token_env=str(value.get("token_env", "SCOREBOARD_PUBLICATION_TOKEN")),
            cot_mode=cot_mode,
            tasks=value.get("tasks", {}),
            timeout=float(value.get("timeout", 3600)),
            control_timeout=float(value.get("control_timeout", 30)),
            retries=int(value.get("retries", 2)),
            retry_delay=float(value.get("retry_delay", 1)),
            rerun_reason=value.get("rerun_reason"),
            dry_run=value.get("dry_run") is True,
        )


def _leaf_upload(
    leaf_name: str,
    leaf_doc: dict[str, Any],
    policy: dict[str, Any],
    samples: _SampleEvidence | None,
    metric_hint: str | None = None,
) -> _LeafUpload:
    """Derive one leaf's docs / completions / score / truncation.

    Every number comes from the leaf's own artifacts: the doc count and the
    completion count the run recorded (falling back to the sample evidence
    for artifacts written before those were recorded), and the average the
    results actually carry -- the ``<metric>_avg@k`` value when the task ran
    more than one sample per doc, the plain metric otherwise. Preferring the
    recorded count is what makes the published ``avg_k`` agree with the
    published score; reading only the plain metric (the previous behavior)
    published a single-sample number under an avg@k label.
    """
    task_config = leaf_doc["configs"][leaf_name]
    row = leaf_doc["results"][leaf_name]
    metrics = _metrics(leaf_doc, leaf_name)
    primary = _leaf_primary_metric(metrics, task_config, policy, metric_hint)

    recorded_docs = _recorded_count(row, "n_samples")
    recorded_completions = _recorded_count(row, "n_completions")
    recorded_k = (
        recorded_completions / recorded_docs
        if recorded_docs and recorded_completions
        else None
    )
    planned = task_config.get("repeats")
    planned_k = (
        int(planned)
        if isinstance(planned, (int, float)) and not isinstance(planned, bool)
        else 1
    )
    # What the samples themselves show, for artifacts whose metrics carry no
    # avg@k average to read k off (see `_leaf_metric_key`).
    observed_k = (
        max(1, round(samples.completions / samples.docs))
        if samples is not None and samples.docs
        else None
    )

    metric_key, k = _leaf_metric_key(
        metrics, primary, recorded_k, planned_k, leaf_name, observed_k
    )
    if observed_k and observed_k > 1 and metric_key == primary:
        eval_logger.info(
            "leaf %s: results carry no <metric>_avg@k average, so avg_k is read "
            "from the samples: %d completions over %d docs (k=%g).",
            leaf_name,
            samples.completions,
            samples.docs,
            samples.completions / samples.docs,
        )
    if planned_k > 1 and k < planned_k:
        eval_logger.warning(
            "leaf %s: the run planned k=%d, but the average being published is "
            "%s (k=%d) and neither the results nor the samples carry more. This "
            "artifact predates the avg@k coverage fix (or was resumed from a "
            "checkpoint that restored the doc's requests), so the published score "
            "and avg_k describe the average that is actually present.",
            leaf_name,
            planned_k,
            metric_key,
            k,
        )

    if recorded_docs:
        docs = int(recorded_docs)
    elif samples is not None:
        docs = samples.docs
    else:
        eval_logger.warning(
            "leaf %s: no recorded sample count and no sample evidence; falling "
            "back to n-samples.effective, which under-reports docs evaluated at "
            "avg@k on pre-fix artifacts.",
            leaf_name,
        )
        docs = int(leaf_doc["n-samples"][leaf_name]["effective"])
    if recorded_completions:
        completions = recorded_completions
    elif metric_key != primary:
        # The published score is the ``<metric>_avg@k`` average this key
        # names, so it stands on `docs * k` completions even when the samples
        # file kept fewer rows per doc than the run took.
        completions = float(docs * k)
    elif samples is not None and samples.completions:
        completions = float(samples.completions)
    else:
        completions = float(docs * k)

    if "truncation_rate" in row:
        truncation_rate = float(row["truncation_rate"])
    elif samples is not None:
        truncation_rate = samples.truncation_rate
    else:
        truncation_rate = 0.0
    return _LeafUpload(
        docs=docs,
        completions=completions,
        score=metrics[metric_key],
        truncation_rate=truncation_rate,
        metric_name=primary,
        metric_key=metric_key,
        k=k,
    )


def _leaf_primary_metric(
    metrics: dict[str, float],
    task_config: dict[str, Any],
    policy: dict[str, Any],
    hint: str | None,
) -> str:
    """The leaf's headline metric key.

    A benchmark names its headline metric at the benchmark level (the
    group's first aggregated metric, e.g. ``age_bias`` for ``discrim_eval``,
    or an explicit ``publication.tasks.<name>.primary_metric``), which is
    what the board used to publish; a leaf's own first `metric_list` entry is
    only the fallback for a leaf whose benchmark states nothing.
    """
    declared = policy.get("primary_metric")
    for candidate in (declared, hint):
        if candidate is None:
            continue
        key = _match_metric_key(metrics, candidate)
        if key is not None:
            return key
        if candidate is declared:
            raise ScoreboardError(
                f"primary metric {declared!r} is not present in results"
            )
    return _primary_metric(metrics, task_config, policy)


def build_score(
    results: dict[str, Any],
    config: PublicationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one (benchmark_info, score) pair ready for /api/add_benchmarks
    and /api/upload from an aggregated benchmark artifact.

    `results` is the benchmark-level results.json written by
    `save_benchmark_artifacts` -- it never carries leaf sub-entries, only the
    aggregated score(s), plus pointers (`leaf_artifact_paths`) to each leaf's
    own artifacts so sample evidence can still be sourced without
    duplicating it on disk.

    Everything the board receives is derived from those leaf artifacts, the
    way lighteval-rwkv's producer derives it from its own: docs scored and
    completions sampled are *read off the run* (``num_samples``,
    ``avg_k = n_completions / n_samples``), the headline score is the
    per-sample average the results actually carry, and a multi-leaf benchmark
    blends its leaves by completions after dropping any leaf whose documents a
    sibling leaf already covers (see `_parent_leaves`). The hand-written
    ``avg_k`` in the publication policy is only an assertion from here on --
    if it disagrees with what was run, publication fails instead of labelling
    one number with another's sampling.
    """
    benchmark_name = next(iter(results["results"]))
    benchmark = results.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ScoreboardError(
            f"benchmark {benchmark_name!r} has no saved benchmark.field; re-run "
            "with the explicit benchmark configuration before publication"
        )
    if benchmark.get("name") != benchmark_name:
        raise ScoreboardError(
            f"saved benchmark.name {benchmark.get('name')!r} does not match "
            f"results key {benchmark_name!r}"
        )
    benchmark_field = benchmark.get("field")
    leaf_artifact_paths: dict[str, dict[str, str]] = (
        results.get("leaf_artifact_paths") or {}
    )
    if not leaf_artifact_paths:
        raise ScoreboardError(
            f"benchmark {benchmark_name!r} has no leaf artifacts to publish; "
            "publish a benchmark-level results.json produced by "
            "save_benchmark_artifacts"
        )
    # The run directory that holds these artifacts is named after the served
    # weight, so it is the fallback source for the weight's context window.
    artifact_hint_paths = [
        value
        for paths in leaf_artifact_paths.values()
        for key in ("results", "samples")
        if (value := paths.get(key))
    ]

    leaf_docs = {
        leaf_name: json.loads(Path(paths["results"]).read_text("utf-8"))
        for leaf_name, paths in leaf_artifact_paths.items()
    }
    # A group may score an unfiltered task and filtered subsets of it
    # (winogender's `all` alongside its female/male/neutral filters). Those
    # documents would otherwise be counted once per leaf that covers them, so
    # the parent leaves are dropped and the children carry the benchmark.
    parents = _parent_leaves(leaf_docs, leaf_artifact_paths)
    publish_leaves = {
        leaf_name: paths
        for leaf_name, paths in leaf_artifact_paths.items()
        if leaf_name not in parents
    }
    if parents and not publish_leaves:
        eval_logger.warning(
            "benchmark %s: every leaf covers another leaf's documents (%s); "
            "publishing all of them rather than none.",
            benchmark_name,
            ", ".join(sorted(parents)),
        )
        publish_leaves = dict(leaf_artifact_paths)
    else:
        for parent_name, children in sorted(parents.items()):
            eval_logger.warning(
                "benchmark %s: leaf %s re-scores documents already covered by %s, "
                "so it is excluded from the published score and sample count.",
                benchmark_name,
                parent_name,
                ", ".join(children),
            )
    policy = _task_policy(config, benchmark_name, benchmark_name)

    run_config = results["config"]
    if run_config.get("model") in {
        "rwkv7-lightning-cuda",
        "rwkv7-remote",
        "rwkv7-remote-chat",
    }:
        from lm_eval.api.checkpoint import EVALUATION_PROTOCOL_VERSION
        if run_config.get("evaluation_protocol") != EVALUATION_PROTOCOL_VERSION:
            raise ScoreboardError("Historical generative artifacts require a versioned raw-answer replay before publication")
    model_args = run_config["model_args"]
    model_name = str(
        model_args.get("model") or model_args.get("pretrained") or run_config["model"]
    )
    # The served weight's revision (and thus its ctx_len) usually can't live in
    # model_name: that string is also the literal API model= value, which must
    # stay the backend's short serving alias (e.g. "rwkv7-g1j-1.5b") rather than
    # a release filename stem. metadata.weight_revision carries the full
    # revision string instead, so _weight_ctx_len can read ctx<N> from it
    # without model_name needing to carry anything the API would reject.
    weight_revision = (run_config.get("metadata") or {}).get("weight_revision")

    uploaded_counts = dict.fromkeys(("passed", "wrong", "failed"), 0)
    merged_buckets: dict[str, list[dict[str, Any]]] = {
        "passed": [],
        "wrong": [],
        "failed": [],
    }
    uploads: list[_LeafUpload] = []
    namespaced_scores: dict[str, float] = {}
    # The benchmark's own headline metric (a group's first aggregated metric;
    # `primary_metric_hint` on the artifact) or an explicit policy override.
    metric_hint = policy.get("primary_metric") or results.get("primary_metric_hint")
    for leaf_name, paths in publish_leaves.items():
        leaf_doc = leaf_docs[leaf_name]
        leaf_policy = _task_policy(config, leaf_name, benchmark_name)
        leaf_primary = _leaf_primary_metric(
            _metrics(leaf_doc, leaf_name),
            leaf_doc["configs"][leaf_name],
            leaf_policy,
            metric_hint,
        )
        sample_metric = str(
            leaf_policy.get("outcome_metric") or leaf_primary.split(",", 1)[0]
        )
        samples_path = paths.get("samples")
        evidence = (
            _details(Path(samples_path), sample_metric, uploaded_counts, leaf_primary.partition(",")[2] or None)
            if samples_path
            else None
        )
        upload = _leaf_upload(
            leaf_name, leaf_doc, leaf_policy, evidence, metric_hint
        )
        uploads.append(upload)
        namespaced_scores[f"{leaf_name}/{upload.metric_key}"] = upload.score
        if evidence is not None:
            for outcome, samples in merged_buckets.items():
                samples.extend(evidence.buckets[outcome])

    num_samples = sum(upload.docs for upload in uploads)
    completions = sum(upload.completions for upload in uploads)
    if not num_samples or not completions:
        raise ScoreboardError(
            f"benchmark {benchmark_name!r} has no evaluated samples to publish"
        )
    headline = None
    averaged = results.get("averaged_results") or {}
    headline_key = _match_metric_key(averaged, policy.get("primary_metric") or results.get("primary_metric_hint"))
    if headline_key is None and len({upload.metric_name for upload in uploads}) == 1:
        headline_key = _match_metric_key(averaged, uploads[0].metric_name)
    if headline_key is not None and isinstance(averaged[headline_key], (int, float)):
        headline = float(averaged[headline_key])
    if headline is None and all(upload.k == 1 for upload in uploads):
        # Nothing was averaged per doc, so the benchmark-level metric the run
        # already aggregated is the authoritative headline -- it is the only
        # source that honours a group's own aggregation (e.g. mmlu's macro
        # average over subtasks rather than a pooled one).
        benchmark_metrics = _metrics(results, benchmark_name)
        key = _match_metric_key(
            benchmark_metrics, policy.get("primary_metric")
        ) or _match_metric_key(benchmark_metrics, results.get("primary_metric_hint"))
        if key is not None:
            headline = benchmark_metrics[key]
    if headline is None and len({upload.metric_name for upload in uploads}) == 1:
        # Legacy artifacts without an authoritative group average can only
        # blend a shared metric by documents, preserving equal document weight.
        headline = (
            sum(upload.score * upload.docs for upload in uploads) / num_samples
        )
    if headline is None:
        # A benchmark whose leaves report different metrics (e.g. cruxeval /
        # infinitebench mixing task types) has no meaningful blended score;
        # require a declared benchmark aggregation rather than choosing a leaf.
        raise ScoreboardError("Benchmark leaves use different metrics; an explicit benchmark aggregation is required")
    truncation_rate = (
        sum(upload.truncation_rate * upload.completions for upload in uploads)
        / completions
    )

    # A single-leaf (standalone) benchmark carries its task's own generation
    # kwargs/metadata, which the run's sampling config must include; a
    # multi-leaf benchmark has no single such task.
    sole_config = (
        leaf_docs[next(iter(leaf_docs))]["configs"][next(iter(leaf_docs))]
        if len(leaf_docs) == 1
        else {}
    )
    benchmark_info = _benchmark_info(
        benchmark_name,
        benchmark_field,
        num_samples,
        completions / num_samples,
        policy,
    )
    model = _rwkv_model(
        model_name,
        _weight_ctx_len(
            model_name, *artifact_hint_paths, weight_revision=weight_revision
        ),
    )
    payload = {
        "benchmark_id": None,
        "model": model,
        "cot_mode": config.cot_mode,
        "sampling_config": _sampling_config(
            results,
            sole_config,
            model_args,
            sole_config.get("metadata") or {},
        ),
        "score": headline,
        "truncation_rate": truncation_rate,
        "passed_details": merged_buckets["passed"],
        "wrong_details": merged_buckets["wrong"],
        "failed_details": merged_buckets["failed"],
    }
    return benchmark_info, payload


class ScoreboardPublisher:
    """Own benchmark registration, score upload, retries, and conflicts."""

    def __init__(self, config: PublicationConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.token = os.environ[config.token_env]
        self._health_checked = False

    def _send(
        self,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        data = None
        if body is not None:
            data = _json(body)
            headers["Content-Type"] = "application/json"
        url = self.base_url + path
        for attempt in range(self.config.retries + 1):
            try:
                request = Request(  # noqa: S310 - publication URL is configured.
                    url, data=data, method=method, headers=headers
                )
                with urlopen(  # noqa: S310 - publication URL is configured.
                    request, timeout=timeout or self.config.timeout
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                if error.code == 409:
                    raise ScoreboardError(
                        f"scoreboard {method} {path} returned HTTP 409"
                    ) from error
                if (
                    error.code not in RETRYABLE_HTTP_STATUSES
                    or attempt == self.config.retries
                ):
                    raise ScoreboardError(
                        f"scoreboard {method} {path} returned HTTP {error.code}"
                    ) from error
            except (TimeoutError, URLError) as error:
                if attempt == self.config.retries:
                    raise ScoreboardError(
                        f"cannot reach scoreboard: {error}"
                    ) from error
            time.sleep(min(self.config.retry_delay * 2**attempt, 30))
        raise AssertionError("unreachable: loop always returns or raises")

    def _upload(
        self,
        score: dict[str, Any],
        *,
        benchmark_name: str | None = None,
        allow_expire: bool = True,
    ) -> dict[str, Any]:
        request = Request(  # noqa: S310 - publication URL is configured.
            self.base_url + "/api/upload",
            data=_json(score),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(self.config.retries + 1):
            try:
                with urlopen(  # noqa: S310 - publication URL is configured.
                    request, timeout=self.config.timeout
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                raw = error.read().decode("utf-8", errors="replace")
                try:
                    body = json.loads(raw or "{}")
                except ValueError:
                    body = {}
                detail = body.get("detail") if isinstance(body, dict) else None
                code = detail.get("code") if isinstance(detail, dict) else None
                if (
                    error.code == HTTP_STATUS_UNPROCESSABLE
                    and benchmark_name is not None
                    and "benchmark_name" not in score
                    and isinstance(detail, list)
                    and any(
                        isinstance(item, dict)
                        and item.get("loc") == ["body", "benchmark_name"]
                        for item in detail
                    )
                ):
                    eval_logger.info(
                        "scoreboard rejected the upload for a missing "
                        "benchmark_name; retrying with the name (older server "
                        "revision keyed by benchmark name)."
                    )
                    return self._upload(
                        {**score, "benchmark_name": benchmark_name},
                        benchmark_name=benchmark_name,
                        allow_expire=allow_expire,
                    )
                if error.code == 409:
                    if (
                        allow_expire
                        and self.config.rerun_reason is not None
                        and code in ("duplicate_score", "active_score_exists")
                        and detail.get("score_id") is not None
                    ):
                        self._send(
                            "/api/expire",
                            body={"score_id": detail["score_id"]},
                            timeout=self.config.control_timeout,
                        )
                        return self._upload(
                            score,
                            benchmark_name=benchmark_name,
                            allow_expire=False,
                        )
                    if code == "duplicate_score" and self.config.rerun_reason is None:
                        return {
                            "success": True,
                            "message": "Score already uploaded (idempotent replay)",
                            "replayed": True,
                            "score_id": detail.get("score_id"),
                            "benchmark_id": score["benchmark_id"],
                        }
                if (
                    error.code not in RETRYABLE_HTTP_STATUSES
                    or attempt == self.config.retries
                ):
                    raise ScoreboardError(
                        f"scoreboard POST /api/upload returned HTTP {error.code}: {detail or raw}"
                    ) from error
            except (TimeoutError, URLError) as error:
                if attempt == self.config.retries:
                    raise ScoreboardError(
                        f"cannot reach scoreboard: {error}"
                    ) from error
            time.sleep(min(self.config.retry_delay * 2**attempt, 30))
        raise AssertionError("unreachable: loop always returns or raises")

    def publish(
        self, benchmark_info: dict[str, Any], score: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._health_checked:
            health = self._send(
                "/api/health", method="GET", timeout=self.config.control_timeout
            )
            if health.get("status") != "ok":
                raise ScoreboardError("scoreboard health check did not return ok")
            self._health_checked = True
        benchmark_receipt = self._send(
            "/api/add_benchmarks",
            body=benchmark_info,
            timeout=self.config.control_timeout,
        )
        benchmark_id = benchmark_receipt["benchmark_id"]
        score_with_id = {**score, "benchmark_id": benchmark_id}
        upload_receipt = self._upload(
            score_with_id, benchmark_name=benchmark_info["name"]
        )
        return {
            "benchmark_id": benchmark_id,
            "benchmark": benchmark_receipt,
            "score": upload_receipt,
        }


class ScoreboardTaskCallback:
    """Read one completed benchmark artifact and publish it immediately."""

    def __init__(self, *, config: PublicationConfig) -> None:
        self.config = config
        self._publisher: ScoreboardPublisher | None = None

    def __call__(
        self, *, task_name: str, artifact_paths: Mapping[str, str | Path]
    ) -> dict[str, Any]:
        status = {
            "task_name": task_name,
            "evaluation": "complete",
            "publication": "failed",
            "uploaded": False,
        }
        try:
            results = json.loads(Path(artifact_paths["results"]).read_text("utf-8"))
            benchmark_info, score = build_score(results, self.config)
            if self.config.dry_run:
                status["publication"] = "validated"
                return status
            if self._publisher is None:
                self._publisher = ScoreboardPublisher(self.config)
            receipt = self._publisher.publish(benchmark_info, score)
            status.update(
                publication="complete",
                uploaded=True,
                benchmark_id=receipt["benchmark_id"],
                receipt=receipt,
            )
        except (KeyError, OSError, ScoreboardError, TypeError, ValueError) as error:
            status["error"] = f"{type(error).__name__}: {error}"
        return status
