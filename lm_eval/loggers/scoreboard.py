"""Convert native lm-eval results and publish them to scoreboard-v1."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests


SCHEMA = "scoreboard-v1"
SOURCE = "lm-eval-harness"
CONTRACT = "lm-eval-scoreboard-v3"
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
SENSITIVE_MODEL_ARGS = {
    "api_key",
    "auth_token",
    "header",
    "headers",
    "password",
    "request_headers_env",
    "secret",
    "token",
}


class ScoreboardError(RuntimeError):
    """Raised when publication data or transport is incomplete."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ScoreboardError(f"value is not strict JSON: {error}") from error


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def json_value(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
        )
    except (TypeError, ValueError) as error:
        raise ScoreboardError(f"lm-eval value is not serializable: {error}") from error


def _text(value: Any) -> str:
    if value is None:
        return ""
    return (
        value if isinstance(value, str) else canonical_json(json_value(value)).decode()
    )


def _trimmed(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ScoreboardError(f"publication.{name} must be a non-empty string")
    return value


def _positive_number(value: Any, name: str, default: float) -> float:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreboardError(f"publication.{name} must be positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ScoreboardError(f"publication.{name} must be positive")
    return value


def _non_negative_int(value: Any, name: str, default: int) -> int:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScoreboardError(f"publication.{name} must be a non-negative integer")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        raise ScoreboardError(f"publication.{name} must be a string array")
    if len(value) != len(set(value)):
        raise ScoreboardError(f"publication.{name} must contain unique values")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class PublicationConfig:
    base_url: str
    token_env: str
    model_sha256: str
    tasks: dict[str, dict[str, Any]]
    model_revision: str | None = None
    timeout: float = 3600.0
    control_timeout: float = 30.0
    retries: int = 2
    retry_delay: float = 1.0
    rerun_reason: str | None = None
    configured_benchmarks: tuple[str, ...] | None = None
    skipped_benchmarks: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> PublicationConfig:
        value = dict(raw or {})
        tasks = value.get("tasks", value.get("task_metadata"))
        if not isinstance(tasks, dict) or not tasks:
            raise ScoreboardError("publication.tasks must configure every task")
        if value.get("enabled", True) is not True:
            raise ScoreboardError("disabled publication must not invoke the uploader")
        if value.get("finalize", True) is not True:
            raise ScoreboardError("scoreboard publication must finalize the campaign")
        model_sha256 = _trimmed(value.get("model_sha256"), "model_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None:
            raise ScoreboardError("publication.model_sha256 must be lowercase SHA-256")
        rerun_reason = value.get("rerun_reason")
        if rerun_reason is not None:
            rerun_reason = _trimmed(rerun_reason, "rerun_reason")
        model_revision = value.get("model_revision")
        if model_revision is not None:
            model_revision = _trimmed(model_revision, "model_revision")
        return cls(
            base_url=_trimmed(value.get("base_url"), "base_url"),
            token_env=_trimmed(value.get("token_env"), "token_env"),
            model_sha256=model_sha256,
            model_revision=model_revision,
            tasks=json_value(tasks),
            timeout=_positive_number(value.get("timeout"), "timeout", 3600.0),
            control_timeout=_positive_number(
                value.get("control_timeout"), "control_timeout", 30.0
            ),
            retries=_non_negative_int(value.get("retries"), "retries", 2),
            retry_delay=_positive_number(value.get("retry_delay"), "retry_delay", 1.0),
            rerun_reason=rerun_reason,
            configured_benchmarks=_string_tuple(
                value.get("configured_benchmarks"), "configured_benchmarks"
            ),
            skipped_benchmarks=_string_tuple(
                value.get("skipped_benchmarks"), "skipped_benchmarks"
            ),
        )

    def token(self) -> str:
        token = os.environ.get(self.token_env)
        if not token:
            raise ScoreboardError(f"publication token is missing from {self.token_env}")
        return token


def campaign_run_key(campaign: dict[str, Any]) -> str:
    value = deepcopy(campaign)
    value.pop("run_key", None)
    for name in ("configured_benchmarks", "resolved_benchmarks", "skipped_benchmarks"):
        value[name] = sorted(value[name])
    tasks = []
    for task in value["expected_tasks"]:
        task = deepcopy(task)
        for name in ("evaluation_splits", "languages", "tags"):
            task[name] = sorted(task[name])
        tasks.append(task)
    value["expected_tasks"] = sorted(
        tasks, key=lambda task: (task["identity"], canonical_json(task))
    )
    return content_digest(value)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _task_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = config.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _metrics(results: dict[str, Any], task_name: str) -> dict[str, float]:
    task_results = results.get("results", {}).get(task_name)
    if not isinstance(task_results, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, value in task_results.items():
        if _finite(value):
            name = str(key).rsplit(".", 1)[-1].split(",", 1)[0]
            if name != "sample_len":
                metrics.setdefault(name, float(value))
    return metrics


def _primary_metric(
    task_config: dict[str, Any], metadata: dict[str, Any], metrics: dict[str, float]
) -> str:
    explicit = metadata.get("primary_metric")
    if isinstance(explicit, str) and explicit in metrics:
        return explicit
    metric_list = task_config.get("metric_list")
    if isinstance(metric_list, list):
        for item in metric_list:
            name = item.get("metric") if isinstance(item, dict) else None
            if name in metrics:
                return name
    return next(
        (name for name in metrics if not name.endswith("_stderr")),
        next(iter(metrics)),
    )


def _token_ids(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ScoreboardError(f"{context} must contain integer token IDs")
    return list(value)


def _model_response(sample: dict[str, Any], context: str) -> dict[str, Any]:
    raw_evidence = sample.get("response_evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ScoreboardError(f"{context} lacks response evidence")
    evidence: list[dict[str, Any]] = []
    texts: list[str] = []
    answers: list[str] = []
    output_tokens: list[list[int]] = []
    logprobs: list[float] = []
    input_tokens: list[int] | None = None
    prompts: list[str] = []
    for request_index, group in enumerate(raw_evidence):
        items = [group] if isinstance(group, dict) else group
        if not isinstance(items, list) or not items:
            raise ScoreboardError(
                f"{context}.response_evidence[{request_index}] is empty"
            )
        for response_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ScoreboardError(
                    f"{context}.response_evidence[{request_index}][{response_index}] is invalid"
                )
            item = json_value(item)
            item["request_index"] = request_index
            evidence.append(item)
            input_ids = _token_ids(item.get("input_token_ids"), f"{context}.input")
            output_ids = _token_ids(item.get("output_token_ids"), f"{context}.output")
            input_tokens = input_tokens or input_ids
            output_tokens.append(output_ids)
            if isinstance(item.get("prompt"), str):
                prompts.append(item["prompt"])
            raw = item.get("raw_response")
            choices = raw.get("choices") if isinstance(raw, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else None
            if not isinstance(choice, dict):
                raise ScoreboardError(f"{context} lacks raw response choice")
            text = choice.get("text")
            answer = item.get("post_processed_answer")
            token_logprobs = (
                choice.get("logprobs", {}).get("token_logprobs")
                if isinstance(choice.get("logprobs"), dict)
                else None
            )
            if isinstance(text, str):
                texts.append(text)
            if isinstance(answer, str):
                answers.append(answer)
            if isinstance(token_logprobs, list):
                values = [float(value) for value in token_logprobs if _finite(value)]
                if values:
                    logprobs.append(sum(values))
            if not isinstance(text, str) and not isinstance(token_logprobs, list):
                raise ScoreboardError(f"{context} lacks completion or logprobs")
            if isinstance(text, str) and not isinstance(answer, str):
                raise ScoreboardError(f"{context} lacks post-processed answer")
    response: dict[str, Any] = {
        "input": prompts[0] if prompts else None,
        "input_tokens": input_tokens or [],
        "output_tokens": output_tokens,
        "raw_resps": json_value(sample.get("resps", [])),
        "filtered_resps": json_value(sample.get("filtered_resps", [])),
        "evidence": evidence,
        "evidence_complete": True,
    }
    if texts:
        response["text"] = texts
    if answers:
        response["text_post_processed"] = answers
    if logprobs:
        response["logprobs"] = logprobs
    return response


def _stop_sequences(
    task_config: dict[str, Any], model_args: dict[str, Any]
) -> list[str]:
    generation = task_config.get("generation_kwargs")
    generation = generation if isinstance(generation, dict) else {}
    value = generation.get("until", generation.get("stop", model_args.get("stop")))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _diagnostics(
    samples: list[dict[str, Any]], stop_sequences: list[str]
) -> dict[str, Any]:
    truncated = violations = completions = 0
    for sample in samples:
        response = sample["model_response"]
        evidence = response["evidence"]
        texts = response.get("text", [])
        outputs = response["output_tokens"]
        completions += len(outputs)
        truncated += int(
            any(
                item.get("truncation") is True
                or item.get("finish_reason") in {"length", "max_tokens"}
                for item in evidence
            )
        )
        violations += int(
            any(stop in text for stop in stop_sequences for text in texts)
        )
    count = len(samples)
    return {
        "samples": count,
        "completions": completions,
        "truncated": truncated,
        "non_truncated": count - truncated,
        "truncation_rate": truncated / count if count else 0.0,
        "turn_boundary_violations": violations,
        "turn_boundary_violation_rate": violations / count if count else 0.0,
    }


def _answer_metadata(
    sample: dict[str, Any], detail: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    metric_name = _trimmed(policy.get("outcome_metric"), "tasks.outcome_metric")
    response_mode = policy.get("response_mode")
    if response_mode not in {"single", "multiple_choice"}:
        raise ScoreboardError(
            "publication task response_mode must be single or multiple_choice"
        )
    response = detail["model_response"]
    if response_mode == "single" and len(response["output_tokens"]) != 1:
        raise ScoreboardError("single response mode requires exactly one response")
    filtered = sample.get("filtered_resps")
    while isinstance(filtered, list) and len(filtered) == 1:
        filtered = filtered[0]
    extracted = _text(filtered)
    metric = detail["metrics"].get(metric_name)
    if _finite(metric) and float(metric) == 1.0:
        outcome, fail_reason = "correct", None
    elif _finite(metric) and float(metric) == 0.0:
        truncated = any(
            item.get("truncation") is True
            or item.get("finish_reason") in {"length", "max_tokens"}
            for item in response["evidence"]
        )
        outcome, fail_reason = (
            "incorrect",
            "truncated" if truncated else "answer_mismatch",
        )
    elif not extracted:
        outcome, fail_reason = "unanswered", "empty_extracted_answer"
    else:
        outcome, fail_reason = "undetermined", "non_binary_outcome_metric"
    ground_truth = _text(sample.get("target"))
    if not ground_truth and policy.get("allow_empty_ground_truth") is not True:
        raise ScoreboardError("comparison answer lacks ground_truth")
    texts = response.get("text", [])
    raw_completion = (
        "\n".join(texts)
        if any(isinstance(text, str) and text for text in texts)
        else _text(response["raw_resps"])
    )
    if not raw_completion:
        raise ScoreboardError("comparison answer lacks raw_completion")
    prompts = list(
        dict.fromkeys(
            item["prompt"]
            for item in response["evidence"]
            if isinstance(item.get("prompt"), str)
        )
    )
    latency = [
        float(item["latency_ms"])
        for item in response["evidence"]
        if _finite(item.get("latency_ms"))
    ]
    repeat_id = sample.get("repeat_id", policy.get("repeat_id", 0))
    if isinstance(repeat_id, bool) or not isinstance(repeat_id, int) or repeat_id < 0:
        raise ScoreboardError("sample repeat_id must be a non-negative integer")
    return {
        "outcome": outcome,
        "problem_id": str(sample["doc_id"]),
        "repeat_id": repeat_id,
        "ground_truth": ground_truth,
        "extracted_answer": extracted,
        "assembled_prompt": prompts[0] if len(prompts) == 1 else _text(prompts),
        "raw_completion": raw_completion,
        "fail_reason": fail_reason,
        "generated_tokens": sum(len(tokens) for tokens in response["output_tokens"]),
        "latency_ms": sum(latency) if latency else None,
    }


def _comparison_metadata(
    policy: dict[str, Any],
    diagnostics: dict[str, Any],
    sample_count: int,
    wkv_mode: str,
    model_args: dict[str, Any],
) -> dict[str, Any] | None:
    if policy.get("history_only") is True:
        if policy.get("comparison") is not None:
            raise ScoreboardError("history_only task must not define comparison")
        return None
    comparison = policy.get("comparison")
    if not isinstance(comparison, dict):
        raise ScoreboardError("task policy requires comparison or history_only=true")
    value = json_value(comparison)
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ScoreboardError("comparison requires evaluation metadata")
    if evaluation.get("precision") != wkv_mode:
        raise ScoreboardError("comparison precision does not match WKV mode")
    runtime_profile = model_args.get("rwkv_generation_prompt")
    if (
        runtime_profile is not None
        and evaluation.get("prompt_profile") != runtime_profile
    ):
        raise ScoreboardError("comparison prompt_profile does not match model args")
    value.update(
        samples=sample_count,
        truncation_rate=diagnostics["truncation_rate"],
    )
    return value


def _public_model_args(model_args: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json_value(value)
        for key, value in model_args.items()
        if key.casefold() not in SENSITIVE_MODEL_ARGS
    }


def _sampling_config(
    results: dict[str, Any], task_config: dict[str, Any], model_args: dict[str, Any]
) -> dict[str, Any]:
    run_config = results.get("config")
    run_config = run_config if isinstance(run_config, dict) else {}
    effective = run_config.get("sampling_config", results.get("sampling_config"))
    value = json_value(effective) if isinstance(effective, dict) else {}
    generation = task_config.get("generation_kwargs")
    generation = generation if isinstance(generation, dict) else {}
    for source_name, target_name in (
        ("max_gen_toks", "max_tokens"),
        ("max_tokens", "max_tokens"),
        ("until", "stop"),
        ("stop", "stop"),
        ("do_sample", "do_sample"),
    ):
        if source_name in generation:
            value[target_name] = json_value(generation[source_name])
    for name in ("batch_size",):
        if run_config.get(name) is not None:
            value[name] = json_value(run_config[name])
    for name in ("max_length", "num_concurrent"):
        if model_args.get(name) is not None:
            value[name] = json_value(model_args[name])
    return value


def _environment(
    results: dict[str, Any],
    model_name: str,
    wkv_mode: str,
    config: PublicationConfig,
    model_args: dict[str, Any],
) -> dict[str, Any]:
    execution = results.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    value = {
        "model_name": model_name,
        "model_revision": config.model_revision,
        "weight_sha256": config.model_sha256,
        "wkv_mode": wkv_mode,
        "model_args": _public_model_args(model_args),
        "lm_eval_version": results.get("lm_eval_version"),
        "backend_version": results.get("backend_version"),
        "backend_revision": execution.get("backend_revision")
        or results.get("backend_commit"),
        "torch_version": results.get("torch_version"),
        "gpu": execution.get("gpu") or results.get("gpu"),
        "chat_template_sha": results.get("chat_template_sha"),
        "git_hash": results.get("git_hash"),
    }
    return json_value(value)


def build_publication(
    results: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    *,
    publication: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = PublicationConfig.from_mapping(publication)
    if not samples:
        raise ScoreboardError("publication requires logged samples")
    run_config = results.get("config")
    run_config = run_config if isinstance(run_config, dict) else {}
    model_args = run_config.get("model_args")
    model_args = model_args if isinstance(model_args, dict) else {}
    model_name = (
        results.get("model_name")
        or model_args.get("model")
        or model_args.get("pretrained")
    )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ScoreboardError("lm-eval results lack model identity")
    evaluator_version = results.get("lm_eval_version")
    if not isinstance(evaluator_version, str) or not evaluator_version.strip():
        raise ScoreboardError("lm-eval results lack evaluator version")
    task_configs = results.get("configs")
    task_configs = task_configs if isinstance(task_configs, dict) else {}
    expected_tasks: list[dict[str, Any]] = []
    task_payloads: list[dict[str, Any]] = []
    for task_name, task_samples in samples.items():
        if not isinstance(task_samples, list) or not task_samples:
            raise ScoreboardError(f"lm-eval task {task_name} has no samples")
        task_config = task_configs.get(task_name)
        task_config = task_config if isinstance(task_config, dict) else {}
        metadata = _task_metadata(task_config)
        wkv_mode = (
            metadata.get("wkv_mode")
            or task_config.get("wkv_mode")
            or model_args.get("wkv_mode")
        )
        wkv_mode = _trimmed(wkv_mode, f"tasks.{task_name}.wkv_mode")
        benchmark = metadata.get("benchmark_name", task_name)
        benchmark = _trimmed(benchmark, f"tasks.{task_name}.benchmark")
        policy = config.tasks.get(task_name, config.tasks.get(benchmark))
        if not isinstance(policy, dict):
            raise ScoreboardError(f"publication task policy is missing for {task_name}")
        versions = results.get("versions")
        version = versions.get(task_name) if isinstance(versions, dict) else None
        split = task_config.get("test_split") or task_config.get("validation_split")
        descriptor = {
            "identity": f"{config.model_sha256}:{wkv_mode}:{task_name}",
            "weight_sha256": config.model_sha256,
            "weight_display_name": model_name,
            "wkv_mode": wkv_mode,
            "benchmark": benchmark,
            "task_name": task_name,
            "task_version": str(
                metadata.get(
                    "task_version", version or metadata.get("version", "unknown")
                )
            ),
            "dataset": task_config.get("dataset_path") or metadata.get("dataset"),
            "subset": task_config.get("dataset_name") or metadata.get("subset"),
            "evaluation_splits": [
                str(value)
                for value in metadata.get("evaluation_splits", [split or "unknown"])
            ],
            "languages": [str(value) for value in metadata.get("languages", [])],
            "tags": [
                str(value)
                for value in metadata.get("tags", metadata.get("upstream_tags", []))
            ],
        }
        metrics = _metrics(results, task_name)
        if not metrics:
            raise ScoreboardError(f"lm-eval task {task_name} has no aggregate metrics")
        primary_metric = _primary_metric(task_config, metadata, metrics)
        details: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(task_samples):
            if not isinstance(sample, dict):
                raise ScoreboardError(f"{task_name}.sample[{sample_index}] is invalid")
            document_index = sample.get("doc_id")
            if (
                isinstance(document_index, bool)
                or not isinstance(document_index, int)
                or document_index < 0
            ):
                raise ScoreboardError(
                    f"{task_name}.sample[{sample_index}] lacks doc_id"
                )
            document = json_value(sample.get("doc", {}))
            if not isinstance(document, dict):
                raise ScoreboardError(
                    f"{task_name}.sample[{sample_index}] document is invalid"
                )
            document.update(
                task_name=task_name, target=json_value(sample.get("target"))
            )
            response = _model_response(sample, f"{task_name}.sample[{sample_index}]")
            sample_metrics = {
                name: sample[name]
                for name in sample.get("metrics", [])
                if isinstance(name, str) and _finite(sample.get(name))
            }
            details.append(
                {
                    "sample_index": sample_index,
                    "document_index": document_index,
                    "document": document,
                    "metrics": sample_metrics,
                    "model_response": response,
                }
            )
        counts = results.get("n-samples", {}).get(task_name)
        if not isinstance(counts, dict) or counts.get("effective") != len(details):
            raise ScoreboardError(f"lm-eval task {task_name} sample count mismatch")
        diagnostics = _diagnostics(details, _stop_sequences(task_config, model_args))
        comparison = _comparison_metadata(
            policy, diagnostics, len(details), wkv_mode, model_args
        )
        if comparison is not None:
            for sample, detail in zip(task_samples, details, strict=True):
                detail["answer"] = _answer_metadata(sample, detail, policy)
        task_value = json_value(task_config)
        generation = task_value.get("generation_kwargs")
        generation = generation if isinstance(generation, dict) else {}
        max_tokens = generation.get("max_gen_toks") or generation.get("max_tokens")
        if not isinstance(max_tokens, int):
            max_tokens = (
                1 if task_config.get("output_type") != "generate_until" else None
            )
        if max_tokens is None:
            raise ScoreboardError(f"lm-eval task {task_name} lacks generation size")
        task_value.update(
            generation_size=max_tokens,
            original_num_docs=counts.get("original"),
            effective_num_docs=counts.get("effective"),
            not_evaluated_num_docs=counts.get("original") - counts.get("effective"),
        )
        payload = {
            "schema_version": SCHEMA,
            "campaign_id": None,
            "task": descriptor,
            "result_files": [
                {"role": "metrics", "path": "publication/raw_results.json"},
                {
                    "role": "samples",
                    "path": f"publication/tasks/{re.sub(r'[^A-Za-z0-9_.-]+', '_', task_name)}.json",
                },
            ],
            "task_config": task_value,
            "environment": _environment(
                results, model_name, wkv_mode, config, model_args
            ),
            "sampling_config": _sampling_config(results, task_config, model_args),
            "primary_metric": primary_metric,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "samples": details,
        }
        if comparison is not None:
            payload["comparison"] = comparison
        expected_tasks.append(descriptor)
        task_payloads.append(payload)
    resolved = list(dict.fromkeys(task["benchmark"] for task in expected_tasks))
    configured = list(config.configured_benchmarks or resolved)
    skipped = list(config.skipped_benchmarks or ())
    if set(resolved).intersection(skipped) or set(resolved).union(skipped) != set(
        configured
    ):
        raise ScoreboardError(
            "configured/resolved/skipped benchmark partition is invalid"
        )
    campaign = {
        "schema_version": SCHEMA,
        "source": SOURCE,
        "config_sha256": content_digest(
            {
                "run_config": run_config,
                "task_configs": task_configs,
                "tasks": list(samples),
                "publication_tasks": config.tasks,
            }
        ),
        "registry_sha256": content_digest(expected_tasks),
        "contract_sha256": content_digest(
            {"contract": CONTRACT, "lm_eval_version": evaluator_version}
        ),
        "configured_benchmarks": configured,
        "resolved_benchmarks": resolved,
        "skipped_benchmarks": skipped,
        "expected_tasks": expected_tasks,
        "rerun_reason": config.rerun_reason,
    }
    campaign["run_key"] = campaign_run_key(campaign)
    return campaign, task_payloads


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_root(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    return root if root.name == "publication" else root / "publication"


def write_publication(
    output_dir: str | Path, campaign: dict[str, Any], tasks: list[dict[str, Any]]
) -> tuple[Path, list[Path]]:
    root = _artifact_root(output_dir)
    campaign_path = root / "campaign.json"
    _write_json(campaign_path, campaign)
    task_paths = []
    for task in tasks:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task["task"]["task_name"])
        path = root / "tasks" / f"{name}.json"
        _write_json(path, task)
        task_paths.append(path)
    return campaign_path, task_paths


def _api_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ScoreboardError("publication.base_url must be absolute http(s)")
    path = parsed.path.rstrip("/")
    if not path.endswith("/api"):
        path += "/api"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class ScoreboardClient:
    def __init__(self, config: PublicationConfig) -> None:
        self.config = config
        self.api_root = _api_root(config.base_url)
        self.token = config.token()

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        data = None
        if payload is not None:
            data = gzip.compress(canonical_json(payload))
            headers.update(
                {"Content-Type": "application/json", "Content-Encoding": "gzip"}
            )
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        url = self.api_root + path
        for attempt in range(self.config.retries + 1):
            try:
                response = requests.request(
                    method,
                    url,
                    data=data,
                    headers=headers,
                    timeout=min(self.config.timeout, timeout)
                    if timeout
                    else self.config.timeout,
                )
            except requests.RequestException as error:
                if attempt < self.config.retries:
                    time.sleep(min(self.config.retry_delay * 2**attempt, 30))
                    continue
                raise ScoreboardError(
                    f"cannot reach scoreboard {url}: {error}"
                ) from error
            if 200 <= response.status_code < 300:
                value = response.json()
                if not isinstance(value, dict):
                    raise ScoreboardError(f"scoreboard returned non-object JSON: {url}")
                return value
            if (
                response.status_code in RETRYABLE_HTTP_STATUSES
                and attempt < self.config.retries
            ):
                time.sleep(min(self.config.retry_delay * 2**attempt, 30))
                continue
            raise ScoreboardError(
                f"scoreboard {method} {url} returned HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )
        raise ScoreboardError(f"scoreboard retries exhausted: {method} {url}")


def publish(
    client: ScoreboardClient,
    campaign: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = [task["identity"] for task in campaign["expected_tasks"]]
    by_identity = {task["task"]["identity"]: task for task in tasks}
    if len(expected) != len(set(expected)) or set(expected) != set(by_identity):
        raise ScoreboardError("task payloads do not match campaign")
    preflight = client.request(
        "GET",
        "/v1/evaluation-publication-preflight",
        timeout=client.config.control_timeout,
    )
    if (
        preflight.get("status") != "ready"
        or preflight.get("schema_version") != SCHEMA
        or SOURCE not in preflight.get("sources", [])
    ):
        raise ScoreboardError(f"scoreboard does not support {SCHEMA}/{SOURCE}")
    campaign_receipt = client.request(
        "POST",
        "/v1/evaluation-campaigns",
        payload=campaign,
        idempotency_key=f"campaign:{campaign['run_key']}",
        timeout=client.config.control_timeout,
    )
    campaign_id = campaign_receipt.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ScoreboardError("campaign creation returned no campaign_id")
    status = client.request(
        "GET",
        f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}",
        timeout=client.config.control_timeout,
    )
    hashes = status.get("task_hashes")
    if not isinstance(hashes, dict):
        raise ScoreboardError("campaign status has invalid task hashes")
    receipts = []
    for identity in expected:
        payload = deepcopy(by_identity[identity])
        payload["campaign_id"] = campaign_id
        digest = content_digest(payload)
        if hashes.get(identity) == digest:
            receipts.append({"identity": identity, "action": "unchanged"})
        elif status.get("status") == "complete":
            raise ScoreboardError(f"complete campaign task differs: {identity}")
        else:
            receipt = client.request(
                "PUT",
                f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/tasks/{quote(identity, safe='')}",
                payload=payload,
                idempotency_key=f"publish:{digest}",
            )
            receipts.append({"identity": identity, **receipt})
    finalized = client.request(
        "POST",
        f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/finalize",
        idempotency_key=f"finalize:{campaign_id}",
        timeout=client.config.control_timeout,
    )
    return {
        "campaign_id": campaign_id,
        "campaign": campaign_receipt,
        "preflight": preflight,
        "tasks": receipts,
        "finalize": finalized,
    }


def publish_lm_eval_evaluation(
    results: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    *,
    output_dir: str | Path,
    publication: dict[str, Any] | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = _artifact_root(output_dir)
    raw_path = root / "raw_results.json"
    status_path = root / "status.json"
    raw = json_value(results)
    raw["samples"] = json_value(samples)
    _write_json(raw_path, raw)
    status: dict[str, Any] = {
        "schema_version": "scoreboard-publication-status-v3",
        "evaluation": "complete",
        "publication": "failed",
        "uploaded": False,
        "raw_results_path": str(raw_path),
        "status_path": str(status_path),
    }
    try:
        config = PublicationConfig.from_mapping(publication)
        campaign, tasks = build_publication(
            results, samples, publication=publication or {}
        )
        campaign_path, task_paths = write_publication(output_dir, campaign, tasks)
        status.update(
            campaign_path=str(campaign_path),
            task_paths=[str(path) for path in task_paths],
            publication="validated" if dry_run else "pending",
        )
        if dry_run:
            _write_json(status_path, status)
            return status
        receipt = publish(ScoreboardClient(config), campaign, tasks)
        status.update(
            publication="complete",
            uploaded=True,
            campaign_id=receipt["campaign_id"],
            receipt=receipt,
        )
    except (KeyError, ScoreboardError, OSError, TypeError, ValueError) as error:
        status.update(
            publication="failed",
            uploaded=False,
            message="evaluation complete, publication incomplete",
            error=f"{type(error).__name__}: {error}",
        )
    _write_json(status_path, status)
    return status
