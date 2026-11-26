#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "Qwen3.5-2B",
)
BENCHMARKS = {
    "ruler": ("4096", "higher", "task_macro", 13),
    "wmdp": ("acc", "higher", "sample_weighted", 3),
    "cruxeval": ("pass@1", "higher", "task_macro", 2),
    "paloma": ("bits_per_byte", "lower", "task_macro", 16),
    "inverse_scaling_prize": ("acc", "higher", "task_macro", 10),
    "model_written_evals": ("acc", "higher", "task_macro", 187),
}
FIVE_BENCHMARKS = {
    "humaneval_infilling": {
        "primary_metric": "pass@1",
        "aggregation": "task_macro",
        "expected_tasks": 4,
        "bad_case_metric": "pass@1",
        "direction": "higher",
    },
    "mutual": {
        "primary_metric": "mrr",
        "aggregation": "task_macro",
        "expected_tasks": 1,
        "bad_case_metric": "r@1",
        "direction": "higher",
    },
    "mc_taco": {
        "primary_metric": "f1",
        "aggregation": "task_macro",
        "expected_tasks": 1,
        "bad_case_metric": "acc",
        "direction": "higher",
    },
    "discrim_eval": {
        "primary_metric": "mean_abs_bias",
        "aggregation": "derived_mean_absolute_native_bias",
        "expected_tasks": 2,
        "bad_case_metric": None,
        "direction": "lower",
    },
    "winogender": {
        "primary_metric": "acc",
        "aggregation": "winogender_all",
        "expected_tasks": 7,
        "bad_case_metric": "acc",
        "direction": "higher",
    },
}
DISCRIM_BIAS_FIELDS = (
    "black_bias",
    "asian_bias",
    "hispanic_bias",
    "native_american_bias",
    "female_bias",
    "non_binary_bias",
    "age_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("six", "five"), default="six")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/formal-six-benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/formal-six-benchmarks/comparison",
    )
    parser.add_argument("--bad-cases-per-task", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_value(values: dict[str, Any], metric: str) -> float | None:
    for key, value in values.items():
        if key != metric and not key.startswith(f"{metric},"):
            continue
        if isinstance(value, int | float) and math.isfinite(value):
            return float(value)
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return float(value.lower() == "true")
    return None


def latest_task_records(benchmark_dir: Path) -> dict[str, tuple[Path, dict]]:
    records: dict[str, tuple[Path, dict]] = {}
    paths = sorted(
        benchmark_dir.glob("**/results_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        for task_name in result.get("n-samples", {}):
            if task_name in result.get("results", {}):
                records[task_name] = (path, result)
    return records


def sample_path(result_path: Path, task_name: str) -> Path:
    timestamp = result_path.stem.removeprefix("results_")
    return result_path.parent / f"samples_{task_name}_{timestamp}.jsonl"


def aggregate_score(
    records: dict[str, tuple[Path, dict]], metric: str, aggregation: str
) -> tuple[float | None, int]:
    values = []
    sample_total = 0
    for task_name, (_, result) in records.items():
        value = metric_value(result["results"][task_name], metric)
        samples = int(result["n-samples"][task_name]["effective"])
        sample_total += samples
        if value is not None:
            values.append((value, samples))
    if not values:
        return None, sample_total
    if aggregation == "sample_weighted":
        denominator = sum(samples for _, samples in values)
        return sum(
            value * samples for value, samples in values
        ) / denominator, sample_total
    return statistics.fmean(value for value, _ in values), sample_total


def inferred_qwen_truncation(
    records: dict[str, tuple[Path, dict]], tokenizer_path: Path
) -> dict[str, dict[str, int]]:
    def response_strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [text for item in value for text in response_strings(item)]
        return []

    pending: list[tuple[str, int, list[str]]] = []
    for task_name, (result_path, _) in records.items():
        path = sample_path(result_path, task_name)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                gen_args = sample.get("arguments", {}).get("gen_args_0", {})
                gen_kwargs = gen_args.get("arg_1")
                if not isinstance(gen_kwargs, dict):
                    continue
                max_tokens = gen_kwargs.get("max_gen_toks")
                if not isinstance(max_tokens, int):
                    continue
                response_texts = response_strings(sample.get("resps", []))
                if response_texts:
                    pending.append((task_name, max_tokens, response_texts))
    if not pending:
        return {}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    inferred: dict[str, dict[str, int]] = {}
    for task_name, max_tokens, response_texts in pending:
        token_ids = tokenizer(
            response_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids
        task_stats = inferred.setdefault(
            task_name,
            {"generated_samples": 0, "truncated_samples": 0},
        )
        task_stats["generated_samples"] += 1
        task_stats["truncated_samples"] += int(
            any(len(tokens) >= max_tokens for tokens in token_ids)
        )
    return inferred


def truncation_stats(
    model_name: str,
    records: dict[str, tuple[Path, dict]],
    tokenizer_path: Path | None,
) -> dict[str, Any]:
    generated = 0
    truncated = 0
    seen = set()
    inferred = {}
    if model_name == "Qwen3.5-2B" and tokenizer_path is not None:
        has_missing_generation_stats = any(
            int(
                result.get("config", {})
                .get("truncation", {})
                .get(task_name, {})
                .get("generated_samples", 0)
            )
            == 0
            for task_name, (_, result) in records.items()
        )
        if has_missing_generation_stats:
            inferred = inferred_qwen_truncation(records, tokenizer_path)
    for task_name, (path, result) in records.items():
        stats = result.get("config", {}).get("truncation")
        if stats is None:
            stats = result.get("config", {}).get("rwkv_truncation", {})
        task_stats = stats.get(task_name, {})
        if int(task_stats.get("generated_samples", 0)) == 0:
            task_stats = inferred.get(task_name, task_stats)
        key = (path, task_name)
        if key in seen:
            continue
        seen.add(key)
        generated += int(task_stats.get("generated_samples", 0))
        truncated += int(task_stats.get("truncated_samples", 0))
    return {
        "generated_samples": generated,
        "truncated_samples": truncated,
        "truncation_rate": truncated / generated if generated else None,
        "source": (
            "posthoc_output_token_count"
            if inferred and generated
            else "api_finish_reason"
            if generated
            else None
        ),
    }


def gpu_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    utilization = []
    used = []
    free = []
    total = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 6 or row[0].strip().lower() == "timestamp":
                continue
            try:
                utilization.append(float(row[2].replace("%", "").strip()))
                total.append(float(row[3].replace("MiB", "").strip()))
                used.append(float(row[4].replace("MiB", "").strip()))
                free.append(float(row[5].replace("MiB", "").strip()))
            except ValueError:
                continue
    if not utilization:
        return {}
    active = [value for value in utilization if value > 0]
    active_mean = statistics.fmean(active) if active else 0.0
    longest_high_load = []
    current_high_load = []
    for value in utilization:
        if value >= 90:
            current_high_load.append(value)
        else:
            if len(current_high_load) > len(longest_high_load):
                longest_high_load = current_high_load
            current_high_load = []
    if len(current_high_load) > len(longest_high_load):
        longest_high_load = current_high_load
    sustained_scope_valid = len(longest_high_load) >= 12
    sustained_mean = statistics.fmean(longest_high_load) if longest_high_load else None
    min_free_ratio = min(free) / max(total)
    return {
        "source": str(path),
        "measurement_scope": (
            "full evaluator process including dataset loading, tokenization, "
            "judging, and serialization"
        ),
        "samples": len(utilization),
        "utilization_mean_percent": statistics.fmean(utilization),
        "utilization_active_mean_percent": active_mean,
        "utilization_peak_percent": max(utilization),
        "sustained_high_load_definition": (
            "longest contiguous 5-second sample run with utilization >= 90%"
        ),
        "sustained_high_load_samples": len(longest_high_load),
        "sustained_high_load_seconds": len(longest_high_load) * 5,
        "sustained_high_load_mean_percent": sustained_mean,
        "sustained_high_load_median_percent": (
            statistics.median(longest_high_load) if longest_high_load else None
        ),
        "memory_peak_mib": max(used),
        "memory_min_free_mib": min(free),
        "memory_min_free_ratio": min_free_ratio,
        "gpu_utilization_target_met": (
            sustained_mean >= 97 if sustained_scope_valid else None
        ),
        "gpu_utilization_target_scope_valid": sustained_scope_valid,
        "gpu_memory_free_target_met": min_free_ratio < 0.1,
    }


def benchmark_gpu_stats(benchmark_dir: Path) -> dict[str, Any]:
    performance_path = benchmark_dir / "performance.json"
    if not performance_path.exists():
        return gpu_stats(benchmark_dir / "gpu_utilization.csv")
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    sustained = performance.get("gpu_utilization_sustained_observed_percent", [])
    return {
        "source": str(performance_path),
        "measurement_scope": performance.get("measurement_scope"),
        "utilization_active_mean_percent": statistics.fmean(sustained)
        if sustained
        else None,
        "utilization_peak_percent": max(sustained) if sustained else None,
        "memory_peak_mib": performance.get("gpu_memory_peak_used_mib"),
        "memory_min_free_mib": performance.get("gpu_memory_min_free_mib"),
        "memory_min_free_ratio": performance.get("gpu_memory_min_free_ratio"),
        "gpu_utilization_target_met": performance.get("gpu_utilization_target_met"),
        "gpu_memory_free_target_met": performance.get("gpu_memory_free_target_met"),
        "request_throughput_per_second": performance.get(
            "request_throughput_per_second"
        ),
    }


def unwrap(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def bounded(value: Any, limit: int = 4000) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    half = limit // 2
    return {"excerpt": f"{text[:half]}\n...\n{text[-half:]}"}


def bad_cases(
    model_name: str,
    benchmark: str,
    metric: str,
    direction: str,
    records: dict[str, tuple[Path, dict]],
    limit: int,
) -> list[dict[str, Any]]:
    output = []
    for task_name, (result_path, _) in records.items():
        path = sample_path(result_path, task_name)
        if not path.exists():
            continue
        candidates = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                score = metric_value(sample, metric)
                if score is None:
                    continue
                candidates.append((score, sample))
        candidates.sort(key=lambda item: item[0], reverse=direction == "lower")
        for score, sample in candidates[:limit]:
            output.append(
                {
                    "model_name": model_name,
                    "benchmark_name": benchmark,
                    "task_name": task_name,
                    "doc_id": sample.get("doc_id"),
                    "metric": metric,
                    "score": score,
                    "target": sample.get("target"),
                    "response": bounded(unwrap(sample.get("resps"))),
                    "filtered_response": bounded(unwrap(sample.get("filtered_resps"))),
                    "truncated": sample.get("truncated"),
                    "doc": bounded(sample.get("doc")),
                    "source_sample": str(path),
                }
            )
    return output


def discrim_logit_yes(sample: dict[str, Any]) -> float | None:
    """Read the per-example normalized yes/no logit emitted by Discrim-Eval."""
    for metric in DISCRIM_BIAS_FIELDS:
        value = sample.get(metric)
        if (
            isinstance(value, list | tuple)
            and len(value) == 3
            and isinstance(value[2], int | float)
            and math.isfinite(value[2])
        ):
            return float(value[2])
    return None


def discrim_bad_cases(
    model_name: str,
    records: dict[str, tuple[Path, dict]],
    limit: int,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, Any, Any], dict[str, Any]],
]:
    """Select raw matched demographic contrasts rather than invent correctness labels."""
    output = []
    by_contrast = {}
    for task_name, (result_path, _) in sorted(records.items()):
        path = sample_path(result_path, task_name)
        if not path.exists():
            continue
        samples = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                doc = sample.get("doc", {})
                logit_yes = discrim_logit_yes(sample)
                if not isinstance(doc, dict) or logit_yes is None:
                    continue
                samples.append(
                    {
                        "sample": sample,
                        "doc_id": sample.get("doc_id"),
                        "question": doc.get("decision_question_id"),
                        "age": doc.get("age"),
                        "gender": str(doc.get("gender", "")).lower(),
                        "race": str(doc.get("race", "")).lower(),
                        "logit_yes": logit_yes,
                    }
                )
        indexed = {
            (row["question"], row["age"], row["gender"], row["race"]): row
            for row in samples
        }
        candidates: dict[str, list[tuple[float, dict, dict]]] = {
            "race": [],
            "gender": [],
            "age": [],
        }
        for compared in samples:
            if compared["race"] != "white":
                control = indexed.get(
                    (
                        compared["question"],
                        compared["age"],
                        compared["gender"],
                        "white",
                    )
                )
                if control is not None:
                    candidates["race"].append(
                        (
                            abs(compared["logit_yes"] - control["logit_yes"]),
                            control,
                            compared,
                        )
                    )
            if compared["gender"] != "male":
                control = indexed.get(
                    (
                        compared["question"],
                        compared["age"],
                        "male",
                        compared["race"],
                    )
                )
                if control is not None:
                    candidates["gender"].append(
                        (
                            abs(compared["logit_yes"] - control["logit_yes"]),
                            control,
                            compared,
                        )
                    )
        age_groups: dict[tuple[Any, str, str], list[dict[str, Any]]] = {}
        for row in samples:
            age_groups.setdefault(
                (row["question"], row["gender"], row["race"]), []
            ).append(row)
        for rows in age_groups.values():
            ordered = sorted(rows, key=lambda row: float(row["age"]))
            control, compared = ordered[0], ordered[-1]
            if control["age"] != compared["age"]:
                candidates["age"].append(
                    (
                        abs(compared["logit_yes"] - control["logit_yes"]),
                        control,
                        compared,
                    )
                )

        for axis, axis_candidates in candidates.items():
            for score, control, compared in axis_candidates:
                contrast_key = (
                    task_name,
                    axis,
                    control["doc_id"],
                    compared["doc_id"],
                )
                by_contrast[contrast_key] = {
                    "model_name": model_name,
                    "task_name": task_name,
                    "contrast_axis": axis,
                    "control_doc_id": control["doc_id"],
                    "compared_doc_id": compared["doc_id"],
                    "score": score,
                    "signed_delta": compared["logit_yes"] - control["logit_yes"],
                    "source_sample": str(path),
                }

        selected = []
        per_axis = max(1, math.ceil(limit / len(candidates)))
        for axis in ("race", "gender", "age"):
            selected.extend(
                (axis, *candidate)
                for candidate in sorted(
                    candidates[axis], key=lambda item: item[0], reverse=True
                )[:per_axis]
            )
        for axis, score, control, compared in selected[:limit]:
            contrast_key = (
                task_name,
                axis,
                control["doc_id"],
                compared["doc_id"],
            )
            record = {
                "model_name": model_name,
                "benchmark_name": "discrim_eval",
                "task_name": task_name,
                "doc_id": {
                    "control": control["doc_id"],
                    "compared": compared["doc_id"],
                },
                "metric": "absolute_logit_yes_demographic_contrast",
                "score": score,
                "case_type": "matched_demographic_contrast",
                "contrast_axis": axis,
                "target": "zero logit difference under demographic-only substitution",
                "response": {
                    "control_logit_yes": control["logit_yes"],
                    "compared_logit_yes": compared["logit_yes"],
                    "signed_delta": compared["logit_yes"] - control["logit_yes"],
                    "control_choice_loglikelihoods": control["sample"].get(
                        "filtered_resps"
                    ),
                    "compared_choice_loglikelihoods": compared["sample"].get(
                        "filtered_resps"
                    ),
                },
                "doc": {
                    "control": bounded(control["sample"].get("doc")),
                    "compared": bounded(compared["sample"].get("doc")),
                },
                "source_sample": str(path),
            }
            output.append(record)
            by_contrast[contrast_key] = record
    return output, by_contrast


def paired_discrim_bad_cases(
    contrasts_by_model: dict[str, dict[tuple[str, str, Any, Any], dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    output = []
    rwkv = contrasts_by_model.get(MODELS[0], {})
    qwen = contrasts_by_model.get(MODELS[1], {})
    common = rwkv.keys() & qwen.keys()
    for task_name in sorted({key[0] for key in common}):
        candidates = []
        for key in common:
            if key[0] != task_name:
                continue
            delta = qwen[key]["score"] - rwkv[key]["score"]
            candidates.append((abs(delta), delta, key))
        for _, delta, key in sorted(candidates, reverse=True)[:limit]:
            output.append(
                {
                    "benchmark_name": "discrim_eval",
                    "task_name": task_name,
                    "doc_id": {
                        "control": key[2],
                        "compared": key[3],
                    },
                    "metric": "absolute_logit_yes_demographic_contrast",
                    "case_type": (
                        "qwen_larger_raw_contrast"
                        if delta > 0
                        else "rwkv_larger_raw_contrast"
                    ),
                    "rwkv": bounded(rwkv[key]),
                    "qwen": bounded(qwen[key]),
                }
            )
    return output


def metric_samples(
    record: tuple[Path, dict], task_name: str, metric: str
) -> dict[Any, tuple[Path, dict, float]]:
    result_path, _ = record
    path = sample_path(result_path, task_name)
    if not path.exists():
        return {}
    samples = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            value = metric_value(sample, metric)
            doc_id = sample.get("doc_id")
            if value is not None and doc_id is not None:
                samples[doc_id] = (path, sample, value)
    return samples


def paired_bad_cases(
    benchmark: str,
    metric: str,
    direction: str,
    records_by_model: dict[str, dict[str, tuple[Path, dict]]],
    limit: int,
) -> list[dict[str, Any]]:
    output = []
    common_tasks = set(records_by_model.get(MODELS[0], {})) & set(
        records_by_model.get(MODELS[1], {})
    )
    for task_name in sorted(common_tasks):
        rwkv_samples = metric_samples(
            records_by_model[MODELS[0]][task_name], task_name, metric
        )
        qwen_samples = metric_samples(
            records_by_model[MODELS[1]][task_name], task_name, metric
        )
        candidates = []
        for doc_id in rwkv_samples.keys() & qwen_samples.keys():
            rwkv_path, rwkv_sample, rwkv_score = rwkv_samples[doc_id]
            qwen_path, qwen_sample, qwen_score = qwen_samples[doc_id]
            quality_delta = qwen_score - rwkv_score
            if direction == "lower":
                quality_delta = -quality_delta
            if quality_delta == 0:
                continue
            candidates.append(
                (
                    quality_delta,
                    doc_id,
                    rwkv_path,
                    rwkv_sample,
                    rwkv_score,
                    qwen_path,
                    qwen_sample,
                    qwen_score,
                )
            )
        selected = []
        selected.extend(
            sorted(
                (candidate for candidate in candidates if candidate[0] < 0),
                key=lambda candidate: candidate[0],
            )[:limit]
        )
        selected.extend(
            sorted(
                (candidate for candidate in candidates if candidate[0] > 0),
                key=lambda candidate: candidate[0],
                reverse=True,
            )[:limit]
        )
        for (
            quality_delta,
            doc_id,
            rwkv_path,
            rwkv_sample,
            rwkv_score,
            qwen_path,
            qwen_sample,
            qwen_score,
        ) in selected:
            output.append(
                {
                    "benchmark_name": benchmark,
                    "task_name": task_name,
                    "doc_id": doc_id,
                    "metric": metric,
                    "case_type": (
                        "qwen_better" if quality_delta > 0 else "rwkv_better"
                    ),
                    "quality_delta_qwen_minus_rwkv": quality_delta,
                    "target": rwkv_sample.get("target"),
                    "doc": bounded(rwkv_sample.get("doc")),
                    "rwkv": {
                        "score": rwkv_score,
                        "response": bounded(unwrap(rwkv_sample.get("resps"))),
                        "filtered_response": bounded(
                            unwrap(rwkv_sample.get("filtered_resps"))
                        ),
                        "source_sample": str(rwkv_path),
                    },
                    "qwen": {
                        "score": qwen_score,
                        "response": bounded(unwrap(qwen_sample.get("resps"))),
                        "filtered_response": bounded(
                            unwrap(qwen_sample.get("filtered_resps"))
                        ),
                        "source_sample": str(qwen_path),
                    },
                }
            )
    return output


def main_six(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.results_root / "protocol.json"
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else {}
    )
    tokenizer_paths = {
        model_name: Path(model.get("weight_path", ""))
        for model_name, model in protocol.get("models", {}).items()
        if model.get("weight_path")
    }
    comparison: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(args.results_root.resolve()),
        "models": list(MODELS),
        "benchmarks": {},
        "provenance": [],
    }
    task_rows = []
    all_bad_cases = []
    all_paired_bad_cases = []

    for benchmark, (
        metric,
        direction,
        aggregation,
        expected_tasks,
    ) in BENCHMARKS.items():
        benchmark_record = {
            "metric": metric,
            "direction": direction,
            "aggregation": aggregation,
            "expected_tasks": expected_tasks,
            "protocol": protocol.get("benchmarks", {}).get(benchmark, {}),
            "models": {},
        }
        protocol_record = protocol.get("benchmarks", {}).get(benchmark, {})
        records_by_model = {}
        if protocol_record.get("status") == "blocked":
            benchmark_record["blocker"] = protocol_record.get("blocker")
        for model_name in MODELS:
            benchmark_dir = args.results_root / model_name / benchmark
            records = latest_task_records(benchmark_dir)
            records_by_model[model_name] = records
            score, sample_total = aggregate_score(records, metric, aggregation)
            status = "complete" if len(records) == expected_tasks else "incomplete"
            if not records and protocol_record.get("status") == "blocked":
                status = "blocked"
            model_settings = protocol_record.get("model_settings", {}).get(
                model_name, {}
            )
            benchmark_record["models"][model_name] = {
                "status": status,
                "score": score,
                "n_samples": sample_total,
                "task_count": len(records),
                "settings": model_settings,
                "truncation": truncation_stats(
                    model_name,
                    records,
                    tokenizer_paths.get(model_name),
                ),
                "gpu": benchmark_gpu_stats(benchmark_dir),
            }
            for task_name, (result_path, result) in sorted(records.items()):
                value = metric_value(result["results"][task_name], metric)
                samples = int(result["n-samples"][task_name]["effective"])
                task_rows.append(
                    {
                        "benchmark_name": benchmark,
                        "model_name": model_name,
                        "task_name": task_name,
                        "metric": metric,
                        "score": value,
                        "n_samples": samples,
                        "cot_mode": model_settings.get("cot_mode"),
                        "prompt_template": model_settings.get("prompt_template"),
                        "wkv_mode": model_settings.get("wkv_mode"),
                    }
                )
                comparison["provenance"].append(
                    {
                        "benchmark_name": benchmark,
                        "model_name": model_name,
                        "task_name": task_name,
                        "result_path": str(result_path),
                        "result_sha256": sha256(result_path),
                    }
                )
            all_bad_cases.extend(
                bad_cases(
                    model_name,
                    benchmark,
                    metric,
                    direction,
                    records,
                    args.bad_cases_per_task,
                )
            )
        benchmark_paired_bad_cases = paired_bad_cases(
            benchmark,
            metric,
            direction,
            records_by_model,
            args.bad_cases_per_task,
        )
        all_paired_bad_cases.extend(benchmark_paired_bad_cases)
        benchmark_record["paired_bad_case_count"] = len(benchmark_paired_bad_cases)
        rwkv_score = benchmark_record["models"][MODELS[0]]["score"]
        qwen_score = benchmark_record["models"][MODELS[1]]["score"]
        benchmark_record["qwen_minus_rwkv"] = (
            qwen_score - rwkv_score
            if rwkv_score is not None and qwen_score is not None
            else None
        )
        comparison["benchmarks"][benchmark] = benchmark_record

    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_task.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(task_rows[0])
            if task_rows
            else [
                "benchmark_name",
                "model_name",
                "task_name",
                "metric",
                "score",
                "n_samples",
                "cot_mode",
                "prompt_template",
                "wkv_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(task_rows)
    with (args.output_dir / "bad_cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_bad_cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "paired_bad_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in all_paired_bad_cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# Six-benchmark comparison",
        "",
        "Full formal evaluation; PALOMA remains explicitly blocked rather than replaced by a proxy dataset.",
        "",
        "| Benchmark | Metric | Samples (RWKV / Qwen) | RWKV7 1.5B | Qwen3.5 2B | Qwen - RWKV | Truncation (RWKV / Qwen) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for benchmark, record in comparison["benchmarks"].items():
        rwkv = record["models"][MODELS[0]]
        qwen = record["models"][MODELS[1]]
        values = [rwkv["score"], qwen["score"], record["qwen_minus_rwkv"]]
        formatted = ["—" if value is None else f"{value:.6f}" for value in values]
        sample_counts = f"{rwkv['n_samples']} / {qwen['n_samples']}"
        truncation_rates = []
        for model_record in (rwkv, qwen):
            rate = model_record["truncation"]["truncation_rate"]
            truncation_rates.append("—" if rate is None else f"{rate:.4%}")
        status = f"{rwkv['status']} / {qwen['status']}"
        lines.append(
            f"| {benchmark} | {record['metric']} | {sample_counts} | "
            f"{formatted[0]} | {formatted[1]} | {formatted[2]} | "
            f"{truncation_rates[0]} / {truncation_rates[1]} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `comparison.json`: aggregate scores, protocol, truncation, GPU, and provenance.",
            "- `per_task.csv`: every leaf-task score and model settings.",
            "- `bad_cases.jsonl`: lowest-scoring raw samples per model and task.",
            "- `paired_bad_cases.jsonl`: same-document RWKV/Qwen disagreements.",
            "- `../protocol.json`: frozen model, backend, dataset, template, and concurrency metadata.",
        ]
    )
    blockers = [
        f"- `{benchmark}`: {record['blocker']}"
        for benchmark, record in comparison["benchmarks"].items()
        if record.get("blocker")
    ]
    if blockers:
        lines.extend(["", "## Blockers", "", *blockers])
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def native_task_metrics(result: dict, task_name: str) -> dict[str, float]:
    metrics = {}
    for key, value in result.get("results", {}).get(task_name, {}).items():
        metric = key.split(",", 1)[0]
        if (
            metric.endswith("_stderr")
            or metric in {"name", "alias", "sample_len"}
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            continue
        metrics[metric] = float(value)
    return metrics


def sandbox_results(benchmark_dir: Path) -> tuple[Path | None, dict]:
    path = benchmark_dir / "sandbox_results.json"
    if not path.exists():
        return None, {}
    return path, json.loads(path.read_text(encoding="utf-8"))


def five_primary_score(
    benchmark: str,
    task_metrics: dict[str, dict[str, float]],
) -> float | None:
    spec = FIVE_BENCHMARKS[benchmark]
    metric = spec["primary_metric"]
    if benchmark == "discrim_eval":
        values = [
            abs(value)
            for metrics in task_metrics.values()
            for value in metrics.values()
        ]
    elif benchmark == "winogender":
        value = task_metrics.get("winogender_all", {}).get(metric)
        return float(value) if value is not None else None
    else:
        values = [
            metrics[metric] for metrics in task_metrics.values() if metric in metrics
        ]
    return statistics.fmean(values) if values else None


def humaneval_bad_cases(
    model_name: str,
    benchmark_dir: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, Any], dict[str, Any]]]:
    path = benchmark_dir / "sandbox_judgements.jsonl"
    if not path.exists():
        return [], {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    by_key = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task_name = record.get("task_name", record.get("task"))
            if task_name is None:
                continue
            key = (task_name, record["doc_id"])
            by_key[key] = record
            if not record["passed"]:
                by_task.setdefault(task_name, []).append(record)
    output = []
    for task_name, records in sorted(by_task.items()):
        for record in records[:limit]:
            output.append(
                {
                    "model_name": model_name,
                    "benchmark_name": "humaneval_infilling",
                    "task_name": task_name,
                    "doc_id": record["doc_id"],
                    "metric": "pass@1",
                    "score": 0.0,
                    "target": bounded(record.get("reference")),
                    "response": bounded(record.get("prediction")),
                    "truncated": record.get("truncated"),
                    "judge_result": record.get("result"),
                    "source_sample": record.get("source_sample"),
                    "source_judgement": str(path),
                }
            )
    return output, by_key


def main_five(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.results_root / "protocol.json"
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else {}
    )
    tokenizer_paths = {
        model_name: Path(model.get("weight_path", ""))
        for model_name, model in protocol.get("models", {}).items()
        if model.get("weight_path")
    }
    comparison: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(args.results_root.resolve()),
        "models": list(MODELS),
        "benchmarks": {},
        "provenance": [],
    }
    task_rows = []
    all_bad_cases = []
    all_paired_bad_cases = []

    for benchmark, spec in FIVE_BENCHMARKS.items():
        benchmark_record: dict[str, Any] = {
            "primary_metric": spec["primary_metric"],
            "aggregation": spec["aggregation"],
            "direction": spec["direction"],
            "expected_tasks": spec["expected_tasks"],
            "protocol": protocol.get("benchmarks", {}).get(benchmark, {}),
            "models": {},
        }
        records_by_model = {}
        judgement_by_model = {}
        discrim_contrasts_by_model = {}
        for model_name in MODELS:
            benchmark_dir = args.results_root / model_name / benchmark
            records = latest_task_records(benchmark_dir)
            records_by_model[model_name] = records
            judge_path, judge = sandbox_results(benchmark_dir)
            if benchmark == "humaneval_infilling":
                metrics_by_task = {
                    task_name: {
                        metric: float(value)
                        for metric, value in (
                            task_record.get("metrics")
                            or {
                                "pass@1": task_record.get("pass@1")
                            }
                        ).items()
                        if value is not None
                    }
                    for task_name, task_record in judge.get("tasks", {}).items()
                }
                sample_total = sum(
                    int(task_record.get("n_samples", 0))
                    for task_record in judge.get("tasks", {}).values()
                )
            else:
                metrics_by_task = {
                    task_name: native_task_metrics(result, task_name)
                    for task_name, (_, result) in records.items()
                }
                sample_total = sum(
                    int(result["n-samples"][task_name]["effective"])
                    for task_name, (_, result) in records.items()
                )
            score = five_primary_score(benchmark, metrics_by_task)
            status = (
                "complete"
                if len(metrics_by_task) == spec["expected_tasks"] and score is not None
                else "incomplete"
            )
            settings = {}
            if records:
                first_task = min(records)
                first_result = records[first_task][1]
                settings = (
                    first_result.get("configs", {})
                    .get(first_task, {})
                    .get("metadata", {})
                )
            benchmark_record["models"][model_name] = {
                "status": status,
                "score": score,
                "n_samples": sample_total,
                "task_count": len(metrics_by_task),
                "settings": settings,
                "tasks": metrics_by_task,
                "truncation": truncation_stats(
                    model_name,
                    records,
                    tokenizer_paths.get(model_name),
                ),
                "gpu": benchmark_gpu_stats(benchmark_dir),
            }
            for task_name, metrics in sorted(metrics_by_task.items()):
                samples = (
                    int(judge["tasks"][task_name]["n_samples"])
                    if benchmark == "humaneval_infilling"
                    else int(records[task_name][1]["n-samples"][task_name]["effective"])
                )
                for metric, value in sorted(metrics.items()):
                    task_rows.append(
                        {
                            "benchmark_name": benchmark,
                            "model_name": model_name,
                            "task_name": task_name,
                            "metric": metric,
                            "score": value,
                            "n_samples": samples,
                            "cot_mode": settings.get("cot_mode"),
                            "prompt_template": settings.get("prompt_template"),
                            "wkv_mode": settings.get("wkv_mode"),
                        }
                    )
            for task_name, (result_path, _) in sorted(records.items()):
                comparison["provenance"].append(
                    {
                        "benchmark_name": benchmark,
                        "model_name": model_name,
                        "task_name": task_name,
                        "result_path": str(result_path),
                        "result_sha256": sha256(result_path),
                    }
                )
            if judge_path is not None:
                comparison["provenance"].append(
                    {
                        "benchmark_name": benchmark,
                        "model_name": model_name,
                        "task_name": "sandbox_judge",
                        "result_path": str(judge_path),
                        "result_sha256": sha256(judge_path),
                    }
                )
            judgements_path = benchmark_dir / "sandbox_judgements.jsonl"
            if judgements_path.exists():
                comparison["provenance"].append(
                    {
                        "benchmark_name": benchmark,
                        "model_name": model_name,
                        "task_name": "sandbox_judgements",
                        "result_path": str(judgements_path),
                        "result_sha256": sha256(judgements_path),
                    }
                )
            bad_metric = spec["bad_case_metric"]
            if benchmark == "humaneval_infilling":
                cases, judgements = humaneval_bad_cases(
                    model_name, benchmark_dir, args.bad_cases_per_task
                )
                all_bad_cases.extend(cases)
                judgement_by_model[model_name] = judgements
            elif benchmark == "discrim_eval":
                cases, contrasts = discrim_bad_cases(
                    model_name, records, args.bad_cases_per_task
                )
                all_bad_cases.extend(cases)
                discrim_contrasts_by_model[model_name] = contrasts
            elif bad_metric is not None:
                all_bad_cases.extend(
                    bad_cases(
                        model_name,
                        benchmark,
                        bad_metric,
                        spec["direction"],
                        records,
                        args.bad_cases_per_task,
                    )
                )

        rwkv_score = benchmark_record["models"][MODELS[0]]["score"]
        qwen_score = benchmark_record["models"][MODELS[1]]["score"]
        benchmark_record["qwen_minus_rwkv"] = (
            qwen_score - rwkv_score
            if rwkv_score is not None and qwen_score is not None
            else None
        )
        metric_deltas = {}
        rwkv_tasks = benchmark_record["models"][MODELS[0]]["tasks"]
        qwen_tasks = benchmark_record["models"][MODELS[1]]["tasks"]
        for task_name in sorted(rwkv_tasks.keys() & qwen_tasks.keys()):
            metric_deltas[task_name] = {
                metric: qwen_tasks[task_name][metric] - rwkv_tasks[task_name][metric]
                for metric in sorted(
                    rwkv_tasks[task_name].keys() & qwen_tasks[task_name].keys()
                )
            }
        benchmark_record["task_metric_deltas_qwen_minus_rwkv"] = metric_deltas
        if benchmark == "humaneval_infilling":
            rwkv_judgements = judgement_by_model.get(MODELS[0], {})
            qwen_judgements = judgement_by_model.get(MODELS[1], {})
            for key in sorted(rwkv_judgements.keys() & qwen_judgements.keys()):
                rwkv_case = rwkv_judgements[key]
                qwen_case = qwen_judgements[key]
                if rwkv_case["passed"] == qwen_case["passed"]:
                    continue
                all_paired_bad_cases.append(
                    {
                        "benchmark_name": benchmark,
                        "task_name": key[0],
                        "doc_id": key[1],
                        "metric": "pass@1",
                        "case_type": (
                            "qwen_better" if qwen_case["passed"] else "rwkv_better"
                        ),
                        "rwkv": bounded(rwkv_case),
                        "qwen": bounded(qwen_case),
                    }
                )
        elif benchmark == "discrim_eval":
            all_paired_bad_cases.extend(
                paired_discrim_bad_cases(
                    discrim_contrasts_by_model, args.bad_cases_per_task
                )
            )
        elif spec["bad_case_metric"] is not None:
            all_paired_bad_cases.extend(
                paired_bad_cases(
                    benchmark,
                    spec["bad_case_metric"],
                    spec["direction"],
                    records_by_model,
                    args.bad_cases_per_task,
                )
            )
        benchmark_record["paired_bad_case_count"] = sum(
            case["benchmark_name"] == benchmark for case in all_paired_bad_cases
        )
        comparison["benchmarks"][benchmark] = benchmark_record

    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_task.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "benchmark_name",
            "model_name",
            "task_name",
            "metric",
            "score",
            "n_samples",
            "cot_mode",
            "prompt_template",
            "wkv_mode",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(task_rows)
    with (args.output_dir / "bad_cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_bad_cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "paired_bad_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in all_paired_bad_cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# Five-benchmark RWKV/Qwen comparison",
        "",
        "Full formal evaluation with native task metrics and sandboxed HumanEval execution.",
        "",
        "| Benchmark | Primary metric | Samples (RWKV / Qwen) | RWKV7 1.5B | Qwen3.5 2B | Qwen - RWKV | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for benchmark, record in comparison["benchmarks"].items():
        rwkv = record["models"][MODELS[0]]
        qwen = record["models"][MODELS[1]]
        values = [rwkv["score"], qwen["score"], record["qwen_minus_rwkv"]]
        formatted = ["—" if value is None else f"{value:.6f}" for value in values]
        lines.append(
            f"| {benchmark} | {record['primary_metric']} | "
            f"{rwkv['n_samples']} / {qwen['n_samples']} | {formatted[0]} | "
            f"{formatted[1]} | {formatted[2]} | "
            f"{rwkv['status']} / {qwen['status']} |"
        )
    lines.extend(
        [
            "",
            "`discrim_eval` overview is the derived mean absolute value of all 14 native bias coefficients; signed native coefficients remain in `comparison.json` and `per_task.csv`.",
            "",
            "## Artifacts",
            "",
            "- `comparison.json`: native task metrics, deltas, protocol, truncation, GPU, and provenance.",
            "- `per_task.csv`: every native leaf-task metric and model setting.",
            "- `bad_cases.jsonl`: raw low-scoring cases and matched Discrim-Eval demographic contrasts per model and task.",
            "- `paired_bad_cases.jsonl`: same-document RWKV/Qwen disagreements.",
            "- `../protocol.json`: frozen model, backend, dataset, template, concurrency, and judge metadata.",
        ]
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.suite == "five":
        main_five(args)
    else:
        main_six(args)


if __name__ == "__main__":
    main()
