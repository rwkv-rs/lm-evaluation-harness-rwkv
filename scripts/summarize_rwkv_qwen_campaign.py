#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "Qwen3.5-2B",
)
BENCHMARKS = {
    "graphwalks": {
        "metrics": ("f1", "flexible_f1"),
        "primary": "flexible_f1",
        "expected_samples": 350,
    },
    "multiblimp": {
        "metrics": ("acc", "acc_norm"),
        "primary": "acc",
        "expected_samples": 121305,
    },
    "logiqa2": {
        "metrics": ("acc", "acc_norm"),
        "primary": "acc",
        "expected_samples": 1572,
    },
    "tmmluplus": {
        "metrics": ("acc", "acc_norm"),
        "primary": "acc",
        "expected_samples": 20160,
    },
    "mmlu_prox": {
        "metrics": ("exact_match",),
        "primary": "exact_match",
        "expected_samples": 341011,
    },
}
PROTOCOLS = {
    MODELS[0]: {
        "backend": "rwkv7-http",
        "cot_mode": "fake_think",
        "prompt_template": "assistant",
        "wkv_mode": "fp32io16",
        "decoding": {
            "mode": "rwkv_profile",
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
        },
        "request_concurrency": 25,
        "inference_concurrency": 24,
    },
    MODELS[1]: {
        "backend": "local-completions",
        "cot_mode": "task_native",
        "prompt_template": "qwen_official",
        "wkv_mode": None,
        "decoding": {"mode": "task_native"},
        "request_concurrency": 5,
        "inference_concurrency": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/formal-five-benchmarks-20260818",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bad-cases-per-benchmark", type=int, default=20)
    return parser.parse_args()


def metric_value(values: dict[str, Any], metric: str) -> float | None:
    for key, value in values.items():
        if key != metric and not key.startswith(f"{metric},"):
            continue
        if isinstance(value, int | float) and math.isfinite(value):
            return float(value)
    return None


def latest_result(benchmark_dir: Path) -> tuple[Path, dict] | None:
    paths = sorted(
        benchmark_dir.glob("**/results_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not paths:
        return None
    path = paths[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def latest_results(benchmark_dir: Path, benchmark: str) -> list[tuple[Path, dict]]:
    shard_root = benchmark_dir / "shards"
    if benchmark == "mmlu_prox" and shard_root.is_dir():
        records = []
        for shard_dir in sorted(path for path in shard_root.iterdir() if path.is_dir()):
            record = latest_result(shard_dir)
            if record is not None:
                records.append(record)
        return records
    record = latest_result(benchmark_dir)
    return [] if record is None else [record]


def leaf_tasks(result: dict) -> list[str]:
    return [
        task_name
        for task_name in result.get("n-samples", {})
        if task_name in result.get("results", {})
    ]


def aggregate_metrics(result: dict, metrics: tuple[str, ...]) -> tuple[dict, int]:
    tasks = leaf_tasks(result)
    effective = sum(
        int(result["n-samples"][task_name]["effective"]) for task_name in tasks
    )
    aggregated = {}
    for metric in metrics:
        values = []
        for task_name in tasks:
            value = metric_value(result["results"][task_name], metric)
            if value is None:
                continue
            samples = int(result["n-samples"][task_name]["effective"])
            values.append((value, samples))
        denominator = sum(samples for _, samples in values)
        aggregated[metric] = (
            sum(value * samples for value, samples in values) / denominator
            if denominator
            else None
        )
    return aggregated, effective


def aggregate_result_metrics(
    results: list[dict], metrics: tuple[str, ...]
) -> tuple[dict, int]:
    per_result = [aggregate_metrics(result, metrics) for result in results]
    effective = sum(result_effective for _, result_effective in per_result)
    aggregated = {}
    for metric in metrics:
        weighted = [
            (values.get(metric), result_effective)
            for values, result_effective in per_result
            if values.get(metric) is not None
        ]
        denominator = sum(samples for _, samples in weighted)
        aggregated[metric] = (
            sum(value * samples for value, samples in weighted) / denominator
            if denominator
            else None
        )
    return aggregated, effective


def sample_paths(result_path: Path, result: dict) -> list[Path]:
    timestamp = result_path.stem.removeprefix("results_")
    return [
        result_path.parent / f"samples_{task_name}_{timestamp}.jsonl"
        for task_name in leaf_tasks(result)
    ]


def sample_metric(sample: dict, metric: str) -> float | None:
    value = sample.get(metric)
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    for key, candidate in sample.items():
        if key.startswith(f"{metric},") and isinstance(candidate, int | float):
            return float(candidate)
    return None


def sample_is_correct(sample: dict, primary_metric: str) -> bool:
    value = sample_metric(sample, primary_metric)
    return value is not None and value >= 1.0


def samples_and_truncation(
    result_path: Path, result: dict, primary_metric: str
) -> tuple[list[dict], dict[str, Any]]:
    samples = []
    generated = 0
    output_limit = 0
    incomplete_at_limit = 0
    for path in sample_paths(result_path, result):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                sample["task_name"] = path.name.split("samples_", 1)[1].rsplit(
                    f"_{result_path.stem.removeprefix('results_')}.jsonl", 1
                )[0]
                samples.append(sample)
                if "truncated" not in sample:
                    continue
                generated += 1
                truncated = bool(sample["truncated"])
                output_limit += int(truncated)
                incomplete_at_limit += int(
                    truncated and not sample_is_correct(sample, primary_metric)
                )
    return samples, {
        "generated_samples": generated,
        "output_limit_samples": output_limit,
        "incomplete_at_output_limit_samples": incomplete_at_limit,
        "output_limit_rate": output_limit / generated if generated else None,
        "truncation_rate": output_limit / generated if generated else None,
        "incomplete_at_output_limit_rate": (
            incomplete_at_limit / generated if generated else None
        ),
        "definition": "finish_reason=length divided by generated samples",
    }


def result_samples_and_truncation(
    records: list[tuple[Path, dict]], primary_metric: str
) -> tuple[list[dict], dict[str, Any]]:
    samples = []
    generated = 0
    output_limit = 0
    incomplete_at_limit = 0
    for result_path, result in records:
        result_samples, truncation = samples_and_truncation(
            result_path, result, primary_metric
        )
        samples.extend(result_samples)
        generated += truncation["generated_samples"]
        output_limit += truncation["output_limit_samples"]
        incomplete_at_limit += truncation["incomplete_at_output_limit_samples"]
    return samples, {
        "generated_samples": generated,
        "output_limit_samples": output_limit,
        "incomplete_at_output_limit_samples": incomplete_at_limit,
        "output_limit_rate": output_limit / generated if generated else None,
        "truncation_rate": output_limit / generated if generated else None,
        "incomplete_at_output_limit_rate": (
            incomplete_at_limit / generated if generated else None
        ),
        "definition": "finish_reason=length divided by generated samples",
    }


def gpu_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "active_utilization_p95": None,
            "peak_memory_ratio": None,
            "minimum_free_memory_ratio": None,
            "meets_target": None,
        }
    utilization = []
    memory_ratios = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                util = float(row["utilization_gpu"])
                used = float(row["memory_used"])
                total = float(row["memory_total"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if util > 0:
                utilization.append(util)
            memory_ratios.append(used / total)
    active_p95 = None
    if utilization:
        ordered = sorted(utilization)
        active_p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
    peak_memory = max(memory_ratios) if memory_ratios else None
    free_memory = 1 - peak_memory if peak_memory is not None else None
    return {
        "active_utilization_mean": (
            statistics.fmean(utilization) if utilization else None
        ),
        "active_utilization_p95": active_p95,
        "peak_memory_ratio": peak_memory,
        "minimum_free_memory_ratio": free_memory,
        "meets_target": (
            active_p95 is not None
            and active_p95 >= 97
            and free_memory is not None
            and free_memory < 0.10
        ),
    }


def compact_bad_case(
    model_name: str, benchmark: str, sample: dict, primary_metric: str
) -> dict:
    return {
        "model_name": model_name,
        "benchmark_name": benchmark,
        "task_name": sample.get("task_name"),
        "doc_id": sample.get("doc_id"),
        "doc_hash": sample.get("doc_hash"),
        "target": sample.get("target"),
        "response": sample.get("resps"),
        "filtered_response": sample.get("filtered_resps"),
        "primary_metric": primary_metric,
        "primary_value": sample_metric(sample, primary_metric),
        "truncated": sample.get("truncated"),
        "finish_reasons": sample.get("finish_reasons"),
        "doc": sample.get("doc"),
    }


def summarize(results_root: Path, bad_case_limit: int) -> tuple[dict, list[dict]]:
    summary = {
        "results_root": str(results_root),
        "models": {},
        "comparison_status": "incomplete",
    }
    bad_cases = []
    complete = True
    for model_name in MODELS:
        model_summary = {
            "protocol": PROTOCOLS[model_name],
            "gpu": gpu_stats(results_root / model_name / "gpu_telemetry.csv"),
            "benchmarks": {},
        }
        for benchmark, benchmark_spec in BENCHMARKS.items():
            records = latest_results(
                results_root / model_name / benchmark, benchmark
            )
            if not records:
                complete = False
                model_summary["benchmarks"][benchmark] = {"status": "missing"}
                continue
            metrics, effective = aggregate_result_metrics(
                [result for _, result in records], benchmark_spec["metrics"]
            )
            samples, truncation = result_samples_and_truncation(
                records, benchmark_spec["primary"]
            )
            status = (
                "complete"
                if effective == benchmark_spec["expected_samples"]
                else "sample_count_mismatch"
            )
            complete &= status == "complete"
            model_summary["benchmarks"][benchmark] = {
                "status": status,
                "result_paths": [str(result_path) for result_path, _ in records],
                "n_result_shards": len(records),
                "n_samples": effective,
                "expected_samples": benchmark_spec["expected_samples"],
                "primary_metric": benchmark_spec["primary"],
                "accuracy": metrics.get(benchmark_spec["primary"]),
                "k_metrics": metrics,
                "truncation": truncation,
            }
            incorrect = [
                sample
                for sample in samples
                if not sample_is_correct(sample, benchmark_spec["primary"])
            ]
            bad_cases.extend(
                compact_bad_case(
                    model_name, benchmark, sample, benchmark_spec["primary"]
                )
                for sample in incorrect[:bad_case_limit]
            )
        summary["models"][model_name] = model_summary
    summary["comparison_status"] = "complete" if complete else "incomplete"
    return summary, bad_cases


def paired_bad_cases(bad_cases: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for case in bad_cases:
        doc_hash = case.get("doc_hash")
        if not isinstance(doc_hash, str):
            continue
        key = (case["benchmark_name"], doc_hash)
        by_key.setdefault(key, {})[case["model_name"]] = case
    return [
        {"benchmark_name": key[0], "doc_hash": key[1], "models": records}
        for key, records in by_key.items()
        if len(records) == len(MODELS)
    ]


def write_outputs(output_dir: Path, summary: dict, bad_cases: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "benchmark_name",
            "model_name",
            "status",
            "n_samples",
            "primary_metric",
            "accuracy",
            "truncation_rate",
            "cot_mode",
            "prompt_template",
            "wkv_mode",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model in summary["models"].items():
            for benchmark, values in model["benchmarks"].items():
                writer.writerow(
                    {
                        "benchmark_name": benchmark,
                        "model_name": model_name,
                        "status": values["status"],
                        "n_samples": values.get("n_samples"),
                        "primary_metric": values.get("primary_metric"),
                        "accuracy": values.get("accuracy"),
                        "truncation_rate": values.get("truncation", {}).get(
                            "truncation_rate"
                        ),
                        "cot_mode": model["protocol"]["cot_mode"],
                        "prompt_template": model["protocol"]["prompt_template"],
                        "wkv_mode": model["protocol"]["wkv_mode"],
                    }
                )
    with (output_dir / "bad_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in bad_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    with (output_dir / "paired_bad_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in paired_bad_cases(bad_cases):
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    lines = [
        "# RWKV7 1.5B / Qwen3.5 2B Benchmark Comparison",
        "",
        f"Status: `{summary['comparison_status']}`",
        "",
        "| Benchmark | Model | Samples | Primary | Score | Truncation | Status |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for benchmark in BENCHMARKS:
        for model_name in MODELS:
            values = summary["models"][model_name]["benchmarks"][benchmark]
            score = values.get("accuracy")
            truncation = values.get("truncation", {}).get("truncation_rate")
            lines.append(
                "| {benchmark} | {model} | {samples} | {metric} | {score} | "
                "{truncation} | {status} |".format(
                    benchmark=benchmark,
                    model=model_name,
                    samples=values.get("n_samples", "—"),
                    metric=values.get("primary_metric", "—"),
                    score="—" if score is None else f"{score:.6f}",
                    truncation=("—" if truncation is None else f"{truncation:.4%}"),
                    status=values["status"],
                )
            )
    lines.extend(["", "## Throughput Checks", ""])
    for model_name in MODELS:
        gpu = summary["models"][model_name]["gpu"]
        lines.append(
            f"- {model_name}: active p95 utilization="
            f"{gpu.get('active_utilization_p95')}, minimum free memory ratio="
            f"{gpu.get('minimum_free_memory_ratio')}, target={gpu.get('meets_target')}"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_root / "comparison"
    summary, bad_cases = summarize(args.results_root, args.bad_cases_per_benchmark)
    write_outputs(output_dir, summary, bad_cases)
    print(
        json.dumps({"status": summary["comparison_status"], "output": str(output_dir)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
