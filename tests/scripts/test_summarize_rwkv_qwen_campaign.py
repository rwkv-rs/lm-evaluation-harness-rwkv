import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/summarize_rwkv_qwen_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_rwkv_qwen_campaign", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_result(root: Path, model: str, benchmark: str, samples: int, metric: str):
    result_dir = root / model / benchmark / model
    result_dir.mkdir(parents=True)
    timestamp = "2026-08-18T00-00-00.000000"
    result = {
        "results": {"leaf": {f"{metric},none": 0.5}},
        "n-samples": {"leaf": {"original": samples, "effective": samples}},
        "config": {},
    }
    result_path = result_dir / f"results_{timestamp}.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    sample = {
        "doc_id": 0,
        "doc_hash": f"{benchmark}-hash",
        "target": "A",
        "resps": [["B"]],
        "filtered_resps": ["B"],
        metric: 0.0,
        "truncated": True,
        "finish_reasons": ["length"],
    }
    (result_dir / f"samples_leaf_{timestamp}.jsonl").write_text(
        json.dumps(sample) + "\n", encoding="utf-8"
    )


def test_summary_records_metrics_truncation_and_bad_cases(tmp_path):
    for model in MODULE.MODELS:
        for benchmark, spec in MODULE.BENCHMARKS.items():
            write_result(
                tmp_path,
                model,
                benchmark,
                spec["expected_samples"],
                spec["primary"],
            )

    summary, bad_cases = MODULE.summarize(tmp_path, bad_case_limit=2)

    assert summary["comparison_status"] == "complete"
    graphwalks = summary["models"][MODULE.MODELS[0]]["benchmarks"]["graphwalks"]
    assert graphwalks["accuracy"] == 0.5
    assert graphwalks["truncation"]["truncation_rate"] == 1.0
    assert len(bad_cases) == len(MODULE.MODELS) * len(MODULE.BENCHMARKS)


def test_tmmluplus_expected_samples_match_pinned_dataset():
    assert MODULE.BENCHMARKS["tmmluplus"]["expected_samples"] == 20160


def test_protocol_records_rwkv_fake_think_decoding_and_concurrency():
    protocol = MODULE.PROTOCOLS[MODULE.MODELS[0]]

    assert protocol["decoding"] == {
        "mode": "rwkv_profile",
        "temperature": 1.0,
        "top_p": 0.28,
        "top_k": 32,
    }
    assert protocol["request_concurrency"] == 25
    assert protocol["inference_concurrency"] == 24


def test_truncation_rate_counts_correct_outputs_at_limit(tmp_path):
    model = MODULE.MODELS[0]
    write_result(tmp_path, model, "graphwalks", 1, "flexible_f1")
    result_path, result = MODULE.latest_result(tmp_path / model / "graphwalks")
    sample_path = MODULE.sample_paths(result_path, result)[0]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["flexible_f1"] = 1.0
    sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    _, truncation = MODULE.samples_and_truncation(
        result_path, result, "flexible_f1"
    )

    assert truncation["truncation_rate"] == 1.0
    assert truncation["incomplete_at_output_limit_rate"] == 0.0


def test_write_outputs_includes_paired_bad_cases(tmp_path):
    cases = [
        {
            "model_name": model,
            "benchmark_name": "graphwalks",
            "doc_hash": "same",
        }
        for model in MODULE.MODELS
    ]

    paired = MODULE.paired_bad_cases(cases)

    assert len(paired) == 1
    assert set(paired[0]["models"]) == set(MODULE.MODELS)


def test_summary_aggregates_mmlu_prox_language_shards(tmp_path):
    model = MODULE.MODELS[0]
    benchmark_dir = tmp_path / model / "mmlu_prox" / "shards"
    per_shard = MODULE.BENCHMARKS["mmlu_prox"]["expected_samples"] // 29
    for index in range(29):
        shard_dir = benchmark_dir / f"lang-{index:02d}" / model
        shard_dir.mkdir(parents=True)
        result = {
            "results": {
                f"leaf-{index}": {"exact_match,custom-extract": 0.25}
            },
            "n-samples": {
                f"leaf-{index}": {
                    "original": per_shard,
                    "effective": per_shard,
                }
            },
        }
        (shard_dir / f"results_{index:02d}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    records = MODULE.latest_results(tmp_path / model / "mmlu_prox", "mmlu_prox")
    metrics, effective = MODULE.aggregate_result_metrics(
        [result for _, result in records], ("exact_match",)
    )

    assert len(records) == 29
    assert effective == MODULE.BENCHMARKS["mmlu_prox"]["expected_samples"]
    assert metrics["exact_match"] == 0.25
