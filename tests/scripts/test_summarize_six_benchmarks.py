import json
from types import SimpleNamespace

from scripts.summarize_six_benchmarks import truncation_stats


def test_qwen_truncation_falls_back_to_output_token_counts(monkeypatch, tmp_path):
    result_path = tmp_path / "results_2026-08-15T00-00-00.json"
    result = {
        "config": {
            "truncation": {
                "cruxeval_output": {
                    "generated_samples": 0,
                    "truncated_samples": 0,
                    "truncation_rate": None,
                }
            }
        }
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    sample_path = tmp_path / "samples_cruxeval_output_2026-08-15T00-00-00.jsonl"
    samples = [
        {
            "arguments": {
                "gen_args_0": {"arg_1": {"max_gen_toks": 3}}
            },
            "resps": [["one two"], ["one two three"]],
        },
        {
            "arguments": {
                "gen_args_0": {"arg_1": {"max_gen_toks": 3}}
            },
            "resps": [["one"], ["one two"]],
        },
    ]
    sample_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return SimpleNamespace(
                input_ids=[text.split() for text in texts]
            )

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    stats = truncation_stats(
        "Qwen3.5-2B",
        {"cruxeval_output": (result_path, result)},
        tmp_path / "tokenizer",
    )

    assert stats == {
        "generated_samples": 2,
        "truncated_samples": 1,
        "truncation_rate": 0.5,
        "source": "posthoc_output_token_count",
    }
