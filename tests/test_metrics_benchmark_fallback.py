from lm_eval.api.metrics import aggregate_benchmark_fallback


def test_aggregate_benchmark_fallback_averages_shared_metric():
    task_metrics = {
        "xquad_en": {
            "name": "xquad_en",
            "alias": "xquad_en",
            "sample_len": 10,
            "exact_match,none": 0.8,
            "exact_match_stderr,none": 0.1,
            "f1,none": 0.9,
            "f1_stderr,none": 0.05,
        },
        "xquad_de": {
            "name": "xquad_de",
            "alias": "xquad_de",
            "sample_len": 30,
            "exact_match,none": 0.4,
            "exact_match_stderr,none": 0.2,
            "f1,none": 0.5,
            "f1_stderr,none": 0.1,
        },
    }
    result, primary = aggregate_benchmark_fallback(
        "xquad", ["xquad_en", "xquad_de"], task_metrics
    )

    assert primary == "exact_match"
    assert result["name"] == "xquad"
    assert result["alias"] == "xquad"
    assert result["sample_len"] == 40
    assert result["exact_match,none"] == (0.8 * 10 + 0.4 * 30) / 40
    assert result["f1,none"] == (0.9 * 10 + 0.5 * 30) / 40
    assert result["exact_match_stderr,none"] != "N/A"
    assert result["sample_count"] == {"exact_match,none": 40, "f1,none": 40}


def test_aggregate_benchmark_fallback_no_shared_metric_produces_no_score():
    task_metrics = {
        "infinitebench_passkey": {
            "name": "infinitebench_passkey",
            "sample_len": 5,
            "int_match,none": 1.0,
            "int_match_stderr,none": 0.0,
        },
        "infinitebench_longbook_sum_en": {
            "name": "infinitebench_longbook_sum_en",
            "sample_len": 5,
            "rouge,none": 0.3,
            "rouge_stderr,none": "N/A",
        },
    }
    result, primary = aggregate_benchmark_fallback(
        "infinitebench",
        ["infinitebench_passkey", "infinitebench_longbook_sum_en"],
        task_metrics,
    )

    assert primary is None
    assert result == {
        "name": "infinitebench",
        "alias": "infinitebench",
        "sample_len": 10,
    }


def test_aggregate_benchmark_fallback_preserves_all_shared_metrics():
    # Mirrors discrim_eval: every leaf shares the same seven named bias
    # metrics -- none of them should be dropped in favor of a single one.
    bias_metrics = [
        "black_bias",
        "asian_bias",
        "hispanic_bias",
        "native_american_bias",
        "female_bias",
        "non_binary_bias",
        "age_bias",
    ]

    def leaf(sample_len: int, value: float) -> dict:
        entry = {"name": "leaf", "sample_len": sample_len}
        for metric in bias_metrics:
            entry[f"{metric},none"] = value
            entry[f"{metric}_stderr,none"] = "N/A"
        return entry

    task_metrics = {
        "discrim_eval_explicit": leaf(50, 0.2),
        "discrim_eval_implicit": leaf(50, 0.6),
    }
    result, primary = aggregate_benchmark_fallback(
        "discrim_eval",
        ["discrim_eval_explicit", "discrim_eval_implicit"],
        task_metrics,
    )

    for metric in bias_metrics:
        assert result[f"{metric},none"] == 0.4
    assert primary == "age_bias"  # alphabetically first, none in priority list


def test_aggregate_benchmark_fallback_skips_missing_leaf():
    task_metrics = {
        "xquad_en": {
            "name": "xquad_en",
            "sample_len": 10,
            "acc,none": 1.0,
            "acc_stderr,none": 0.0,
        },
    }
    result, primary = aggregate_benchmark_fallback(
        "xquad", ["xquad_en", "xquad_missing"], task_metrics
    )

    assert primary == "acc"
    assert result["sample_len"] == 10
    assert result["acc,none"] == 1.0


def test_aggregate_benchmark_fallback_no_leaves_with_results():
    result, primary = aggregate_benchmark_fallback("xquad", ["xquad_en"], {})

    assert primary is None
    assert result == {"name": "xquad", "alias": "xquad", "sample_len": 0}
