import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from lm_eval.tasks.discrim_eval.utils import process_results


_TASK_DIR = Path(__file__).parents[1] / "lm_eval" / "tasks" / "discrim_eval"


def _make_doc(race, gender, age, template_id):
    return {
        "race": race,
        "gender": gender,
        "age": age,
        "decision_question_id": template_id,
    }


def test_shared_task_configs_keep_the_native_multiple_choice_protocol():
    for name in ("discrim_eval_explicit", "discrim_eval_implicit"):
        config = yaml.load(
            (_TASK_DIR / f"{name}.yaml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        assert config["output_type"] == "multiple_choice"
        assert config["doc_to_text"].startswith("{{filled_template}}")
        assert config["doc_to_target"] == "yes"
        assert config["doc_to_choice"] == ["yes", "Yes", "no", "No"]
        assert config["gen_prefix"].startswith("Based on the information provided")
        assert {
            metric["aggregation"] for metric in config["metric_list"]
        } == {"utils.agg_demographic_bias_regression"}


def test_process_results_uses_the_native_four_candidate_logits():
    doc = _make_doc("Black", "Female", 30.0, 0)
    results = [(-1.0, True), (-2.0, False), (-3.0, True), (-4.0, False)]

    out = process_results(doc, results)

    yes_prob = np.exp(-1.0) + np.exp(-2.0)
    no_prob = np.exp(-3.0) + np.exp(-4.0)
    pnorm_yes = yes_prob / (yes_prob + no_prob)
    expected = math.log(pnorm_yes) - math.log1p(-pnorm_yes)
    assert out["black_bias"][:2] == (("black", "female", 30.0, 0), "black_bias")
    assert out["black_bias"][2] == pytest.approx(expected)
    assert set(out) == {
        "black_bias",
        "asian_bias",
        "hispanic_bias",
        "native_american_bias",
        "female_bias",
        "non_binary_bias",
        "age_bias",
    }
