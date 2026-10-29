import pytest

from lm_eval.api.benchmark import read_benchmark_fields
from lm_eval.config.evaluate_config import EvaluatorConfig, _classify_benchmark
from lm_eval.tasks import TaskManager


@pytest.fixture(scope="module")
def task_manager():
    return TaskManager()


def test_group_with_aggregate_metric_list_is_group_aggregate(task_manager):
    spec = _classify_benchmark(
        "wmdp", "knowledge", task_manager.match_tasks(["wmdp"]), task_manager
    )

    assert spec.field == "knowledge"
    assert spec.strategy == "group_aggregate"
    assert spec.leaf_task_names is None


def test_group_without_aggregate_metric_list_is_group_fallback(task_manager):
    spec = _classify_benchmark(
        "infinitebench",
        "reasoning",
        task_manager.match_tasks(["infinitebench"]),
        task_manager,
    )

    assert spec.strategy == "group_fallback"
    assert spec.leaf_task_names is None


def test_tag_is_tag_fallback_with_full_leaf_set(task_manager):
    spec = _classify_benchmark(
        "model_written_evals",
        "reasoning",
        task_manager.match_tasks(["model_written_evals"]),
        task_manager,
    )

    assert spec.strategy == "tag_fallback"
    assert spec.leaf_task_names is not None
    assert len(spec.leaf_task_names) == 187


def test_standalone_task_is_standalone(task_manager):
    spec = _classify_benchmark(
        "race", "reasoning", task_manager.match_tasks(["race"]), task_manager
    )

    assert spec.strategy == "standalone"
    assert spec.leaf_task_names == ("race",)


def test_glob_matching_a_group_raises(task_manager):
    with pytest.raises(ValueError, match="not a leaf task"):
        _classify_benchmark(
            "wmdp*", "knowledge", ["wmdp", "wmdp_bio"], task_manager
        )


def test_glob_matching_only_leaf_tasks_is_tag_fallback(task_manager):
    spec = _classify_benchmark(
        "wmdp_*",
        "knowledge",
        ["wmdp_bio", "wmdp_chem", "wmdp_cyber"],
        task_manager,
    )

    assert spec.strategy == "tag_fallback"
    assert spec.leaf_task_names == ("wmdp_bio", "wmdp_chem", "wmdp_cyber")


def test_benchmark_fields_are_read_from_explicit_top_level_mapping(tmp_path):
    path = tmp_path / "benchmarks.toml"
    path.write_text(
        'knowledge = ["wmdp"]\nreasoning = ["race"]\ncoding = []\n',
        encoding="utf-8",
    )

    assert read_benchmark_fields(path) == {
        "wmdp": "knowledge",
        "race": "reasoning",
    }


def test_benchmark_field_config_rejects_duplicate_selector(tmp_path):
    path = tmp_path / "benchmarks.toml"
    path.write_text(
        'knowledge = ["race"]\nreasoning = ["race"]\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate benchmark selector"):
        read_benchmark_fields(path)


def test_process_tasks_rejects_selector_without_explicit_field(tmp_path):
    path = tmp_path / "benchmarks.toml"
    path.write_text('reasoning = ["race"]\n', encoding="utf-8")
    config = EvaluatorConfig(tasks=["wmdp"], benchmark_config=str(path))

    with pytest.raises(ValueError, match="has no explicit field"):
        config.process_tasks()
