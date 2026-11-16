import json
import random
from pathlib import Path

import pytest

import lm_eval.evaluator as evaluator
from lm_eval.api.benchmark import BenchmarkSpec
from lm_eval.api.metrics import aggregate_benchmark_fallback
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


_TEST_DATA = [
    {"question": "What is 2+2?", "choices": ["1", "2", "3", "4"], "answer": 3},
    {"question": "What is 3+3?", "choices": ["4", "5", "6", "7"], "answer": 2},
    {"question": "What is 4+4?", "choices": ["6", "7", "8", "9"], "answer": 2},
    {"question": "What is 5+5?", "choices": ["8", "9", "10", "11"], "answer": 2},
]

_LEAF_TEMPLATE = """\
task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data_path}
output_type: multiple_choice
doc_to_text: "{{{{question}}}}"
doc_to_target: answer
doc_to_choice: choices
test_split: test
num_fewshot: {num_fewshot}
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""

_GROUP_YAML = """\
group: test_group
task:
  - group_task_fs0
  - group_task_fs1
"""


@pytest.fixture
def benchmark_fixture_task_manager(tmp_path):
    """A `group_fallback` benchmark (no `aggregate_metric_list`) with two leaves
    that share the `acc` metric -- built fresh per-test so the leaves can
    actually run (`build_all_requests`), unlike the load-only
    tests/test_configs/group.yaml fixture used elsewhere.
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    data_path = fixtures_dir / "test_data.json"
    data_path.write_text(json.dumps(_TEST_DATA), encoding="utf-8")

    (fixtures_dir / "task_fs0.yaml").write_text(
        _LEAF_TEMPLATE.format(
            task_name="group_task_fs0", data_path=str(data_path), num_fewshot=0
        ),
        encoding="utf-8",
    )
    (fixtures_dir / "task_fs1.yaml").write_text(
        _LEAF_TEMPLATE.format(
            task_name="group_task_fs1", data_path=str(data_path), num_fewshot=1
        ),
        encoding="utf-8",
    )
    (fixtures_dir / "group.yaml").write_text(_GROUP_YAML, encoding="utf-8")

    return TaskManager(include_path=str(fixtures_dir), include_defaults=False)


@pytest.fixture
def recording_callback():
    calls = []

    def callback(task_name, artifact_paths):
        calls.append({"task_name": task_name, "artifact_paths": artifact_paths})
        return {"publication": "created", "uploaded": True}

    callback.calls = calls
    return callback


def _benchmark_map():
    return {
        "test_group": BenchmarkSpec(
            name="test_group", field="reasoning", strategy="group_fallback"
        )
    }


def _run_test_group(
    task_manager, recording_callback, tmp_path, seed=0, checkpoint_store=None
):
    random.seed(seed)
    task_dict = task_manager.load(["test_group"])
    lm = DummyLM()
    evaluation_tracker = EvaluationTracker(output_path=str(tmp_path / "output"))
    task_callback_context = {
        "config": {"model": "dummy", "model_args": {}},
        "lm_eval_version": "test",
    }

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        checkpoint_store=checkpoint_store,
        task_callback=recording_callback,
        task_callback_context=task_callback_context,
        benchmark_map=_benchmark_map(),
    )
    return recording_callback.calls, evaluation_tracker


def test_leaf_tasks_never_published_only_benchmark_is(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    calls, _ = _run_test_group(
        benchmark_fixture_task_manager, recording_callback, tmp_path
    )

    assert len(calls) == 1
    assert calls[0]["task_name"] == "test_group"
    assert calls[0]["task_name"] not in ("group_task_fs0", "group_task_fs1")
    results_doc = json.loads(
        Path(calls[0]["artifact_paths"]["results"]).read_text("utf-8")
    )
    assert results_doc["benchmark"] == {
        "name": "test_group",
        "field": "reasoning",
    }
    metadata_doc = json.loads(
        Path(calls[0]["artifact_paths"]["metadata"]).read_text("utf-8")
    )
    assert metadata_doc["benchmark"] == results_doc["benchmark"]


def test_simple_evaluate_writes_run_metadata_into_the_published_artifact(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    """The published results.json must carry the run metadata.

    Scoreboard publication reads ``config.metadata.weight_revision`` from the
    artifact to derive the served weight's ``ctx_len``; the API-facing
    ``model_name`` alone is only a short serving alias. ``simple_evaluate``
    builds the artifact's run config, so it has to copy its ``metadata``
    argument there -- both into the live callback context and into the final
    results document.
    """
    metadata = {
        "model_name": "rwkv7-g1j-1.5b",
        "weight_revision": "rwkv7-g1j-1.5b-20260831-ctx16384",
    }

    results = evaluator.simple_evaluate(
        model="dummy",
        model_args=(
            "model=rwkv7-g1j-1.5b,temperature=0.96,top_p=0.76,top_k=32,"
            "presence_penalty=1.0,frequency_penalty=0.1,penalty_decay=0.988,seed=42"
        ),
        tasks=["test_group"],
        limit=1,
        bootstrap_iters=0,
        task_manager=benchmark_fixture_task_manager,
        evaluation_tracker=EvaluationTracker(output_path=str(tmp_path / "output")),
        metadata=metadata,
        task_callback=recording_callback,
        benchmark_map=_benchmark_map(),
    )

    assert results is not None
    assert results["config"]["metadata"] == metadata
    (call,) = recording_callback.calls
    results_doc = json.loads(Path(call["artifact_paths"]["results"]).read_text("utf-8"))
    assert results_doc["config"]["metadata"] == metadata

    # The artifact is what the scoreboard callback reads. A short API alias
    # plus metadata.weight_revision must be enough to build the upload payload.
    from lm_eval.loggers.scoreboard import PublicationConfig, build_score

    _, score = build_score(
        results_doc,
        PublicationConfig.from_mapping(
            {
                "base_url": "https://scoreboard.example",
                "token_env": "SCOREBOARD_PUBLICATION_TOKEN",
                "cot_mode": "CoT",
                "tasks": {"test_group": {"field": "reasoning"}},
                "timeout": 60,
                "control_timeout": 30,
                "retries": 0,
                "retry_delay": 0,
                "dry_run": True,
            }
        ),
    )
    assert score["model"]["ctx_len"] == 16384


def test_published_aggregate_matches_direct_fallback_computation(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    calls, _ = _run_test_group(
        benchmark_fixture_task_manager, recording_callback, tmp_path
    )

    artifact_paths = calls[0]["artifact_paths"]
    results_path = Path(artifact_paths["results"])
    results_doc = json.loads(results_path.read_text("utf-8"))
    published_aggregate = results_doc["results"]["test_group"]

    leaf_artifact_paths = results_doc["leaf_artifact_paths"]
    leaf_metrics = {}
    for leaf_name in ("group_task_fs0", "group_task_fs1"):
        leaf_results_path = Path(leaf_artifact_paths[leaf_name]["results"])
        leaf_doc = json.loads(leaf_results_path.read_text("utf-8"))
        leaf_metrics[leaf_name] = leaf_doc["results"][leaf_name]

    expected_aggregate, expected_primary = aggregate_benchmark_fallback(
        "test_group", ["group_task_fs0", "group_task_fs1"], leaf_metrics
    )

    assert published_aggregate == expected_aggregate
    assert results_doc["primary_metric_hint"] == expected_primary
    assert expected_primary == "acc"


def test_avg_at_k_aggregation_does_not_overwrite_score_with_stderr():
    metrics = {
        "leaf": {
            "acc,none": 0.25,
            "acc_stderr,none": 0.04,
            "acc_avg@2,none": 0.5,
            "acc_avg@2_stderr,none": 0.02,
        }
    }

    averaged = evaluator._apply_avg_at_k(metrics)

    assert averaged["leaf"]["acc,none"] == 0.5
    assert averaged["leaf"]["acc_stderr,none"] == 0.02


def test_resume_with_all_leaves_precompleted_still_publishes_once(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    from lm_eval.api.checkpoint import CheckpointStore

    checkpoint_store = CheckpointStore(str(tmp_path / "checkpoint.sqlite"), "run-1")

    calls, _ = _run_test_group(
        benchmark_fixture_task_manager,
        recording_callback,
        tmp_path,
        checkpoint_store=checkpoint_store,
    )

    assert len(calls) == 1
    assert calls[0]["task_name"] == "test_group"
    benchmark_row = checkpoint_store.get_benchmark("test_group")
    assert benchmark_row["publish_status"] == "succeeded"


def test_already_published_benchmark_skips_callback_but_updates_checkpoint(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    from lm_eval.api.checkpoint import CheckpointStore, compute_task_fingerprint

    checkpoint_store = CheckpointStore(str(tmp_path / "checkpoint.sqlite"), "run-1")
    fingerprint = compute_task_fingerprint(
        {
            "leaves": sorted(["group_task_fs0", "group_task_fs1"]),
            "strategy": "group_fallback",
        }
    )
    checkpoint_store.start_benchmark("test_group", fingerprint)
    checkpoint_store.mark_benchmark_completed("test_group", None, None)
    checkpoint_store.set_publish_status("test_group", "succeeded")

    calls, _ = _run_test_group(
        benchmark_fixture_task_manager,
        recording_callback,
        tmp_path,
        checkpoint_store=checkpoint_store,
    )

    assert len(calls) == 0
    benchmark_row = checkpoint_store.get_benchmark("test_group")
    assert benchmark_row["publish_status"] == "succeeded"
