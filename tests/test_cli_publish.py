import argparse
import json
import random
from pathlib import Path

import pytest

from lm_eval import evaluator
from lm_eval._cli.publish import Publish
from lm_eval.api.benchmark import BenchmarkSpec
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


_TEST_DATA = [
    {"question": "What is 2+2?", "answer": "lol"},
    {"question": "What is 3+3?", "answer": "not-lol"},
]

_LEAF_TEMPLATE = """\
task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data_path}
output_type: generate_until
doc_to_text: "{{{{question}}}}"
doc_to_target: answer
test_split: test
num_fewshot: 0
generation_kwargs:
  until: []
  max_gen_toks: 32
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
metadata:
  benchmark_name: {task_name}
"""

_CONFIG_TOML = """\
[publication]
base_url = "https://scoreboard.example"
token_env = "SCOREBOARD_PUBLICATION_TOKEN"
cot_mode = "NoCoT"

[publication.tasks.solo_leaf]
field = "reasoning"
avg_k = 1
"""


def _write_run_and_get_results_json(tmp_path: Path) -> Path:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    data_path = fixtures_dir / "test_data.json"
    data_path.write_text(json.dumps(_TEST_DATA), encoding="utf-8")
    (fixtures_dir / "solo_leaf.yaml").write_text(
        _LEAF_TEMPLATE.format(task_name="solo_leaf", data_path=str(data_path)),
        encoding="utf-8",
    )
    (fixtures_dir / "group.yaml").write_text(
        "group: unused\ntask:\n  - solo_leaf\n", encoding="utf-8"
    )

    task_manager = TaskManager(include_path=str(fixtures_dir), include_defaults=False)
    task_dict = task_manager.load(["solo_leaf"])

    random.seed(0)
    lm = DummyLM()
    output_path = tmp_path / "output"
    evaluation_tracker = EvaluationTracker(output_path=str(output_path))

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        task_callback=lambda **_kwargs: None,
        task_callback_context={
            "config": {
                "model": "rwkv7-g1j-1.5b",
                "model_args": {
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "top_k": 0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "penalty_decay": 0.0,
                    "seed": 42,
                },
                "max_length": 4096,
                "metadata": {
                    "model_name": "rwkv7-g1j-1.5b",
                    "weight_revision": "rwkv7-g1j-1.5b-20260831-ctx16384",
                },
            },
            "lm_eval_version": "test",
        },
        benchmark_map={
            "solo_leaf": BenchmarkSpec(
                name="solo_leaf",
                field="reasoning",
                strategy="standalone",
                leaf_task_names=("solo_leaf",),
            )
        },
    )

    (results_json_path,) = output_path.rglob("results.json")
    return results_json_path


def test_discover_results_skips_files_without_leaf_artifact_paths(tmp_path):
    root = tmp_path / "task_artifacts"
    benchmark_dir = root / "some_benchmark"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "results.json").write_text(
        json.dumps({"leaf_artifact_paths": {}}), encoding="utf-8"
    )
    leaf_dir = root / "some_leaf"
    leaf_dir.mkdir(parents=True)
    (leaf_dir / "results.json").write_text(json.dumps({"no": "leaf paths here"}))

    discovered = Publish._discover_results([str(root)])

    assert discovered == [benchmark_dir / "results.json"]


def test_resolve_artifact_path_rebuilds_relative_task_artifacts_path(tmp_path):
    real_root = tmp_path / "output" / "model" / "task_artifacts" / "solo_leaf"
    real_root.mkdir(parents=True)
    real_path = real_root / "results.json"
    real_path.write_text("{}", encoding="utf-8")

    anchor = tmp_path / "output" / "model" / "task_artifacts" / "benchmark" / "results.json"
    stale_recorded = "/some/other/cwd/output/model/task_artifacts/solo_leaf/results.json"

    resolved = Publish._resolve_artifact_path(stale_recorded, anchor)

    assert Path(resolved) == real_path


def test_publish_dry_run_validates_real_artifact_without_uploading(tmp_path, capsys):
    results_json_path = _write_run_and_get_results_json(tmp_path)
    config_path = tmp_path / "publication.toml"
    config_path.write_text(_CONFIG_TOML, encoding="utf-8")

    args = argparse.Namespace(
        results=[str(results_json_path)],
        config=str(config_path),
        dry_run=True,
        rerun_reason=None,
        fail_fast=False,
    )

    Publish._execute(args)

    captured = capsys.readouterr()
    assert "[dry-run] solo_leaf" in captured.out


def test_publish_repairs_an_artifact_written_before_weight_revision_metadata(
    tmp_path, capsys
):
    """Older artifacts only name the short API alias.

    The publish config already carries the full weight revision (top-level for
    a single-model file, per ``[[models]]`` entry for a campaign), so replaying
    such an artifact must use it instead of failing on the missing ctx_len.
    """
    results_json_path = _write_run_and_get_results_json(tmp_path)
    results_doc = json.loads(results_json_path.read_text("utf-8"))
    results_doc["config"].pop("metadata", None)
    results_json_path.write_text(json.dumps(results_doc), encoding="utf-8")

    config_path = tmp_path / "publication.toml"
    config_path.write_text(
        """\
[[models]]
model_name = "rwkv7-g1j-1.5b"
weight_revision = "rwkv7-g1j-1.5b-20260831-ctx16384"

[publication]
base_url = "https://scoreboard.example"
token_env = "SCOREBOARD_PUBLICATION_TOKEN"
cot_mode = "NoCoT"

[publication.tasks.solo_leaf]
field = "reasoning"
avg_k = 1
""",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        results=[str(results_json_path)],
        config=str(config_path),
        dry_run=True,
        rerun_reason=None,
        fail_fast=False,
    )

    Publish._execute(args)

    captured = capsys.readouterr()
    assert "[dry-run] solo_leaf" in captured.out


def test_publish_reports_failure_and_exits_nonzero_on_bad_policy(tmp_path, capsys):
    results_json_path = _write_run_and_get_results_json(tmp_path)
    config_path = tmp_path / "publication.toml"
    config_path.write_text(
        """\
[publication]
base_url = "https://scoreboard.example"
token_env = "SCOREBOARD_PUBLICATION_TOKEN"
cot_mode = "NoCoT"

[publication.tasks.solo_leaf]
field = "coding"
""",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        results=[str(results_json_path)],
        config=str(config_path),
        dry_run=True,
        rerun_reason=None,
        fail_fast=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        Publish._execute(args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "[failed]" in captured.err
