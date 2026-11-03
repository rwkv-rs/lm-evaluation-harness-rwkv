"""Integration tests for sequential benchmark execution in ``evaluator.py``.

Benchmarks run one at a time -- lighteval-rwkv's evaluation loop -- in
``_order_benchmarks`` order: benchmarks that do not generate first, then
fewest pending rollouts first. There is no per-benchmark worker thread and no
cross-benchmark concurrency any more, so the endpoint's single flat request
budget is never multiplied.
"""

import json
from pathlib import Path

import lm_eval.evaluator as evaluator
from lm_eval.api.benchmark import BenchmarkSpec
from lm_eval.api.checkpoint import CheckpointStore, compute_config_fingerprint
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


_DATA_BY_SIZE = {
    1: [
        {"question": "1+1?", "choices": ["1", "2", "3"], "answer": 1},
    ],
    2: [
        {"question": "1+1?", "choices": ["1", "2", "3"], "answer": 1},
        {"question": "2+2?", "choices": ["3", "4", "5"], "answer": 1},
    ],
    4: [
        {"question": "1+1?", "choices": ["1", "2", "3"], "answer": 1},
        {"question": "2+2?", "choices": ["3", "4", "5"], "answer": 1},
        {"question": "3+3?", "choices": ["5", "6", "7"], "answer": 1},
        {"question": "4+4?", "choices": ["7", "8", "9"], "answer": 1},
    ],
}

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
num_fewshot: 0
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""

_GENERATIVE_TEMPLATE = """\
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
  until:
    - "\\n"
  max_gen_toks: 8
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
"""


def _write_task(
    fixtures_dir: Path,
    task_name: str,
    num_docs: int,
    *,
    template: str = _LEAF_TEMPLATE,
) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    data_path = fixtures_dir / f"{task_name}_data.json"
    data_path.write_text(json.dumps(_DATA_BY_SIZE[num_docs]), encoding="utf-8")
    (fixtures_dir / f"{task_name}.yaml").write_text(
        template.format(task_name=task_name, data_path=str(data_path)),
        encoding="utf-8",
    )


def _load_tasks(
    tmp_path: Path,
    sized_tasks: dict[str, int],
    *,
    generative_tasks: tuple[str, ...] = (),
) -> dict:
    fixtures_dir = tmp_path / "fixtures"
    for task_name, num_docs in sized_tasks.items():
        _write_task(
            fixtures_dir,
            task_name,
            num_docs,
            template=(
                _GENERATIVE_TEMPLATE if task_name in generative_tasks else _LEAF_TEMPLATE
            ),
        )
    task_manager = TaskManager(include_path=str(fixtures_dir), include_defaults=False)
    return task_manager.load(list(sized_tasks))


def _standalone_benchmark_map(task_names: list) -> dict:
    return {
        task_name: BenchmarkSpec(
            name=task_name,
            field="reasoning",
            strategy="standalone",
            leaf_task_names=(task_name,),
        )
        for task_name in task_names
    }


class _RecordingDummyLM(DummyLM):
    """DummyLM that records, in dispatch order, which benchmark each batch
    belongs to -- so tests can prove benchmarks never interleave."""

    def __init__(self):
        super().__init__()
        self.dispatched_tasks: list[str] = []

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        self.dispatched_tasks.extend(request.task_name for request in requests)
        return super().loglikelihood(requests, disable_tqdm=True)

    def generate_until(self, requests, disable_tqdm: bool = False):
        self.dispatched_tasks.extend(request.task_name for request in requests)
        return super().generate_until(requests, disable_tqdm=True)


def _publishing_callback(calls: list):
    def callback(task_name, artifact_paths):
        calls.append(task_name)

    return callback


def test_benchmarks_publish_fewest_rollouts_first(tmp_path):
    task_dict = _load_tasks(
        tmp_path, {"solo_small_a": 2, "solo_small_b": 2, "solo_big": 4}
    )
    calls: list[str] = []

    evaluator.evaluate(
        lm=DummyLM(),
        task_dict=task_dict,
        bootstrap_iters=0,
        task_callback=_publishing_callback(calls),
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_standalone_benchmark_map(
            ["solo_small_a", "solo_small_b", "solo_big"]
        ),
    )

    # Smallest benchmarks first (ties keep the configured order), the big one last.
    assert calls == ["solo_small_a", "solo_small_b", "solo_big"]


def test_generative_benchmarks_run_after_scored_ones(tmp_path):
    task_dict = _load_tasks(
        tmp_path,
        {"gen_tiny": 1, "scored_big": 4},
        generative_tasks=("gen_tiny",),
    )
    calls: list[str] = []

    evaluator.evaluate(
        lm=DummyLM(),
        task_dict=task_dict,
        bootstrap_iters=0,
        task_callback=_publishing_callback(calls),
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_standalone_benchmark_map(["gen_tiny", "scored_big"]),
    )

    assert calls == ["scored_big", "gen_tiny"]


def test_benchmarks_never_interleave_requests(tmp_path):
    task_dict = _load_tasks(tmp_path, {"seq_first": 4, "seq_second": 2})
    lm = _RecordingDummyLM()

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        task_callback=_publishing_callback([]),
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_standalone_benchmark_map(["seq_first", "seq_second"]),
    )

    # seq_second (fewer rollouts) runs first, and each benchmark's requests
    # form one contiguous block: a task name never reappears after another
    # task name has been dispatched.
    ordered_names = list(dict.fromkeys(lm.dispatched_tasks))
    assert ordered_names == ["seq_second", "seq_first"]
    blocks = [
        name
        for index, name in enumerate(lm.dispatched_tasks)
        if index == 0 or lm.dispatched_tasks[index - 1] != name
    ]
    assert blocks == ["seq_second", "seq_first"]


def test_every_benchmark_is_published_exactly_once(tmp_path):
    task_dict = _load_tasks(tmp_path, {"once_a": 2, "once_b": 2, "once_c": 4})
    calls: list[str] = []

    evaluator.evaluate(
        lm=DummyLM(),
        task_dict=task_dict,
        bootstrap_iters=0,
        task_callback=_publishing_callback(calls),
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_standalone_benchmark_map(["once_a", "once_b", "once_c"]),
    )

    assert sorted(calls) == ["once_a", "once_b", "once_c"]


def test_sequential_run_marks_all_benchmarks_completed_in_checkpoint(tmp_path):
    task_dict = _load_tasks(tmp_path, {"ckpt_a": 2, "ckpt_b": 4})
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint = compute_config_fingerprint({"config": "sequential-test"})
    checkpoint_store = CheckpointStore(db_path, fingerprint)

    evaluator.evaluate(
        lm=DummyLM(),
        task_dict=task_dict,
        bootstrap_iters=0,
        checkpoint_store=checkpoint_store,
        benchmark_map=_standalone_benchmark_map(["ckpt_a", "ckpt_b"]),
    )

    for task_name in ("ckpt_a", "ckpt_b"):
        benchmark = checkpoint_store.get_benchmark(task_name)
        assert benchmark["status"] == "completed"
    checkpoint_store.close()
