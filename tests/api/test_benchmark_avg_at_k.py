"""Unit tests for avg@k rollout sizing.

Mirrors lighteval-rwkv's ``src/lighteval/models/rwkv/pipeline.py``
``_evaluation_plan``: pick the power-of-two ``k`` whose total rollout count
(``k * num_docs``) lands in ``[3000, 6000]``, closest to 4096; a limited run
always degrades to ``k=1``.

``num_docs`` is a *benchmark's* question count, not a leaf task's: sizing
each leaf separately multiplied a many-subtask benchmark's own question count
by its per-leaf ``k`` (a 57-subtask benchmark of ~250 docs each picked
``k=16`` and so took ~225k rollouts for a 14k-question benchmark).
"""

import json

import pytest

from lm_eval.api.benchmark import (
    BenchmarkSpec,
    avg_at_k_plan,
    benchmark_leaves,
    plan_avg_at_k,
)
from lm_eval.filters import build_filter_ensemble


@pytest.mark.parametrize(
    "num_docs, expected_k",
    [
        (30, 128),
        (500, 8),
        (4096, 1),
        (10000, 1),
    ],
)
def test_avg_at_k_plan_unlimited(num_docs, expected_k):
    k, metric_name = avg_at_k_plan(num_docs, limited=False)
    assert k == expected_k
    assert metric_name == f"avg@{expected_k}"


def test_avg_at_k_plan_limited_forces_k_1():
    k, metric_name = avg_at_k_plan(30, limited=True)
    assert k == 1
    assert metric_name == "avg@1"


def test_avg_at_k_plan_zero_docs():
    k, metric_name = avg_at_k_plan(0, limited=False)
    assert k == 1
    assert metric_name == "avg@1"


def test_avg_at_k_plan_stays_within_budget():
    for num_docs in (1, 2, 7, 17, 100, 999, 3001, 4095, 4097):
        k, _ = avg_at_k_plan(num_docs, limited=False)
        total = k * num_docs
        assert k == 1 or 3000 <= total <= 6000


class _SizedTask:
    """A task as the planner sees one: a name and its evaluation documents."""

    def __init__(self, name, docs):
        self.task_name = name
        self.eval_docs = list(range(docs))


class _SizedGroup:
    """A group as the planner sees one: a name and flat leaf tasks."""

    def __init__(self, name, leaves):
        self.name = name
        self._leaves = list(leaves)

    def get_all_tasks(self, recursive=True):
        return list(self._leaves)


def _plan(tasks, groups=None, benchmark_map=None, limited=False):
    return plan_avg_at_k(
        {task.task_name: task for task in tasks},
        groups or {},
        benchmark_map,
        limited=limited,
    )


def test_benchmark_total_docs_sizes_every_leaf_at_once():
    """Three 1000-doc sub-tasks are one 3000-question benchmark.

    Sized per leaf each of them would take ``k=4`` (4000 rollouts) and the
    benchmark would run 12000; sized as a benchmark the total already sits in
    the budget window, so every leaf samples once per doc.
    """
    leaves = [_SizedTask(name, 1000) for name in ("sub_a", "sub_b", "sub_c")]
    plans = _plan(leaves, {"bench": _SizedGroup("bench", leaves)})

    assert {plan.num_docs for plan in plans.values()} == {3000}
    assert {plan.k for plan in plans.values()} == {1}
    assert {plan.metric_name for plan in plans.values()} == {"avg@1"}
    assert {plan.benchmark for plan in plans.values()} == {"bench"}
    assert avg_at_k_plan(1000, limited=False)[0] == 4  # the old per-leaf sizing


def test_small_benchmark_spends_one_budget_for_the_whole_group():
    leaves = [_SizedTask(name, 200) for name in ("sub_a", "sub_b", "sub_c")]
    plans = _plan(leaves, {"bench": _SizedGroup("bench", leaves)})

    assert {plan.num_docs for plan in plans.values()} == {600}
    assert {plan.k for plan in plans.values()} == {8}
    for plan in plans.values():
        assert 3000 <= plan.num_docs * plan.k <= 6000


def test_many_subtask_benchmark_is_not_multiplied_by_k():
    """The 14k-question MMLU case: 57 leaves of ~246 docs each."""
    leaves = [_SizedTask(f"mmlu:{index}", 246) for index in range(57)]
    plans = _plan(leaves, {"mmlu": _SizedGroup("mmlu", leaves)})

    assert {plan.num_docs for plan in plans.values()} == {246 * 57}
    assert {plan.k for plan in plans.values()} == {1}
    assert {plan.num_docs * plan.k for plan in plans.values()} == {246 * 57}


def test_standalone_task_is_sized_from_its_own_docs():
    plans = _plan([_SizedTask("race", 500)])

    assert plans["race"].benchmark == "race"
    assert plans["race"].num_docs == 500
    assert plans["race"].k == 8


def test_limited_run_pins_k_to_1():
    leaves = [_SizedTask(name, 200) for name in ("sub_a", "sub_b")]
    plans = _plan(
        leaves, {"bench": _SizedGroup("bench", leaves)}, limited=True
    )

    assert {plan.k for plan in plans.values()} == {1}
    assert {plan.metric_name for plan in plans.values()} == {"avg@1"}


def test_spec_leaf_names_are_used_without_a_group():
    """Tags and standalone entries carry their leaves from config time."""
    tasks = [_SizedTask("race_a", 50), _SizedTask("race_b", 450)]
    benchmark_map = {
        "race_bench": BenchmarkSpec(
            name="race_bench",
            field="reasoning",
            strategy="tag_fallback",
            leaf_task_names=("race_a", "race_b"),
        )
    }
    plans = _plan(tasks, benchmark_map=benchmark_map)

    assert {plan.benchmark for plan in plans.values()} == {"race_bench"}
    assert {plan.num_docs for plan in plans.values()} == {500}
    assert {plan.k for plan in plans.values()} == {8}


def test_leaf_that_rescores_its_siblings_is_counted_once():
    """`all` next to the disjoint female/male/neutral subsets it contains."""
    leaves = [
        _SizedTask("winogender_all", 300),
        _SizedTask("winogender_female", 100),
        _SizedTask("winogender_male", 100),
        _SizedTask("winogender_neutral", 100),
    ]
    plans = _plan(leaves, {"winogender": _SizedGroup("winogender", leaves)})

    assert {plan.num_docs for plan in plans.values()} == {300}
    assert {plan.k for plan in plans.values()} == {16}


def test_two_halves_of_one_dataset_are_not_treated_as_parent_and_child():
    leaves = [_SizedTask("half_a", 400), _SizedTask("half_b", 400)]
    plans = _plan(leaves, {"bench": _SizedGroup("bench", leaves)})

    assert {plan.num_docs for plan in plans.values()} == {800}


def test_shared_leaf_is_sized_by_the_largest_benchmark():
    shared = _SizedTask("shared", 100)
    small_extra = _SizedTask("small_extra", 100)
    small = _SizedGroup("small", [shared, small_extra])
    tasks = [shared, small_extra, _SizedTask("big_extra", 1000)]
    benchmark_map = {
        "big": BenchmarkSpec(
            name="big",
            field="reasoning",
            strategy="tag_fallback",
            leaf_task_names=("shared", "big_extra"),
        )
    }
    plans = _plan(tasks, {"small": small}, benchmark_map)

    # 1100 docs beat 200 docs, so the shared leaf runs at the larger
    # benchmark's (smaller) k instead of one k for half of its run.
    assert plans["shared"].benchmark == "big"
    assert plans["shared"].k == 4
    assert plans["small_extra"].benchmark == "small"


def test_benchmark_leaves_lists_groups_and_standalones():
    leaves = [_SizedTask("sub_a", 10), _SizedTask("sub_b", 10)]
    tasks = {task.task_name: task for task in leaves + [_SizedTask("solo", 5)]}
    groups = {"bench": _SizedGroup("bench", leaves)}

    assert benchmark_leaves(tasks, groups) == {
        "bench": ("sub_a", "sub_b"),
        "solo": ("solo",),
    }


_TINY_MC_DATA = [
    {"question": "What is 2+2?", "choices": ["1", "2", "3", "4"], "answer": 3}
]

_TINY_LEAF_YAML = """\
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


@pytest.fixture
def tiny_budget(monkeypatch):
    """The real 4096/3000/6000 budget, scaled to a two-document benchmark."""
    from lm_eval.api import benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "_TARGET_COMPLETIONS", 4)
    monkeypatch.setattr(benchmark_module, "_MIN_COMPLETIONS", 3)
    monkeypatch.setattr(benchmark_module, "_MAX_COMPLETIONS", 6)


@pytest.fixture
def one_doc_group_task_manager(tmp_path):
    """A group of two one-document tasks, built fresh so it can actually run."""
    from lm_eval.tasks import TaskManager

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    data_path = fixtures / "test_data.json"
    data_path.write_text(json.dumps(_TINY_MC_DATA), encoding="utf-8")
    for task_name in ("bench_leaf_a", "bench_leaf_b"):
        (fixtures / f"{task_name}.yaml").write_text(
            _TINY_LEAF_YAML.format(task_name=task_name, data_path=data_path),
            encoding="utf-8",
        )
    (fixtures / "group.yaml").write_text(
        "group: bench_tiny\ntask:\n  - bench_leaf_a\n  - bench_leaf_b\n",
        encoding="utf-8",
    )
    return TaskManager(include_path=str(fixtures), include_defaults=False)


def test_a_run_samples_the_benchmark_budget_not_each_leaf(
    one_doc_group_task_manager, tiny_budget
):
    """A run's `repeats` comes from the benchmark's question count.

    Two one-document leaves are one two-document benchmark, which buys `k=2`;
    sizing each leaf on its own (the old behavior) would give `k=4` for both
    and take four times the benchmark's own question count in rollouts.
    """
    from lm_eval.api.benchmark import BenchmarkSpec
    from lm_eval.models.dummy import DummyLM

    import lm_eval.evaluator as evaluator

    lm = DummyLM()
    lm.SUPPORTS_AVG_AT_K = True
    lm._avg_k_samples = {}
    # `DummyLM.tokenizer` downloads `allenai/Olmo-3-32B-Think`; results metadata
    # only needs *a* tokenizer, and this run has nothing to do with one.
    lm.tokenizer = None

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["bench_tiny"],
        task_manager=one_doc_group_task_manager,
        benchmark_map={
            "bench_tiny": BenchmarkSpec(
                name="bench_tiny", field="reasoning", strategy="group_fallback"
            )
        },
        bootstrap_iters=0,
        log_samples=False,
    )

    assert {config["repeats"] for config in results["configs"].values()} == {2}
    assert avg_at_k_plan(1, limited=False)[0] == 4  # the old per-leaf sizing
    assert results["n-samples"]["bench_leaf_a"] == {"original": 1, "effective": 1}


def _default_filters():
    """The single-response-set pipeline a task without a `filter_list` gets."""
    return [build_filter_ensemble("none", [["take_first", None]])]


class _FakeTask:
    def __init__(self, output_type, process_results, filters=None):
        self.OUTPUT_TYPE = output_type
        self.process_results = process_results
        self._aggregation_list = {"acc": "mean_fn", "exact_match": "mean_fn"}
        self._higher_is_better = {"acc": True, "exact_match": True}
        self._filters = _default_filters() if filters is None else filters


class _FakeLM:
    def __init__(self):
        self._avg_k_samples = {}


def test_structured_metrics_retain_every_rollout_for_aggregation():
    from lm_eval.evaluator import _RepeatedMetricValues, _install_avg_at_k_patch

    task = _FakeTask("generate_until", lambda doc, values: {"bias": ("demographic", values[0])})
    task._aggregation_list["bias"] = list
    _install_avg_at_k_patch(task, _FakeLM(), 3, "avg@3")
    metrics = task.process_results({}, [["yes", "no", "yes"]])
    assert isinstance(metrics["bias_avg@3"], _RepeatedMetricValues)
    assert metrics["bias_avg@3"] == [("demographic", "yes"), ("demographic", "no"), ("demographic", "yes")]
    assert task._aggregation_list["bias_avg@3"] is list


def test_avg_at_k_survives_a_checkpoint_restored_doc():
    """A doc whose requests were restored instead of re-generated never
    reaches the model's ``loglikelihood()`` again, so ``lm._avg_k_samples``
    has no entry for it and it used to drop out of the benchmark's average
    entirely. The per-rollout record carried by the responses themselves
    keeps it in both the average and the counted completions.
    """
    from lm_eval.evaluator import _install_avg_at_k_patch
    from lm_eval.models.rwkv_choice import RWKVLoglikelihoodResponse

    def process_results(doc, results):
        (ll, is_greedy) = results[0]
        return {"acc": 1.0 if is_greedy else 0.0}

    task = _FakeTask("multiple_choice", process_results)
    lm = _FakeLM()  # nothing recorded in memory: this doc was restored
    _install_avg_at_k_patch(task, lm, k=4, metric_name="avg@4")

    rollouts = [
        {"index": 0, "truncated": False},
        {"index": 1, "truncated": False},
        {"index": 0, "truncated": True},
        {"index": 0, "truncated": False},
    ]
    responses = [
        RWKVLoglikelihoodResponse(
            0.0 if choice == 0 else -1e6, choice == 0, rollouts=rollouts
        )
        for choice in (0, 1)
    ]

    task._avg_k_current_doc_id = 7
    metrics = task.process_results({}, responses)

    assert metrics["acc"] == 1.0
    assert metrics["acc_avg@4"] == pytest.approx(3 / 4)
    assert task._rollout_stats[7] == {"completions": 4, "truncated": 1}


def test_install_avg_at_k_patch_multiple_choice_averages_samples():
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        (ll, is_greedy) = results[0]
        return {"acc": 1.0 if is_greedy else 0.0}

    task = _FakeTask("multiple_choice", process_results)
    lm = _FakeLM()
    lm._avg_k_samples[42] = [
        [(0.0, True), (-1e6, False)],
        [(-1e6, False), (0.0, True)],
        [(0.0, True), (-1e6, False)],
    ]

    _install_avg_at_k_patch(task, lm, k=3, metric_name="avg@3")

    task._avg_k_current_doc_id = 42
    metrics = task.process_results({}, [(0.0, True)])

    assert metrics["acc"] == 1.0
    assert metrics["acc_avg@3"] == pytest.approx(2 / 3)
    # doc_id popped so it can't leak into a later doc
    assert 42 not in lm._avg_k_samples


def test_install_avg_at_k_patch_generate_until_averages_k_generations():
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        return {"exact_match": 1.0 if results[0] == doc["gold"] else 0.0}

    task = _FakeTask("generate_until", process_results)
    lm = _FakeLM()

    _install_avg_at_k_patch(task, lm, k=4, metric_name="avg@4")

    # the task's own pipeline is kept, not replaced
    assert task._filters[0].name == "none"

    doc = {"gold": "42"}
    metrics = task.process_results(doc, [["42", "7", "42", "42"]])

    assert metrics["exact_match"] == 1.0
    assert metrics["exact_match_avg@4"] == pytest.approx(3 / 4)


def test_per_repeat_filter_runs_a_collapsing_pipeline_once_per_repeat():
    from lm_eval.filters.selection import PerRepeatFilter

    ensemble = build_filter_ensemble(
        "regex_match",
        [("regex", {"regex_pattern": r"([A-D])"}), ("take_first", None)],
    )

    filtered = PerRepeatFilter(ensemble).apply(
        [["reasoning... answer: A", "reasoning... answer: B", "answer: C", "answer: D"]],
        [{}],
    )

    # one value per repeat, each shaped like a `repeats: 1` result
    assert filtered == [["A", "B", "C", "D"]]


def test_per_repeat_filter_keeps_per_repeat_extractors_wrapped():
    from lm_eval.filters.selection import PerRepeatFilter

    ensemble = build_filter_ensemble(
        "albatross_mcq", [["albatross_mcq", {"choice_labels": "ABCD"}]]
    )

    filtered = PerRepeatFilter(ensemble).apply(
        [["<think>hmm</think>The answer is A", "The answer is B", "no letter", "D"]],
        [{}],
    )

    # `albatross_mcq` already yields a per-response list, so the wrapper adds
    # the repeat dimension without flattening it
    assert filtered == [[["A"], ["B"], ["[invalid]"], ["D"]]]


def test_install_avg_at_k_patch_generate_until_scores_the_extracted_answer():
    """Regression: the avg@k patch used to swap in `take_first_k`, so
    `process_results` saw raw completions and mutual-style lettered tasks
    scored 0 on every doc."""
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        predicted = results[0][0] if results and results[0] else None
        return {"acc": int(predicted == doc["gold"])}

    task = _FakeTask(
        "generate_until",
        process_results,
        filters=[
            build_filter_ensemble(
                "albatross_mcq", [["albatross_mcq", {"choice_labels": "ABCD"}]]
            )
        ],
    )
    lm = _FakeLM()

    _install_avg_at_k_patch(task, lm, k=4, metric_name="avg@4")

    # filter name (and therefore the metric key) survives the patch
    assert [ensemble.name for ensemble in task._filters] == ["albatross_mcq"]

    # `finish_benchmark` hands over one instance's filtered_resps: the
    # per-repeat extracted answers, not the k raw completions
    doc = {"gold": "B"}
    metrics = task.process_results(doc, [[["A"], ["B"], ["C"], ["B"]]])

    assert metrics["acc"] == 0.0
    assert metrics["acc_avg@4"] == pytest.approx(0.5)


def test_install_avg_at_k_patch_noop_when_k_is_one():
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        return {"acc": 1.0}

    task = _FakeTask("multiple_choice", process_results)
    lm = _FakeLM()

    _install_avg_at_k_patch(task, lm, k=1, metric_name="avg@1")

    assert task.process_results is process_results
