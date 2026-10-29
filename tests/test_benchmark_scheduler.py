"""Ordering tests for lighteval-rwkv's `_order_benchmarks` port.

The evaluator runs one benchmark at a time in this order; there is no
cross-benchmark concurrency left to schedule.
"""

from types import SimpleNamespace

from lm_eval.api.benchmark_scheduler import (
    benchmark_pending_rollouts,
    order_benchmarks,
    pending_rollouts_for_task,
    requires_generation,
)
from lm_eval.api.checkpoint import CheckpointStore, ItemKey, compute_config_fingerprint


def _task(*repeats, request_type="loglikelihood"):
    return SimpleNamespace(
        instances=[
            SimpleNamespace(repeats=r, request_type=request_type) for r in repeats
        ]
    )


def _store(tmp_path):
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint = compute_config_fingerprint({"config": "x"})
    return CheckpointStore(db_path, fingerprint)


def _register_and_complete(store, task_name, n_completed, n_total):
    keys = [
        ItemKey(
            item_id=f"{task_name}-{i}",
            request_fingerprint="fp",
            idempotency_key=f"idem-{task_name}-{i}",
            task_name=task_name,
            doc_id=0,
            instance_idx=0,
            request_type="loglikelihood",
            repeat_ord=i,
        )
        for i in range(n_total)
    ]
    store.register_pending_items(keys)
    store.mark_items_completed(
        [(keys[i].item_id, "{}") for i in range(n_completed)]
    )


def test_pending_rollouts_for_task_no_checkpoint():
    task = _task(1, 1, 2)
    assert pending_rollouts_for_task(task, "t", None) == 4


def test_pending_rollouts_for_task_subtracts_completed(tmp_path):
    store = _store(tmp_path)
    task = _task(1, 1, 1, 1)
    _register_and_complete(store, "t", n_completed=3, n_total=4)
    assert pending_rollouts_for_task(task, "t", store) == 1
    store.close()


def test_pending_rollouts_for_task_never_negative(tmp_path):
    store = _store(tmp_path)
    task = _task(1)
    _register_and_complete(store, "t", n_completed=1, n_total=1)
    assert pending_rollouts_for_task(task, "t", store) == 0
    store.close()


def test_benchmark_pending_rollouts_sums_members():
    members = [("t1", _task(1, 1)), ("t2", _task(3))]
    assert benchmark_pending_rollouts(members, None) == 5


def test_requires_generation_reads_request_type():
    assert requires_generation(_task(1)) is False
    assert requires_generation(_task(1, request_type="generate_until")) is True
    assert requires_generation(_task(1, 1, request_type="generate_until")) is True


def test_order_benchmarks_fewest_pending_rollouts_first():
    units = {
        "big": [("big", _task(*([1] * 10)))],
        "small": [("small", _task(1))],
        "medium": [("medium", _task(*([1] * 5)))],
    }
    assert order_benchmarks(units) == ["small", "medium", "big"]


def test_order_benchmarks_stable_on_ties():
    units = {"a": [("a", _task(1))], "b": [("b", _task(1))], "c": [("c", _task(1))]}
    assert order_benchmarks(units) == ["a", "b", "c"]


def test_order_benchmarks_runs_non_generative_first():
    units = {
        "generative_small": [("generative_small", _task(1, request_type="generate_until"))],
        "scored_big": [("scored_big", _task(*([1] * 50)))],
    }
    assert order_benchmarks(units) == ["scored_big", "generative_small"]


def test_order_benchmarks_uses_checkpoint_progress(tmp_path):
    store = _store(tmp_path)
    units = {
        "half_done": [("half_done", _task(*([1] * 10)))],
        "untouched": [("untouched", _task(*([1] * 6)))],
    }
    _register_and_complete(store, "half_done", n_completed=8, n_total=10)
    # Untouched still has 6 rollouts left, half_done only 2.
    assert order_benchmarks(units, store) == ["half_done", "untouched"]
    store.close()


def test_order_benchmarks_orders_leaves_of_one_benchmark_together():
    units = {
        "group": [("group_b", _task(1)), ("group_a", _task(1))],
        "other": [("other", _task(*([1] * 3)))],
    }
    # A unit's members stay together and in their original order.
    assert order_benchmarks(units) == ["group", "other"]
    assert [task_name for task_name, _ in units["group"]] == ["group_b", "group_a"]
