"""Benchmark execution order, mirroring lighteval-rwkv's ``_order_benchmarks``.

lighteval-rwkv evaluates one benchmark at a time
(``src/rwkv_eval/main.py``): the next benchmark starts only after the previous
wave has been collected. Which benchmark goes first is decided by
``_order_benchmarks`` -- benchmarks that do not need generation (no CoT) run
before generative ones, and within each group the benchmark with the fewest
samples runs first, so short benchmarks finish (and publish to the scoreboard)
promptly instead of waiting behind one very large benchmark.

This module holds that ordering rule and nothing else: there is no
cross-benchmark concurrency left to schedule. The evaluator runs every
benchmark strictly sequentially in the order returned by
:func:`order_benchmarks`, which also keeps the endpoint's single flat request
budget from being multiplied by concurrent benchmarks.

A "benchmark" here is a scheduling unit: either every leaf task of a
benchmark group (aggregated and published together), or a single standalone
task treated as its own one-leaf unit. Both are represented the same way --
a list of ``(task_name, task)`` pairs -- so this module never needs to know
which case it's looking at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from lm_eval.api.checkpoint import CheckpointStore
    from lm_eval.api.task import Task


def pending_rollouts_for_task(
    task: "Task", task_name: str, checkpoint_store: "CheckpointStore | None"
) -> int:
    """Rollouts (repeated ``Instance`` draws) not yet completed for one task.

    A fresh run has no checkpoint data to consult, so the total is read
    straight out of ``task.instances`` (already populated by
    ``task.build_all_requests()`` before this is called). Resuming a run
    subtracts what the checkpoint store already has recorded as completed.
    """
    total = sum(instance.repeats for instance in task.instances)
    if checkpoint_store is not None:
        total -= checkpoint_store.count_completed(task_name)
    return max(total, 0)


def benchmark_pending_rollouts(
    members: list[tuple[str, "Task"]],
    checkpoint_store: "CheckpointStore | None",
) -> int:
    """Sum of pending rollouts across every task in one scheduling unit."""
    return sum(
        pending_rollouts_for_task(task, task_name, checkpoint_store)
        for task_name, task in members
    )


def requires_generation(task: "Task") -> bool:
    """Whether a task's leaves ask the model to generate, not just to score.

    The harness analogue of lighteval-rwkv's ``requires_cot``: tasks whose
    rollouts are ``generate_until`` requests (open-ended answers, CoT) run
    after tasks that can be scored from fixed continuations.
    """
    return any(instance.request_type == "generate_until" for instance in task.instances)


def order_benchmarks(
    units: dict[str, list[tuple[str, "Task"]]],
    checkpoint_store: "CheckpointStore | None" = None,
) -> list[str]:
    """Benchmark names ordered by ascending pending-rollout count.

    Ties keep the original (insertion) order, since ``dict`` iteration order
    is preserved and :func:`sorted` is stable.
    """
    return [
        name
        for name, _ in sorted(
            units.items(),
            key=lambda item: (
                int(any(requires_generation(task) for _, task in item[1])),
                benchmark_pending_rollouts(item[1], checkpoint_store),
            ),
        )
    ]
