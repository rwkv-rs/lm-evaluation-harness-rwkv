from lm_eval.api.checkpoint import (
    CheckpointStore,
    ItemKey,
    compute_config_fingerprint,
    compute_item_id,
    compute_request_fingerprint,
    compute_task_fingerprint,
    serialize_response,
)
from lm_eval.api.instance import Instance
from lm_eval.evaluator import _build_checkpointed_dispatch


def _make_store(tmp_path):
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint = compute_config_fingerprint({"config": "test"})
    return CheckpointStore(db_path, fingerprint)


def _instance(doc_id, idx, arguments, repeats=1):
    return Instance(
        request_type="loglikelihood",
        doc={"doc_id": doc_id},
        arguments=arguments,
        idx=idx,
        metadata=("task", doc_id, repeats),
    )


def test_all_pending_dispatches_everything(tmp_path):
    store = _make_store(tmp_path)
    reqs = [_instance(0, 0, ("ctx", "cont"))]

    dispatch_reqs, dispatch_keys = _build_checkpointed_dispatch(
        store, "task", "task-fp", "model-fp", reqs, "loglikelihood"
    )

    assert len(dispatch_reqs) == 1
    assert dispatch_keys[0] is not None
    assert dispatch_keys[0].repeat_ord == 0
    assert reqs[0].resps == [None]
    store.close()


def test_completed_item_is_restored_and_skipped(tmp_path):
    store = _make_store(tmp_path)
    reqs = [_instance(0, 0, ("ctx", "cont"))]

    item_id = compute_item_id("task", 0, 0, "loglikelihood", 0)
    request_fp = compute_request_fingerprint(
        item_id, reqs[0].args, "task-fp", "model-fp"
    )
    store.register_pending_items(
        [
            ItemKey(
                item_id=item_id,
                request_fingerprint=request_fp,
                idempotency_key="idem",
                task_name="task",
                doc_id=0,
                instance_idx=0,
                request_type="loglikelihood",
                repeat_ord=0,
            )
        ]
    )
    store.mark_items_completed([(item_id, serialize_response((0.5, True)))])

    dispatch_reqs, dispatch_keys = _build_checkpointed_dispatch(
        store, "task", "task-fp", "model-fp", reqs, "loglikelihood"
    )

    assert dispatch_reqs == []
    assert dispatch_keys == []
    assert reqs[0].resps == [(0.5, True)]
    store.close()


def test_partial_repeat_resume_dispatches_only_missing_ords(tmp_path):
    store = _make_store(tmp_path)
    reqs = [_instance(0, 0, ("ctx", "gen_kwargs"), repeats=3)]
    reqs[0].request_type = "generate_until"

    # Pre-complete repeat_ord=1 only.
    item_id_1 = compute_item_id("task", 0, 0, "generate_until", 1)
    request_fp_1 = compute_request_fingerprint(
        item_id_1, reqs[0].args, "task-fp", "model-fp"
    )

    store.register_pending_items(
        [
            ItemKey(
                item_id=item_id_1,
                request_fingerprint=request_fp_1,
                idempotency_key="idem",
                task_name="task",
                doc_id=0,
                instance_idx=0,
                request_type="generate_until",
                repeat_ord=1,
            )
        ]
    )
    store.mark_items_completed([(item_id_1, serialize_response("cached text"))])

    dispatch_reqs, dispatch_keys = _build_checkpointed_dispatch(
        store, "task", "task-fp", "model-fp", reqs, "generate_until"
    )

    # Only repeat_ord 0 and 2 need dispatch.
    assert len(dispatch_reqs) == 2
    assert [k.repeat_ord for k in dispatch_keys] == [0, 2]
    assert reqs[0].resps[1] == "cached text"
    assert reqs[0].resps[0] is None
    assert reqs[0].resps[2] is None

    # Simulate responses coming back out of order relative to repeat_ord,
    # written by index (as run_requests does), and confirm final ordering.
    for response, request, key in zip(
        ["gen-for-2", "gen-for-0"],
        [dispatch_reqs[1], dispatch_reqs[0]],
        [dispatch_keys[1], dispatch_keys[0]],
        strict=True,
    ):
        request.resps[key.repeat_ord] = response

    assert reqs[0].resps == ["gen-for-0", "cached text", "gen-for-2"]
    store.close()


def test_content_change_invalidates_cached_item(tmp_path):
    store = _make_store(tmp_path)
    reqs = [_instance(0, 0, ("ctx", "cont"))]

    item_id = compute_item_id("task", 0, 0, "loglikelihood", 0)
    stale_fp = "stale-fingerprint"

    store.register_pending_items(
        [
            ItemKey(
                item_id=item_id,
                request_fingerprint=stale_fp,
                idempotency_key="idem",
                task_name="task",
                doc_id=0,
                instance_idx=0,
                request_type="loglikelihood",
                repeat_ord=0,
            )
        ]
    )
    store.mark_items_completed([(item_id, serialize_response((0.1, False)))])

    # request content now differs from what's recorded (task/model fingerprint
    # or args changed) -> must redispatch rather than trust stale content.
    dispatch_reqs, dispatch_keys = _build_checkpointed_dispatch(
        store, "task", "task-fp", "model-fp", reqs, "loglikelihood"
    )

    assert len(dispatch_reqs) == 1
    assert dispatch_keys[0] is not None
    store.close()


def _task_config(**operational):
    """A dumped task config shaped like the real one: run metadata is merged in."""
    return {
        "task": "task",
        "generation_kwargs": {"until": [], "max_gen_toks": 8},
        "metadata": {
            "model_name": "rwkv7-g1k-1.5b",
            "cot_mode": "open_think",
            "prompt_template": "assistant",
            **operational,
        },
    }


def test_scheduling_knobs_do_not_invalidate_completed_items(tmp_path):
    """Resuming with a different concurrency/timeout/retry budget must restore.

    Regression: the run id already ignored those knobs, but the task fingerprint
    is computed from ``task.dump_config()``, which carries the run metadata --
    so bumping the request concurrency (or the client timeout) on resume used to
    flip every per-item fingerprint and re-run the whole benchmark.
    """
    store = _make_store(tmp_path)
    reqs = [_instance(0, 0, ("ctx", "cont"))]

    original = compute_task_fingerprint(
        _task_config(num_concurrent=64, timeout=300, max_retries=3)
    )
    resumed = compute_task_fingerprint(
        _task_config(num_concurrent=200, timeout=1800, max_retries=10)
    )
    assert resumed == original

    item_id = compute_item_id("task", 0, 0, "loglikelihood", 0)
    store.register_pending_items(
        [
            ItemKey(
                item_id=item_id,
                request_fingerprint=compute_request_fingerprint(
                    item_id, reqs[0].args, original, "model-fp"
                ),
                idempotency_key="idem",
                task_name="task",
                doc_id=0,
                instance_idx=0,
                request_type="loglikelihood",
                repeat_ord=0,
            )
        ]
    )
    store.mark_items_completed([(item_id, serialize_response((0.5, True)))])

    dispatch_reqs, dispatch_keys = _build_checkpointed_dispatch(
        store, "task", resumed, "model-fp", reqs, "loglikelihood"
    )

    assert dispatch_reqs == []
    assert dispatch_keys == []
    assert reqs[0].resps == [(0.5, True)]
    store.close()


def test_non_scheduling_change_still_invalidates_completed_items(tmp_path):
    """Dropping the scheduling knobs must not blunt real content changes."""
    base = _task_config(num_concurrent=200)
    assert compute_task_fingerprint(base) != compute_task_fingerprint(
        {**base, "generation_kwargs": {"until": [], "max_gen_toks": 16}}
    )
    assert compute_task_fingerprint(base) != compute_task_fingerprint(
        {**base, "metadata": {**base["metadata"], "prompt_template": "bot"}}
    )
    assert compute_task_fingerprint(base) != compute_task_fingerprint(
        {**base, "metadata": {**base["metadata"], "model_name": "rwkv7-g1k-2.9b"}}
    )
