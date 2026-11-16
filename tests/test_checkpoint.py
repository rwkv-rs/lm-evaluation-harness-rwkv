import pytest

from lm_eval.api.checkpoint import (
    CheckpointConfigMismatchError,
    CheckpointStore,
    ItemKey,
    compute_config_fingerprint,
    compute_idempotency_key,
    compute_item_id,
    compute_request_fingerprint,
    compute_task_fingerprint,
    deserialize_response,
    serialize_response,
    without_operational_args,
)


def _make_store(tmp_path, config=("model", "a")):
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint = compute_config_fingerprint({"config": config})
    return CheckpointStore(db_path, fingerprint)


def test_completion_resume_retains_seed_and_thinking_contract():
    from lm_eval.filters.extraction import AlbatrossMultiChoiceFilter
    from lm_eval.models.openai_completions import _RWKVCompletion

    answer = _RWKVCompletion(">The answer is B", "length")
    answer.requires_think_close = True
    answer.seed = 104
    restored = deserialize_response(serialize_response(answer))
    assert restored.seed == 104
    assert restored.requires_think_close is True
    assert AlbatrossMultiChoiceFilter().apply([[restored]], [{}]) == [["[invalid]"]]


def _item_key(task_name="task", item_id="item-1", repeat_ord=0):
    return ItemKey(
        item_id=item_id,
        request_fingerprint="fp",
        idempotency_key=compute_idempotency_key(item_id, "fp"),
        task_name=task_name,
        doc_id=0,
        instance_idx=0,
        request_type="loglikelihood",
        repeat_ord=repeat_ord,
    )


def test_fresh_db_starts_empty(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_benchmark("task") is None
    assert store.get_items("task") == {}
    store.close()


def test_without_operational_args_drops_only_scheduling_knobs():
    values = {
        "model": "rwkv7-g1k-1.5b",
        "num_concurrent": 200,
        "timeout": 1800,
        "max_retries": 10,
        "max_length": 25600,
    }

    assert without_operational_args(values) == {
        "model": "rwkv7-g1k-1.5b",
        "max_length": 25600,
    }
    assert without_operational_args(None) == {}
    assert without_operational_args({}) == {}


def test_task_fingerprint_ignores_scheduling_knobs_at_any_level():
    """Run metadata reaches the task config, so knobs must be dropped there too."""
    metadata = {
        "model_name": "rwkv7-g1k-1.5b",
        "prompt_template": "assistant",
        "num_concurrent": 64,
        "timeout": 300,
        "max_retries": 3,
    }
    base = {"task": "mc_taco", "metadata": metadata, "num_concurrent": 64}
    resumed = {
        "task": "mc_taco",
        "metadata": {
            **metadata,
            "num_concurrent": 200,
            "timeout": 1800,
            "max_retries": 10,
        },
        "num_concurrent": 200,
    }

    assert compute_task_fingerprint(base) == compute_task_fingerprint(resumed)

    # Anything that actually changes what is asked still changes the fingerprint.
    assert compute_task_fingerprint(base) != compute_task_fingerprint(
        {**base, "metadata": {**metadata, "prompt_template": "bot"}}
    )


def test_reopen_same_config_resumes(tmp_path):
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint = compute_config_fingerprint({"config": ("model", "a")})
    store = CheckpointStore(db_path, fingerprint)
    store.start_benchmark("task", "task-fp")
    store.close()

    store2 = CheckpointStore(db_path, fingerprint)
    assert store2.get_benchmark("task")["status"] == "running"
    store2.close()


def test_reopen_different_config_raises(tmp_path):
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint_a = compute_config_fingerprint({"config": ("model", "a")})
    fingerprint_b = compute_config_fingerprint({"config": ("model", "b")})
    store = CheckpointStore(db_path, fingerprint_a)
    store.close()

    with pytest.raises(CheckpointConfigMismatchError):
        CheckpointStore(db_path, fingerprint_b)


def test_fresh_true_discards_mismatched_state(tmp_path):
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint_a = compute_config_fingerprint({"config": ("model", "a")})
    fingerprint_b = compute_config_fingerprint({"config": ("model", "b")})
    store = CheckpointStore(db_path, fingerprint_a)
    store.start_benchmark("task", "task-fp")
    store.close()

    store2 = CheckpointStore(db_path, fingerprint_b, fresh=True)
    assert store2.get_benchmark("task") is None
    store2.close()


def test_register_and_mark_items_completed(tmp_path):
    store = _make_store(tmp_path)
    key = _item_key()
    store.register_pending_items([key])

    items = store.get_items(key.task_name)
    assert items[key.item_id]["status"] == "pending"
    assert items[key.item_id]["result_json"] is None

    store.mark_items_completed([(key.item_id, serialize_response("hello"))])

    items = store.get_items(key.task_name)
    assert items[key.item_id]["status"] == "completed"
    assert deserialize_response(items[key.item_id]["result_json"]) == "hello"
    store.close()


def test_register_pending_items_is_idempotent(tmp_path):
    store = _make_store(tmp_path)
    key = _item_key()
    store.register_pending_items([key])
    store.mark_items_completed([(key.item_id, serialize_response("hello"))])
    # Re-registering (e.g. on resume, before a redispatch decision) must never
    # clobber an already-completed result back to pending.
    store.register_pending_items([key])

    items = store.get_items(key.task_name)
    assert items[key.item_id]["status"] == "completed"
    store.close()


def test_count_completed_matches_get_items(tmp_path):
    store = _make_store(tmp_path)
    keys = [_item_key(item_id=f"item-{i}", repeat_ord=i) for i in range(3)]
    store.register_pending_items(keys)
    assert store.count_completed("task") == 0

    store.mark_items_completed(
        [(keys[0].item_id, serialize_response("a")), (keys[1].item_id, serialize_response("b"))]
    )

    items = store.get_items("task")
    expected = sum(1 for item in items.values() if item["status"] == "completed")
    assert store.count_completed("task") == expected == 2
    store.close()


def test_count_completed_unknown_task_is_zero(tmp_path):
    store = _make_store(tmp_path)
    assert store.count_completed("nonexistent-task") == 0
    store.close()


def test_mark_benchmark_completed_and_publish_status(tmp_path):
    store = _make_store(tmp_path)
    store.start_benchmark("task", "task-fp")
    store.mark_benchmark_completed("task", "results.json", "metadata.json")

    benchmark = store.get_benchmark("task")
    assert benchmark["status"] == "completed"
    assert benchmark["artifact_results_path"] == "results.json"
    assert benchmark["publish_status"] == "none"

    store.set_publish_status("task", "succeeded")
    assert store.get_benchmark("task")["publish_status"] == "succeeded"
    store.close()


def test_mark_items_completed_async(tmp_path):
    import asyncio

    store = _make_store(tmp_path)
    key = _item_key()
    store.register_pending_items([key])

    asyncio.run(
        store.mark_items_completed_async([(key.item_id, serialize_response("x"))])
    )

    items = store.get_items(key.task_name)
    assert items[key.item_id]["status"] == "completed"
    store.close()


def test_serialize_deserialize_tuple_roundtrip():
    data = serialize_response((0.5, True))
    assert deserialize_response(data) == (0.5, True)


def test_serialize_deserialize_plain_string_roundtrip():
    data = serialize_response("plain text")
    assert deserialize_response(data) == "plain text"


def test_serialize_deserialize_rwkv_completion_roundtrip():
    from lm_eval.models.openai_completions import _RWKVCompletion

    response = _RWKVCompletion(
        "generated text", "stop", raw_response={"choices": [{"text": "generated text"}]}
    )
    data = serialize_response(response)
    restored = deserialize_response(data)
    assert restored == "generated text"
    assert restored.finish_reason == "stop"
    assert restored.raw_response == {"choices": [{"text": "generated text"}]}


def test_serialize_deserialize_rwkv_loglikelihood_roundtrip():
    from lm_eval.models.rwkv_choice import RWKVLoglikelihoodResponse

    response = RWKVLoglikelihoodResponse(
        0.0,
        True,
        prompt="ctx\n\nA. one\nB. two\n\nAnswer with only the letter of the correct option.",
        raw_response={"choices": [{"text": "The answer is B."}]},
        finish_reason="stop",
        output_token_ids=[1, 2, 3],
        truncated=False,
    )
    data = serialize_response(response)
    restored = deserialize_response(data)

    # Resuming from a checkpoint must not silently fall back to the fake,
    # reconstructed-from-score prompt/completion: the real evidence has to
    # survive the round trip exactly like a fresh run would produce it.
    assert restored == (0.0, True)
    assert isinstance(restored, tuple)
    assert restored.prompt == response.prompt
    assert restored.raw_response == {"choices": [{"text": "The answer is B."}]}
    assert restored.finish_reason == "stop"
    assert restored.output_token_ids == [1, 2, 3]
    assert restored.truncated is False


def test_serialize_deserialize_keeps_the_per_rollout_record():
    """The per-rollout record is what lets a restored doc still be scored at
    avg@k (and counted in the truncation rate), so it has to survive the
    round trip even when the response carries no other evidence.
    """
    from lm_eval.models.rwkv_choice import RWKVLoglikelihoodResponse

    rollouts = [
        {"index": 0, "truncated": False},
        {"index": -1, "truncated": True},
    ]
    response = RWKVLoglikelihoodResponse(0.0, True, rollouts=rollouts)

    restored = deserialize_response(serialize_response(response))

    assert restored == (0.0, True)
    assert restored.rollouts == rollouts


def test_responses_without_a_rollout_record_round_trip_as_plain_tuples():
    """Older responses (and every non-RWKV backend) keep serializing as bare
    2-tuples: the per-rollout record must not change their wire shape.
    """
    assert deserialize_response(serialize_response((0.5, False))) == (0.5, False)


def test_compute_item_id_stable_and_distinguishes_repeat_ord():
    id_0 = compute_item_id("task", 3, 0, "loglikelihood", 0)
    id_0_again = compute_item_id("task", 3, 0, "loglikelihood", 0)
    id_1 = compute_item_id("task", 3, 0, "loglikelihood", 1)
    assert id_0 == id_0_again
    assert id_0 != id_1


def test_compute_request_fingerprint_changes_with_args():
    item_id = compute_item_id("task", 0, 0, "loglikelihood", 0)
    fp1 = compute_request_fingerprint(item_id, ("ctx", "cont"), "task-fp", "model-fp")
    fp2 = compute_request_fingerprint(item_id, ("ctx", "other"), "task-fp", "model-fp")
    assert fp1 != fp2


def test_idempotency_key_is_bounded_length():
    item_id = compute_item_id("task", 0, 0, "loglikelihood", 0)
    fp = compute_request_fingerprint(item_id, ("ctx", "cont"), "task-fp", "model-fp")
    key = compute_idempotency_key(item_id, fp)
    assert len(key) <= 32
