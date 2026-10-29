import json
from unittest.mock import patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.rwkv_choice import build_choice_prompt
from lm_eval.models.rwkv_remote_chat import RWKV7RemoteChat


def _build(**kwargs):
    kwargs.setdefault("model", "rwkv7-g1i-1.5b")
    kwargs.setdefault("num_concurrent", 1)
    return RWKV7RemoteChat(**kwargs)






def test_chat_choice_scoring_preserves_roles_and_all_rollouts(monkeypatch):
    model = _build()
    history = [
        {"role": "system", "content": "Be precise"},
        {"role": "user", "content": "Example"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "Which pet?"},
    ]
    context = JsonChatStr(json.dumps(history))
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(context, choice),
            idx=index,
            metadata=("choice_task", 0, 2),
        )
        for index, choice in enumerate(["cats", "dogs"])
        for _ in range(2)
    ]
    captured = []

    def generate(instances):
        captured.extend(instances)
        return ["A", "B"]

    monkeypatch.setattr(model, "generate_until", generate)
    responses = model.loglikelihood(requests)
    assert len(captured) == 2
    sent = json.loads(captured[0].args[0].prompt)
    assert sent[:3] == history[:3]
    assert "A. cats" in sent[-1]["content"]
    assert "B. dogs" in sent[-1]["content"]
    assert json.loads(responses[0].prompt) == sent
    assert len(responses[0].rollouts) == 2
    assert responses[0].rollouts[1]["index"] == 1




def test_rwkv7_remote_chat_construction_defaults(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.base_url == RWKV7RemoteChat.DEFAULT_BASE_URL
    assert model.base_url.endswith("/v1/chat/completions")
    assert model.tokenizer_backend is None
    assert model.tokenizer is None


def test_rwkv7_remote_chat_rejects_bad_prompt_template(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    with pytest.raises(ValueError, match="rwkv_prompt_template"):
        _build(rwkv_prompt_template="nope")


def test_rwkv7_remote_chat_rejects_bad_generation_prompt(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    with pytest.raises(ValueError, match="rwkv_generation_prompt"):
        _build(rwkv_generation_prompt="nope")


def test_rwkv7_remote_chat_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("RWKV_API_KEY", raising=False)
    model = _build()

    with pytest.raises(ValueError, match="RWKV_API_KEY"):
        _ = model.api_key


def test_rwkv7_remote_chat_registered_in_model_mapping(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    assert get_model("rwkv7-remote-chat") is RWKV7RemoteChat


def test_rwkv7_remote_chat_get_model_info_reports_fixed_sampling_config(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.get_model_info() == {
        "sampling_config": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "do_sample": True,
            "stop": ["✿", "\nUser:", "\n### User"],
        }
    }


def test_rwkv7_remote_chat_tokenizer_name_has_no_tokenizer_dependency(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_prompt_template="bot", rwkv_generation_prompt="open_think")

    assert model.tokenizer_name == "rwkv7-g1i-1.5b:bot:open_think"


def test_rwkv7_remote_chat_create_payload_sends_chat_template_kwargs(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_prompt_template="bot", rwkv_generation_prompt="open_think")

    payload = model._create_payload(
        [{"role": "user", "content": "hi"}],
        generate=True,
        gen_kwargs={"until": [], "max_gen_toks": 32},
    )

    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["chat_template_kwargs"] == {
        "rwkv_prompt_template": "bot",
        "rwkv_generation_prompt": "open_think",
    }
    assert "generation_config" not in payload
    assert payload["top_p"] == 0.76
    assert payload["penalty_decay"] == 0.988



def test_rwkv7_remote_chat_no_think_sends_template_and_deterministic_profile(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_prompt_template="assistant", rwkv_generation_prompt="no_think")

    payload = model._create_payload(
        [{"role": "user", "content": "hi"}],
        generate=True,
        gen_kwargs={"until": [], "max_gen_toks": 32},
    )

    assert payload["chat_template_kwargs"] == {
        "rwkv_prompt_template": "assistant",
        "rwkv_generation_prompt": "no_think",
    }
    generation_config = json.loads(payload["generation_config"])
    assert generation_config["temperature"] == 0.0
    assert generation_config["do_sample"] is False


def test_rwkv7_remote_chat_create_payload_passthrough_for_non_generate(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    payload = model._create_payload([{"role": "user", "content": "hi"}], generate=False)

    assert "chat_template_kwargs" not in payload


def test_rwkv7_remote_chat_parse_generations_reads_message_content():
    outputs = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            },
        ]
    }

    (generation,) = RWKV7RemoteChat.parse_generations(outputs)

    assert str(generation) == "hello"
    assert generation.finish_reason == "stop"
    assert generation.raw_response == outputs


def _mc_requests(context, choices, doc_id=0, repeats=1):
    # Mirrors evaluator.py's `cloned_reqs = [req for req in reqs for _ in
    # range(req.repeats)]`: avg@k's k is read off the actual number of
    # cloned instances per choice, not off the `repeats` metadata value
    # alone, so tests must clone here too.
    return [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(context, choice),
            idx=idx,
            metadata=("some_task", doc_id, repeats),
        )
        for idx, choice in enumerate(choices)
        for _ in range(repeats)
    ]


def test_rwkv7_remote_chat_resolve_choices_wraps_prompt_as_chat_message(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    calls = []

    def fake_generate_until(instances):
        calls.append(instances)
        return ["A"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    prompt = build_choice_prompt("What is 1+1?", ["one", "two"])
    indices = model._resolve_choices(prompt, ["A", "B"], ["one", "two"], 1)

    assert indices == [0]
    sent = calls[0][0].args[0]
    # A bare string would blow up LocalChatCompletion's `_create_payload`
    # assertion, so `_resolve_choices` must hand generate_until a JsonChatStr
    # that unwraps into a proper [{"role", "content"}, ...] message list.
    assert isinstance(sent, JsonChatStr)
    messages = json.loads(sent.prompt)
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    assert "A. one" in messages[0]["content"]
    assert "B. two" in messages[0]["content"]


def test_rwkv7_remote_chat_avg_at_k_samples_k_generations_via_generate_until(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    calls = []

    def fake_generate_until(instances):
        calls.append(instances)
        return ["A", "B", "A", "A"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests("What is 1+1?", ["one", "two"], doc_id=7, repeats=4)
    results = model.loglikelihood(requests)

    # All k rollouts are dispatched through a single generate_until call so
    # the harness's existing concurrency (num_concurrent semaphore, batching,
    # multi-replica pool dispatch) parallelizes them instead of blocking on
    # one rollout at a time.
    assert len(calls) == 1
    assert len(calls[0]) == 4
    sample_encodings = model._avg_k_samples[7]
    assert len(sample_encodings) == 4
    assert [enc[0][1] for enc in sample_encodings] == [True, False, True, True]
    assert model._avg_k_parsed[7] == [True, True, True, True]
    # `requests` has 4 cloned instances per choice, so `results` mirrors that
    # shape: 4 outcomes for choice "one" (the winner) then 4 for "two".
    assert results == [(0.0, True)] * 4 + [(-1e6, False)] * 4


def test_rwkv7_remote_chat_avg_at_k_ignores_unparsed_rollouts_in_majority_vote(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    def fake_generate_until(instances):
        # Only one of four rollouts names a valid letter; the other three
        # generations don't contain anything parse_choice_index recognizes.
        return ["gibberish", "B", "gibberish", "gibberish"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests("What is 1+1?", ["one", "two"], doc_id=9, repeats=4)
    results = model.loglikelihood(requests)

    # The majority vote is decided purely by the single valid rollout ("B"),
    # not diluted/overruled by unparseable ones defaulting to option A.
    assert results == [(-1e6, False)] * 4 + [(0.0, True)] * 4
    assert model._avg_k_parsed[9] == [False, True, False, False]


def test_rwkv7_remote_chat_loglikelihood_attaches_real_generation_as_evidence(monkeypatch):
    """See test_rwkv7_remote_loglikelihood_attaches_real_generation_as_evidence
    in test_rwkv_remote.py -- identical contract for the chat backend."""
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    real_generation = RWKV7RemoteChat.parse_generations(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The answer is B."},
                    "finish_reason": "stop",
                }
            ]
        }
    )[0]
    monkeypatch.setattr(model, "generate_until", lambda instances: [real_generation])

    requests = _mc_requests("ctx", ["one", "two"], doc_id=1, repeats=1)
    results = model.loglikelihood(requests)

    assert results == [(-1e6, False), (0.0, True)]
    for outcome in results:
        assert outcome.raw_response is real_generation.raw_response
        assert outcome.finish_reason == "stop"
        assert "Answer with only the letter" in outcome.prompt
        assert "A. one" in outcome.prompt
        assert "B. two" in outcome.prompt
