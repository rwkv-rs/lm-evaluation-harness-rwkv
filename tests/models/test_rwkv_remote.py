from unittest.mock import patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model
from lm_eval.models.rwkv_lightning import RWKV7LightningCUDA
from lm_eval.models.rwkv_remote import RWKV7Remote


def _build(**kwargs):
    kwargs.setdefault("model", "rwkv7-g1i-1.5b")
    kwargs.setdefault("num_concurrent", 1)
    return RWKV7Remote(**kwargs)


def test_remote_sends_sampling_fields_to_vllm_directly():
    model = _build(rwkv_generation_prompt="open_think")
    payload = model._create_payload("question", generate=True,
                                    gen_kwargs={"max_gen_toks": 32, "until": []})
    assert "generation_config" not in payload
    for name in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay"):
        assert payload[name] == model._generation_profile[name]
    assert payload["stop"] == [model.PROMPT_STOPS[model.rwkv_prompt_template]]


def test_remote_independent_seeds_share_one_concurrent_dispatch(monkeypatch):
    model = _build(num_concurrent=4, seed=100)
    captured = []

    async def dispatch(requests, cache_keys, **kwargs):
        captured.append((requests, cache_keys, kwargs))
        return [["B"] for _ in requests]

    monkeypatch.setattr(model, "get_batched_requests", dispatch)
    request = Instance(request_type="generate_until", doc={}, arguments=("question", {"until": [], "max_gen_toks": 32}), idx=0, metadata=("task", 7, 4))
    model.generate_until([request] * 4)
    assert len(captured) == 1
    seeds = sorted(key[1].get("seed") for key in captured[0][1])
    assert seeds == [100, 101, 102, 103]
    assert "seed" not in request.args[1]


def test_remote_context_budget_preserves_all_prompt_tokens(monkeypatch):
    model = _build(num_concurrent=4, max_length=100)
    model.tokenized_requests = True
    monkeypatch.setattr(model, "tok_encode", lambda texts, **kw: [list(range(len(text))) for text in texts])
    captured = []

    async def dispatch(requests, cache_keys, **kwargs):
        captured.append((requests, cache_keys))
        return [["B"] for _ in requests]

    monkeypatch.setattr(model, "get_batched_requests", dispatch)
    request = Instance(request_type="generate_until", doc={}, arguments=("q" * 80, {"until": [], "max_gen_toks": 8192}), idx=0)
    model.generate_until([request])
    assert len(captured[0][0][0]) == 80
    assert captured[0][1][0][1]["max_gen_toks"] == 20


def test_remote_checkpoint_identity_includes_prompt_mode():
    from lm_eval.api.checkpoint import model_fingerprint
    first = _build(rwkv_prompt_template="assistant")
    second = _build(rwkv_prompt_template="bot")
    assert model_fingerprint(first) != model_fingerprint(second)


def test_remote_retains_every_raw_rollout(monkeypatch):
    model = _build()
    monkeypatch.setattr(model, "generate_until", lambda instances: ["A", "B"])
    responses = model.loglikelihood(_mc_requests("question", ["cats", "dogs"], repeats=2))
    assert [row["text"] for row in responses[0].rollouts] == ["A", "B"]


def test_remote_repeat_seeds_are_concurrent_and_reproducible(monkeypatch):
    import asyncio

    model = _build(num_concurrent=4, seed=100)
    seen = []
    active = 0
    peak = 0

    async def call(*, sem, gen_kwargs, **kwargs):
        nonlocal active, peak
        async with sem:
            active += 1
            peak = max(peak, active)
            seen.append(gen_kwargs["seed"])
            await asyncio.sleep(0.01)
            active -= 1
        return ["B"]

    monkeypatch.setattr(model, "amodel_call", call)
    request = Instance("generate_until", {}, ("question", {"max_gen_toks": 32, "until": []}), 0, ("task", 7, 4))
    values = model.generate_until([request] * 4)
    assert sorted(seen) == [100, 101, 102, 103]
    assert peak == 4
    assert [value.seed for value in values] == [100, 101, 102, 103]


def test_remote_writes_thinking_contract_to_checkpoint_before_return(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from lm_eval.api.checkpoint import CheckpointStore, ItemKey, deserialize_response

    monkeypatch.setenv("RWKV_API_KEY", "unit-test")
    model = _build(rwkv_generation_prompt="open_think", seed=100)
    store = CheckpointStore(str(tmp_path / "run.sqlite3"), "test")
    key = ItemKey("slot", "request", "idempotency", "task", 0, 0, "generate_until", 2)
    store.register_pending_items([key])
    model.set_checkpoint_store(store)
    response = Mock(ok=True)
    response.json.return_value = {"choices": [{"index": 0, "text": ">The answer is B", "finish_reason": "length"}]}
    monkeypatch.setattr("lm_eval.models.api_models.requests.post", lambda *a, **kw: response)
    request = Instance("generate_until", {}, ("question", {"max_gen_toks": 32, "until": []}), 0, ("task", 0, 3))
    model.generate_until([request], item_keys=[key])
    restored = deserialize_response(store.get_items("task")["slot"]["result_json"])
    assert restored.seed == 102
    assert restored.requires_think_close is True
    store.close()


def test_remote_resume_preserves_repeat_ord_seed(monkeypatch):
    from types import SimpleNamespace

    model = _build(num_concurrent=4, seed=100)
    seen = []

    async def dispatch(requests, cache_keys, **kwargs):
        seen.extend(key[1]["seed"] for key in cache_keys)
        return [["B"] for _ in requests]

    monkeypatch.setattr(model, "get_batched_requests", dispatch)
    request = Instance("generate_until", {}, ("question", {"max_gen_toks": 32, "until": []}), 0, ("task", 7, 4))
    values = model.generate_until([request, request], item_keys=[SimpleNamespace(repeat_ord=1), SimpleNamespace(repeat_ord=3)])
    assert sorted(seen) == [101, 103]
    assert [value.seed for value in values] == [101, 103]


@pytest.mark.parametrize("gold", range(4))
def test_remote_invalid_output_earns_zero_for_every_gold(monkeypatch, gold):
    from types import SimpleNamespace
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.evaluator import _install_avg_at_k_patch

    model = _build()
    monkeypatch.setattr(model, "generate_until", lambda instances: ["I cannot answer this question."] * len(instances))
    responses = model.loglikelihood(_mc_requests("question", ["cats", "dogs", "birds", "fish"], repeats=4))[::4]
    task = SimpleNamespace(config=SimpleNamespace(process_results=None), OUTPUT_TYPE="multiple_choice",
                           _metric_fn_list={"acc": None}, multiple_input=False, multiple_target=False,
                           doc_to_choice=lambda doc: ["cats", "dogs", "birds", "fish"], doc_to_target=lambda doc: gold,
                           _aggregation_list={"acc": None}, _higher_is_better={"acc": True}, _avg_k_current_doc_id=0)
    task.process_results = lambda doc, results: ConfigurableTask.process_results(task, doc, results)
    _install_avg_at_k_patch(task, model, 4, "avg@4")
    score = task.process_results({}, responses)
    assert score == {"acc": 0.0, "acc_avg@4": 0.0}


def test_remote_rejects_partial_choice_groups(monkeypatch):
    model = _build()
    monkeypatch.setattr(model, "generate_until", lambda instances: ["B"] * len(instances))
    with pytest.raises(ValueError, match="complete"):
        model.loglikelihood(_mc_requests("question", ["cats", "dogs"], repeats=2)[:-1])


def test_remote_does_not_copy_history_into_each_quoted_option(monkeypatch):
    model = _build()
    model._chat_template_source = _ASSISTANT_STYLE_TEMPLATE.replace(
        "{% if message.role == 'user' %}",
        "{% if message.role == 'system' %}System: {{ message.content }}\n\n{% elif message.role == 'user' %}",
    )
    prompt = model.apply_chat_template([
        {"role": "system", "content": "Unique system instruction"},
        {"role": "user", "content": "Unique demonstration"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": 'My answer would be "'},
    ])
    sent = []

    def generate(instances):
        sent.extend(inst.args[0] for inst in instances)
        return ["B"] * len(instances)

    monkeypatch.setattr(model, "generate_until", generate)
    model.loglikelihood(_mc_requests(prompt, ["yes", "no"]))
    assert sent[0].count("Unique system instruction") == 1
    assert sent[0].count("Unique demonstration") == 1
    assert 'A. My answer would be "yes' in sent[0]


def test_rwkv7_remote_construction_defaults(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.base_url == RWKV7Remote.DEFAULT_BASE_URL
    assert model.tokenizer_backend is None
    assert model.tokenizer is None


def test_rwkv7_remote_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("RWKV_API_KEY", raising=False)
    model = _build()

    with pytest.raises(ValueError, match="RWKV_API_KEY"):
        _ = model.api_key


def test_rwkv7_remote_api_key_used_in_header(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "sk-test-fixture-key")
    model = _build()

    assert model.api_key == "sk-test-fixture-key"
    assert model.header == {"Authorization": "Bearer sk-test-fixture-key"}


def test_rwkv7_remote_registered_in_model_mapping(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    assert get_model("rwkv7-remote") is RWKV7Remote


def test_rwkv7_remote_tokenizer_enables_tokenized_requests(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.return_value.chat_template = "{{ messages }}"
        model = _build(tokenizer="some/repo")

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.args[0] == "some/repo"
    assert model.tokenizer_backend == "huggingface"
    assert model.tokenized_requests is True


def test_rwkv7_remote_without_tokenizer_keeps_defaults(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.tokenizer_backend is None
    assert model.tokenized_requests is False


def test_rwkv7_remote_without_tokenizer_reports_fixed_sampling_config(monkeypatch):
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
            "stop": ["\nUser:"],
        }
    }


def test_rwkv7_remote_get_model_info_reports_sampling_config_with_tokenizer(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 0
        bos_token_id = 1
        chat_template = "{{ messages }}"

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()
    ):
        model = _build(tokenizer="some/repo")

    assert model.get_model_info() == {
        "sampling_config": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "do_sample": True,
            "stop": ["\nUser:"],
        }
    }


def test_rwkv7_remote_open_think_uses_its_own_fixed_sampling_config(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_generation_prompt="open_think")

    assert model.get_model_info() == {
        "sampling_config": {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
            "do_sample": True,
            "stop": ["\nUser:"],
        }
    }


def test_rwkv7_remote_no_think_uses_deterministic_sampling_config(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_generation_prompt="no_think")

    assert model.get_model_info() == {
        "sampling_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 1.0,
            "do_sample": False,
            "stop": ["\nUser:"],
        }
    }


def test_rwkv7_remote_no_think_renders_legacy_official_template(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 0
        bos_token_id = 1
        # The official tokenizer template rejects values other than open_think
        # and fake_think before selecting its generation cue.
        chat_template = (
            "{% if rwkv_generation_prompt not in ['open_think', 'fake_think'] %}"
            "{{ raise_exception('Unsupported RWKV generation prompt') }}"
            "{% endif %}Assistant: "
            "{% if rwkv_generation_prompt == 'open_think' %}<think"
            "{% else %}<think></think{% endif %}"
        )

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
        model = _build(tokenizer="some/repo", rwkv_generation_prompt="no_think")

    rendered = model.apply_chat_template(
        [{"role": "user", "content": "Question?"}], add_generation_prompt=True
    )
    assert rendered.endswith("Assistant: ")
    assert "<think" not in rendered
    assert model.get_model_info()["sampling_config"]["do_sample"] is False


def test_rwkv7_remote_parse_generations_reports_finish_reason():
    outputs = {
        "choices": [
            {"index": 0, "text": "hello", "finish_reason": "length"},
        ]
    }

    (generation,) = RWKV7Remote.parse_generations(outputs)

    assert str(generation) == "hello"
    assert generation.finish_reason == "length"
    assert generation.truncated is True


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


def test_rwkv7_remote_avg_at_k_samples_k_generations_via_generate_until(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    calls = []

    def fake_generate_until(instances):
        calls.append(instances)
        return ["A", "B", "A", "A"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests("What is 1+1?", ["one", "two"], doc_id=7, repeats=4)
    results = model.loglikelihood(requests)

    # avg@k exercises real generation (not echo/logprobs), so the sampling
    # profile in _create_payload actually applies -- all k samples for a doc
    # are dispatched as one generate_until call (not k blocking calls) so
    # the harness's existing concurrency parallelizes the rollouts.
    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert "A. one" in calls[0][0].args[0]
    assert "B. two" in calls[0][0].args[0]

    # k=4 sample encodings recorded for the process_results avg@k patch,
    # one full per-choice (ll, is_greedy) list per independent generation.
    sample_encodings = model._avg_k_samples[7]
    assert len(sample_encodings) == 4
    assert [enc[0][1] for enc in sample_encodings] == [True, False, True, True]
    assert [enc[1][1] for enc in sample_encodings] == [False, True, False, False]
    assert model._avg_k_parsed[7] == [True, True, True, True]

    # The single (ll, is_greedy) list returned still encodes a majority-vote
    # winner (choice "A" won 3/4), so acc/acc_norm work without any patch.
    # `requests` has 4 cloned instances per choice, so `results` mirrors that
    # shape: 4 outcomes for choice "one" (the winner) then 4 for "two".
    assert results == [(0.0, True)] * 4 + [(-1e6, False)] * 4


def test_rwkv7_remote_avg_at_k_ignores_unparsed_rollouts_in_majority_vote(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    def fake_generate_until(instances):
        # Only one of four rollouts names a valid letter; the rest can't be
        # parsed into a choice at all.
        return ["gibberish", "B", "gibberish", "gibberish"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests("What is 1+1?", ["one", "two"], doc_id=9, repeats=4)
    results = model.loglikelihood(requests)

    # The majority vote is decided purely by the single valid rollout ("B"),
    # not diluted/overruled by unparseable ones silently defaulting to A.
    assert results == [(-1e6, False)] * 4 + [(0.0, True)] * 4
    assert model._avg_k_parsed[9] == [False, True, False, False]


def test_rwkv7_remote_avg_at_k_defaults_to_one_sample(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    monkeypatch.setattr(model, "generate_until", lambda instances: ["B"])

    requests = _mc_requests("ctx", ["one", "two"], doc_id=1, repeats=1)
    results = model.loglikelihood(requests)

    assert len(model._avg_k_samples[1]) == 1
    assert results == [(-1e6, False), (0.0, True)]


# A minimal jinja template shaped like the real RWKV assistant-style chat
# template: a fixed system preamble, User/Assistant turns, then a
# generation-prompt cue appended independently of `messages`' content --
# exactly like every real chat template's generation-prompt cue behaves.
_ASSISTANT_STYLE_TEMPLATE = (
    "System: You are RWKV, an AI assistant.\n\n"
    "{% for message in messages %}"
    "{% if message.role == 'user' %}User: {{ message.content }}\n\n"
    "{% elif message.role == 'assistant' %}Assistant: {{ message.content }}\n\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: "
    "{% if rwkv_generation_prompt == 'open_think' %}<think"
    "{% else %}<think></think{% endif %}"
    "{% endif %}"
)


def test_rwkv7_remote_splices_choices_before_pretemplated_generation_cue(monkeypatch):
    """When the harness applies the chat template itself (apply_chat_template =
    true), `Instance.args[0]` arrives already ending in the model's turn-start
    cue (e.g. "Assistant: <think"). Choice content must be spliced in *before*
    that cue, and the prompt must be rendered exactly once -- not appended
    after the cue, and not chat-templated a second time on top of itself."""
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 0
        bos_token_id = 1
        chat_template = _ASSISTANT_STYLE_TEMPLATE

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
        model = _build(tokenizer="some/repo", rwkv_generation_prompt="open_think")

    question = "Which gas is most abundant in Earth's atmosphere?"
    pretemplated_context = model.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True
    )
    assert pretemplated_context.endswith("Assistant: <think")

    captured = {}

    def fake_generate_until(instances):
        captured["prompt"] = instances[0].args[0]
        return [">B"] * len(instances)

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests(pretemplated_context, ["Oxygen", "Nitrogen"], doc_id=1)
    model.loglikelihood(requests)

    prompt = captured["prompt"]
    # The cue is still the exact final thing in the prompt -- nothing landed
    # after it, so the model's completion region is untouched.
    assert prompt.endswith("Assistant: <think")
    # The choice menu was inserted before the cue, inside the user turn.
    assert prompt.index("A. Oxygen") < prompt.index("Assistant: <think")
    assert prompt.index("B. Nitrogen") < prompt.index("Assistant: <think")
    # Exactly one render pass: a second `apply_chat_template()` call on top
    # of the already-templated context would nest a second "Assistant:"
    # turn-start marker inside the outer one.
    assert prompt.count("Assistant:") == 1
    assert prompt.count("System:") == 1


def test_rwkv7_remote_loglikelihood_attaches_real_generation_as_evidence(monkeypatch):
    """The Scoreboard-published assembled_prompt/raw_completion must come
    from the real generative rollout, not be reconstructed from the score
    (see RWKVLoglikelihoodResponse) -- k=1, the only mode currently used in
    production, has exactly one real generation per doc, so this checks it
    is threaded through unmodified."""
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    real_generation = RWKV7Remote.parse_generations(
        {
            "choices": [
                {"index": 0, "text": "The answer is B.", "finish_reason": "stop"}
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


class _LightningTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 0
    bos_token_id = 1
    chat_template = _ASSISTANT_STYLE_TEMPLATE

    def __call__(self, text, **kwargs):
        from types import SimpleNamespace

        if isinstance(text, str):
            input_ids = list(range(len(text)))
        else:
            input_ids = [list(range(len(value))) for value in text]
        return SimpleNamespace(input_ids=input_ids)


def _build_lightning(**kwargs):
    kwargs.setdefault("model", "rwkv7-g1k-1.5b")
    kwargs.setdefault("tokenizer", "official-tokenizer")
    kwargs.setdefault("num_concurrent", 1)
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=_LightningTokenizer(),
    ):
        return RWKV7LightningCUDA(**kwargs)


def test_rwkv7_lightning_payload_uses_native_batch_fields_and_context_budget():
    model = _build_lightning(
        batch_size=2,
        max_length=20,
        rwkv_generation_prompt="open_think",
    )

    payload = model._create_payload(
        ["short", "tenletters"],
        generate=True,
        gen_kwargs={
            "max_gen_toks": 16,
            "until": ["END"],
            "seed": 99,
            "presence_penalty": 9.0,
        },
    )

    assert payload["contents"] == ["short", "tenletters"]
    assert payload["max_tokens"] == 10
    assert payload["temperature"] == 0.96
    assert payload["top_p"] == 0.76
    assert payload["top_k"] == 32
    assert payload["alpha_presence"] == 1.0
    assert payload["alpha_frequency"] == 0.1
    assert payload["alpha_decay"] == 0.988
    assert payload["force_reasoning"] is True
    assert payload["stop"] == ["END", *model.TRANSPORT_STOPS]
    assert "prompt" not in payload
    assert "seed" not in payload
    assert "presence_penalty" not in payload
    assert "chat_template_kwargs" not in payload


def test_rwkv7_lightning_native_batch_dispatches_one_http_request(monkeypatch):
    model = _build_lightning(batch_size=2, max_length=64, stream=False)
    captured = []

    class Response:
        ok = True

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "first"},
                        "finish_reason": "stop",
                    },
                    {
                        "index": 1,
                        "message": {"role": "assistant", "content": "second"},
                        "finish_reason": "length",
                    },
                ]
            }

    def post(url, **kwargs):
        captured.append((url, kwargs["json"]))
        return Response()

    monkeypatch.setattr("lm_eval.models.api_models.requests.post", post)
    requests = [
        Instance("generate_until", {}, (prompt, {"max_gen_toks": 8, "until": []}), index)
        for index, prompt in enumerate(("question one", "question two"))
    ]

    generations = model.generate_until(requests)

    assert [str(value) for value in generations] == ["first", "second"]
    assert [value.truncated for value in generations] == [False, True]
    assert len(captured) == 1
    assert captured[0][0].endswith("/v1/batch/completions")
    assert captured[0][1]["contents"] == ["question one", "question two"]


def test_rwkv7_lightning_strips_retained_matched_stop_from_scored_text():
    output = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is B.\nUser:",
                },
                "finish_reason": "stop",
                "matched_stop": "\nUser:",
            }
        ]
    }

    (generation,) = RWKV7LightningCUDA.parse_generations(output)

    assert str(generation) == "The answer is B."
    assert generation.finish_reason == "stop"
    assert generation.raw_response is output


def test_rwkv7_lightning_is_registered():
    assert get_model("rwkv7-lightning-cuda") is RWKV7LightningCUDA


def test_rwkv7_lightning_no_think_uses_minimum_supported_temperature():
    model = _build_lightning(rwkv_generation_prompt="no_think")

    payload = model._create_payload(
        "question", generate=True, gen_kwargs={"max_gen_toks": 8, "until": []}
    )

    assert payload["temperature"] == 0.01


def test_rwkv7_lightning_no_think_renders_closed_think_prefix():
    model = _build_lightning(rwkv_generation_prompt="no_think")

    rendered = model.apply_chat_template(
        [{"role": "user", "content": "question"}], add_generation_prompt=True
    )

    assert rendered.endswith("Assistant: <think></think>\n")
    assert model.rwkv_generation_prompt == "no_think"


def test_rwkv7_lightning_reads_cloudflare_access_headers(monkeypatch):
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "client-id")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "client-secret")
    model = _build_lightning()

    assert model.header["CF-Access-Client-Id"] == "client-id"
    assert model.header["CF-Access-Client-Secret"] == "client-secret"


def test_rwkv7_lightning_reports_actual_transport_sampling_contract():
    model = _build_lightning(rwkv_generation_prompt="no_think")

    sampling = model.get_model_info()["sampling_config"]

    assert sampling["temperature"] == 0.01
    assert sampling["stop"] == list(model.TRANSPORT_STOPS)
    assert sampling["seed"] == -1
    assert sampling["seed_strategy"] == "unsupported"


def test_rwkv7_lightning_requires_complete_sse_stream():
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "OK<EOD>"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "matched_stop": "<EOD>",
                }
            ]
        },
    ]

    output = RWKV7LightningCUDA._sse_output(events, expected=1, saw_done=True)
    (generation,) = RWKV7LightningCUDA.parse_generations(output)

    assert str(generation) == "OK"
    with pytest.raises(Exception, match="without data: \\[DONE\\]"):
        RWKV7LightningCUDA._sse_output(events, expected=1, saw_done=False)
