import asyncio
import hashlib
import json
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.api_models import (
    ApiRequestError,
    ContextLengthExhausted,
    context_error_prompt_tokens,
    create_image_prompt,
    shrink_max_tokens,
)
from lm_eval.models.openai_completions import LocalCompletionsAPI


TEST_AUTH_TOKEN = "secure-token"  # noqa: S105 - fixed test sentinel


@pytest.fixture
def api():
    return LocalCompletionsAPI(
        base_url="http://test-url.com", tokenizer_backend=None, model="gpt-3.5-turbo"
    )


@pytest.fixture
def api_tokenized():
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        model="EleutherAI/pythia-1b",
        tokenizer_backend="huggingface",
    )


@pytest.fixture
def api_batch_ssl_tokenized():
    return LocalCompletionsAPI(
        base_url="https://test-url.com",
        model="EleutherAI/pythia-1b",
        verify_certificate=False,
        num_concurrent=2,
        tokenizer_backend="huggingface",
    )


def test_create_payload_generate(api):
    messages = ["Generate a story"]
    gen_kwargs = {
        "max_tokens": 100,
        "temperature": 0.7,
        "until": ["The End"],
        "do_sample": True,
        "seed": 1234,
    }
    payload = api._create_payload(messages, generate=True, gen_kwargs=gen_kwargs)

    assert payload == {
        "prompt": ["Generate a story"],
        "model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["The End"],
        "seed": 1234,
    }


def test_create_payload_loglikelihood(api):
    messages = ["The capital of France is"]
    payload = api._create_payload(messages, generate=False, gen_kwargs=None)

    assert payload == {
        "model": "gpt-3.5-turbo",
        "prompt": ["The capital of France is"],
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
        "temperature": 0,
        "seed": 1234,
    }


def test_local_completions_preserves_finish_reason(api):
    generations = api.parse_generations(
        {
            "choices": [
                {"index": 0, "text": "complete", "finish_reason": "stop"},
                {"index": 1, "text": "cut off", "finish_reason": "length"},
            ]
        }
    )

    assert generations == ["complete", "cut off"]
    assert generations[0].finish_reason == "stop"
    assert generations[1].finish_reason == "length"

    restored = pickle.loads(pickle.dumps(generations))
    assert restored == generations
    assert restored[0].finish_reason == "stop"
    assert restored[1].finish_reason == "length"


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            ["Hello, how are"],
            True,
            {"max_gen_toks": 100, "temperature": 0.7, "until": ["hi"]},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "temperature": 0.7,
                "stop": ["hi"],
                "seed": 1234,
            },
        ),
        (
            ["Hello, how are", "you"],
            True,
            {},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 256,
                "temperature": 0,
                "stop": [],
                "seed": 1234,
            },
        ),
    ],
)
def test_model_generate_call_usage(
    api, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}


def test_model_call_passes_configured_timeout():
    api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="gpt-3.5-turbo",
        timeout=17,
    )

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        api.model_call(["Hello"], generate=True, gen_kwargs={})

        assert mock_post.call_args.kwargs["timeout"] == 17


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            [[1, 2, 3, 4, 5]],
            False,
            None,
            {
                "model": "EleutherAI/pythia-1b",
                "prompt": [[1, 2, 3, 4, 5]],
                "max_tokens": 1,
                "logprobs": 1,
                "echo": True,
                "seed": 1234,
                "temperature": 0,
            },
        ),
    ],
)
def test_model_tokenized_call_usage(
    api_tokenized, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api_tokenized.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(
            "LocalChatCompletion",
            id="local-chat-completions",
        ),
        pytest.param(
            "OpenAIChatCompletion",
            id="openai-chat-completions",
        ),
    ],
)
def test_chat_template_payload_does_not_add_top_level_type(model_cls):
    from lm_eval.models import openai_completions

    model = getattr(openai_completions, model_cls)(
        base_url="http://test-url.com",
        model="test-model",
    )
    chat = [{"role": "user", "content": "Reply with one word: hello"}]

    messages = model.create_message((model.apply_chat_template(chat),))
    payload = model._create_payload(messages, generate=True, gen_kwargs={})

    assert payload["messages"] == chat
    assert "type" not in payload["messages"][0]


@pytest.mark.parametrize("include_legacy_type", [False, True])
def test_create_image_prompt_uses_content_parts_without_top_level_type(
    include_legacy_type,
):
    class DummyImage:
        def save(self, buf, format):
            buf.write(b"image-bytes")

    chat = [{"role": "user", "content": "Describe this image"}]
    if include_legacy_type:
        chat[0]["type"] = "text"

    messages = create_image_prompt([DummyImage()], json.loads(json.dumps(chat)))

    assert "type" not in messages[-1]
    assert messages[-1]["content"][0]["type"] == "image_url"
    assert messages[-1]["content"][1] == {
        "type": "text",
        "text": "Describe this image",
    }


class DummyAsyncContextManager:
    def __init__(self, result):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.parametrize(
    "expected_inputs, expected_ctxlens, expected_cache_keys",
    [
        (
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            [3, 3, 3, 3],
            ["cache_key1", "cache_key2", "cache_key3", "cache_key4"],
        ),
    ],
)
def test_get_batched_requests_with_no_ssl(
    api_batch_ssl_tokenized, expected_inputs, expected_ctxlens, expected_cache_keys
):
    with (
        patch(
            "lm_eval.models.api_models.TCPConnector", autospec=True
        ) as mock_connector,
        patch(
            "lm_eval.models.api_models.ClientSession", autospec=True
        ) as mock_client_session,
        patch(
            "lm_eval.models.openai_completions.LocalCompletionsAPI.parse_logprobs",
            autospec=True,
        ) as mock_parse,
    ):
        mock_session_instance = AsyncMock()
        mock_post_response = AsyncMock()
        mock_post_response.status = 200
        mock_post_response.ok = True
        mock_post_response.json = AsyncMock(return_value={"mocked": "response"})
        mock_post_response.raise_for_status = lambda: None
        mock_session_instance.post = lambda *args, **kwargs: DummyAsyncContextManager(
            mock_post_response
        )
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        mock_parse.return_value = [(1.23, True), (4.56, False)]

        async def run():
            return await api_batch_ssl_tokenized.get_batched_requests(
                expected_inputs,
                expected_cache_keys,
                generate=False,
                ctxlens=expected_ctxlens,
            )

        result_batches = asyncio.run(run())

        mock_connector.assert_called_with(limit=2, ssl=False)
        assert result_batches


def test_local_completionsapi_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token=TEST_AUTH_TOKEN,
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == TEST_AUTH_TOKEN


def test_local_completionsapi_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None


def test_localchatcompletion_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token=TEST_AUTH_TOKEN,
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == TEST_AUTH_TOKEN


def test_localchatcompletion_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None


# ---------------------------------------------------------------------------
# Single-endpoint request dispatch: one flat concurrency budget, no replica
# pool. The endpoint's declared capacity is the whole scheduling story, so
# every request goes to `self.base_url` and failures are retried in place by
# the caller's tenacity policy (see `get_batched_requests`).
# ---------------------------------------------------------------------------


@pytest.fixture
def endpoint_api():
    return LocalCompletionsAPI(
        base_url="http://test-url.com/v1/completions",
        tokenizer_backend=None,
        model="test-model",
    )


def test_context_error_prompt_tokens_matches():
    body = "Error: prompt contains at least 4096 input tokens, please reduce."
    assert context_error_prompt_tokens(400, body) == 4096


def test_context_error_prompt_tokens_ignores_other_status():
    body = "prompt contains at least 4096 input tokens"
    assert context_error_prompt_tokens(500, body) is None


def test_context_error_prompt_tokens_no_match():
    assert context_error_prompt_tokens(400, "some other error") is None


def test_shrink_max_tokens_shrinks():
    payload = {"max_tokens": 512}
    adjusted = shrink_max_tokens(payload, available=100)
    assert adjusted == {"max_tokens": 100}
    assert payload == {"max_tokens": 512}  # original untouched


def test_shrink_max_tokens_handles_max_completion_tokens():
    payload = {"max_completion_tokens": 512, "other": 1}
    adjusted = shrink_max_tokens(payload, available=50)
    assert adjusted == {"max_completion_tokens": 50, "other": 1}


def test_shrink_max_tokens_no_budget():
    assert shrink_max_tokens({"max_tokens": 512}, available=0) is None
    assert shrink_max_tokens({"max_tokens": 512}, available=-5) is None


def test_shrink_max_tokens_no_shrinkable_key():
    assert shrink_max_tokens({"prompt": "x"}, available=100) is None


def test_shrink_max_tokens_already_smaller_than_available():
    # Nothing to shrink -- available is not tighter than current value.
    assert shrink_max_tokens({"max_tokens": 10}, available=100) is None


def test_context_length_exhausted_is_an_api_request_error():
    assert issubclass(ContextLengthExhausted, ApiRequestError)


def test_model_call_shrinks_max_tokens_on_context_overflow(endpoint_api):
    overflow_response = MagicMock()
    overflow_response.ok = False
    overflow_response.status_code = 400
    overflow_response.text = (
        "prompt contains at least 2040 input tokens, please reduce."
    )

    success_response = MagicMock()
    success_response.ok = True
    success_response.json.return_value = {"result": "shrunk-ok"}

    with patch(
        "requests.post", side_effect=[overflow_response, success_response]
    ) as mock_post:
        result = endpoint_api.model_call(
            ["hello"], generate=True, gen_kwargs={"max_tokens": 256, "until": []}
        )

    assert result == {"result": "shrunk-ok"}
    assert mock_post.call_count == 2
    # Both attempts hit the one configured endpoint.
    assert {call.args[0] for call in mock_post.call_args_list} == {
        "http://test-url.com/v1/completions"
    }
    first_payload = mock_post.call_args_list[0].kwargs["json"]
    second_payload = mock_post.call_args_list[1].kwargs["json"]
    assert first_payload["max_tokens"] == 256
    assert second_payload["max_tokens"] == endpoint_api.max_length - 2040


def test_model_call_raises_context_length_exhausted_when_unshrinkable(endpoint_api):
    overflow_response = MagicMock()
    overflow_response.ok = False
    overflow_response.status_code = 400
    overflow_response.text = (
        f"prompt contains at least {endpoint_api.max_length + 100} input tokens."
    )

    with patch("requests.post", return_value=overflow_response):
        with pytest.raises(ContextLengthExhausted):
            endpoint_api.model_call(
                ["hello"], generate=True, gen_kwargs={"max_tokens": 256, "until": []}
            )


def test_model_call_raises_api_request_error_on_other_failures(endpoint_api):
    failure_response = MagicMock()
    failure_response.ok = False
    failure_response.status_code = 503
    failure_response.text = "service unavailable"

    with patch("requests.post", return_value=failure_response):
        with pytest.raises(ApiRequestError, match="HTTP 503"):
            endpoint_api.model_call(["hello"], generate=False)


def test_generate_until_converts_exhaustion_to_placeholder(endpoint_api):
    overflow_response = MagicMock()
    overflow_response.ok = False
    overflow_response.status_code = 400
    overflow_response.text = (
        f"prompt contains at least {endpoint_api.max_length + 100} input tokens."
    )

    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=("hello", {"until": [], "max_tokens": 16}),
        idx=0,
    )

    with patch("requests.post", return_value=overflow_response):
        results = endpoint_api.generate_until([instance])

    assert len(results) == 1
    assert str(results[0]) == ""
    assert results[0].finish_reason == "length"
    assert results[0].truncated is True


def _async_response(*, ok, status=200, text="", json_body=None):
    resp = AsyncMock()
    resp.ok = ok
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    resp.json = AsyncMock(return_value=json_body or {})
    return resp


def test_amodel_call_shrinks_in_place_on_context_overflow(endpoint_api):
    overflow = _async_response(
        ok=False, status=400, text="prompt contains at least 2040 input tokens."
    )
    success = _async_response(
        ok=True, json_body={"choices": [{"index": 0, "text": "ok"}]}
    )

    post_calls = []

    def fake_post(url, json, headers):
        post_calls.append((url, dict(json)))
        return DummyAsyncContextManager(overflow if len(post_calls) == 1 else success)

    session = MagicMock()
    session.post = fake_post
    sem = asyncio.Semaphore(1)

    async def run():
        return await endpoint_api.amodel_call(
            session,
            sem,
            ["hello"],
            generate=True,
            gen_kwargs={"max_tokens": 256, "until": []},
        )

    with patch.object(LocalCompletionsAPI, "parse_generations", return_value=["ok"]):
        answers = asyncio.run(run())

    assert answers == ["ok"]
    assert len(post_calls) == 2
    assert post_calls[0][0] == post_calls[1][0] == endpoint_api.base_url
    assert post_calls[1][1]["max_tokens"] == endpoint_api.max_length - 2040


def test_amodel_call_raises_api_request_error_on_other_failures(endpoint_api):
    failure = _async_response(ok=False, status=503, text="unavailable")

    session = MagicMock()
    session.post = lambda url, json, headers: DummyAsyncContextManager(failure)
    sem = asyncio.Semaphore(1)

    async def run():
        return await endpoint_api.amodel_call(
            session,
            sem,
            ["hello"],
            generate=True,
            gen_kwargs={"max_tokens": 16, "until": []},
        )

    with pytest.raises(ApiRequestError, match="HTTP 503"):
        asyncio.run(run())


def test_get_batched_requests_cancels_siblings_when_one_task_fails_hard(endpoint_api):
    """When one batch hits a non-retryable, non-context-length error,
    `get_batched_requests` must cancel the other in-flight `amodel_call` tasks
    itself before propagating, instead of letting `tqdm_asyncio.gather` raise
    while they keep running against a session the enclosing `async with` is
    about to close.
    """
    endpoint_api.max_retries = 1  # avoid tenacity's exponential backoff delay
    still_running = []
    cancelled = []

    async def fake_amodel_call(self, session, sem, messages, **kwargs):
        # `get_batched_requests` chunks `requests` with `batch_size=1`, so
        # each call receives a single-element list wrapping one of our
        # original request entries, e.g. `[["fail"]]` for the `["fail"]`
        # request.
        if messages == [["fail"]]:
            raise ApiRequestError("HTTP 404: Unknown model")
        still_running.append(messages)
        try:
            await asyncio.sleep(5)
            return ["should not get here"]
        except asyncio.CancelledError:
            cancelled.append(messages)
            raise

    with patch.object(
        type(endpoint_api), "amodel_call", side_effect=fake_amodel_call, autospec=True
    ):
        with pytest.raises(ApiRequestError):
            asyncio.run(
                asyncio.wait_for(
                    endpoint_api.get_batched_requests(
                        [["fail"], ["slow-1"], ["slow-2"]],
                        cache_keys=[None, None, None],
                        generate=True,
                        gen_kwargs={"until": []},
                    ),
                    timeout=3,
                )
            )

    assert sorted(still_running) == [[["slow-1"]], [["slow-2"]]]
    assert sorted(cancelled) == [[["slow-1"]], [["slow-2"]]]
