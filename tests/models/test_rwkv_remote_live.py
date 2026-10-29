"""Live network diagnostics for the hosted RWKV service (e.g. api.rwkv.rs).

Unlike test_rwkv_remote.py / test_rwkv_remote_chat.py (pure unit tests, no
network access), everything here makes real HTTP requests. It is opt-in and
skipped entirely unless RWKV_API_KEY is set in the environment, so it never
runs in normal CI. Point it at a different deployment with RWKV_LIVE_BASE_URL
and RWKV_LIVE_MODEL if needed.

Usage:
    RWKV_API_KEY=... RWKV_LIVE_MODEL=rwkv7-g1i-7.2b \
        pytest tests/models/test_rwkv_remote_live.py -v
"""

import json
import os

import pytest
import requests

from lm_eval.api.instance import Instance
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.rwkv_remote import RWKV7Remote
from lm_eval.models.rwkv_remote_chat import RWKV7RemoteChat


API_KEY = os.environ.get("RWKV_API_KEY")
BASE_URL = os.environ.get("RWKV_LIVE_BASE_URL", "https://api.rwkv.rs")
MODEL = os.environ.get("RWKV_LIVE_MODEL", "rwkv7-g1i-7.2b")

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="live RWKV endpoint tests require RWKV_API_KEY to be set",
)


def _headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def test_health_endpoint_is_up():
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_missing_auth_is_rejected():
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        headers={"Content-Type": "application/json"},
        json={"model": MODEL, "prompt": "test", "max_tokens": 4},
        timeout=10,
    )
    assert response.status_code == 401


def test_bad_auth_is_rejected():
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        headers={"Authorization": "Bearer not-a-real-key", "Content-Type": "application/json"},
        json={"model": MODEL, "prompt": "test", "max_tokens": 4},
        timeout=10,
    )
    assert response.status_code == 401


def test_v1_models_lists_the_configured_model():
    """Surfaces server-side model-registry problems directly.

    A 500 here (rather than a 200 with a `data` list) means the server's
    model registry itself is broken -- every /v1/completions or
    /v1/chat/completions call will 404 "Unknown model" regardless of the
    name used, no matter how it's spelled.
    """
    response = requests.get(f"{BASE_URL}/v1/models", headers=_headers(), timeout=10)
    assert response.status_code == 200, (
        f"/v1/models returned {response.status_code}: {response.text!r} -- "
        "the server's model registry looks broken or empty"
    )
    model_ids = {entry.get("id") for entry in response.json().get("data", [])}
    assert MODEL in model_ids, f"{MODEL!r} not in served models: {sorted(model_ids)}"


def test_completions_endpoint_returns_text():
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        headers=_headers(),
        json={"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    choices = response.json()["choices"]
    assert choices and choices[0]["text"].strip()


def test_chat_completions_endpoint_returns_message():
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=_headers(),
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "chat_template_kwargs": {
                "rwkv_prompt_template": "assistant",
                "rwkv_generation_prompt": "fake_think",
            },
            "max_tokens": 32,
        },
        timeout=30,
    )
    assert response.status_code == 200, response.text
    choices = response.json()["choices"]
    assert choices and choices[0]["message"]["content"].strip()


def test_rwkv7_remote_generate_until_via_real_network():
    model = RWKV7Remote(model=MODEL, num_concurrent=1)
    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=("The capital of France is", {"until": [], "max_gen_toks": 16}),
        idx=0,
    )
    (generated,) = model.generate_until([instance])
    assert str(generated).strip()


def test_rwkv7_remote_chat_generate_until_via_real_network():
    model = RWKV7RemoteChat(model=MODEL, num_concurrent=1)
    # LocalChatCompletion's _create_payload requires a [{"role", "content"}, ...]
    # message list; generate_until never wraps a bare string into one (only
    # apply_chat_template / _resolve_choices do), so the context must already
    # be a JsonChatStr here.
    chat_prompt = JsonChatStr(
        json.dumps([{"role": "user", "content": "Say hello in one word."}])
    )
    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=(chat_prompt, {"until": [], "max_gen_toks": 32}),
        idx=0,
    )
    (generated,) = model.generate_until([instance])
    assert str(generated).strip()
