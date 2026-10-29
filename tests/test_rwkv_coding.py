"""RWKV coding contracts, extraction, replay, and real local execution."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lm_eval.rwkv_coding import (
    EXTRACTOR,
    PROTOCOL,
    CodingConfig,
    DockerJudge,
    JudgeError,
    _hash,
    extract_code,
    load_problems,
    pass_at_k,
    read_samples,
    run_coding,
    score,
)


@pytest.fixture
def suite(tmp_path):
    row = {
        "task_id": "local/add",
        "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
        "canonical_solution": "    return a + b\n",
    }
    path = tmp_path / "problems.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


@pytest.fixture
def config(suite, tmp_path):
    return CodingConfig(
        model={
            "model": "rwkv7-remote",
            "model_args": {
                "model": "rwkv7-g1k-1.5b",
                "base_url": "http://localhost:8000/v1/completions",
                "tokenizer": "official",
                "max_length": 32768,
                "max_gen_toks": 16384,
                "num_concurrent": 2,
                "seed": 40,
                "rwkv_prompt_template": "assistant",
                "rwkv_generation_prompt": "open_think",
            },
            "metadata": {"weight_revision": "rwkv7-g1k-1.5b-20260930-ctx32768"},
        },
        problems=suite,
        output_dir=tmp_path / "run",
        num_samples=2,
        pass_k=(1, 2),
        workers=2,
        test_timeout=0.5,
        memory_mb=256,
        docker_image="python:3.11-slim",
    )


@pytest.mark.parametrize(
    "raw, status",
    [
        ("<think>def add(a,b): return a+b", "unfinished_thinking"),
        (
            ">reasoning</think>\n```text\nx + y\n```\n```python\ndef add(a,b):\n    return a+b\n```\n",
            "extracted",
        ),
        (
            ">reasoning</think>\n```python\nimport math\n```\n```python\ndef add(a,b):\n    return a+b\n```\n",
            "extracted",
        ),
        (">reasoning</think>\n    return a + b\n", "extracted"),
        (">reasoning</think>\n```python\ndef add(a,b):\n", "invalid_code"),
        (
            ">reasoning</think>\n```\nexample\n```\n```python\ndef add(a,b): return a+b\n```\n```python\ndef add(a,b): return a-b\n```\n",
            "ambiguous_code",
        ),
    ],
)
def test_extraction_obeys_rwkv_final_and_implementation_contract(suite, raw, status):
    result = extract_code(raw, load_problems(suite)[0], thinking=True)
    assert result["status"] == status
    if status == "extracted":
        assert "return a+b" in result["code"] or "return a + b" in result["code"]


def test_fake_think_preserves_literal_tag_and_indentation(suite):
    raw = 'def add(a, b):\n    marker = "</think>"\n    return a + b\n'
    result = extract_code(raw, load_problems(suite)[0], thinking=False)
    assert result["code"] == raw


def test_open_think_preserves_literal_closing_tag_in_final_code(suite):
    code = 'def add(a, b):\n    marker = "</think>"\n    return a + b\n'
    result = extract_code(
        ">thought</think>\n" + code, load_problems(suite)[0], thinking=True
    )
    assert result["code"] == code


def test_deep_invalid_python_is_an_invalid_answer(suite):
    raw = "def add(a,b):\n    return " + "+".join(["1"] * 10000)
    result = extract_code(raw, load_problems(suite)[0], thinking=False)
    assert result["code"] is None
    assert result["status"] == "invalid_code"


@pytest.mark.parametrize(
    "n,c,k,expected", [(8, 0, 8, 0), (8, 1, 8, 1), (8, 4, 1, 0.5), (4, 2, 2, 5 / 6)]
)
def test_unbiased_pass_at_k(n, c, k, expected):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_rejects_too_few_samples():
    with pytest.raises(ValueError):
        pass_at_k(1, 0, 2)


def _rows(config):
    return [
        {
            "protocol": PROTOCOL,
            "generation_fingerprint": _hash(config.identity()),
            "runtime_model_fingerprint": "fixed-model",
            "task_id": "local/add",
            "repeat_ord": ordinal,
            "seed": 40 + ordinal,
            "raw_completion": raw,
            "finish_reason": reason,
        }
        for ordinal, raw, reason in [
            (
                0,
                ">reasoning</think>\n```python\ndef add(a,b): return a+b\n```\n",
                "stop",
            ),
            (1, ">still reasoning", "length"),
        ]
    ]


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "wrong_seed", "wrong_weight", "mixed_model"]
)
def test_replay_rejects_incomplete_or_mixed_identity(config, tmp_path, change):
    rows = _rows(config)
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    elif change == "wrong_seed":
        rows[0]["seed"] = 99
    elif change == "wrong_weight":
        rows[0]["generation_fingerprint"] = "other"
    else:
        rows[0]["runtime_model_fingerprint"] = "other"
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        read_samples(path, config, load_problems(config.problems))


@pytest.fixture
def docker_judge(config):
    # Optional integration checks use only small trusted test fixtures.
    try:
        return DockerJudge(config)
    except JudgeError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize(
    "code,status",
    [
        ("def add(a,b): return a+b", "passed"),
        ("def add(a,b): return a-b", "assertion_error"),
        ("def add(a,b):\n    while True: pass", "timeout"),
        ("import os\nos._exit(0)", "runtime_error"),
        ('print(\'{"status":"passed"}\')\nraise SystemExit(0)', "runtime_error"),
    ],
)
def test_real_docker_verdicts(docker_judge, suite, code, status):
    result = docker_judge.check(code, load_problems(suite)[0].tests)
    assert result["status"] == status
    assert result["passed"] is (status == "passed")


def test_real_offline_replay_keeps_raw_answers_and_metrics(
    config, docker_judge, tmp_path, monkeypatch
):
    rows = _rows(config)
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        "lm_eval.rwkv_coding._model",
        lambda c: pytest.fail("Offline replay must not initialize an HTTP model"),
    )
    out = tmp_path / "replay"
    result = run_coding(config, phase="score", samples=path, output_dir=out)
    assert result["metrics"] == {"pass@1": 0.5, "pass@2": 1.0, "avg@2": 0.5}
    assert result["truncation_rate"] == 0.5
    assert result["unfinished_thinking_rate"] == 0.5
    assert result["extractor"] == EXTRACTOR
    assert [
        json.loads(line)["raw_completion"]
        for line in (out / "samples.jsonl").read_text().splitlines()
    ] == [row["raw_completion"] for row in rows]
    with pytest.raises(ValueError, match="not empty"):
        run_coding(config, phase="score", samples=path, output_dir=out)


def test_reference_control_failure_produces_no_model_score(config, tmp_path):
    class BrokenJudge:
        image_id = "fake"

        def check(self, code, tests):
            return {"status": "runtime_error", "passed": False}

    with pytest.raises(JudgeError, match="control failed"):
        score(
            config,
            load_problems(config.problems),
            _rows(config),
            tmp_path,
            BrokenJudge(),
        )
    assert not (tmp_path / "results.json").exists()


def test_cli_registers_coding_without_loading_datasets(monkeypatch):
    import sys

    from lm_eval._cli.harness import HarnessCLI

    monkeypatch.setattr(
        sys, "argv", ["lm-eval", "coding", "-C", "coding.toml", "--phase", "generate"]
    )
    args = HarnessCLI().parse_args()
    assert args.command == "coding"
    assert args.phase == "generate"


def test_example_config_loads():
    config = CodingConfig.load(
        Path(__file__).parents[1] / "configs/eval/rwkv7_coding_local.toml"
    )
    assert config.num_samples == 8
    assert config.pass_k == (1, 4, 8)
    assert config.model["model_args"]["seed"] == 1234


@pytest.fixture
def wire_model(config, monkeypatch):
    from types import SimpleNamespace

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        bos_token_id = 0
        eos_token = "<eos>"  # noqa: S105 - tokenizer fixture, not a credential.
        chat_template = "{% for m in messages %}{{m.role}}: {{m.content}}\n\n{% endfor %}{% if add_generation_prompt %}Assistant: {% if rwkv_generation_prompt == 'open_think' %}<think{% else %}<think></think>\n{% endif %}{% endif %}"

        def __call__(self, values, **kwargs):
            return SimpleNamespace(
                input_ids=[[ord(char) for char in value] for value in values]
            )

        def decode(self, values, **kwargs):
            return "".join(chr(value) for value in values)

    monkeypatch.setenv("RWKV_API_KEY", "test-coding-secret")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: Tokenizer()
    )
    model = {
        **config.model,
        "model_args": {**config.model["model_args"], "num_concurrent": 1},
    }
    return replace(config, model=model)


def _completion(seed):
    from unittest.mock import Mock

    response = Mock(ok=True)
    text = (
        ">reasoning</think>\n```python\ndef add(a,b): return a+b\n```\n"
        if seed == 40
        else ">unfinished reasoning"
    )
    response.json.return_value = {
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop" if seed == 40 else "length",
            }
        ]
    }
    return response


def test_http_generation_resume_reuses_only_completed_ordinals(wire_model, monkeypatch):
    seen = []
    interrupt = True

    def post(url, *, json, **kwargs):
        seen.append(json["seed"])
        if interrupt and json["seed"] == 41:
            raise RuntimeError("simulated interrupt")
        return _completion(json["seed"])

    monkeypatch.setattr("lm_eval.models.api_models.requests.post", post)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        run_coding(wire_model, phase="generate")
    partial = (wire_model.output_dir / "samples.jsonl").read_text()
    assert [json.loads(line)["seed"] for line in partial.splitlines()] == [40]
    interrupt = False
    seen.clear()
    result = run_coding(wire_model, phase="generate", resume=True)
    assert seen == [41]
    assert result["n_completions"] == 2
    rows = read_samples(
        wire_model.output_dir / "samples.jsonl",
        wire_model,
        load_problems(wire_model.problems),
    )
    assert [row["seed"] for row in rows] == [40, 41]
    assert rows[0]["raw_response"]["choices"][0]["finish_reason"] == "stop"
    assert (
        "test-coding-secret"
        not in (wire_model.output_dir / "manifest.json").read_text()
    )


def test_http_to_real_local_execution(wire_model, docker_judge, monkeypatch):
    sent = []

    def post(url, *, json, **kwargs):
        sent.append(json)
        assert kwargs["timeout"] == 300
        assert "generation_config" not in json
        assert json["temperature"] == 0.96
        assert json["top_p"] == 0.76
        assert json["penalty_decay"] == 0.988
        prompt = "".join(chr(token) for token in json["prompt"][0])
        assert prompt.endswith("Assistant: <think")
        assert "assert candidate" not in prompt
        return _completion(json["seed"])

    monkeypatch.setattr("lm_eval.models.api_models.requests.post", post)
    result = run_coding(wire_model)
    assert sorted(payload["seed"] for payload in sent) == [40, 41]
    assert result["metrics"] == {"pass@1": 0.5, "pass@2": 1.0, "avg@2": 0.5}
    assert result["outcomes"] == {"passed": 1, "unfinished_thinking": 1}
    assert (
        wire_model.output_dir / "problems.source.jsonl"
    ).read_bytes() == wire_model.problems.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        run_coding(wire_model, resume=True)


def test_judge_failure_is_detected_before_http_generation(config, monkeypatch):
    def unavailable(cfg):
        raise JudgeError("image unavailable")

    monkeypatch.setattr("lm_eval.rwkv_coding.DockerJudge", unavailable)
    monkeypatch.setattr(
        "lm_eval.rwkv_coding._model",
        lambda c: pytest.fail("Generation must follow judge preflight"),
    )
    with pytest.raises(JudgeError, match="image unavailable"):
        run_coding(config)


def test_mbpp_local_problem_preserves_setup_and_tests(suite):
    row = {
        "task_id": 7,
        "entry_point": "add",
        "text": "Add a and b",
        "code": "def add(a,b): return a+b",
        "test_setup_code": "import math",
        "test_list": ["assert add(1,2)==3"],
    }
    suite.write_text(json.dumps(row) + "\n")
    problem = load_problems(suite)[0]
    assert problem.task_id == "7"
    assert "import math" in problem.tests
    assert "assert add(1,2)==3" in problem.prompt
    assert "def add" not in problem.prompt


def test_reference_control_rejects_tests_that_do_not_call_candidate(
    config, docker_judge
):
    problem = replace(load_problems(config.problems)[0], tests="assert 1 == 1")
    with pytest.raises(JudgeError, match="control failed"):
        score(config, [problem], _rows(config), config.output_dir, docker_judge)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repeat_ord", 1.0),
        ("repeat_ord", True),
        ("seed", True),
        ("runtime_model_fingerprint", None),
    ],
)
def test_replay_requires_exact_identity_types(config, tmp_path, field, value):
    rows = _rows(config)
    rows[0][field] = value
    path = tmp_path / "bad-samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        read_samples(path, config, load_problems(config.problems))


@pytest.mark.parametrize("mode", ["open_think", "fake_think"])
@pytest.mark.parametrize("template", ["assistant", "bot", "function_calling"])
def test_coding_config_accepts_each_native_rwkv_profile(tmp_path, mode, template):
    original = Path(__file__).parents[1] / "configs/eval/rwkv7_coding_local.toml"
    text = (
        original.read_text()
        .replace('generation_prompt = "open_think"', f'generation_prompt = "{mode}"')
        .replace('prompt_template = "assistant"', f'prompt_template = "{template}"')
    )
    path = tmp_path / "coding.toml"
    path.write_text(text)
    cfg = CodingConfig.load(path)
    assert cfg.thinking is (mode == "open_think")
    assert cfg.model["model_args"]["rwkv_prompt_template"] == template


@pytest.mark.parametrize(
    "old,new",
    [
        ("num_samples = 8", "num_samples = true"),
        ("pass_k = [1, 4, 8]", "pass_k = [9]"),
        ("test_timeout = 5.0", "test_timeout = -1"),
        ("workers = 2", "workers = 0"),
        ("memory_mb = 4096", "memory_mb = 0"),
        ('backend = "rwkv7-remote"', 'backend = "qwen-remote-chat"'),
        ('generation_prompt = "open_think"', 'generation_prompt = "no_think"'),
        ("apply_chat_template = true", "apply_chat_template = false"),
        ("log_samples = true", "log_samples = false"),
        ('docker_image = "python:3.11-slim"', 'docker_image = "--privileged"'),
    ],
)
def test_coding_config_rejects_invalid_evaluation_contract(tmp_path, old, new):
    original = Path(__file__).parents[1] / "configs/eval/rwkv7_coding_local.toml"
    path = tmp_path / "coding.toml"
    path.write_text(original.read_text().replace(old, new))
    with pytest.raises((ValueError, TypeError)):
        CodingConfig.load(path)


def test_failed_resume_preserves_original_problem_snapshot(wire_model, monkeypatch):
    monkeypatch.setattr(
        "lm_eval.models.api_models.requests.post",
        lambda url, *, json, **kw: _completion(json["seed"]),
    )
    run_coding(wire_model, phase="generate")
    snapshot = wire_model.output_dir / "problems.source.jsonl"
    original = snapshot.read_bytes()
    wire_model.problems.write_text(
        wire_model.problems.read_text().replace("Add two", "Sum two")
    )
    with pytest.raises(ValueError, match="identity changed"):
        run_coding(wire_model, phase="generate", resume=True)
    assert snapshot.read_bytes() == original


def test_real_container_has_no_host_credentials_or_write_access(
    docker_judge, suite, monkeypatch
):
    monkeypatch.setenv("RWKV_API_KEY", "local-only-test-value")
    code = """import os
def add(a, b):
    assert os.geteuid() != 0
    assert "RWKV_API_KEY" not in os.environ
    assert not os.path.exists("/var/run/docker.sock")
    try:
        open("/cannot-write-on-root", "w")
    except OSError:
        return a+b
    raise AssertionError("root filesystem writable")
"""
    assert docker_judge.check(code, load_problems(suite)[0].tests)["passed"]


def test_cli_real_http_chat_and_docker(suite, tmp_path, docker_judge, monkeypatch):
    import subprocess
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    sent = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sent.append(payload)
            content = (
                ">reasoning</think>\n```python\ndef add(a,b): return a+b\n```\n"
                if payload["seed"] == 40
                else ">unfinished reasoning"
            )
            response = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": content},
                            "finish_reason": "stop"
                            if payload["seed"] == 40
                            else "length",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("RWKV_API_KEY", "test-local-http-key")
    original = Path(__file__).parents[1] / "configs/eval/rwkv7_coding_local.toml"
    text = original.read_text().replace(
        'backend = "rwkv7-remote"', 'backend = "rwkv7-remote-chat"'
    )
    text = text.replace('tokenizer_name = "rwkv-rs/rwkv7-g1-st"\n', "")
    text = text.replace(
        "http://127.0.0.1:8000/v1/completions",
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
    )
    text = text.replace(
        'problems = "../../data/coding/problems.jsonl"', 'problems = "problems.jsonl"'
    )
    text = text.replace("seed = [0, 1234, 1234, 1234]", "seed = [0, 1234, 40, 1234]")
    text = text.replace("num_samples = 8", "num_samples = 2").replace(
        "pass_k = [1, 4, 8]", "pass_k = [1, 2]"
    )
    path = tmp_path / "coding.toml"
    path.write_text(text)
    out = tmp_path / "cli-run"
    try:
        result = subprocess.run(  # noqa: S603 - this project's CLI with local fixture paths.
            [
                sys.executable,
                "-m",
                "lm_eval",
                "coding",
                "-C",
                str(path),
                "--output-dir",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode == 0, result.stderr
    assert sorted(payload["seed"] for payload in sent) == [40, 41]
    assert all(
        payload["chat_template_kwargs"]
        == {"rwkv_prompt_template": "assistant", "rwkv_generation_prompt": "open_think"}
        for payload in sent
    )
    assert all(
        payload["temperature"] == 0.96 and "generation_config" not in payload
        for payload in sent
    )
    report = json.loads((out / "results.json").read_text())
    assert report["metrics"] == {"pass@1": 0.5, "pass@2": 1.0, "avg@2": 0.5}
    assert "test-local-http-key" not in result.stdout
