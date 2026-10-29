"""RWKV HTTP generation and reproducible, local Python functional evaluation.

This runner consumes local problem files; it does not register copies of coding
benchmarks that are already provided by other evaluation frameworks.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from lm_eval.api.checkpoint import (
    CheckpointStore,
    ItemKey,
    compute_idempotency_key,
    compute_item_id,
    compute_request_fingerprint,
    deserialize_response,
    model_fingerprint,
    serialize_response,
)
from lm_eval.api.instance import Instance
from lm_eval.config.evaluate_config import EvaluatorConfig


PROTOCOL = "rwkv-coding-v1"
EXTRACTOR = "rwkv-python-v1"
TASK = "rwkv_coding_local"
_FENCES = re.compile(
    r"^[ \t]*```([\w+-]*)[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)",
    re.MULTILINE | re.DOTALL,
)
_LENGTH_FINISHES = {"length", "max_tokens", "max_new_tokens", "max_output_tokens"}


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _positive(value, name, *, integer=True):
    kind = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kind):
        raise TypeError(f"{name} must be numeric, without boolean values")
    if value <= 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(
            f"{name} must be a positive {'integer' if integer else 'number'}"
        )
    return value


@dataclass(frozen=True)
class CodingConfig:
    model: dict
    problems: Path
    output_dir: Path
    num_samples: int
    pass_k: tuple[int, ...]
    workers: int
    test_timeout: float
    memory_mb: int
    docker_image: str

    @classmethod
    def load(cls, filename):
        path = Path(filename).expanduser().resolve()
        raw = dict(EvaluatorConfig._load_config_mapping(str(path)))
        coding = raw.pop("coding", None)
        if not isinstance(coding, dict):
            raise TypeError("A [coding] table is required")
        allowed = {
            "problems",
            "num_samples",
            "pass_k",
            "workers",
            "test_timeout",
            "memory_mb",
            "docker_image",
        }
        if unknown := coding.keys() - allowed:
            raise ValueError("Unknown coding fields: " + ", ".join(sorted(unknown)))
        if raw.get("backend") not in {"rwkv7-remote", "rwkv7-remote-chat"}:
            raise ValueError("Local coding requires rwkv7-remote or rwkv7-remote-chat")
        if raw.get("benchmarks") != [TASK]:
            raise ValueError(f"coding requires benchmarks = ['{TASK}']")
        if raw.get("backend") == "rwkv7-remote" and not raw.get("tokenizer_name"):
            raise ValueError("rwkv7-remote coding requires the official tokenizer_name")
        if raw.get("apply_chat_template", True) is not True:
            raise ValueError("RWKV coding requires apply_chat_template = true")
        if any(
            name in raw
            for name in (
                "limit",
                "samples",
                "use_cache",
                "checkpoint",
                "task_overrides",
                "models",
                "model_overrides",
            )
        ):
            raise ValueError(
                "coding owns its sample plan and checkpoint; use a separate config without run-only overrides"
            )
        model = EvaluatorConfig._normalize_versioned_config(raw)
        if model["model_args"].get("rwkv_generation_prompt") not in {
            "open_think",
            "fake_think",
        }:
            raise ValueError("coding requires open_think or fake_think generation")
        if (model.get("publication") or {}).get("enabled"):
            raise ValueError(
                "Coding results are local artifacts; automatic publication is not supported"
            )
        if not model["log_samples"] or model["batch_size"] != 1:
            raise ValueError("coding requires log_samples = true and batch_size = 1")
        if not re.match(r"rwkv7[a-z]?-g1", raw["model_name"], re.IGNORECASE):
            raise ValueError("model_name must identify a specific RWKV7 G1 weight")
        revision = raw.get("weight_revision", "")
        if (
            not isinstance(revision, str)
            or re.search(r"ctx[1-9][0-9]*", revision) is None
        ):
            raise ValueError(
                "weight_revision must identify the exact weight and its ctx<number>"
            )
        endpoint = urlsplit(raw["base_url"])
        if (
            endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "Use an endpoint without embedded credentials or query parameters"
            )
        if (
            not isinstance(coding.get("problems"), str)
            or not coding["problems"].strip()
        ):
            raise ValueError("coding.problems must name a local JSONL file")
        n = _positive(coding.get("num_samples", 8), "coding.num_samples")
        ks = coding.get("pass_k", sorted({1, n}))
        if not isinstance(ks, list) or not ks:
            raise ValueError("coding.pass_k must be a nonempty list")
        for k in ks:
            if _positive(k, "coding.pass_k") > n:
                raise ValueError("pass@k requires at least k samples for every problem")
        image = coding.get("docker_image", "python:3.11-slim")
        if not isinstance(image, str) or not image.strip() or image.startswith("-"):
            raise ValueError("coding.docker_image must name a local Python image")
        return cls(
            model=model,
            problems=(path.parent / coding["problems"]).resolve(),
            output_dir=(path.parent / raw["output_dir"]).resolve(),
            num_samples=n,
            pass_k=tuple(sorted(set(ks))),
            workers=_positive(coding.get("workers", 2), "coding.workers"),
            test_timeout=_positive(
                coding.get("test_timeout", 5.0), "coding.test_timeout", integer=False
            ),
            memory_mb=_positive(coding.get("memory_mb", 4096), "coding.memory_mb"),
            docker_image=image,
        )

    @property
    def thinking(self):
        return self.model["model_args"]["rwkv_generation_prompt"] == "open_think"

    def identity(self):
        # Operational knobs may change on resume; generation semantics may not.
        args = {
            k: v
            for k, v in self.model["model_args"].items()
            if k not in {"num_concurrent", "timeout", "max_retries"}
        }
        return {
            "protocol": PROTOCOL,
            "backend": self.model["model"],
            "model_args": args,
            "weight_revision": self.model["metadata"]["weight_revision"],
            "wkv_mode": self.model["metadata"].get("wkv_mode"),
            "num_samples": self.num_samples,
            "problems_sha256": hashlib.sha256(self.problems.read_bytes()).hexdigest(),
        }


@dataclass(frozen=True)
class Problem:
    task_id: str
    prompt: str
    entry_point: str
    tests: str
    reference: str
    prefix: str = ""


def _tree(code):
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None


def _defines(code, entry_point):
    tree = _tree(code)
    return tree is not None and any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == entry_point
        for node in tree.body
    )


def load_problems(path) -> list[Problem]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    problems, seen = [], set()
    with opener(path, "rt", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("task_id"), (str, int))
                or isinstance(row.get("task_id"), bool)
            ):
                raise ValueError(f"Problem line {lineno} requires a task_id")  # noqa: TRY004 - dataset validation error.
            task_id = str(row["task_id"])
            if not task_id or task_id in seen:
                raise ValueError(f"Duplicate or empty task_id: {task_id!r}")
            seen.add(task_id)
            entry = row.get("entry_point")
            if not isinstance(entry, str) or not entry.isidentifier():
                raise ValueError(
                    f"{task_id}: an explicit Python entry_point is required"
                )
            if "test_list" in row:
                tests = row["test_list"]
                if (
                    not isinstance(tests, list)
                    or not tests
                    or any(not isinstance(t, str) for t in tests)
                ):
                    raise ValueError(
                        f"{task_id}: test_list must contain Python assertions"
                    )
                text, reference = row.get("text"), row.get("code")
                if not isinstance(text, str) or not isinstance(reference, str):
                    raise ValueError(
                        f"{task_id}: MBPP-style problems require text and reference code"
                    )
                prompt = (
                    f"Write a complete Python implementation for this task:\n{text}\nYour code should pass these tests:\n"
                    + "\n".join(tests)
                )
                tests = row.get("test_setup_code", "") + "\n" + "\n".join(tests)
                prefix = ""
            else:
                prefix, tests, solution = (
                    row.get("prompt"),
                    row.get("test"),
                    row.get("canonical_solution"),
                )
                if any(
                    not isinstance(value, str) or not value.strip()
                    for value in (prefix, tests, solution)
                ):
                    raise ValueError(
                        f"{task_id}: requires prompt, test, and canonical_solution"
                    )
                prompt = (
                    "Complete the following Python function. Return only the complete Python implementation.\n```python\n"
                    + prefix
                    + "\n```"
                )
                reference = solution if _defines(solution, entry) else prefix + solution
                test_tree = _tree(tests)
                if test_tree is None or not any(
                    isinstance(n, ast.FunctionDef) and n.name == "check"
                    for n in test_tree.body
                ):
                    raise ValueError(f"{task_id}: test must define check(candidate)")
                tests += f"\ncheck({entry})\n"
            tree = _tree(tests)
            if (
                tree is None
                or not any(isinstance(n, ast.Assert) for n in ast.walk(tree))
                or not _defines(reference, entry)
            ):
                raise ValueError(
                    f"{task_id}: invalid tests or reference implementation"
                )
            problems.append(Problem(task_id, prompt, entry, tests, reference, prefix))
    if not problems:
        raise ValueError("The problem file contains no problems")
    return problems


def extract_code(raw: str, problem: Problem, *, thinking: bool) -> dict:
    """Frozen before grading: never select a candidate using test outcomes."""
    final, closed = raw, False
    if thinking or raw.lstrip().startswith("<think"):
        # The native opening prefix has exactly one matching close. Later
        # occurrences can be literal strings in the generated Python program.
        _, tag, final = raw.partition("</think>")
        closed = bool(tag)
        if not closed:
            return {
                "code": None,
                "status": "unfinished_thinking",
                "thinking_closed": False,
            }
    blocks = [(m.group(1).lower(), m.group(2)) for m in _FENCES.finditer(final)]
    eligible = [
        (i, code)
        for i, (language, code) in enumerate(blocks)
        if language in {"", "python", "py", "python3"}
    ]

    def complete(code):
        if _defines(code, problem.entry_point):
            return code
        first = next((line for line in code.splitlines() if line.strip()), "")
        if (
            problem.prefix
            and first.startswith((" ", "\t"))
            and _defines(problem.prefix + code, problem.entry_point)
        ):
            return problem.prefix + code
        return None

    if blocks:
        candidates = [(i, complete(code)) for i, code in eligible]
        candidates = [(i, code) for i, code in candidates if code is not None]
        # Preserve an existing first complete implementation. Otherwise require
        # one unique complete implementation after example/formula blocks.
        if (
            candidates
            and eligible
            and candidates[0][0] == eligible[0][0]
            or len(candidates) == 1
        ):
            chosen, code = candidates[0]
        else:
            return {
                "code": None,
                "status": "ambiguous_code" if candidates else "invalid_code",
                "thinking_closed": closed,
            }
        imports = []
        for index, preceding in eligible:
            if index >= chosen:
                break
            tree = _tree(preceding)
            if (
                tree is not None
                and tree.body
                and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body)
            ):
                imports.append(preceding)
        if imports:
            code = "\n".join(imports) + "\n" + code
    else:
        code = complete(final.lstrip("\r\n"))
        if code is None:
            return {"code": None, "status": "invalid_code", "thinking_closed": closed}
    return {"code": code, "status": "extracted", "thinking_closed": closed}


# Candidate stdout/stderr never becomes the host's verdict channel. A zero exit
# without the checker's completion record is a failure, including os._exit(0).
_CHECKER = r"""
import json, os, sys
data = json.load(sys.stdin)
report = os.dup(1)
write = os.write
encode = json.dumps
with open(os.devnull, "w") as sink:
    os.dup2(sink.fileno(), 1)
    os.dup2(sink.fileno(), 2)
namespace = {"__name__": "__main__"}
try:
    exec(compile(data["code"], "<candidate>", "exec"), namespace)
    exec(compile(data["tests"], "<tests>", "exec"), namespace)
except BaseException as error:
    result = {"status": "assertion_error" if isinstance(error, AssertionError) else "runtime_error",
              "error": type(error).__name__}
else:
    result = {"status": "passed"}
write(report, encode(result).encode())
"""

_CONTAINER_RUNNER = r"""
import json, subprocess, sys
data = json.load(sys.stdin)
try:
    child = subprocess.run([sys.executable, "-I", "-c", data["checker"]],
        input=json.dumps({"code": data["code"], "tests": data["tests"]}).encode(),
        capture_output=True, timeout=data["timeout"])
    try:
        result = json.loads(child.stdout) if child.returncode == 0 else {"status": "runtime_error"}
    except (ValueError, TypeError):
        result = {"status": "runtime_error"}
    if not isinstance(result, dict) or result.get("status") not in {"passed", "assertion_error", "runtime_error"}:
        result = {"status": "runtime_error"}
except subprocess.TimeoutExpired:
    result = {"status": "timeout"}
result["token"] = data["token"]
print(json.dumps(result))
"""


class JudgeError(RuntimeError):
    """Execution infrastructure failed; do not score this as a wrong answer."""


class DockerJudge:
    def __init__(self, config: CodingConfig):
        self.config = config
        self.executable = shutil.which("docker")
        if self.executable is None:
            raise JudgeError("Docker must be installed before local coding evaluation")
        try:
            result = subprocess.run(  # noqa: S603 - explicit arguments, no shell.
                [
                    self.executable,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    config.docker_image,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise JudgeError(
                "Docker and the configured local Python image must be available before evaluation"
            ) from error
        self.image_id = result.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_id):
            raise JudgeError("Docker did not return an immutable image identity")

    def check(self, code: str, tests: str) -> dict:
        name = "lm-eval-rwkv-coding-" + uuid.uuid4().hex
        token = uuid.uuid4().hex
        memory = str(self.config.memory_mb) + "m"
        command = [
            self.executable,
            "run",
            "--rm",
            "--name",
            name,
            "--pull=never",
            "-i",
            "--network=none",
            "--read-only",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--cpus=1",
            "--memory=" + memory,
            "--memory-swap=" + memory,
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--workdir=/tmp",
            "--entrypoint=python",
            self.image_id,
            "-I",
            "-c",
            _CONTAINER_RUNNER,
        ]
        payload = json.dumps(
            {
                "code": code,
                "tests": tests,
                "checker": _CHECKER,
                "timeout": self.config.test_timeout,
                "token": token,
            }
        )
        try:
            result = subprocess.run(  # noqa: S603 - restricted container, no shell.
                command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout + 15,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise JudgeError(
                "Docker did not complete within its wall-clock budget"
            ) from error
        except OSError as error:
            raise JudgeError("Docker could not start") from error
        finally:
            # Only this attempt's unique container can be removed.
            try:
                subprocess.run(  # noqa: S603 - remove only this generated container.
                    [self.executable, "rm", "-f", name],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        if result.returncode in {137, -9}:
            return {"status": "resource_limit", "passed": False}
        if result.returncode != 0:
            raise JudgeError(f"Docker failed with exit status {result.returncode}")
        try:
            verdict = json.loads(result.stdout)
        except ValueError as error:
            raise JudgeError("Docker returned no valid completion record") from error
        if (
            not isinstance(verdict, dict)
            or verdict.pop("token", None) != token
            or verdict.get("status")
            not in {"passed", "assertion_error", "runtime_error", "timeout"}
        ):
            raise JudgeError("Docker returned an invalid verdict")
        return {**verdict, "passed": verdict["status"] == "passed"}


def _atomic(path: Path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = isinstance(value, bytes)
    with tempfile.NamedTemporaryFile(
        "wb" if binary else "w",
        dir=path.parent,
        **({} if binary else {"encoding": "utf-8"}),
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            if binary:
                stream.write(value)
            elif jsonl:
                for row in value:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _fresh_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            "Output directory is not empty; use --resume for incomplete generation, or a separate directory for replay"
        )
    path.mkdir(parents=True, exist_ok=True)


def _model(config):
    if config.model["model"] == "rwkv7-remote":
        from lm_eval.models.rwkv_remote import RWKV7Remote

        cls = RWKV7Remote
    else:
        from lm_eval.models.rwkv_remote_chat import RWKV7RemoteChat

        cls = RWKV7RemoteChat
    return cls(**config.model["model_args"])


def generate(config, problems, output, *, resume=False):
    """Use the project's SQLite checkpoint and stable per-repeat item keys."""
    model = _model(config)
    identity = config.identity()
    fingerprint = _hash(identity)
    runtime_fp = model_fingerprint(model)
    manifest = {
        **identity,
        "generation_fingerprint": fingerprint,
        "runtime_model_fingerprint": runtime_fp,
        "model_info": model.get_model_info(),
        "extractor": EXTRACTOR,
    }
    if resume:
        previous = json.loads((output / "manifest.json").read_text())
        if previous != manifest:
            raise ValueError(
                "Resume identity changed; keep the old artifacts and use a separate output directory"
            )
    with_store = CheckpointStore(str(output / "run.sqlite3"), _hash(manifest))
    model.set_checkpoint_store(with_store)
    prompts, keys, requests = {}, [], []
    export_ready = False
    try:
        _atomic(output / "manifest.json", manifest)
        for doc_id, problem in enumerate(problems):
            prompt = model.apply_chat_template(
                [{"role": "user", "content": problem.prompt}]
            )
            prompts[doc_id] = (
                prompt.prompt if hasattr(prompt, "prompt") else str(prompt)
            )
            for ordinal in range(config.num_samples):
                request = Instance(
                    "generate_until",
                    {},
                    (
                        prompt,
                        {
                            "until": [],
                            "max_gen_toks": config.model["model_args"]["max_gen_toks"],
                        },
                    ),
                    0,
                    (TASK, doc_id, config.num_samples),
                )
                item_id = compute_item_id(TASK, doc_id, 0, "generate_until", ordinal)
                request_fp = compute_request_fingerprint(
                    item_id, request.args, fingerprint, runtime_fp
                )
                keys.append(
                    ItemKey(
                        item_id,
                        request_fp,
                        compute_idempotency_key(item_id, request_fp),
                        TASK,
                        doc_id,
                        0,
                        "generate_until",
                        ordinal,
                    )
                )
                requests.append(request)
        with_store.register_pending_items(keys)
        existing = with_store.get_items(TASK)
        if any(
            existing[key.item_id]["request_fingerprint"] != key.request_fingerprint
            for key in keys
        ):
            raise ValueError(
                "Rendered coding prompts changed; use a separate output directory"
            )
        export_ready = True
        pending = [
            (req, key)
            for req, key in zip(requests, keys, strict=True)
            if existing[key.item_id]["status"] != "completed"
        ]
        if pending:
            # Keep one semaphore busy; each response is checkpointed separately.
            reqs, item_keys = zip(*pending, strict=True)
            responses = model.generate_until(list(reqs), item_keys=list(item_keys))
            with_store.mark_items_completed(
                [
                    (key.item_id, serialize_response(value))
                    for key, value in zip(item_keys, responses, strict=True)
                ]
            )
    finally:
        items = with_store.get_items(TASK)
        rows = []
        for key in keys:
            item = items.get(key.item_id)
            if item is None or item["status"] != "completed":
                continue
            value = deserialize_response(item["result_json"])
            rows.append(
                {
                    "protocol": PROTOCOL,
                    "generation_fingerprint": fingerprint,
                    "runtime_model_fingerprint": runtime_fp,
                    "task_id": problems[key.doc_id].task_id,
                    "repeat_ord": key.repeat_ord,
                    "seed": config.model["model_args"]["seed"] + key.repeat_ord,
                    "prompt": prompts[key.doc_id],
                    "raw_completion": str(value),
                    "finish_reason": getattr(value, "finish_reason", None),
                    "raw_response": getattr(value, "raw_response", None),
                }
            )
        try:
            if export_ready:
                _atomic(output / "samples.jsonl", rows, jsonl=True)
        finally:
            with_store.close()
            model.set_checkpoint_store(None)
    return rows


def read_samples(path, config, problems):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected = {(p.task_id, r) for p in problems for r in range(config.num_samples)}
    seen, runtimes = set(), set()
    fingerprint = _hash(config.identity())
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("task_id"), str)
            or type(row.get("repeat_ord")) is not int
            or type(row.get("seed")) is not int
        ):
            raise ValueError("Invalid sample task ID, repeat ordinal, or seed")
        key = (row.get("task_id"), row.get("repeat_ord"))
        if key not in expected or key in seen or isinstance(key[1], bool):
            raise ValueError("Sample file contains unknown or duplicate attempts")
        if (
            row.get("protocol") != PROTOCOL
            or row.get("generation_fingerprint") != fingerprint
        ):
            raise ValueError(
                "Sample identity differs from the configured generation and problem file"
            )
        if row.get("seed") != config.model["model_args"]["seed"] + key[
            1
        ] or not isinstance(row.get("raw_completion"), str):
            raise ValueError("Invalid sample seed or raw completion")
        runtime = row.get("runtime_model_fingerprint")
        if not isinstance(runtime, str) or not runtime:
            raise ValueError("Missing runtime model identity")
        runtimes.add(runtime)
        seen.add(key)
    if seen != expected or len(runtimes) != 1 or None in runtimes:
        raise ValueError(
            "Every problem must have exactly the declared repeats from one model identity"
        )
    return sorted(rows, key=lambda row: (row["task_id"], row["repeat_ord"]))


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(
            "pass@k requires 0 <= correct <= samples and 1 <= k <= samples"
        )
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def check_controls(problems, output, judge):
    controls = []
    # A broken environment must not turn reference implementations into low
    # model scores. Negative controls also detect tests that never call code.
    previous = output / "controls.jsonl"
    if previous.exists():
        previous.rename(output / f"controls.previous.{uuid.uuid4().hex}.jsonl")
    for problem in problems:
        positive = judge.check(problem.reference, problem.tests)
        negative = judge.check(
            f"def {problem.entry_point}(*args, **kwargs):\n    raise AssertionError('negative control')\n",
            problem.tests,
        )
        controls.append(
            {"task_id": problem.task_id, "positive": positive, "negative": negative}
        )
        _atomic(output / "controls.jsonl", controls, jsonl=True)
        if not positive["passed"] or negative["status"] != "assertion_error":
            raise JudgeError(
                f"Reference/negative control failed for {problem.task_id}; no model score was produced"
            )
    return controls


def score(config, problems, rows, output, judge, *, controls=None):
    if controls is None:
        check_controls(problems, output, judge)
    by_id = {problem.task_id: problem for problem in problems}

    def grade(row):
        problem = by_id[row["task_id"]]
        extracted = extract_code(
            row["raw_completion"], problem, thinking=config.thinking
        )
        verdict = (
            judge.check(extracted["code"], problem.tests)
            if extracted["code"] is not None
            else {"status": extracted["status"], "passed": False}
        )
        return {
            **row,
            "extraction": extracted,
            "verdict": verdict,
            "truncated": str(row.get("finish_reason") or "").lower()
            in _LENGTH_FINISHES,
        }

    reviews = []
    try:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(grade, row) for row in rows]
            for future in as_completed(futures):
                reviews.append(future.result())
    finally:
        reviews.sort(key=lambda row: (row["task_id"], row["repeat_ord"]))
        previous = output / "reviews.jsonl"
        if previous.exists():
            previous.rename(output / f"reviews.previous.{uuid.uuid4().hex}.jsonl")
        _atomic(previous, reviews, jsonl=True)
    metrics = {f"pass@{k}": 0.0 for k in config.pass_k}
    total_correct = 0
    for problem in problems:
        correct = sum(
            row["verdict"]["passed"]
            for row in reviews
            if row["task_id"] == problem.task_id
        )
        total_correct += correct
        for k in config.pass_k:
            metrics[f"pass@{k}"] += pass_at_k(config.num_samples, correct, k) / len(
                problems
            )
    metrics[f"avg@{config.num_samples}"] = total_correct / len(rows)
    result = {
        "protocol": PROTOCOL,
        "extractor": EXTRACTOR,
        "benchmark_name": TASK,
        "model_name": config.model["model_args"]["model"],
        "weight_revision": config.model["metadata"]["weight_revision"],
        "cot_mode": config.model["model_args"]["rwkv_generation_prompt"],
        "prompt_template": config.model["model_args"]["rwkv_prompt_template"],
        "generation": config.identity(),
        "generation_fingerprint": _hash(config.identity()),
        "runtime_model_fingerprint": rows[0]["runtime_model_fingerprint"],
        "scoring_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "docker_image_id": judge.image_id,
        "test_timeout": config.test_timeout,
        "memory_mb": config.memory_mb,
        "n_problems": len(problems),
        "n_completions": len(rows),
        "n_passed": total_correct,
        "metrics": metrics,
        "truncation_rate": sum(row["truncated"] for row in reviews) / len(rows),
        "unfinished_thinking_rate": sum(
            row["extraction"]["status"] == "unfinished_thinking" for row in reviews
        )
        / len(rows),
        "outcomes": dict(Counter(row["verdict"]["status"] for row in reviews)),
    }
    _atomic(output / "results.json", result)
    return result


def run_coding(
    config: CodingConfig, *, phase="run", samples=None, output_dir=None, resume=False
):
    output = (
        Path(output_dir).expanduser().resolve() if output_dir else config.output_dir
    )
    problems = load_problems(config.problems)
    if phase not in {"run", "generate", "score"}:
        raise ValueError("phase must be run, generate, or score")
    source_identity = _hash(config.identity())
    if phase == "score":
        if samples is None or resume:
            raise ValueError(
                "Offline score requires --samples and a separate output directory; --resume applies to generation"
            )
        rows = read_samples(samples, config, problems)
        _fresh_output(output)
        _atomic(output / "samples.jsonl", rows, jsonl=True)
        _atomic(
            output / "manifest.json",
            {
                **config.identity(),
                "replay_source": str(Path(samples).resolve()),
                "replay_sha256": hashlib.sha256(Path(samples).read_bytes()).hexdigest(),
            },
        )
    else:
        if samples is not None:
            raise ValueError("--samples is only supported with --phase score")
        if resume:
            if (
                not (output / "run.sqlite3").is_file()
                or (output / "results.json").exists()
            ):
                raise ValueError(
                    "--resume requires an incomplete coding generation directory"
                )
            previous = json.loads((output / "manifest.json").read_text())
            if previous.get("generation_fingerprint") != source_identity:
                raise ValueError(
                    "Resume identity changed; use a separate output directory"
                )
        else:
            _fresh_output(output)

    source_name = (
        "problems.source.jsonl.gz"
        if config.problems.suffix == ".gz"
        else "problems.source.jsonl"
    )
    _atomic(output / source_name, config.problems.read_bytes())
    judge = DockerJudge(config) if phase != "generate" else None
    controls = check_controls(problems, output, judge) if judge is not None else None
    if _hash(config.identity()) != source_identity:
        raise ValueError("Problem file changed during preflight")
    if phase != "score":
        rows = generate(config, problems, output, resume=resume)
        if phase == "generate":
            return {
                "n_completions": len(rows),
                "samples": str(output / "samples.jsonl"),
            }
        rows = read_samples(output / "samples.jsonl", config, problems)
    return score(config, problems, rows, output, judge, controls=controls)
