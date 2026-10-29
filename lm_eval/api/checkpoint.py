"""Crash-safe, resumable per-run checkpoint store for evaluation runs.

Scoped, opt-in support for ``lm_eval run --resume`` (see ``lm_eval/_cli/run.py``
and ``lm_eval/config/evaluate_config.py``'s ``[checkpoint]`` TOML table). One
SQLite database file == one Run, by convention (matches "one TOML == one Run").

Three distinct identities are used throughout this module and its callers:

* ``item_id`` -- the *slot* identity of one evaluation item: task + doc_id +
  instance_idx + request_type + repeat_ord. Stable regardless of prompt
  content; this is "have we already produced a result for this exact slot".
* ``request_fingerprint`` -- the *content* identity: what would change if the
  actual request content (args, task config, model config) changed between
  runs even though the slot identity stayed the same.
* ``idempotency_key`` -- derived from a single item, sent as an HTTP header on
  that item's own request for remote retry safety. Never a hash of a whole
  HTTP batch.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional


eval_logger = logging.getLogger(__name__)

# Scheduling/resource knobs. They decide how many requests are in flight, how
# long one request may take, and how often it is retried -- never what is asked
# or how an answer is scored. Checkpoint identity must therefore be invariant
# under them: both the run id (``lm_eval/_cli/run.py``) and every per-item
# request fingerprint drop them. Without that, "raise the concurrency and
# resume" silently invalidates every stored answer and re-runs the benchmark.
EVALUATION_PROTOCOL_VERSION = "rwkv-generative-v2"

OPERATIONAL_MODEL_ARGS: tuple[str, ...] = (
    "num_concurrent",
    "timeout",
    "max_retries",
)


class CheckpointConfigMismatchError(RuntimeError):
    """Raised when an existing checkpoint DB belongs to a different effective config."""


def _stable_repr(obj: Any) -> str:
    """JSON ``default=`` fallback for values ``dump_config()`` can embed
    (functions, filter objects, etc.) that plain ``str()``/``repr()`` would
    render with a per-process memory address (e.g. ``<function f at
    0x7f...>``), making the hash spuriously differ across otherwise-identical
    runs. Use a stable module-qualified name instead wherever possible.
    """
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module is not None and qualname is not None:
        return f"{module}.{qualname}"
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=_stable_repr, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_config_fingerprint(effective_config: dict) -> str:
    """``run_id``: sha256 over the entire effective, resolved evaluation config."""
    return _canonical_hash(effective_config)


def without_operational_args(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy of ``values`` without the scheduling knobs in ``OPERATIONAL_MODEL_ARGS``."""
    if not values:
        return {}
    return {
        key: value for key, value in values.items() if key not in OPERATIONAL_MODEL_ARGS
    }


def compute_task_fingerprint(task_config: dict) -> str:
    """Hash of a single task's resolved config (``task.dump_config()``).

    The dumped config embeds the run's model metadata (``TaskFactory`` merges
    the run metadata into every task), and that metadata carries the scheduling
    knobs in ``OPERATIONAL_MODEL_ARGS``. They are dropped here for the same
    reason they are dropped from the run id: a resume that only changes
    concurrency/timeout/retries must still match its stored answers. Everything
    else -- prompt, filters, metrics, generation kwargs, model identity -- keeps
    invalidating them.
    """
    metadata = task_config.get("metadata")
    payload = without_operational_args(task_config)
    if isinstance(metadata, Mapping):
        payload["metadata"] = without_operational_args(metadata)
    return _canonical_hash(payload)


def compute_item_id(
    task_name: str, doc_id: int, instance_idx: int, request_type: str, repeat_ord: int
) -> str:
    """Slot identity -- independent of prompt content."""
    return _canonical_hash([task_name, doc_id, instance_idx, request_type, repeat_ord])


def compute_request_fingerprint(
    item_id: str, args: Any, task_fingerprint: str, model_fingerprint: str
) -> str:
    """Content identity -- what the actual LM request looks like for this item."""
    return _canonical_hash([item_id, args, task_fingerprint, model_fingerprint])


def compute_idempotency_key(item_id: str, request_fingerprint: str) -> str:
    """Per single logical request -- never a hash of a whole HTTP batch."""
    return _canonical_hash(["idempotency", item_id, request_fingerprint])[:32]


def model_fingerprint(lm: Any) -> str:
    info = {}
    get_model_info = getattr(lm, "get_model_info", None)
    if callable(get_model_info):
        try:
            info = dict(get_model_info())
        except Exception:  # noqa: BLE001
            info = {}
    info.setdefault("model", getattr(lm, "model", type(lm).__name__))
    info.setdefault("base_url", getattr(lm, "base_url", None))
    if getattr(lm, "EVALUATION_PROTOCOL_VERSION", None):
        info["evaluation_protocol"] = lm.EVALUATION_PROTOCOL_VERSION
        info["prompt_identity"] = getattr(lm, "tokenizer_name", "")
        for name in ("rwkv_prompt_template", "rwkv_generation_prompt", "enable_thinking", "_seed", "_max_gen_toks", "max_length"):
            info[name] = getattr(lm, name, None)
    return _canonical_hash(info)


def serialize_response(response: Any) -> str:
    # Checked ahead of the generic `isinstance(response, tuple)` branch below:
    # `RWKVLoglikelihoodResponse` *is* a tuple (so every other consumer keeps
    # unpacking/zipping it like a bare 2-tuple), but a plain `list(response)`
    # would silently drop its raw_response/finish_reason evidence -- the same
    # loss this type exists to prevent -- so it needs its own `kind` to round
    # -trip that evidence across a checkpoint resume.
    if isinstance(response, tuple) and (
        getattr(response, "raw_response", None) is not None
        or getattr(response, "finish_reason", None) is not None
        or getattr(response, "unparsed", False)
        or getattr(response, "rollouts", None) is not None
    ):
        return json.dumps(
            {
                "kind": "rwkv_loglikelihood",
                "value": list(response),
                "prompt": getattr(response, "prompt", None),
                "text": getattr(response, "text", None),
                "raw_response": response.raw_response,
                "finish_reason": response.finish_reason,
                "output_token_ids": getattr(response, "output_token_ids", None),
                "truncated": bool(getattr(response, "truncated", False)),
                "unparsed": bool(getattr(response, "unparsed", False)),
                "parsed_answer": getattr(response, "parsed_answer", None),
                # Per-rollout record of what the doc's k sampled generations
                # actually answered (and whether each hit the output cap).
                # Restoring it is what keeps a resumed run's avg@k scoring
                # and truncation rate covering every doc instead of only the
                # ones re-generated after the resume -- see
                # `RWKVLoglikelihoodResponse.rollouts`.
                "rollouts": getattr(response, "rollouts", None),
            },
            default=str,
            ensure_ascii=False,
        )
    if isinstance(response, tuple):
        return json.dumps({"kind": "tuple", "value": list(response)}, default=str)
    if isinstance(response, str):
        finish_reason = getattr(response, "finish_reason", None)
        raw_response = getattr(response, "raw_response", None)
        if finish_reason is not None or raw_response is not None or hasattr(response, "requires_think_close"):
            return json.dumps(
                {
                    "kind": "rwkv_completion",
                    "text": str(response),
                    "requires_think_close": bool(getattr(response, "requires_think_close", False)),
                    "seed": getattr(response, "seed", None),
                    "finish_reason": finish_reason,
                    "raw_response": raw_response,
                },
                default=str,
                ensure_ascii=False,
            )
        return json.dumps({"kind": "str", "text": str(response)}, ensure_ascii=False)
    return json.dumps({"kind": "raw", "value": response}, default=str, ensure_ascii=False)


def deserialize_response(data: str) -> Any:
    payload = json.loads(data)
    kind = payload.get("kind")
    if kind == "tuple":
        return tuple(payload["value"])
    if kind == "rwkv_loglikelihood":
        from lm_eval.models.rwkv_choice import RWKVLoglikelihoodResponse

        loglikelihood, is_greedy = payload["value"]
        return RWKVLoglikelihoodResponse(
            loglikelihood,
            is_greedy,
            prompt=payload.get("prompt"),
            text=payload.get("text"),
            raw_response=payload.get("raw_response"),
            finish_reason=payload.get("finish_reason"),
            output_token_ids=payload.get("output_token_ids"),
            truncated=bool(payload.get("truncated", False)),
            unparsed=bool(payload.get("unparsed", False)),
            parsed_answer=payload.get("parsed_answer"),
            rollouts=payload.get("rollouts"),
        )
    if kind == "rwkv_completion":
        from lm_eval.models.openai_completions import _RWKVCompletion

        value = _RWKVCompletion(payload["text"], payload.get("finish_reason"), raw_response=payload.get("raw_response"))
        value.requires_think_close = bool(payload.get("requires_think_close", False))
        value.seed = payload.get("seed")
        return value
    if kind == "str":
        return payload["text"]
    return payload.get("value")


@dataclass
class ItemKey:
    """Identity for one logical request, threaded alongside ``cache_keys`` through dispatch."""

    item_id: str
    request_fingerprint: str
    idempotency_key: str
    task_name: str
    doc_id: Any
    instance_idx: int
    request_type: str
    repeat_ord: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    config_fingerprint TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmarks (
    run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    task_fingerprint TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at REAL,
    completed_at REAL,
    artifact_results_path TEXT,
    artifact_metadata_path TEXT,
    publish_status TEXT NOT NULL DEFAULT 'none',
    PRIMARY KEY (run_id, task_name)
);
CREATE TABLE IF NOT EXISTS items (
    run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    item_id TEXT NOT NULL,
    request_fingerprint TEXT,
    idempotency_key TEXT,
    doc_id TEXT,
    instance_idx INTEGER,
    request_type TEXT,
    repeat_ord INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    result_json TEXT,
    error TEXT,
    updated_at REAL,
    PRIMARY KEY (run_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_items_task ON items(run_id, task_name, status);
"""


class CheckpointStore:
    """Crash-safe, resumable per-run checkpoint of benchmark/item completion state.

    One SQLite DB file == one Run, by convention. Not safe to share across
    concurrent processes. All *write* methods commit ``result``/``status``
    together in a single ``BEGIN IMMEDIATE`` transaction -- never as two
    separate writes -- so a crash can never leave a result without a
    completed status (or vice versa).

    The synchronous methods on this class are meant to be called from
    ordinary (non-event-loop) code, exactly like the existing
    ``evaluation_tracker.save_task_artifacts`` synchronous file I/O they sit
    beside. ``mark_items_completed_async`` is the one method meant to be
    awaited from inside a running asyncio event loop (the concurrent HTTP
    dispatch path in ``lm_eval/models/api_models.py``); it offloads the same
    synchronous sqlite3 call to a dedicated executor (not asyncio's shared
    default one -- see ``self._executor``) so it can never block the event
    loop driving concurrent requests, nor queue behind unrelated
    ``asyncio.to_thread`` callers.
    """

    def __init__(self, db_path: str, config_fingerprint: str, fresh: bool = False) -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        # All writes are serialized by `self._lock` anyway, so this doesn't
        # need to be large -- just large enough that the `max_num_seqs`
        # HTTP requests in flight (all potentially calling
        # `mark_items_completed_async` at once) don't queue behind asyncio's
        # small shared default executor (`min(32, cpu_count+4)`), which is
        # also used by unrelated `asyncio.to_thread` callers elsewhere in the
        # process.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="checkpoint-store"
        )
        weakref.finalize(self, self._executor.shutdown, wait=False)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.run_id = config_fingerprint
        self._init_run(config_fingerprint, fresh)

    def _init_run(self, config_fingerprint: str, fresh: bool) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT run_id FROM runs LIMIT 1").fetchone()
                if row is not None:
                    existing_run_id = row[0]
                    if fresh:
                        eval_logger.warning(
                            "checkpoint.fresh=true: discarding existing checkpoint "
                            "state at %s",
                            self.db_path,
                        )
                        self._conn.execute("DELETE FROM items")
                        self._conn.execute("DELETE FROM benchmarks")
                        self._conn.execute("DELETE FROM runs")
                        row = None
                    elif existing_run_id != config_fingerprint:
                        self._conn.execute("ROLLBACK")
                        raise CheckpointConfigMismatchError(
                            f"Checkpoint store at {self.db_path!r} belongs to a "
                            "different effective config (run_id mismatch). Refusing "
                            "to silently reuse it -- pass checkpoint.fresh=true to "
                            "intentionally discard the existing state, or point "
                            "checkpoint.dir elsewhere."
                        )
                now = time.time()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO runs "
                        "(run_id, config_fingerprint, created_at, updated_at, status) "
                        "VALUES (?, ?, ?, ?, 'running')",
                        (config_fingerprint, config_fingerprint, now, now),
                    )
                else:
                    self._conn.execute(
                        "UPDATE runs SET updated_at=? WHERE run_id=?",
                        (now, config_fingerprint),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get_benchmark(self, task_name: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, artifact_results_path, artifact_metadata_path, "
                "publish_status FROM benchmarks WHERE run_id=? AND task_name=?",
                (self.run_id, task_name),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": row[0],
            "artifact_results_path": row[1],
            "artifact_metadata_path": row[2],
            "publish_status": row[3],
        }

    def start_benchmark(self, task_name: str, task_fingerprint: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                "INSERT INTO benchmarks "
                "(run_id, task_name, task_fingerprint, status, started_at, publish_status) "
                "VALUES (?, ?, ?, 'running', ?, 'none') "
                "ON CONFLICT(run_id, task_name) DO UPDATE SET "
                "task_fingerprint=excluded.task_fingerprint",
                (self.run_id, task_name, task_fingerprint, now),
            )
            self._conn.execute("COMMIT")

    def get_items(self, task_name: str) -> dict[str, dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT item_id, request_fingerprint, repeat_ord, status, result_json "
                "FROM items WHERE run_id=? AND task_name=?",
                (self.run_id, task_name),
            ).fetchall()
        return {
            r[0]: {
                "request_fingerprint": r[1],
                "repeat_ord": r[2],
                "status": r[3],
                "result_json": r[4],
            }
            for r in rows
        }

    def count_completed(self, task_name: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM items "
                "WHERE run_id=? AND task_name=? AND status='completed'",
                (self.run_id, task_name),
            ).fetchone()
        return row[0] if row is not None else 0

    def register_pending_items(self, items: list[ItemKey]) -> None:
        if not items:
            return
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.executemany(
                "INSERT OR IGNORE INTO items "
                "(run_id, task_name, item_id, request_fingerprint, idempotency_key, "
                " doc_id, instance_idx, request_type, repeat_ord, status, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                [
                    (
                        self.run_id,
                        it.task_name,
                        it.item_id,
                        it.request_fingerprint,
                        it.idempotency_key,
                        str(it.doc_id),
                        it.instance_idx,
                        it.request_type,
                        it.repeat_ord,
                        now,
                    )
                    for it in items
                ],
            )
            self._conn.execute("COMMIT")

    def mark_items_completed(self, results: list[tuple[str, str]]) -> None:
        """``results``: ``(item_id, result_json)`` pairs.

        ``result_json`` and ``status='completed'`` are set together in a
        single transaction so a crash can never separate them.
        """
        if not results:
            return
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.executemany(
                "UPDATE items SET result_json=?, status='completed', updated_at=? "
                "WHERE run_id=? AND item_id=?",
                [
                    (result_json, now, self.run_id, item_id)
                    for item_id, result_json in results
                ],
            )
            self._conn.execute("COMMIT")

    async def mark_items_completed_async(self, results: list[tuple[str, str]]) -> None:
        """Awaitable wrapper for use inside a running asyncio event loop.

        Offloads the same synchronous sqlite3 write to ``self._executor``
        (not ``asyncio.to_thread``'s shared default pool) so the event loop
        driving concurrent HTTP dispatch is never blocked by disk I/O, and
        this write never queues behind unrelated ``to_thread`` callers.
        """
        await asyncio.wrap_future(
            self._executor.submit(self.mark_items_completed, results)
        )

    def mark_benchmark_completed(
        self,
        task_name: str,
        artifact_results_path: Optional[str],
        artifact_metadata_path: Optional[str],
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                "UPDATE benchmarks SET status='completed', completed_at=?, "
                "artifact_results_path=?, artifact_metadata_path=? "
                "WHERE run_id=? AND task_name=?",
                (now, artifact_results_path, artifact_metadata_path, self.run_id, task_name),
            )
            self._conn.execute("COMMIT")

    def set_publish_status(self, task_name: str, status: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                "UPDATE benchmarks SET publish_status=? WHERE run_id=? AND task_name=?",
                (status, self.run_id, task_name),
            )
            self._conn.execute("COMMIT")
