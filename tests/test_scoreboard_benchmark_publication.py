import json
import random
from pathlib import Path
from urllib.error import HTTPError

import pytest

from lm_eval import evaluator
from lm_eval.api.benchmark import AvgAtKPlan, BenchmarkSpec
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.loggers.scoreboard import (
    MAX_SAMPLES_PER_OUTCOME,
    PublicationConfig,
    ScoreboardError,
    ScoreboardPublisher,
    _detail,
    _messages,
    _rwkv_model,
    _weight_ctx_len,
    build_score,
)
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


_TEST_DATA = [
    {"question": "What is 2+2?", "answer": "lol"},
    {"question": "What is 3+3?", "answer": "not-lol"},
    {"question": "What is 4+4?", "answer": "not-lol"},
    {"question": "What is 5+5?", "answer": "not-lol"},
]

_LEAF_TEMPLATE = """\
task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data_path}
output_type: generate_until
doc_to_text: "{{{{question}}}}"
doc_to_target: answer
test_split: test
num_fewshot: 0
generation_kwargs:
  until: []
  max_gen_toks: 32
metric_list:
  - metric: {metric}
    aggregation: mean
    higher_is_better: true
metadata:
  field: test-field
  benchmark_name: {task_name}
"""

_MODEL_NAME = "rwkv7-g1j-1.5b"
# Scoreboard records the *weight's* context window as part of the model
# identity, so run output lives under a directory named after the weight
# release (see `_weight_ctx_len`). Keep this fixture dir conventions-complete.
_RUN_DIR_NAME = "rwkv7-g1j-1.5b-20260101-ctx4096-remote-unit"
_MODEL_ARGS = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "penalty_decay": 0.0,
    "seed": 42,
}


def _write_data_and_leaves(fixtures_dir: Path, leaves: list[tuple[str, str]]) -> Path:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    data_path = fixtures_dir / "test_data.json"
    data_path.write_text(json.dumps(_TEST_DATA), encoding="utf-8")
    for task_name, metric in leaves:
        (fixtures_dir / f"{task_name}.yaml").write_text(
            _LEAF_TEMPLATE.format(
                task_name=task_name, data_path=str(data_path), metric=metric
            ),
            encoding="utf-8",
        )
    return data_path


def _publish_config(tasks: dict[str, dict] | None = None) -> PublicationConfig:
    return PublicationConfig(
        base_url="https://scoreboard.example",
        token_env="SCOREBOARD_PUBLICATION_TOKEN",  # noqa: S106 - env var name, not a secret
        cot_mode="NoCoT",
        tasks=tasks or {},
        timeout=60,
        control_timeout=30,
        retries=0,
        retry_delay=0,
        rerun_reason=None,
        dry_run=True,
    )


def _callback_context() -> dict:
    return {
        "config": {
            "model": _MODEL_NAME,
            "model_args": dict(_MODEL_ARGS),
            "max_length": 4096,
        },
        "lm_eval_version": "test",
    }


def _run_benchmark(
    fixtures_dir: Path,
    group_yaml: str,
    benchmark_name: str,
    strategy: str,
    tmp_path: Path,
    seed: int = 0,
) -> dict:
    """Runs a real `evaluate()` pass and returns the aggregated `results.json`
    document for the benchmark, exactly as `ScoreboardTaskCallback` reads it
    in production.
    """
    (fixtures_dir / "group.yaml").write_text(group_yaml, encoding="utf-8")
    task_manager = TaskManager(include_path=str(fixtures_dir), include_defaults=False)
    task_dict = task_manager.load([benchmark_name])

    random.seed(seed)
    lm = DummyLM()
    evaluation_tracker = EvaluationTracker(output_path=str(tmp_path / _RUN_DIR_NAME))

    captured: dict = {}

    def callback(task_name, artifact_paths):
        captured["task_name"] = task_name
        captured["artifact_paths"] = artifact_paths

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        task_callback=callback,
        task_callback_context=_callback_context(),
        benchmark_map={
            benchmark_name: BenchmarkSpec(
                name=benchmark_name,
                field="reasoning",
                strategy=strategy,
                leaf_task_names=(benchmark_name,) if strategy == "standalone" else None,
            )
        },
    )

    return json.loads(Path(captured["artifact_paths"]["results"]).read_text("utf-8"))


def test_standalone_benchmark_produces_score_with_real_metric(tmp_path):
    fixtures_dir = tmp_path / "fixtures_standalone"
    _write_data_and_leaves(fixtures_dir, [("solo_leaf", "exact_match")])
    group_yaml = "group: unused\ntask:\n  - solo_leaf\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "solo_leaf", "standalone", tmp_path
    )

    config = _publish_config({"solo_leaf": {"field": "reasoning", "avg_k": 1}})
    benchmark_info, score = build_score(results_doc, config)

    assert benchmark_info == {
        "name": "solo_leaf",
        "field": "reasoning",
        "num_samples": 4,
        "avg_k": 1,
    }
    assert score["model"] == {
        "arch_version": "rwkv7",
        "data_version": "G1j",
        "param_size": "1.5b",
        "ctx_len": 4096,
    }
    assert score["cot_mode"] == "NoCoT"
    assert score["score"] == 0.25
    assert len(score["passed_details"]) == 1
    assert len(score["wrong_details"]) == 3
    assert score["passed_details"][0]["is_passed"] is True
    assert score["passed_details"][0]["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "lol"},
    ]


def test_group_fallback_merges_leaves_into_one_score(tmp_path):
    fixtures_dir = tmp_path / "fixtures_shared"
    _write_data_and_leaves(
        fixtures_dir, [("shared_leaf_a", "exact_match"), ("shared_leaf_b", "exact_match")]
    )
    group_yaml = "group: bench_shared\ntask:\n  - shared_leaf_a\n  - shared_leaf_b\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "bench_shared", "group_fallback", tmp_path
    )

    config = _publish_config({"bench_shared": {"field": "reasoning", "avg_k": 1}})
    benchmark_info, score = build_score(results_doc, config)

    assert benchmark_info["num_samples"] == 8
    assert len(score["passed_details"]) == 2
    assert len(score["wrong_details"]) == 6


def test_saved_benchmark_field_does_not_require_upload_policy_field(tmp_path):
    fixtures_dir = tmp_path / "fixtures_missing_field"
    _write_data_and_leaves(fixtures_dir, [("no_field_leaf", "exact_match")])
    group_yaml = "group: unused\ntask:\n  - no_field_leaf\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "no_field_leaf", "standalone", tmp_path
    )

    config = _publish_config({"no_field_leaf": {"avg_k": 1}})
    benchmark_info, _score = build_score(results_doc, config)

    assert benchmark_info["field"] == "reasoning"


def test_upload_policy_cannot_override_saved_benchmark_field(tmp_path):
    fixtures_dir = tmp_path / "fixtures_mismatched_field"
    _write_data_and_leaves(fixtures_dir, [("mismatched_field_leaf", "exact_match")])
    group_yaml = "group: unused\ntask:\n  - mismatched_field_leaf\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "mismatched_field_leaf", "standalone", tmp_path
    )

    config = _publish_config({"mismatched_field_leaf": {"field": "coding"}})
    with pytest.raises(ScoreboardError, match="saved benchmark.field"):
        build_score(results_doc, config)


def test_upload_requires_field_saved_by_evaluation(tmp_path):
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {"leaf": _leaf_artifacts(run_dir, "leaf", docs=1, k=1)}
    results_doc = _benchmark_doc(run_dir, leaves)
    results_doc.pop("benchmark")

    with pytest.raises(ScoreboardError, match="no saved benchmark.field"):
        build_score(results_doc, _publish_config())


def test_avg_k_missing_from_policy_raises(tmp_path):
    """avg_k is derived from the artifacts (completions per sample), so a
    policy that omits it must still publish; a policy that *states* a
    different value than the run used must fail instead of labelling the
    score with the wrong sampling.
    """
    fixtures_dir = tmp_path / "fixtures_derived_avg_k"
    _write_data_and_leaves(fixtures_dir, [("no_avg_k_leaf", "exact_match")])
    group_yaml = "group: unused\ntask:\n  - no_avg_k_leaf\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "no_avg_k_leaf", "standalone", tmp_path
    )

    benchmark_info, _score = build_score(
        results_doc, _publish_config({"no_avg_k_leaf": {"field": "reasoning"}})
    )
    assert benchmark_info["avg_k"] == 1
    assert benchmark_info["num_samples"] == 4

    config = _publish_config({"no_avg_k_leaf": {"field": "reasoning", "avg_k": 8}})
    with pytest.raises(ScoreboardError, match="avg_k"):
        build_score(results_doc, config)


def test_publication_config_validates_cot_mode():
    with pytest.raises(ScoreboardError, match="cot_mode"):
        PublicationConfig.from_mapping(
            {"base_url": "https://scoreboard.example", "cot_mode": "not-a-mode"}
        )


def test_rwkv_model_parses_name_with_explicit_ctx_len():
    assert _rwkv_model("rwkv7-g1j-1.5b", 4096) == {
        "arch_version": "rwkv7",
        "data_version": "G1j",
        "param_size": "1.5b",
        "ctx_len": 4096,
    }
    assert _rwkv_model("rwkv7-g1i-1.5b-20260805-ctx16384", 16384) == {
        "arch_version": "rwkv7",
        "data_version": "G1i",
        "param_size": "1.5b",
        "ctx_len": 16384,
    }


def test_weight_ctx_len_reads_the_weight_revision_not_the_eval_budget():
    """ctx_len identifies the weight, so it never comes from max_length.

    Evaluating one weight at a shorter harness `max_length` must not fork it
    into a second model on the board, so the revision is read from
    ``model_name`` when present and otherwise from the run directory, which is
    named after the served weight.
    """
    assert (
        _weight_ctx_len(
            "rwkv7-g1k-13.3b-20260930-ctx25600",
            "results/rwkv7-g1k-13.3b-20260930-ctx25600-remote-x/task_artifacts/a/results.json",
        )
        == 25600
    )
    # model_name without a revision: fall back to the artifact path.
    assert (
        _weight_ctx_len(
            "rwkv7-g1k-13.3b",
            "results/rwkv7-g1k-13.3b-20260930-ctx25600-remote-x/task_artifacts/a/results.json",
        )
        == 25600
    )
    # A revision in the name wins over a directory that disagrees.
    assert (
        _weight_ctx_len(
            "rwkv7-g1k-13.3b-20260930-ctx25600", "results/stale-ctx8192-remote/x/a.json"
        )
        == 25600
    )


def test_weight_ctx_len_refuses_unidentifiable_weight():
    with pytest.raises(ScoreboardError, match="ctx<number>"):
        _weight_ctx_len("rwkv7-g1j-1.5b", "output/a/results.json")


def test_weight_ctx_len_prefers_weight_revision_over_model_name_and_paths():
    """weight_revision exists because model_name doubles as the literal API
    model= value -- a serving alias like "rwkv7-g1j-1.5b" that the backend
    must recognize, so it can't carry a release filename stem. weight_revision
    carries that stem instead, and must win over both model_name and the
    artifact-path fallback when all three disagree.
    """
    assert (
        _weight_ctx_len(
            "rwkv7-g1j-1.5b",
            "results/rwkv7-g1j-1.5b-remote-x/task_artifacts/a/results.json",
            weight_revision="rwkv7-g1j-1.5b-20260831-ctx16384",
        )
        == 16384
    )
    # No weight_revision given: falls back to today's model_name/path search.
    assert (
        _weight_ctx_len(
            "rwkv7-g1j-1.5b",
            "results/rwkv7-g1j-1.5b-20260831-ctx16384-remote-x/task_artifacts/a/results.json",
            weight_revision=None,
        )
        == 16384
    )


def test_build_score_reads_ctx_from_metadata_weight_revision(tmp_path):
    """A run whose model_name is a bare serving alias still publishes.

    model_name is the literal API ``model=`` value the inference backend must
    recognize, so it cannot carry a release filename stem, and a run directory
    need not be named after the weight either. ``metadata.weight_revision`` is
    then the only source of the weight's ctx_len; without it, build_score used
    to raise ScoreboardError and the live callback downgraded that to a
    warning, silently publishing nothing.
    """
    run_dir = tmp_path / "rwkv7-g1j-1.5b-remote-unit"
    leaves = {"leaf_a": _leaf_artifacts(run_dir, "leaf_a", docs=10, k=1, plain=0.5)}
    doc = _benchmark_doc(run_dir, leaves)
    doc["config"] = {
        "model": _MODEL_NAME,
        "model_args": dict(_MODEL_ARGS),
        "metadata": {"weight_revision": "rwkv7-g1j-1.5b-20260831-ctx16384"},
    }

    _, score = build_score(doc, _publish_config({"bench": {"field": "reasoning"}}))

    assert score["model"] == {
        "arch_version": "rwkv7",
        "data_version": "G1j",
        "param_size": "1.5b",
        "ctx_len": 16384,
    }


def test_rwkv_model_rejects_unknown_shape():
    with pytest.raises(ScoreboardError):
        _rwkv_model("not-a-known-model-name", 4096)


def test_messages_parses_json_chat_str_on_disk_shape():
    arguments = {
        "gen_args_0": {
            "arg_0": ['[{"role": "user", "content": "hi"}]'],
            "arg_1": {"until": []},
        }
    }
    assert _messages(arguments) == [{"role": "user", "content": "hi"}]


def test_messages_falls_back_to_plain_string_prompt():
    arguments = {"gen_args_0": {"arg_0": "plain prompt text", "arg_1": {}}}
    assert _messages(arguments) == [{"role": "user", "content": "plain prompt text"}]


def _generative_row(
    *,
    answer: str,
    target: str,
    evidence_text: str | None,
    filtered_answer: str | None = None,
) -> dict:
    # `filtered_resps` is one list of filtered strings (one per repeat) per
    # document request -- [[answer]] mirrors the real on-disk shape for a
    # single-repeat generate_until task (see samples_<task>_*.jsonl), not a
    # flat [answer]. `filtered_answer` defaults to `answer` so existing
    # callers that don't care about filtering behave as before.
    return {
        "arguments": {"gen_args_0": {"arg_0": "the prompt", "arg_1": {}}},
        "filtered_resps": [[filtered_answer if filtered_answer is not None else answer]],
        "resps": [[answer]],
        "target": target,
        "response_evidence": (
            [{"text": evidence_text, "finish_reason": "stop"}]
            if evidence_text is not None
            else []
        ),
        "exact_match": 1.0 if answer == target else 0.0,
    }


def test_detail_buckets_passed_wrong_and_failed():
    passed_detail, passed_outcome, _ = _detail(
        _generative_row(answer="4", target="4", evidence_text="4"), "exact_match"
    )
    assert passed_outcome == "passed"
    assert passed_detail["is_passed"] is True
    assert passed_detail["messages"] == [
        {"role": "user", "content": "the prompt"},
        {"role": "assistant", "content": "4"},
    ]

    wrong_detail, wrong_outcome, _ = _detail(
        _generative_row(answer="5", target="4", evidence_text="5"), "exact_match"
    )
    assert wrong_outcome == "wrong"
    assert wrong_detail["is_passed"] is False
    assert wrong_detail["messages"][-1] == {"role": "assistant", "content": "5"}

    failed_detail, failed_outcome, _ = _detail(
        _generative_row(answer="", target="4", evidence_text=""), "exact_match"
    )
    assert failed_outcome == "failed"
    assert failed_detail["is_passed"] is False
    assert failed_detail["messages"][-1] == {"role": "assistant", "content": ""}


def test_detail_answer_uses_filtered_resps_not_raw_completion():
    """A `filter_list` (e.g. `yes_no`/`albatross_mcq`) exists to reduce a
    long chain-of-thought completion down to the clean value actually used
    for scoring -- `answer` must show that distilled value, not the raw
    completion the filter exists to cut through (the bug: Scoreboard was
    showing the entire multi-KB reasoning dump as "the answer" for every
    `generate_until` task that uses a filter).
    """
    row = _generative_row(
        answer="C",
        target="C",
        evidence_text="<think>lots of reasoning...</think>C. the full option text",
        filtered_answer="C",
    )
    detail, _outcome, _truncated = _detail(row, "exact_match")
    assert detail["answer"] == "C"
    assert detail["messages"][-1]["content"].startswith("<think>")


def test_detail_answer_falls_back_to_dash_when_filter_could_not_extract():
    """The `yes_no`/`albatross_mcq` filters' fallback sentinel
    ("[invalid]") means no answer could be extracted at all -- that must
    show as "-", not the literal sentinel string nor the raw completion.
    """
    row = _generative_row(
        answer="whatever",
        target="C",
        evidence_text="rambling with no clear final answer",
        filtered_answer="[invalid]",
    )
    detail, _outcome, _truncated = _detail(row, "exact_match")
    assert detail["answer"] == "-"


def test_detail_tuple_valued_metric_with_usable_answer_passes():
    """Tasks like discrim_eval score each doc with an aggregation-input
    tuple (`(demographics, bias_name, chose_yes)`), not a binary 0/1 --
    there's no "wrong answer" concept there (any yes/no is legitimate), so
    a doc with a real extracted answer must be "passed", not blanket
    "failed" just because the metric value isn't a plain scalar.
    """
    row = _generative_row(
        answer="yes",
        target="yes",
        evidence_text="<think>reasoning</think>yes",
        filtered_answer="yes",
    )
    row["exact_match"] = [["black", "female", 30.0, 1], "black_bias", 1]
    detail, outcome, _truncated = _detail(row, "exact_match")
    assert outcome == "passed"
    assert detail["is_passed"] is True
    assert detail["answer"] == "yes"


def test_detail_tuple_valued_metric_with_unparseable_answer_fails():
    """Same tuple-valued metric shape, but the `yes_no` filter's fallback
    sentinel means no answer could be extracted at all -- that is a real
    failure, unlike every other doc that did get a usable answer.
    """
    row = _generative_row(
        answer="rambling with no verdict",
        target="yes",
        evidence_text="rambling with no verdict",
        filtered_answer="[invalid]",
    )
    row["exact_match"] = [["black", "female", 30.0, 1], "black_bias", None]
    detail, outcome, _truncated = _detail(row, "exact_match")
    assert outcome == "failed"
    assert detail["is_passed"] is False
    assert detail["answer"] == "-"


def test_detail_answer_falls_back_to_raw_completion_without_filtered_resps():
    """A task with no `filter_list` at all (just the implicit `take_first`
    pass-through) has no meaningfully different filtered value -- the raw
    completion is already the genuine answer there.
    """
    row = _generative_row(answer="4", target="4", evidence_text="4")
    del row["filtered_resps"]
    detail, _outcome, _truncated = _detail(row, "exact_match")
    assert detail["answer"] == "4"


def test_samples_cap_is_shared_across_leaves_of_one_benchmark(tmp_path):
    """A benchmark's per-outcome cap must apply across ALL of its leaves
    combined, not be re-applied independently per leaf. Otherwise a
    benchmark with N sub-tasks uploads up to N * MAX_SAMPLES_PER_OUTCOME
    "passed" samples instead of MAX_SAMPLES_PER_OUTCOME total.
    """
    from lm_eval.loggers.scoreboard import _details

    num_leaves = 7
    docs_per_leaf = MAX_SAMPLES_PER_OUTCOME
    shared_uploaded_counts: dict[str, int] = dict.fromkeys(
        ("passed", "wrong", "failed"), 0
    )
    total_uploaded = 0
    for leaf_index in range(num_leaves):
        rows = [
            _generative_row(answer="4", target="4", evidence_text="4")
            for _ in range(docs_per_leaf)
        ]
        samples_path = tmp_path / f"leaf_{leaf_index}.jsonl"
        samples_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        evidence = _details(samples_path, "exact_match", shared_uploaded_counts)
        total_uploaded += len(evidence.buckets["passed"])
        assert evidence.docs == docs_per_leaf

    assert total_uploaded == MAX_SAMPLES_PER_OUTCOME
    assert shared_uploaded_counts["passed"] == MAX_SAMPLES_PER_OUTCOME


def _http_error(
    code: str, score_id: int = 4, *, status: int = 409, detail: list | None = None
) -> HTTPError:
    import io

    payload = {"detail": detail if detail is not None else {"code": code, "score_id": score_id}}
    body = json.dumps(payload).encode()
    error = HTTPError(
        url="https://scoreboard.example/api/upload",
        code=status,
        msg="Conflict" if status == 409 else "Unprocessable Entity",
        hdrs=None,
        fp=io.BytesIO(body),
    )
    return error


def _publisher(monkeypatch, rerun_reason: str | None) -> ScoreboardPublisher:
    import dataclasses

    monkeypatch.setenv("SCOREBOARD_PUBLICATION_TOKEN", "test-token")
    config = dataclasses.replace(
        _publish_config(), rerun_reason=rerun_reason, dry_run=False
    )
    return ScoreboardPublisher(config)


def test_upload_duplicate_score_without_rerun_reason_is_idempotent_replay(
    monkeypatch,
):
    """Without --rerun-reason, a duplicate_score conflict is a harmless
    no-op replay of the existing score -- no expire, no retry.
    """
    publisher = _publisher(monkeypatch, rerun_reason=None)
    monkeypatch.setattr(
        "lm_eval.loggers.scoreboard.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error("duplicate_score")),
    )

    receipt = publisher._upload({"benchmark_id": 1})

    assert receipt["replayed"] is True
    assert receipt["score_id"] == 4


def test_upload_duplicate_score_with_rerun_reason_expires_and_reuploads(
    monkeypatch,
):
    """With --rerun-reason set, a duplicate_score conflict must NOT be
    silently replayed: the stale score is expired and the new payload
    (e.g. containing a bugfix that doesn't change the numeric score) is
    genuinely re-uploaded, not swallowed.
    """
    publisher = _publisher(monkeypatch, rerun_reason="fixed messages bug")
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if request.full_url.endswith("/api/upload") and len(calls) == 1:
            raise _http_error("duplicate_score")
        if request.full_url.endswith("/api/expire"):
            return _fake_response({"success": True})
        return _fake_response({"success": True, "score_id": 99, "message": "ok"})

    monkeypatch.setattr("lm_eval.loggers.scoreboard.urlopen", fake_urlopen)

    receipt = publisher._upload({"benchmark_id": 1})

    assert calls == [
        "https://scoreboard.example/api/upload",
        "https://scoreboard.example/api/expire",
        "https://scoreboard.example/api/upload",
    ]
    assert receipt.get("replayed") is not True
    assert receipt["score_id"] == 99


def test_upload_retries_with_benchmark_name_for_an_older_server(monkeypatch):
    """A scoreboard revision that predates `Score.benchmark_id` keys an upload
    by the benchmark's name and rejects a payload without one.

    The publisher registers the benchmark by id, so it has to notice that
    rejection and re-send the same score with the name attached -- otherwise
    every run against such a deployment fails to publish.
    """
    publisher = _publisher(monkeypatch, rerun_reason=None)
    payloads = []

    def fake_urlopen(request, timeout=None):
        payloads.append(json.loads(request.data.decode()))
        if len(payloads) == 1:
            raise _http_error(
                "missing",
                status=422,
                detail=[
                    {
                        "type": "missing",
                        "loc": ["body", "benchmark_name"],
                        "msg": "Field required",
                    }
                ],
            )
        return _fake_response({"success": True, "score_id": 77})

    monkeypatch.setattr("lm_eval.loggers.scoreboard.urlopen", fake_urlopen)

    receipt = publisher._upload({"benchmark_id": 1}, benchmark_name="mmlu")

    assert receipt["score_id"] == 77
    assert "benchmark_name" not in payloads[0]
    assert payloads[1]["benchmark_name"] == "mmlu"
    assert payloads[1]["benchmark_id"] == 1


def _fake_response(payload: dict):
    import io
    from contextlib import contextmanager

    @contextmanager
    def cm():
        yield io.BytesIO(json.dumps(payload).encode())

    return cm()


def _choice_row(index: int, score: float = 1.0) -> dict:
    return {
        "doc_id": index,
        "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
        "resps": [["A"]],
        "filtered_resps": [["A"]],
        "target": "A",
        "filter": "none",
        "response_evidence": [{"text": "A", "finish_reason": "stop"}],
        "acc": score,
    }


def _leaf_artifacts(
    run_dir: Path,
    leaf_name: str,
    *,
    docs: int,
    k: int,
    avg_at_k: float | None = None,
    plain: float = 0.25,
    samples_rows: int | None = None,
    recorded_counts: bool = True,
    repeats: int | None = None,
) -> dict[str, str]:
    """Write one leaf's results.json + samples jsonl and return its paths.

    `docs` is how many docs the leaf evaluated, `k` how many samples each was
    scored with, `samples_rows` how many per-doc lines its samples file holds
    (defaults to `docs`), and `avg_at_k=None` writes no ``<metric>_avg@k``
    average at all -- the shape of an artifact whose avg@k coverage was
    incomplete, e.g. one resumed from a checkpoint.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = docs if samples_rows is None else samples_rows
    samples_path = run_dir / f"samples_{leaf_name}.jsonl"
    samples_path.write_text(
        "\n".join(json.dumps(_choice_row(index)) for index in range(rows)) + "\n",
        encoding="utf-8",
    )

    metrics: dict = {"acc,none": plain}
    if avg_at_k is not None:
        metrics[f"acc_avg@{k},none"] = avg_at_k
    if recorded_counts:
        metrics |= {
            "n_samples": docs,
            "n_completions": docs * max(k, 1),
            "n_truncated": docs // 10,
            "truncation_rate": 0.1,
        }
    results_path = run_dir / f"{leaf_name}_results.json"
    results_path.write_text(
        json.dumps(
            {
                "results": {
                    leaf_name: {
                        "name": leaf_name,
                        "alias": leaf_name,
                        "sample_len": docs,
                        **metrics,
                    }
                },
                "configs": {
                    leaf_name: {
                        "metric_list": [
                            {
                                "metric": "acc",
                                "aggregation": "mean",
                                "higher_is_better": True,
                            }
                        ],
                        "repeats": max(k, 1) if repeats is None else repeats,
                    }
                },
                "n-samples": {leaf_name: {"original": docs, "effective": docs}},
                "config": {"model": _MODEL_NAME, "model_args": dict(_MODEL_ARGS)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "results": str(results_path),
        "samples": str(samples_path),
        "metadata": str(run_dir / f"{leaf_name}_metadata.json"),
    }


def _benchmark_doc(run_dir: Path, leaves: dict[str, dict[str, str]], **extra) -> dict:
    return {
        "benchmark": {"name": "bench", "field": "reasoning"},
        "results": {"bench": {"name": "bench", "alias": "bench", "sample_len": 1}},
        "config": {"model": _MODEL_NAME, "model_args": dict(_MODEL_ARGS)},
        "n-samples": {"bench": {"original": 1, "effective": 1}},
        "strategy": "tag_fallback",
        "leaf_artifact_paths": leaves,
        **extra,
    }


def test_published_counts_and_score_match_what_the_run_actually_did(tmp_path):
    """num_samples/avg_k/score are read off the run, not hand-written.

    The benchmark-level doc deliberately claims ``sample_len: 1`` (the old
    code's source for the published sample count, and the shape a metric's
    value list takes when only part of a benchmark reports it): the upload
    must instead report the docs each leaf evaluated and the per-sample
    average each leaf actually carries.
    """
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {
        "leaf_small": _leaf_artifacts(
            run_dir, "leaf_small", docs=100, k=2, avg_at_k=0.75, plain=0.5
        ),
        "leaf_large": _leaf_artifacts(
            run_dir, "leaf_large", docs=300, k=8, avg_at_k=0.6, plain=0.25
        ),
    }

    benchmark_info, score = build_score(
        _benchmark_doc(run_dir, leaves),
        _publish_config({"bench": {"field": "reasoning"}}),
    )

    assert benchmark_info == {
        "name": "bench",
        "field": "reasoning",
        "num_samples": 400,
        # The leaves ran at different k (2 and 8), averaging 6.5 completions
        # per doc; the board's avg_k is an integer label, so it publishes the
        # rate rounded to whole samples per doc.
        "avg_k": pytest.approx(7),
    }
    # Completion-weighted blend of the two leaves' avg@k averages -- never
    # the plain, single-sample `acc` they also report.
    assert score["score"] == pytest.approx((100 * 0.75 + 300 * 0.6) / 400)
    assert score["truncation_rate"] == pytest.approx(0.1)


def test_pre_fix_artifact_reports_docs_from_its_sample_evidence(tmp_path):
    """Artifacts written before the run recorded its own counts still must
    not publish a metric's value-list length as the benchmark's size: the
    samples file has one line per evaluated doc, and the k of the average
    that is actually present is read off the ``<metric>_avg@k`` key.
    """
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {
        "leaf_partial": _leaf_artifacts(
            run_dir,
            "leaf_partial",
            docs=720,
            k=8,
            avg_at_k=0.47,
            plain=0.74,
            samples_rows=720,
            recorded_counts=False,
        )
    }

    benchmark_info, score = build_score(
        _benchmark_doc(run_dir, leaves),
        _publish_config({"bench": {"field": "reasoning"}}),
    )

    assert benchmark_info["num_samples"] == 720
    assert benchmark_info["avg_k"] == 8
    assert score["score"] == pytest.approx(0.47)


def test_publication_uses_actual_profile_over_task_sampling_metadata():
    from lm_eval.loggers.scoreboard import _sampling_config

    results = {"config": {"model_args": {}, "sampling_config": dict(_MODEL_ARGS)}}
    sampling = _sampling_config(
        results,
        {"generation_kwargs": {"temperature": 99, "max_gen_toks": 16}},
        {},
        {"temperature": 42},
    )
    assert sampling["temp"] == 0.7
    assert sampling["max_generated_tokens"] == 16


@pytest.mark.parametrize("name", ["rwkv7-x1j-1.5b", "rwkv7-g2j-1.5b"])
def test_rwkv_model_rejects_incorrect_data_version_prefix(name):
    with pytest.raises(ScoreboardError, match="data_version"):
        _rwkv_model(name, 4096)


@pytest.mark.parametrize("failure", ["timeout", "503"])
def test_upload_retries_transient_failures(monkeypatch, failure):
    import dataclasses
    import io

    publisher = _publisher(monkeypatch, rerun_reason=None)
    publisher.config = dataclasses.replace(publisher.config, retries=1)
    calls = []

    def send(request, timeout=None):
        calls.append(request)
        if len(calls) == 1:
            if failure == "timeout":
                raise TimeoutError("temporary timeout")
            raise HTTPError(
                request.full_url,
                503,
                "Unavailable",
                None,
                io.BytesIO(b"<html>unavailable</html>"),
            )
        return _fake_response({"success": True, "score_id": 22})

    monkeypatch.setattr("lm_eval.loggers.scoreboard.urlopen", send)
    assert publisher._upload({"benchmark_id": 1})["score_id"] == 22
    assert len(calls) == 2


def test_upload_conflict_after_expire_is_bounded(monkeypatch):
    publisher = _publisher(monkeypatch, rerun_reason="new run")
    expirations = []

    def send(request, timeout=None):
        if request.full_url.endswith("/api/expire"):
            expirations.append(request)
            return _fake_response({"success": True})
        raise _http_error("active_score_exists")

    monkeypatch.setattr("lm_eval.loggers.scoreboard.urlopen", send)
    with pytest.raises(ScoreboardError, match="409"):
        publisher._upload({"benchmark_id": 1})
    assert len(expirations) == 1


def test_publisher_preserves_benchmarks_macro_average(tmp_path):
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {"small": _leaf_artifacts(run_dir, "small", docs=1, k=2, avg_at_k=1),
              "large": _leaf_artifacts(run_dir, "large", docs=9, k=2, avg_at_k=0)}
    document = _benchmark_doc(run_dir, leaves, averaged_results={"acc,none": 0.5}, primary_metric_hint="acc")
    info, score = build_score(document, _publish_config({"bench": {"field": "reasoning"}}))
    assert score["score"] == 0.5
    assert info["num_samples"] == 10


def test_sample_details_follow_the_selected_metric_filter(tmp_path):
    from lm_eval.loggers.scoreboard import _details

    path = tmp_path / "samples.jsonl"
    first = {**_choice_row(0, 0), "filter": "wrong_filter"}
    second = {**_choice_row(0, 1), "filter": "selected_filter"}
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    evidence = _details(path, "acc", dict.fromkeys(("passed", "wrong", "failed"), 0), "selected_filter")
    assert evidence.docs == 1
    assert len(evidence.buckets["passed"]) == 1
    assert evidence.buckets["wrong"] == []


def test_historical_remote_results_cannot_silently_publish_under_new_parser(tmp_path):
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {"leaf": _leaf_artifacts(run_dir, "leaf", docs=1, k=1)}
    document = _benchmark_doc(run_dir, leaves)
    document["config"]["model"] = "rwkv7-remote"
    with pytest.raises(ScoreboardError, match="versioned raw-answer replay"):
        build_score(document, _publish_config({"bench": {"field": "reasoning"}}))


@pytest.mark.parametrize("output_type", ["generate_until", "multiple_choice"])
def test_rwkv_remote_wire_to_scoreboard_with_open_thinking(tmp_path, monkeypatch, output_type):
    from types import SimpleNamespace
    from unittest.mock import Mock

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        bos_token_id = 0
        eos_token = "<eos>"
        chat_template = "{% for m in messages %}{{m.role}}: {{m.content}}\n\n{% endfor %}{% if add_generation_prompt %}Assistant: <think{% endif %}"

        def __call__(self, values, **kwargs):
            return SimpleNamespace(input_ids=[[ord(char) for char in value] for value in values])

    fixtures = tmp_path / "fixtures"
    data = _write_data_and_leaves(fixtures, [("remote_leaf", "exact_match")])
    if output_type == "multiple_choice":
        data.write_text(json.dumps([{"question": "Which pet?", "choices": ["cats", "dogs"], "answer": 1}]))
        path = fixtures / "remote_leaf.yaml"
        path.write_text(path.read_text().replace("output_type: generate_until", "output_type: multiple_choice\ndoc_to_choice: choices").replace("metric: exact_match", "metric: acc"))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setenv("RWKV_API_KEY", "unit-test")
    def two_rollouts(tasks, _groups, _benchmark_map, *, limited=False):
        assert not limited
        return {
            name: AvgAtKPlan(
                benchmark=name,
                leaves=(name,),
                num_docs=len(task.eval_docs),
                k=2,
                metric_name="avg@2",
            )
            for name, task in tasks.items()
        }

    monkeypatch.setattr(evaluator, "plan_avg_at_k", two_rollouts)
    sent = []

    def post(url, *, json, **kwargs):
        sent.append(json)
        assert kwargs["timeout"] == 300
        assert "generation_config" not in json
        assert json["top_p"] == 0.76
        assert json["top_k"] == 32
        prompt = "".join(chr(token) for token in json["prompt"][0])
        assert prompt.endswith("Assistant: <think")
        if output_type == "multiple_choice":
            assert prompt.index("B. dogs") < prompt.index("Assistant: <think")
            answer = "B" if json["seed"] == 40 else "A"
        else:
            answer = "lol"
        response = Mock(ok=True)
        response.json.return_value = {"choices": [{"index": 0, "text": f">tentative reasoning</think>{answer}", "finish_reason": "stop"}]}
        return response

    monkeypatch.setattr("lm_eval.models.api_models.requests.post", post)
    captured = {}

    def callback(task_name, artifact_paths):
        document = json.loads(Path(artifact_paths["results"]).read_text())
        captured["publication"] = build_score(document, _publish_config({"remote_leaf": {"field": "reasoning"}}))
        return {"publication": "validated", "uploaded": False}

    evaluator.simple_evaluate(
        model="rwkv7-remote", model_args={"model": "rwkv7-g1j-1.5b", "tokenizer": "unit-tokenizer",
            "base_url": "http://localhost:8000/v1/completions", "rwkv_generation_prompt": "open_think",
            "max_length": 4096, "seed": 40},
        tasks=["remote_leaf"], task_manager=TaskManager(include_path=str(fixtures), include_defaults=False),
        apply_chat_template=True, log_samples=True, bootstrap_iters=0,
        metadata={"weight_revision": "rwkv7-g1j-1.5b-ctx4096"},
        evaluation_tracker=EvaluationTracker(output_path=str(tmp_path / "run")), task_callback=callback,
        benchmark_map={
            "remote_leaf": BenchmarkSpec(
                name="remote_leaf",
                field="reasoning",
                strategy="standalone",
                leaf_task_names=("remote_leaf",),
            )
        },
    )
    info, score = captured["publication"]
    assert info["avg_k"] == 2
    assert score["score"] == (0.5 if output_type == "multiple_choice" else 0.25)
    assert sorted({payload["seed"] for payload in sent}) == [40, 41]


def _write_sample_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _hashed_row(index: int, completions: int = 1, answer: str = "4") -> dict:
    """One samples row for a document that carries a content hash.

    `doc_hash` is the document's identity (it is what nesting detection
    compares); `resps[0]` holds one entry per sample the run took for the
    document.
    """
    return _generative_row(answer=answer, target=answer, evidence_text=answer) | {
        "doc_id": index,
        "doc_hash": f"doc-{index}",
        "resps": [[answer] * completions],
    }


def test_parent_leaf_documents_are_not_counted_twice(tmp_path):
    """A group that also scores its own subsets publishes the subsets only.

    `winogender_all` scores the same documents as its female/male/neutral
    filters, and `winogender_gotcha` the same documents as its `_female` /
    `_male` filters. Counting the parent as well would double every shared
    document and would weight the parent's own average by its full document
    count -- the lighteval producer skips exactly this case (`all` rows are
    dropped before it blends a benchmark's subtasks).
    """
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {
        "bench_all": _leaf_artifacts(
            run_dir, "bench_all", docs=4, k=1, plain=0.0, recorded_counts=False
        ),
        "bench_left": _leaf_artifacts(
            run_dir, "bench_left", docs=2, k=1, plain=0.5, recorded_counts=False
        ),
        "bench_right": _leaf_artifacts(
            run_dir, "bench_right", docs=2, k=1, plain=1.0, recorded_counts=False
        ),
    }
    # One document pool: the parent scores all four documents, each child one
    # disjoint half of them.
    for leaf_name, doc_indices in (
        ("bench_all", range(4)),
        ("bench_left", range(2)),
        ("bench_right", range(2, 4)),
    ):
        _write_sample_rows(
            Path(leaves[leaf_name]["samples"]),
            [_hashed_row(index) for index in doc_indices],
        )

    benchmark_info, score = build_score(
        _benchmark_doc(run_dir, leaves),
        _publish_config({"bench": {"field": "reasoning"}}),
    )

    assert benchmark_info["num_samples"] == 4
    assert benchmark_info["avg_k"] == 1
    assert score["score"] == pytest.approx((2 * 0.5 + 2 * 1.0) / 4)


def test_avg_k_falls_back_to_the_completions_the_samples_hold(tmp_path):
    """An artifact that records neither its own completion count nor a
    ``<metric>_avg@k`` average (race and wmdp_chem before the avg@k coverage
    fix) still shows how many samples per doc it took: the published avg_k is
    read off the samples instead of defaulting to a single sample.
    """
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {
        "leaf_sampled": _leaf_artifacts(
            run_dir,
            "leaf_sampled",
            docs=3,
            k=4,
            avg_at_k=None,
            plain=0.5,
            recorded_counts=False,
        )
    }
    _write_sample_rows(
        Path(leaves["leaf_sampled"]["samples"]),
        [_hashed_row(index, completions=4) for index in range(3)],
    )

    benchmark_info, score = build_score(
        _benchmark_doc(run_dir, leaves),
        _publish_config({"bench": {"field": "reasoning"}}),
    )

    assert benchmark_info["num_samples"] == 3
    assert benchmark_info["avg_k"] == 4
    assert score["score"] == pytest.approx(0.5)


def test_mixed_leaf_k_publishes_an_integer_avg_k(tmp_path):
    """A benchmark whose leaves were sized by the avg@k budget individually
    has a fractional rate -- wmdp's leaves run 4 / 8 / 2 samples per doc, i.e.
    3.36 on average -- but the board's avg_k is an integer label it renders as
    ``avg@<k>``. Publish that rate rounded to whole samples per doc, not the
    fraction.
    """
    run_dir = tmp_path / _RUN_DIR_NAME
    leaves = {
        "leaf_two": _leaf_artifacts(
            run_dir, "leaf_two", docs=100, k=2, avg_at_k=0.5, plain=0.5
        ),
        "leaf_eight": _leaf_artifacts(
            run_dir, "leaf_eight", docs=30, k=8, avg_at_k=0.5, plain=0.5
        ),
    }

    benchmark_info, _score = build_score(
        _benchmark_doc(run_dir, leaves),
        _publish_config({"bench": {"field": "reasoning"}}),
    )

    assert benchmark_info["num_samples"] == 130
    # (100 * 2 + 30 * 8) / 130 = 3.38 completions per doc
    assert benchmark_info["avg_k"] == 3
