from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import lm_eval.loggers.scoreboard as MODULE


def comparison_policy() -> dict:
    model = {
        "label": "RWKV7 G1I 1.5B",
        "architecture": "RWKV7",
        "generation": "G1I",
        "parameters": "1.5B",
    }
    return {
        "outcome_metric": "acc",
        "response_mode": "single",
        "comparison": {
            "model": model,
            "benchmark": {
                "label": "RACE",
                "categories": [
                    {"id": "reasoning", "label": "Reasoning"},
                ],
                "evaluation_method": "multiple_choice",
                "score_multiplier": 100.0,
            },
            "evaluation": {
                "prompt_profile": "fake_think",
                "prompt_template": "User: {task.problem}\n\nAssistant: <think></think>",
                "precision": "fp32io16",
            },
            "coordinates": [
                {
                    "comparison": {
                        "id": "precision",
                        "label": "fp16 vs fp32io16",
                        "short_label": "precision",
                        "a_label": "fp16",
                        "b_label": "fp32io16",
                        "contract": "Same checkpoint and prompt; only WKV precision changes.",
                    },
                    "parameter_group": {
                        "id": "1.5b",
                        "label": "1.5B",
                        "a_model": model,
                        "b_model": model,
                        "parameter_delta_percent": 0.0,
                        "comparable": True,
                    },
                    "arm": "b",
                }
            ],
        },
    }


def publication_config(*, history_only: bool = False) -> dict:
    policy = {"history_only": True} if history_only else comparison_policy()
    return {
        "base_url": "https://eval.rwkv.rs/test",
        "token_env": "TEST_SCOREBOARD_TOKEN",
        "model_sha256": "e" * 64,
        "model_revision": "local",
        "retries": 0,
        "tasks": {"race": policy},
    }


def native_results(*, complete: bool = True) -> tuple[dict, dict]:
    task_name = "race"
    results = {
        "model_name": "rwkv7-g1i-1.5b-20260805-ctx16384",
        "config": {
            "model": "rwkv7-http",
            "model_args": {
                "model": "rwkv7-g1i-1.5b-20260805-ctx16384",
                "rwkv_prompt_template": "assistant",
                "rwkv_generation_prompt": "fake_think",
                "rwkv_sampling_mode": "profile",
                "wkv_mode": "fp32io16",
                "num_concurrent": 4,
                "max_length": 1024,
            },
            "sampling_config": {
                "temperature": 1.0,
                "top_p": 0.28,
                "top_k": 32,
                "rwkv_sampling_mode": "profile",
                "rwkv_prompt_template": "assistant",
                "rwkv_generation_prompt": "fake_think",
            },
            "batch_size": "1",
        },
        "configs": {
            task_name: {
                "task": task_name,
                "dataset_path": "EleutherAI/race",
                "dataset_name": "high",
                "test_split": "test",
                "output_type": "generate_until",
                "generation_kwargs": {"max_gen_toks": 16},
                "metric_list": [{"metric": "acc", "aggregation": "mean"}],
                "metadata": {
                    "version": 1,
                    "benchmark_name": "race",
                    "languages": ["english"],
                    "tags": ["reading-comprehension"],
                },
            }
        },
        "results": {
            task_name: {
                "sample_len": 2,
                "acc,none": 0.5,
                "acc_stderr,none": 0.5,
            }
        },
        "n-samples": {task_name: {"original": 2, "effective": 2}},
        "lm_eval_version": "0.4.13.dev0",
        "backend_version": "0.23.1",
        "backend_commit": "f" * 40,
        "torch_version": "2.13.0+cu130",
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
    }
    samples = []
    for index, (answer, acc) in enumerate((("A", 1.0), ("B", 0.0))):
        completion = f"Reasoning for question {index}. Final Answer: [{answer}]"
        evidence = {
            "prompt": f"Question {index}",
            "input_token_ids": [1, index + 2],
            "output_token_ids": [10 + index],
            "raw_response": {"choices": [{"text": completion}]},
            "post_processed_answer": completion,
            "finish_reason": "length" if index else "stop",
            "truncation": bool(index),
        }
        if not complete and index == 0:
            evidence.pop("output_token_ids")
        samples.append(
            {
                "doc_id": index,
                "doc": {"question": f"Question {index}"},
                "target": "A",
                "metrics": ["acc"],
                "acc": acc,
                "resps": [[completion]],
                "filtered_resps": [answer],
                "response_evidence": [[evidence]],
            }
        )
    return results, {task_name: samples}


def test_builds_complete_comparison_and_answers() -> None:
    results, samples = native_results()
    campaign, tasks = MODULE.build_publication(
        results, samples, publication=publication_config()
    )

    assert campaign["source"] == "lm-eval-harness"
    assert campaign["contract_sha256"]
    task = tasks[0]
    assert task["primary_metric"] == "acc"
    assert task["metrics"] == {"acc": 0.5, "acc_stderr": 0.5}
    assert task["comparison"]["samples"] == 2
    assert task["comparison"]["truncation_rate"] == 0.5
    assert task["sampling_config"] == {
        "temperature": 1.0,
        "top_p": 0.28,
        "top_k": 32,
        "rwkv_sampling_mode": "profile",
        "rwkv_prompt_template": "assistant",
        "rwkv_generation_prompt": "fake_think",
        "max_tokens": 16,
        "batch_size": "1",
        "max_length": 1024,
        "num_concurrent": 4,
    }
    assert [sample["answer"]["outcome"] for sample in task["samples"]] == [
        "correct",
        "incorrect",
    ]
    assert task["samples"][1]["answer"]["fail_reason"] == "truncated"
    assert task["samples"][0]["answer"]["extracted_answer"] == "A"
    assert task["samples"][0]["answer"]["raw_completion"].startswith("Reasoning")
    assert all(
        sample["model_response"]["evidence_complete"] for sample in task["samples"]
    )


def test_missing_policy_fails_closed() -> None:
    results, samples = native_results()
    config = publication_config()
    config["tasks"] = {}
    with pytest.raises(MODULE.ScoreboardError, match="publication.tasks"):
        MODULE.build_publication(
            results,
            samples,
            publication=config,
        )


def test_history_only_requires_explicit_policy() -> None:
    results, samples = native_results()
    _, tasks = MODULE.build_publication(
        results, samples, publication=publication_config(history_only=True)
    )
    assert "comparison" not in tasks[0]
    assert all("answer" not in sample for sample in tasks[0]["samples"])


def test_incomplete_token_evidence_is_rejected() -> None:
    results, samples = native_results(complete=False)
    with pytest.raises(MODULE.ScoreboardError, match="integer token IDs"):
        MODULE.build_publication(results, samples, publication=publication_config())


def test_writes_stable_publication_files(tmp_path: Path) -> None:
    results, samples = native_results()
    campaign, tasks = MODULE.build_publication(
        results, samples, publication=publication_config()
    )
    campaign_path, task_paths = MODULE.write_publication(tmp_path, campaign, tasks)
    assert campaign_path == tmp_path / "publication/campaign.json"
    assert task_paths == [tmp_path / "publication/tasks/race.json"]
    assert json.loads(task_paths[0].read_text())["comparison"]["samples"] == 2


class FakeResponse:
    status_code = 200

    def __init__(self, value: dict):
        self.value = value
        self.text = json.dumps(value)

    def json(self):
        return self.value


def test_publish_uses_gzip_idempotency_and_resume(monkeypatch) -> None:
    results, samples = native_results()
    campaign, tasks = MODULE.build_publication(
        results, samples, publication=publication_config()
    )
    campaign_id = "00000000-0000-0000-0000-000000000001"
    acknowledged: dict[str, str] = {}
    requests = []

    def fake_request(method, url, *, data, headers, timeout):
        payload = json.loads(gzip.decompress(data)) if data else None
        path = urlsplit(url).path
        requests.append(
            (method, path, headers.get("Idempotency-Key"), payload, timeout)
        )
        if path.endswith("publication-preflight"):
            return FakeResponse(
                {
                    "status": "ready",
                    "schema_version": "scoreboard-v1",
                    "sources": ["lm-eval-harness"],
                }
            )
        if path.endswith("evaluation-campaigns"):
            return FakeResponse(
                {
                    "campaign_id": campaign_id,
                    "action": "created",
                    "status": "incomplete",
                    "expected_task_count": 1,
                    "task_hashes": {},
                }
            )
        if path.endswith(campaign_id):
            return FakeResponse(
                {
                    "campaign_id": campaign_id,
                    "status": "incomplete",
                    "expected_task_count": 1,
                    "task_hashes": acknowledged,
                    "missing_tasks": [],
                }
            )
        if "/tasks/" in path:
            digest = MODULE.content_digest(payload)
            acknowledged[payload["task"]["identity"]] = digest
            return FakeResponse(
                {
                    "evaluation_id": "evaluation",
                    "task_identity": payload["task"]["identity"],
                    "content_sha256": digest,
                    "action": "created",
                }
            )
        if path.endswith("finalize"):
            return FakeResponse(
                {"campaign_id": campaign_id, "status": "complete", "task_count": 1}
            )
        raise AssertionError(path)

    monkeypatch.setenv("TEST_SCOREBOARD_TOKEN", "secret")
    monkeypatch.setattr(MODULE.requests, "request", fake_request)
    client = MODULE.ScoreboardClient(
        MODULE.PublicationConfig.from_mapping(publication_config())
    )
    receipt = MODULE.publish(client, campaign, tasks)
    assert receipt["finalize"]["status"] == "complete"
    assert requests[0][1] == "/test/api/v1/evaluation-publication-preflight"
    assert requests[1][2] == f"campaign:{campaign['run_key']}"
    assert any(request[2].startswith("publish:") for request in requests if request[2])

    requests.clear()
    receipt = MODULE.publish(client, campaign, tasks)
    assert receipt["tasks"][0]["action"] == "unchanged"
    assert not any("/tasks/" in request[1] for request in requests)


def test_publication_failure_preserves_raw_results(tmp_path: Path) -> None:
    results, samples = native_results()
    status = MODULE.publish_lm_eval_evaluation(
        results,
        samples,
        output_dir=tmp_path,
        publication={
            "base_url": "https://eval.rwkv.rs",
            "token_env": "TEST_SCOREBOARD_TOKEN",
            "model_sha256": "e" * 64,
            "tasks": {},
        },
        dry_run=True,
    )
    assert status["evaluation"] == "complete"
    assert status["publication"] == "failed"
    assert Path(status["raw_results_path"]).exists()


def test_transport_failure_does_not_remain_pending(tmp_path: Path, monkeypatch) -> None:
    results, samples = native_results()
    monkeypatch.setenv("TEST_SCOREBOARD_TOKEN", "secret")

    def reject(*_args, **_kwargs):
        raise MODULE.ScoreboardError("remote rejected publication")

    monkeypatch.setattr(MODULE, "publish", reject)
    status = MODULE.publish_lm_eval_evaluation(
        results,
        samples,
        output_dir=tmp_path,
        publication=publication_config(),
    )

    assert status["evaluation"] == "complete"
    assert status["publication"] == "failed"
    assert status["uploaded"] is False
    assert "remote rejected publication" in status["error"]
