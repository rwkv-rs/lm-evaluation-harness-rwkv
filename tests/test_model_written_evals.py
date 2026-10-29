from pathlib import Path

import datasets
from datasets import Dataset

from lm_eval.tasks.model_written_evals import utils as model_written_utils
from lm_eval.tasks.model_written_evals.sycophancy.utils import process_docs


def test_sycophancy_normalizes_list_valued_negative_answer():
    list_dataset = Dataset.from_list([{"answer_not_matching_behavior": [" (A)"]}])
    string_dataset = Dataset.from_list([{"answer_not_matching_behavior": " (B)"}])

    normalized_list = process_docs(list_dataset)
    normalized_string = process_docs(string_dataset)

    assert normalized_list[0]["answer_not_matching_behavior"] == " (A)"
    assert normalized_string[0]["answer_not_matching_behavior"] == " (B)"


def test_model_written_loader_uses_materialized_source(monkeypatch, tmp_path):
    url = "https://raw.githubusercontent.com/anthropics/evals/revision/task.jsonl"
    source_path = model_written_utils.local_source_path(url, tmp_path)
    source_path.write_text('{"question":"test"}\n', encoding="utf-8")
    calls = []

    def fake_load_dataset(path, data_files):
        calls.append((path, data_files))
        return {"validation": []}

    monkeypatch.setenv(model_written_utils.DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    result = model_written_utils.load_json_dataset(data_files={"validation": url})

    assert result == {"validation": []}
    assert calls == [("json", {"validation": str(Path(source_path))})]
