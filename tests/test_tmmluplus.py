from pathlib import Path

from lm_eval.tasks._yaml_loader import load_yaml


def test_tmmluplus_csv_choices_remain_strings(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_text(
        "question,A,B,C,D,answer\ntrain question,1,2,3,4,A\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "question,A,B,C,D,answer\ntest question,4.5,2,3,4,A\n",
        encoding="utf-8",
    )

    config = load_yaml(
        "lm_eval/tasks/tmmluplus/default/"
        "tmmluplus_statistics_and_machine_learning.yaml"
    )
    dataset = config["custom_dataset"](
        data_files={"train": str(train_path), "test": str(test_path)}
    )
    processed = config["process_docs"](dataset["test"])

    assert all(feature.dtype == "string" for feature in dataset["test"].features.values())
    assert processed[0]["choices"] == ["4.5", "2", "3", "4"]
    assert processed[0]["goal"] == 0
