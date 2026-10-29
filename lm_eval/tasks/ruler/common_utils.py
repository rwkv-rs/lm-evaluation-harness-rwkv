import logging
import re
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Union

import requests
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]


class RemoteRulerTokenizer:
    def __init__(self, base_url: str, model: str, timeout: float = 30):
        self.base_url = base_url.removesuffix("/v1/completions").rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def encode(self, text: str) -> list[int]:
        response = self.session.post(
            f"{self.base_url}/tokenize",
            json={
                "model": self.model,
                "prompt": text,
                "add_special_tokens": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise RuntimeError("Malformed response from remote /tokenize endpoint.")
        return tokens[1:] if tokens and tokens[0] == 0 else tokens

    def __call__(self, text: str, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))


@cache
def get_tokenizer(
    tokenizer=None,
    pretrained=None,
    tokenizer_base_url=None,
    tokenizer_model=None,
    model=None,
    **kwargs,
) -> Union[
    RemoteRulerTokenizer,
    "transformers.PreTrainedTokenizer",
    "transformers.PreTrainedTokenizerFast",
]:
    if tokenizer_base_url:
        remote_model = tokenizer_model or model
        assert remote_model, "No model provided for the remote tokenizer."
        eval_logger.info(
            "Using remote tokenizer %s for synthetic tasks.", tokenizer_base_url
        )
        return RemoteRulerTokenizer(tokenizer_base_url, remote_model)
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
