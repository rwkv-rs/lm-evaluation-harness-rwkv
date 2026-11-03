import evaluate as hf_evaluate


compute_ = None


def _get_code_eval_metric():
    global compute_
    if compute_ is None:
        compute_ = hf_evaluate.load("code_eval")
    return compute_


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = _get_code_eval_metric().compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + r + doc["suffix"] for r in resp]
        for resp, doc in zip(resps, docs)
    ]
