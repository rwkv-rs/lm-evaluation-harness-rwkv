def process_results(doc, results):
    """Score the model's free-form "yes"/"no" judgment of plausibility.

    `results[0]` is the `yes_no` filter's output (see
    `lm_eval/filters/extraction.py`) for the single generation: already
    normalized to "yes"/"no", or the filter's fallback sentinel
    ("[invalid]") when no verdict could be extracted. An unparseable
    generation is scored as a wrong/"no" guess rather than excluded, since
    `acc`/`f1` (unlike discrim_eval's bias-diff aggregations) have no
    "skip this doc" convention to fall back on.
    """
    answer = results[0][0] if results and results[0] else None
    predicted = 1 if answer == "yes" else 0
    gold = int(doc["label"])
    return {"acc": int(predicted == gold), "f1": (gold, predicted)}
