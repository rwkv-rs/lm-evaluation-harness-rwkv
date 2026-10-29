_LETTERS = ["A", "B", "C", "D"]


def process_docs(dataset):
    def _detokenize(text):
        text = text.replace(" '", "'")
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" n't", "n't")
        text = text.replace("`` ", '"')
        text = text.replace("''", '"')
        # punctuation
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        return text

    def _process(doc):
        article = _detokenize(doc["article"])
        options = [_detokenize(option) for option in doc["options"]]
        # Bake the lettered option menu directly into the text handed to
        # doc_to_text, so the chat template wraps this exact, complete
        # question exactly once. Building the menu after the chat template
        # had already been applied (as the old multiple_choice path did,
        # via rwkv_choice.build_choice_prompt inside the model's
        # loglikelihood() override) is what produced malformed, doubly- or
        # triply-templated prompts for tasks like discrim_eval -- see the
        # same fix applied there.
        menu = "\n".join(
            f"{letter}. {option}" for letter, option in zip(_LETTERS, options)
        )
        prompt = (
            f"{article}\n\nWhich of the following best completes the passage below?\n\n"
            f"{menu}\n\nAnswer with only the letter of the correct option."
        )
        return {"article": article, "options": options, "prompt": prompt}

    return dataset.map(_process)


def process_results(doc, results):
    """Score the model's single extracted answer letter against gold.

    `results[0]` is the `albatross_mcq` filter's output (see
    `lm_eval/tasks/mutual/mutual.yaml` and
    `lm_eval/filters/extraction.py`) for the one generation: already
    normalized to "A"/"B"/"C"/"D", or the filter's fallback sentinel when
    no letter could be extracted (which then simply never matches `gold`).

    Unlike the previous multiple_choice scoring, this only yields a single
    chosen letter per doc rather than a relative ranking over all four
    options, so `r@2` (top-2 recall) and `mrr` can no longer be computed
    meaningfully and have been dropped; `acc` (did it name the correct
    letter) is the direct generalization of the old `r@1`.
    """
    predicted = results[0][0] if results and results[0] else None
    return {"acc": int(predicted == doc["answers"])}
