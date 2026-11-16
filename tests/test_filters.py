import importlib.util
from pathlib import Path

import pytest
import yaml

from lm_eval.filters.extraction import (
    AlbatrossMultiChoiceFilter,
    MultiChoiceRegexFilter,
    extract_albatross_mcq_answer,
)
from lm_eval.filters.transformation import SPANFilter
from lm_eval.tasks.mmlu_prox.lang_libs import LANG_LIBS


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_choice_text():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["alpha"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(A)"]]


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_bare_letter():
    filt = MultiChoiceRegexFilter(regex_pattern=r"()()")

    resps = [[": B"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(B)"]]


def test_albatross_mcq_prefers_last_explicit_answer_after_think():
    response = "<think>answer is A</think>\nAnswer: C\nFinal answer is (J)"

    assert extract_albatross_mcq_answer(response, choice_labels="ABCDEFGHIJ") == "J"


def test_albatross_mcq_accepts_boxed_and_standalone_answers():
    filt = AlbatrossMultiChoiceFilter(choice_labels="ABCDEFGHIJ")

    assert filt.apply([[r"Reasoning. \boxed{\text{F}}", "\n(B)."]], [{}, {}]) == [
        ["F", "B"]
    ]


def test_albatross_mcq_rejects_unmarked_reasoning_letters():
    filt = AlbatrossMultiChoiceFilter(choice_labels="ABCD")

    assert filt.apply([["A may work, but B is stronger."]], [{}]) == [["[invalid]"]]


def test_albatross_mcq_can_require_closed_think():
    assert (
        extract_albatross_mcq_answer(
            "<think>Final answer is C", require_think_close=True
        )
        is None
    )


def test_albatross_mcq_honors_localized_task_regex():
    filt = AlbatrossMultiChoiceFilter(
        choice_labels="ABCDEFGHIJ",
        regex_pattern=r"Die antwoord is \(?([ABCDEFGHIJ])\)?",
    )

    assert filt.apply([["Die antwoord is (H)"], ["irrelevant"]], [{}, {}]) == [
        ["H"],
        ["[invalid]"],
    ]


def test_albatross_mcq_prefers_localized_final_answer():
    filt = AlbatrossMultiChoiceFilter(
        choice_labels="ABCDEFGHIJ",
        regex_pattern=r"Die antwoord is \(?([ABCDEFGHIJ])\)?",
    )

    response = "The answer is A in the quoted example. Die antwoord is (H)."
    assert filt.apply([[response]], [{}]) == [["H"]]


def test_albatross_mcq_localized_regex_respects_required_think_close():
    filt = AlbatrossMultiChoiceFilter(
        choice_labels="ABCDEFGHIJ",
        require_think_close=True,
        regex_pattern=r"Die antwoord is \(?([ABCDEFGHIJ])\)?",
    )

    assert filt.apply([["Die antwoord is (H)"]], [{}]) == [["[invalid]"]]


def test_albatross_mcq_extracts_all_mmlu_prox_localized_suffixes():
    task_root = Path(__file__).parents[1] / "lm_eval/tasks/mmlu_prox"

    for language, language_prompts in LANG_LIBS.items():
        template = task_root / language / f"_{language}_template_yaml"
        pattern_scalar = next(
            line.split(":", 1)[1].strip()
            for line in template.read_text(encoding="utf-8").splitlines()
            if "regex_pattern:" in line
        )
        filt = AlbatrossMultiChoiceFilter(
            choice_labels="ABCDEFGHIJ",
            regex_pattern=yaml.safe_load(pattern_scalar),
        )

        assert filt.apply([[language_prompts[5].format("J")]], [{}]) == [["J"]]


@pytest.mark.parametrize("text, expected", [
    ("My final choice is (B)", "B"),
    ("The correct one is C.", "C"),
    ("The answer is A. On reflection, my final choice is (B)", "B"),
    ("The answer is B.\nD", "D"),
    ("The answer is B or C.", "[invalid]"),
    ("<think>Final answer is C</think>", "[invalid]"),
])
def test_generation_and_multiple_choice_share_final_answer_rules(text, expected):
    assert AlbatrossMultiChoiceFilter().apply([[text]], [{}]) == [[expected]]


@pytest.mark.parametrize("raw, expected", [
    ("<think>wrong answer</think>\n\nright answer", "right answer"),
    ("<think>right answer but unfinished", ""),
    (">still thinking", ""),
    ("<think>plan</think>\n\ndef solve():\n    return 1\n", "def solve():\n    return 1\n"),
])
def test_default_generation_metrics_only_receive_final_answers(raw, expected):
    from lm_eval.filters.extraction import FinalAnswerFilter
    result = FinalAnswerFilter(require_think_close=True).apply([[raw]], [{}])
    assert result == [[expected]]


def test_final_answer_stage_retains_thinking_attestation_for_strict_mcq_filter():
    from lm_eval.filters.extraction import FinalAnswerFilter
    raw = "<think>The answer is A</think>My final choice is (B)"
    final = FinalAnswerFilter(True).apply([[raw]], [{}])
    assert AlbatrossMultiChoiceFilter(require_think_close=True).apply(final, [{}]) == [["B"]]


def test_format_span_normalizes_label_only():
    # Labels are normalized, but entity text containing label-words as
    # substrings (e.g. "Company", "Country", "George") must be left intact.
    filt = SPANFilter()
    resps = [["ORGANIZATION: Shell Company $ LOCATION: Country Club $ PERSON: George"]]

    assert filt.apply(resps, [{}]) == [
        ["org: shell company $ loc: country club $ per: george"]
    ]


def test_multi_choice_regex_prefix_choice_does_not_shadow_longer_choice():
    # When one choice's text is a prefix of another, naming the longer choice in the
    # response must map to the longer choice's letter. Regression: the fallback regex
    # joined choices in list order, and leftmost-alternation let the shorter prefix
    # ("Guilty") shadow "Guilty of Romance", returning (A) instead of (B).
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["the answer is Guilty of Romance"]]
    docs = [{"choices": ["Guilty", "Guilty of Romance"]}]

    assert filt.apply(resps, docs) == [["(B)"]]


def test_multi_choice_regex_prefix_fix_holds_under_task_config():
    # Every task that uses this filter passes group_select=-1 and a "(\\([A-Z]\\))"
    # pattern, not the defaults; the shorter choice named alone must still win.
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
    )
    docs = [{"choices": ["Guilty", "Guilty of Romance"]}]

    assert filt.apply([["the answer is Guilty of Romance"]], docs) == [["(B)"]]
    assert filt.apply([["the answer is Guilty"]], docs) == [["(A)"]]


@pytest.mark.parametrize("variant", ["zeroshot", "cot_zeroshot"])
def test_bbh_multi_choice_regex_prefix_choice_does_not_shadow_longer_choice(variant):
    # bbh keeps its own copy of MultiChoiceRegexFilter, wired via
    # `!function utils.MultiChoiceRegexFilter`, so it needs the same longest-first
    # ordering. Mirrors the bbh_zeroshot_movie_recommendation filter config.
    path = Path(__file__).parent.parent / f"lm_eval/tasks/bbh/{variant}/utils.py"
    spec = importlib.util.spec_from_file_location(f"bbh_{variant}_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    filt = utils.MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=0,
        ignore_case=True,
        ignore_punctuation=True,
    )
    docs = [
        {
            "input": "Find a movie similar to Batman Begins:\nOptions:\n"
            "(A) Batman\n(B) Batman Returns\n(C) Alien\n(D) Titanic"
        }
    ]

    assert filt.apply([["The answer is Batman Returns"]], docs) == [["(B)"]]
    assert filt.apply([["The answer is Batman"]], docs) == [["(A)"]]
