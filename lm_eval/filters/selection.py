from collections import Counter

from lm_eval.api.filter import Filter, FilterEnsemble
from lm_eval.api.registry import register_filter


# TODO: implement "arg_max" filter. either it should take in an arbitrary "scoring"/reward function
# that takes an input and returns a scalar and then should select the max reward,
# or should implement different filters for different ways of handling a reward model's inference.


@register_filter("take_first")
class TakeFirstFilter(Filter):
    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
        """
        return map(lambda r: r[0], resps)


@register_filter("take_first_k")
class TakeKFilter(Filter):
    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop("k")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        # need resp to be subscriptable to check below
        resps = list(resps)
        # check we have at least k responses per doc, else we can't take the first k
        assert len(resps[0]) >= self.k, (
            f"Need at least {self.k} responses per doc to take first {self.k}, but got {len(resps[0])} only! Please increase TaskConfig.repeats ."
        )
        return map(lambda r: r[: self.k], resps)


class PerRepeatFilter(Filter):
    """Apply a whole filter pipeline to each repeat independently.

    ``avg@k`` generates ``k`` responses per doc and scores each of them, but
    a task's ``filter_list`` is written for a single response set: pipelines
    that end in ``take_first`` (the common answer-extraction shape) collapse
    the ``k`` responses down to one, and even filters that keep a per-response
    list only get one pass over the ensemble. Wrapping a task's ensemble in
    this filter keeps that ensemble in charge of extraction -- each repeat is
    fed through the wrapped pipeline on its own, so the instance's
    ``filtered_resps`` ends up as one filtered value per repeat, each shaped
    exactly like the value a ``repeats: 1`` run would produce.
    """

    def __init__(self, ensemble: FilterEnsemble) -> None:
        self.ensemble = ensemble

    def apply(self, resps, docs):
        resps = list(resps)
        repeats = len(resps[0]) if resps else 0
        filtered: list[list] = [[] for _ in resps]
        for repeat_ord in range(repeats):
            # Replay the wrapped pipeline on a one-response response set, so
            # collapse-style filters (take_first, majority_vote) see the same
            # shape they would in a single-repeat run.
            values = [[response_set[repeat_ord]] for response_set in resps]
            for build_filter in self.ensemble.filters:
                values = build_filter().apply(values, docs)
            for doc_index, value in enumerate(values):
                filtered[doc_index].append(value)
        return filtered


@register_filter("majority_vote")
class MajorityVoteFilter(Filter):
    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Each entry of `resps` is a list of model responses.
        We select the response that occurs most frequently in each entry of `resps`.
        """

        def select_majority(resp):
            counts = Counter(resp)
            vote = counts.most_common(1)[0][0]
            return vote

        return map(lambda r: [select_majority(r)], resps)
