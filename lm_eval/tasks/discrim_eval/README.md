# Discrim-Eval

### Paper

Title: Evaluating and Mitigating Discrimination in Language Model Decisions

Abstract: https://arxiv.org/abs/2312.03689

This benchmark consists of prompts for 70 decision-making scenarios. Each prompt asks whether a person should receive a favorable outcome ("Yes" or "No"). Persons vary by age, gender, and race, yielding 135 prompts per scenario (a full race x gender x age cross). The dataset includes both explicit and implicit mentions of demographics.

This is scored generatively (`output_type: generate_until`): the model answers
the prompt in its own words, and the `yes_no` filter (see
`lm_eval/filters/extraction.py`) extracts the hard "Yes"/"No" choice it
actually made from that free-form text, rather than computing a soft
probability over fixed continuations -- this works for backends (e.g.
`rwkv7-remote`) that are generation-only and never return per-token logprobs.

Because each scenario is a balanced race x gender x age cross, no regression is
needed to control for confounders: for each demographic contrast (e.g. black vs.
white, female vs. male) we compute, within each scenario, the difference in the
fraction of "Yes" choices between the target and reference group, then average
that difference across the 70 scenarios. Age has no reference group, so instead
we compute, within each scenario, the slope of the "Yes" choice against
standardized age, and average the magnitude of that slope across scenarios.

`--limit N` samples whole scenario blocks (shuffled with a fixed seed) rather
than the dataset's first N rows, so any `--limit >= 135` includes at least one
complete, balanced scenario. Larger limits (ideally multiples of 135, up to the
full 9450) give a statistically stronger estimate.

Homepage: https://huggingface.co/datasets/Anthropic/discrim-eval

### Citation

```
@misc{tamkin2023discrim,
      title={Evaluating and Mitigating Discrimination in Language Model Decisions},
      author={Alex Tamkin and Amanda Askell and Liane Lovitt and Esin Durmus and Nicholas Joseph and Shauna Kravec and Karina Nguyen and Jared Kaplan and Deep Ganguli},
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
