# Configuration Guide

This guide explains how to use TOML or YAML configuration files with `lm-eval` to define reusable evaluation settings. Task and benchmark definitions remain YAML.

## Overview

Instead of passing many CLI arguments, define evaluation parameters in a TOML file:

```bash
# Instead of:
lm-eval run --model hf --model_args pretrained=gpt2,dtype=float32 --tasks hellaswag arc_easy --num_fewshot 5 --batch_size 8 --device cuda:0

# Use:
lm-eval run --config eval_config.toml
```

CLI arguments override config file values, so you can set defaults in a config file and override specific settings:

```bash
lm-eval run --config eval_config.toml --tasks mmlu --limit 100
```

## Quick Reference

Unversioned configuration keys correspond directly to CLI arguments. The
versioned RWKV preset uses the smaller public schema documented below. See the
[CLI Reference](interface.md#lm-eval-run) for detailed descriptions of native
options.

### RWKV7 complete run

Start the independent `vllm-rwkv` HTTP service from its own repository and uv environment:

```bash
cd /path/to/vllm-rwkv
# Start the repository's RWKV7 server launcher here.
```

Configure the launcher to pass `--enable-tokenizer-info-endpoint`. The service
must expose `/tokenizer_info` so lm-eval receives the same official RWKV chat
template used by the inference backend.

`configs/eval/lm_eval.toml` is the versioned, validated preset for
`rwkv7-g1i-1.5b-20260805-ctx16384`. Run it directly with:

```bash
uv run --no-sync python -m lm_eval run -C configs/eval/lm_eval.toml
```

The `rwkv7-http` backend itself accepts any RWKV7 served model name. For another
checkpoint size or revision, copy the preset and change `model_name`,
`max_length`, `benchmarks`, and `output_dir`. The loader derives lm-eval's
`model_args` and result metadata from this public schema, removing duplicated
model and profile fields. `rwkv_profile.wkv_mode` remains a declared server-side
fact and must be checked against the vllm-rwkv startup log. The TOML owns the
exact model version, endpoint, benchmark selectors, served endpoint capacity
(`max_num_seqs`), RWKV profile, and result location. The `benchmarks` array
keeps its configured order as the stable tie-break, but benchmarks execute one
at a time in ascending pending-rollout order (see
`lm_eval/api/benchmark_scheduler.py`). For `rwkv7-http`, these ordinary names are resolved through the
model-specific task adapter declared in the adapted task YAML; internal RWKV
task names never appear in this file. Each benchmark YAML owns its prompt,
answer extraction, metrics, stop conditions, and `max_gen_toks`. See the
[RWKV evaluation guide](../RWKV_EVALUATION_GUIDE.md) for the complete server and
evaluation workflow. The deprecated `temp/` launch scripts are not used.

The preset currently exposes 15 benchmark families in configured run order:
RACE, DROP, XQuAD, BABILong, InfiniteBench, RULER, WMDP, CRUXEval,
Inverse Scaling Prize, Model-Written Evals, HumanEval-Infilling, MuTual,
MC-TACO, Discrim-Eval, and Winogender. PALOMA remains excluded because dataset
access is still blocked; it must not be represented as a completed benchmark.

## Config Schema

The repository preset follows the same strict configuration approach as the
Helicopter LightEval runner: `schema_version = 1`, an allowlist of public fields,
required-field checks, duplicate benchmark rejection, and derived internal
arguments. Unknown fields fail before model or dataset initialization.

| Versioned RWKV field | Type | Default | Description |
|----------------------|------|---------|-------------|
| `schema_version` | int | required | Must be `1` |
| `backend` | string | required | `rwkv7-http`, `rwkv7-remote`, `rwkv7-remote-chat`, or `rwkv7-lightning-cuda` |
| `model_name` | string | required | Model id sent to the inference service as the literal API `model=` value (top level for a single-model file, per `[[models]]` entry for a campaign); normally the short serving alias, and may carry the full release stem for older configs |
| `weight_revision` | string | unset | Full served weight filename stem, including the release date and `ctx<N>` (e.g. `rwkv7-g1j-1.5b-20260831-ctx16384`); recorded in run metadata and used by scoreboard publication to derive `ctx_len` without putting it in the API-facing `model_name` |
| `base_url` | string | required | HTTP(S) endpoint: `/v1/completions`, `/v1/chat/completions`, or lightning CUDA `/v1/batch/completions`, according to `backend` |
| `benchmarks` | list | required | Unique public benchmark names; configured order is the stable tie-break for the runtime ascending-pending-rollout order |
| `benchmark_config` | string | `configs/benchmarks.toml` | Explicit top-level field-to-selector mapping; relative paths are resolved from the run config |
| `output_dir` | string | required | Result directory (top level for a single-model file, per `[[models]]` entry for a campaign) |
| `max_length` | int | required | Positive service context limit |
| `max_gen_toks` | int | `8192` | Remote RWKV choice-generation output budget; preserve the full input and reduce output to the remaining context window |
| `batch_size` | int | `1` | lm-eval request batch size; for `rwkv7-lightning-cuda`, this is the number of prompts in each native `contents` request |
| `max_num_seqs` | int | required | Served endpoint's sequence capacity; lightning CUDA derives `ceil(max_num_seqs / batch_size) + 1` HTTP requests so one batch may wait in its admission queue |
| `request_concurrency` | int | derived | Lightning CUDA only: explicit number of simultaneous HTTP requests; when omitted, it is derived from `max_num_seqs` and `batch_size` |
| `device` | string | `"cpu"` | Evaluation-side device |
| `rwkv_profile` | table | required | Complete RWKV execution profile |
| `rwkv_profile.prompt_template` | string | required | Official RWKV template mode |
| `rwkv_profile.generation_prompt` | string | required | `no_think`, `fake_think`, or `open_think` |
| `rwkv_profile.wkv_mode` | string | required | Recorded server WKV mode |
| `models` | array of tables | unset | The model ladder of a campaign; see [One file per benchmark set](#one-file-per-benchmark-set) |
| `model_overrides` | table | `{}` | Versioned fields forced onto every `[[models]]` entry last; the launch-time overlay hook |
| `apply_chat_template` | bool | `true` | Render the service's official template |
| `fewshot_as_multiturn` | bool | chat setting | Render few-shot examples as turns |
| `log_samples` | bool | `true` | Save inputs and model outputs |
| `seed` | list[int] | `[0,1234,1234,1234]` | Four lm-eval random seeds |
| `limit` | number | unset | Smoke-only sample limit |
| `use_cache` | string | unset | Response cache path |
| `weight_revision` | string | unset | Exact weight identity, including `ctx<number>` for Scoreboard; separate from the evaluation's `max_length` |

### Benchmark fields

Benchmark domains are configured only in `configs/benchmarks.toml`. Each
top-level key is one supported field and its array contains opaque benchmark
selectors:

```toml
knowledge = ["wmdp"]
reasoning = ["race", "drop"]
coding = ["cruxeval", "humaneval_infilling"]
```

The evaluator looks up the requested selector before `TaskManager` expands it
into tasks, groups, or tags. It never derives a field from the selector text or
from an expanded leaf name. A missing selector, unknown field, or selector
listed under two fields fails configuration validation.

The selected field is carried by `BenchmarkSpec`, saved in each aggregate
artifact as `benchmark.field`, and read back from that artifact during
Scoreboard publication. `[publication.tasks.<selector>.field]` is no longer a
source of truth; when present for compatibility, it may only agree with the
saved field and cannot override it.

### Remote RWKV evaluation protocol

`rwkv7-remote` renders the official local tokenizer's template and sends token
IDs to `/v1/completions`. Set `tokenizer_name` to the same official tokenizer
used by the server and keep `apply_chat_template = true`. The native
open-thinking prefix ends in `<think` intentionally. A completed `</think>`
is required before final-answer grading; raw responses remain in the artifacts.
Without a tokenizer, this backend only passes raw text through and cannot apply
a chat template. `rwkv7-remote-chat` delegates template rendering to the server.

`rwkv7-lightning-cuda` also requires the official local `tokenizer_name`, but
sends the rendered prompt strings through rwkv-lightning-cuda's native
`/v1/batch/completions` `contents` field. It maps the fixed RWKV sampling preset
to `alpha_presence`, `alpha_frequency`, and `alpha_decay`, preserves the full
input, and reduces the shared batch output budget to the space remaining for its
longest prompt. The endpoint currently exposes no request seed, so avg@k
rollouts are stochastic but cannot be replayed from lm-eval's seed alone.
Lightning rejects `temperature = 0`; this adapter sends `0.01` for `no_think`.
Cloudflare Access deployments read `CF_ACCESS_CLIENT_ID` and
`CF_ACCESS_CLIENT_SECRET` from the environment and add the corresponding HTTP
headers without storing credentials in the evaluation config.
`rwkv_profile.wkv_mode = "fp32io16"` requires starting the CUDA service with
`--wkv32`; its current status response does not expose this flag, so verify it
from the server startup log before publishing scores.

RWKV sampling presets are sent as top-level vLLM fields. The custom server must
support `penalty_decay` and its required sampler. Each document's repeats use
`seed[2] + repeat_ord`, including after resume, and share one concurrency budget.
The full input is preserved. `max_gen_toks` defaults to 8192 for choice rollouts;
set it explicitly when comparing models with different reasoning lengths.
A prompt that exhausts the context window aborts rather than receiving a score
for an empty completion. Multi-leaf benchmarks share one avg@k sizing budget.

The protocol identity is `rwkv-generative-v2`. Use a separate output/checkpoint
directory for runs made before this version. Old raw answers need a versioned
replay before their scores can be published using the new protocol.

### Local RWKV coding

`lm-eval coding -C configs/eval/rwkv7_coding_local.toml` uses the same validated
RWKV model/profile fields and adds a strict `[coding]` table. This entry point
reads a local JSONL problem suite, generates over HTTP, and executes Python tests
in local Docker containers. It does not resolve `rwkv_coding_local` through the
ordinary task registry. Model metadata and samples are always saved locally;
automatic publication and generic run-only task/checkpoint overrides are rejected.

| Coding field | Default | Meaning |
| --- | --- | --- |
| `coding.problems` | required | Local JSONL or JSONL.gz file; relative to the config |
| `coding.num_samples` | `8` | Fixed independent generations per problem |
| `coding.pass_k` | `[1, num_samples]` | Values of k, each no greater than num_samples |
| `coding.workers` | `2` | Local grading workers; separate from inference concurrency |
| `coding.test_timeout` | `5.0` | Per-candidate execution timeout in seconds |
| `coding.memory_mb` | `4096` | Memory limit per container |
| `coding.docker_image` | `python:3.11-slim` | Existing local image, resolved to an immutable ID |

The exact `weight_revision` must include `ctx<number>`. Raw completions require
the official local `tokenizer_name`; the chat backend delegates the native RWKV
template to its server. Both require `apply_chat_template = true`. `output_dir`
also resolves relative to the config. Use `--phase generate` without Docker or
`--phase score --samples ... --output-dir ...` for offline replay into an empty
directory. Only an incomplete generation may use `--resume`; scoring identities
and full problem/repeat coverage are verified before accepting saved answers.


## One file per benchmark set

A full campaign evaluates the same benchmark set on every RWKV7 size, so the
per-size TOMLs differ only in `model_name`, `weight_revision`, `tokenizer_name`,
`output_dir`, and the endpoint capacity each size is served with. Declare those
once as `[[models]]` entries and keep everything else in the shared top-level
table:

```toml
# configs/eval/rwkv7_g1k_campaign_five_benchmarks_full_scoreboard.toml
schema_version = 1
backend = "rwkv7-remote"

base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["mc_taco", "mutual", "race", "winogender", "wmdp"]
include_path = ["configs/eval/offline_six_tasks"]
max_length = 25600
apply_chat_template = true
fewshot_as_multiturn = true
log_samples = true

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "open_think"
wkv_mode = "fp32io16"

[[models]]
model_name = "rwkv7-g1k-1.5b"
weight_revision = "rwkv7-g1k-1.5b-20260930-ctx25600"
tokenizer_name = "/home/link/weight/rwkv-g1k-tokenizer/1.5b"
max_num_seqs = 64
output_dir = "results/rwkv7-g1k-1.5b-...-full"

[[models]]
model_name = "rwkv7-g1k-2.9b"
weight_revision = "rwkv7-g1k-2.9b-20260930-ctx25600"
tokenizer_name = "/home/link/weight/rwkv-g1k-tokenizer/2.9b"
max_num_seqs = 64
output_dir = "results/rwkv7-g1k-2.9b-...-full"

# ... one entry per size, in run order
```

The top-level table holds what the models share; an entry holds what is specific
to one model and may carry any versioned field. Precedence, lowest to highest:
top-level table, `[[models]]` entry, `[model_overrides]`. Each entry must name
its own `model_name` and `output_dir`, and the resolved `output_dir`s and
`model_name`s must be distinct so that no two models write into the same result
directory or publish under the same name. Give every published entry its own
`weight_revision` (or keep the full release stem in `output_dir`, which
publication also reads); it carries the exact release date and `ctx_len`, which
are part of the scoreboard's model identity even when the API alias is stable
across releases.

`lm-eval run -C <campaign.toml>` then evaluates every declared model: one child
process per model, running at the same time by default, because a campaign only
keeps the served endpoint busy when the whole ladder has requests in flight.
Each child re-reads the same file with `--only-model <model_name>`, so it keeps
its own request budget, checkpoint, result directory, and publication settings
— and resolves to exactly the config the equivalent single-model file
produced, checkpoint fingerprint included, so an in-progress run stays
resumable across the switch to one file. Per-model console output is appended to
`<output_dir>/console.log`. `--jobs N` caps how many models run at once;
`--only-model <model_name>` runs one model without fanning out (that is what the
children do, and it is also the way to re-run a single size by hand).

Launch-time overlays keep working: because the overlay cannot know which
per-model values it should beat, put campaign-wide launch knobs under
`[model_overrides]`.

```toml
# local launch overlay (gitignored temp/, one file for the whole ladder)
schema_version = 1
include = ["../configs/eval/rwkv7_g1k_campaign_five_benchmarks_full_scoreboard.toml"]

[model_overrides]
max_num_seqs = 200
timeout = 1800
max_retries = 6
```

The campaign exits non-zero if any model failed, after reporting a one-line
summary per model, so a batch launcher can tell success from partial failure.
`lm-eval publish -C <campaign.toml>` reads the top-level `[publication]` table;
pass `--rerun_reason` when a single model's note has to differ from the shared
one.

Unversioned TOML and YAML files remain compatible with upstream lm-eval. Their
keys correspond directly to CLI arguments:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"hf"` | Model type/provider |
| `model_args` | dict | `{}` | Model constructor arguments |
| `tasks` | list/string | required | Tasks to evaluate |
| `num_fewshot` | int/null | `null` | Few-shot example count |
| `batch_size` | int/string | `1` | Batch size or "auto" |
| `max_batch_size` | int/null | `null` | Max batch size for auto |
| `device` | string/null | `"cuda:0"` | Device to use |
| `limit` | float/null | `null` | Example limit per task |
| `samples` | dict/null | `null` | Specific sample indices |
| `use_cache` | string/null | `null` | Response cache path |
| `cache_requests` | string/dict | `{}` | Request cache settings |
| `output_path` | string/null | `null` | Results output path |
| `log_samples` | bool | `false` | Save model I/O |
| `predict_only` | bool | `false` | Skip metrics |
| `apply_chat_template` | bool/string | `false` | Chat template |
| `system_instruction` | string/null | `null` | System prompt |
| `fewshot_as_multiturn` | bool/null | `null` | Multi-turn few-shot |
| `include_path` | string/null | `null` | External tasks path |
| `gen_kwargs` | dict | `{}` | Generation arguments |
| `wandb_args` | dict | `{}` | W&B init arguments |
| `hf_hub_log_args` | dict | `{}` | HF Hub logging |
| `seed` | list/int | `[0,1234,1234,1234]` | Random seeds |
| `trust_remote_code` | bool | `false` | Trust remote code |
| `metadata` | dict | `{}` | Task metadata |

---

## Example

```toml
# basic_eval.toml
model = "hf"
tasks = ["hellaswag", "arc_easy"]
num_fewshot = 0
batch_size = "auto"
device = "cuda:0"
output_path = "./results/gpt2/"
log_samples = true

[model_args]
pretrained = "gpt2"
dtype = "float32"
```

## Inheriting a run config

Small or smoke runs can include a formal config and override only their scope:

```toml
include = "lm_eval.toml"
benchmarks = ["race"]
limit = 10
output_dir = "results/smoke/race"
```

Included paths are relative to the including file. Nested tables are merged, and local values override included values. CLI arguments still have the highest priority.

The array order is the run order; do not hide a campaign sequence in a
model-specific group YAML. Benchmark prompt construction (`doc_to_text`), answer
extraction and filters, metrics, stop conditions (`until`), and `max_gen_toks`
belong in each task YAML and are rejected by the versioned global schema. Each
`benchmarks` entry is a native lm-eval-harness task, group, or tag name,
resolved directly by `TaskManager` — there is no RWKV-specific adapter or
renaming layer. The `rwkv_profile` table selects only the official model
template and RWKV generation profile; sampling parameters (temperature,
top_p/top_k, penalties) always come from that profile and cannot be
overridden by task YAML. The Python CLI runner only loads this configuration,
selects the model adapter, and executes lm-eval; it contains no
benchmark-specific strategy.

---

## Programmatic Usage

For loading config files in Python, see the [Python API Guide](python-api.md#using-evaluatorconfig).

---

## Validation

Validate your configuration before running:

```bash
# Check that tasks exist
lm-eval validate --tasks hellaswag,arc_easy

# With external tasks
lm-eval validate --tasks my_task --include_path /path/to/tasks
```

---

## Tips

1. **Start simple**: Begin with minimal config and add options as needed
2. **Use CLI overrides**: Set defaults in config, override with CLI for experiments
3. **Separate concerns**: Create different configs for different model families or task sets
4. **Version control**: Commit config files alongside results for reproducibility
5. **Use comments**: TOML and YAML support `#` comments to document your choices
