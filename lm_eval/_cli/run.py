import argparse
import io
import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from lm_eval._cli.subcommand import SubCommand
from lm_eval._cli.utils import (
    MergeDictAction,
    SplitArgs,
    _int_or_none_list_arg_type,
    request_caching_arg_to_dict,
    try_parse_json,
)


def _finish_reasons(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [reason for item in value for reason in _finish_reasons(item)]
    reason = getattr(value, "finish_reason", None)
    return [reason] if isinstance(reason, str) else []


def _add_truncation_stats(results: dict, samples: dict) -> dict:
    task_stats = {}
    for task_name, task_samples in samples.items():
        generated_samples = 0
        truncated_samples = 0
        for sample in task_samples:
            finish_reasons = _finish_reasons(sample.get("resps", []))
            if not finish_reasons:
                continue
            generated_samples += 1
            truncated = "length" in finish_reasons
            truncated_samples += int(truncated)
            sample["finish_reasons"] = finish_reasons
            sample["truncated"] = truncated
        task_stats[task_name] = {
            "generated_samples": generated_samples,
            "truncated_samples": truncated_samples,
            "truncation_rate": (
                truncated_samples / generated_samples if generated_samples else None
            ),
        }
    results["config"]["truncation"] = task_stats
    return task_stats


class Run(SubCommand):
    """Command for running language model evaluation."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "run",
            help="Run the evaluation harness on specified tasks",
            description="Evaluate language models on various benchmarks and tasks.",
            usage="lm-eval run --model <model> --tasks <task> <task> --model_args <arg=value> <arg=value> [options]",
            epilog=textwrap.dedent("""
                examples:
                  # Basic evaluation with HuggingFace model
                  $ lm-eval run --model hf --model_args pretrained=gpt2 dtype=float32 --tasks hellaswag

                  # Evaluate on multiple tasks with few-shot examples
                  $ lm-eval run --model vllm --model_args pretrained=EleutherAI/gpt-j-6B --tasks arc_easy arc_challenge --num_fewshot 5

                  # Evaluation with custom generation parameters
                  $ lm-eval run --model hf --model_args pretrained=gpt2 --tasks lambada --gen_kwargs temperature=0.8 top_p=0.95 'stop=["\\n\\n"]'

                  # Use configuration file
                  $ lm-eval run --config my_config.yaml --tasks mmlu

                For more information, see: https://github.com/EleutherAI/lm-evaluation-harness
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_args()
        self._parser.set_defaults(func=self._execute)

    def _add_args(self) -> None:
        self._parser = self._parser

        # Defaults are set in config/evaluate_config.py
        config_group = self._parser.add_argument_group("configuration")
        config_group.add_argument(
            "--config",
            "-C",
            default=None,
            type=str,
            metavar="<path>",
            help="Set initial arguments from YAML config",
        )
        config_group.add_argument(
            "--only-model",
            type=str,
            default=None,
            metavar="<model_name>",
            help=textwrap.dedent("""
                Evaluate only the named model of a config that declares
                [[models]]. The campaign fan-out passes this to each child
                process; single-model configs never need it.
            """).strip(),
        )

        # Model and Tasks
        model_group = self._parser.add_argument_group("model and tasks")
        model_group.add_argument(
            "--tasks",
            "-t",
            default=None,
            nargs="+",
            metavar="<task>",
            action=SplitArgs,
            help=textwrap.dedent("""
                Space (or comma-separated) list of task names or groupings.
                Use 'lm-eval ls tasks' to see all available tasks.
            """).strip(),
        )
        model_group.add_argument(
            "--model",
            "-M",
            type=str,
            default=None,
            metavar="<model>",
            help="Model name (default: hf)",
        )
        model_group.add_argument(
            "--model_args",
            "-a",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<arg>",
            help="Model arguments as 'key=val,key2=val2' or `key=val` `key2=val2`",
        )
        model_group.add_argument(
            "--apply_chat_template",
            type=str,
            nargs="?",
            const=True,
            default=argparse.SUPPRESS,
            metavar="<template>",
            help="Apply chat template to prompts (optional template name)",
        )
        model_group.add_argument(
            "--limit",
            "-L",
            type=float,
            default=None,
            metavar="<limit>",
            help="Limit examples per task (integer count or fraction)",
        )
        model_group.add_argument(
            "--use_cache",
            "-c",
            type=str,
            default=None,
            metavar="<path>",
            help="Path to cache model responses (skips repeated inference)",
        )
        model_group.add_argument(
            "--resume",
            action="store_true",
            default=argparse.SUPPRESS,
            help=(
                "Resume a crash-safe checkpointed run (equivalent to "
                "checkpoint.enabled=true in [checkpoint]). See [checkpoint] in "
                "the TOML config for dir/fresh options."
            ),
        )

        # Evaluation Settings
        eval_group = self._parser.add_argument_group("evaluation settings")
        eval_group.add_argument(
            "--num_fewshot",
            "-f",
            type=int,
            default=None,
            metavar="<n>",
            help="Number of examples in few-shot context",
        )
        eval_group.add_argument(
            "--batch_size",
            "-b",
            type=str,
            default=argparse.SUPPRESS,
            metavar="<size>",
            help=textwrap.dedent(
                "Batch size: 'auto', 'auto:N' (auto-tune N times), or integer (default: 1)"
            ),
        )
        eval_group.add_argument(
            "--max_batch_size",
            type=int,
            default=None,
            metavar="<n>",
            help="Maximum batch size when using --batch_size auto",
        )
        eval_group.add_argument(
            "--device",
            type=str,
            default=None,
            metavar="<device>",
            help="Device to use (e.g. cuda, cuda:0, cpu, mps)",
        )
        eval_group.add_argument(
            "--gen_kwargs",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<arg>",
            help=textwrap.dedent(
                'Generation arguments as `temperature=0,stop=["stop"]` or `key=val` `key2=val2`.'
                "Values should be parsable with ast.literal_eval."
            ),
        )

        # Data and Output
        data_group = self._parser.add_argument_group(
            "data and output (see also: --limit)"
        )
        data_group.add_argument(
            "--output_path",
            "-o",
            default=None,
            type=str,
            metavar="<path>",
            help="Output dir or json file for results (and samples)",
        )
        data_group.add_argument(
            "--log_samples",
            "-s",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Save all model outputs and documents for post-hoc analysis",
        )
        data_group.add_argument(
            "--samples",
            "-E",
            default=None,
            type=try_parse_json,
            metavar="<json>",
            help="JSON mapping task names to sample indices, e.g. '{\"task1\": [0,1,2]}'. Incompatible with --limit.",
        )

        # Caching and Performance
        cache_group = self._parser.add_argument_group(
            "caching and performance (see also: --use_cache)"
        )
        cache_group.add_argument(
            "--cache_requests",
            type=request_caching_arg_to_dict,
            nargs="?",
            const="true",
            default=None,
            metavar="true|refresh|delete",
            help="Cache preprocessed prompts; bare flag defaults to 'true'",
        )
        cache_group.add_argument(
            "--check_integrity",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Run task test suite validation",
        )

        # Prompt Formatting
        template_group = self._parser.add_argument_group(
            "instruct formatting (see also: --apply_chat_template)"
        )
        template_group.add_argument(
            "--system_instruction",
            type=str,
            default=None,
            metavar="<text>",
            help="Add custom system instruction.",
        )
        template_group.add_argument(
            "--fewshot_as_multiturn",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            nargs="?",
            const=True,
            default=argparse.SUPPRESS,
            metavar="<bool>",
            help="Use fewshot as multi-turn conversation. Auto-enabled with --apply_chat_template. Use 'false' to disable.",
        )

        # Task Management
        task_group = self._parser.add_argument_group("task management")
        task_group.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="<path>",
            help="Additional directory for external tasks",
        )

        # Logging and Tracking
        logging_group = self._parser.add_argument_group("logging and tracking")
        logging_group.add_argument(
            "--verbosity",
            "-v",
            type=str.upper,
            default=None,
            metavar="<level>",
            help="(Deprecated) Log level. Use LMEVAL_LOG_LEVEL env var instead",
        )
        logging_group.add_argument(
            "--write_out",
            "-w",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Print prompts for first few documents",
        )
        logging_group.add_argument(
            "--show_config",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Display full task configuration after evaluation",
        )
        logging_group.add_argument(
            "--wandb_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Weights & Biases init arguments key=val key2=val2",
        )
        logging_group.add_argument(
            "--wandb_config_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Weights & Biases config arguments key=val key2=val2",
        )
        logging_group.add_argument(
            "--trackio_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Trackio init arguments key=val key2=val2 (e.g. project=my-evals)",
        )
        logging_group.add_argument(
            "--hf_hub_log_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Hugging Face Hub logging arguments key=val key2=val2",
        )

        # Advanced Options
        advanced_group = self._parser.add_argument_group("advanced options")
        advanced_group.add_argument(
            "--predict_only",
            "-x",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Save predictions only, skip metric computation",
        )
        default_seed_string = "0,1234,1234,1234"
        advanced_group.add_argument(
            "--seed",
            type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
            default=None,
            metavar="<seed>",
            help=textwrap.dedent(f"""
                Random seeds for python,numpy,torch,fewshot (default: {default_seed_string}).
                Use single integer for all, or comma-separated list of 4 values.
                Use 'None' to skip setting a seed. Example: --seed 42 or --seed 0,None,8,52
            """).strip(),
        )
        advanced_group.add_argument(
            "--trust_remote_code",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Allow executing remote code from Hugging Face Hub",
        )
        advanced_group.add_argument(
            "--confirm_run_unsafe_code",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Confirm understanding of unsafe code execution risks",
        )
        advanced_group.add_argument(
            "--jobs",
            type=int,
            default=None,
            metavar="<n>",
            help=textwrap.dedent("""
                How many models of a [[models]] campaign config to evaluate at
                once (default: all of them, one child process per model). Each
                model still keeps its own request budget, checkpoint, and
                result directory.
            """).strip(),
        )
        advanced_group.add_argument(
            "--metadata",
            type=json.loads,
            default=None,
            metavar="<arg>",
            help=textwrap.dedent(
                """`key=val` `key2=val` args parsable by ast.literal_eval (merged with model_args),
                required for some tasks such as RULER"""
            ),
        )

    @staticmethod
    def _execute(args: argparse.Namespace) -> None:
        """Runs the evaluation harness with the provided arguments.

        A config that declares `[[models]]` describes a campaign: every declared
        model is evaluated, one child process per model, so a single TOML covers
        the whole model ladder. Any other config runs in this process exactly as
        before.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from lm_eval.config.evaluate_config import EvaluatorConfig

        config_path = getattr(args, "config", None)
        if config_path and not getattr(args, "only_model", None):
            configs = EvaluatorConfig.load_configs(config_path)
            if len(configs) > 1:
                Run._execute_campaign(args, configs)
                return

        Run._execute_single(args)

    @staticmethod
    def _campaign_child_command(model_name: str, argv: Sequence[str]) -> list[str]:
        """Command that re-runs this same invocation for a single campaign model."""
        return [sys.executable, "-m", "lm_eval", *argv, "--only-model", model_name]

    @staticmethod
    def _start_campaign_model(
        model: tuple[str, str], argv: Sequence[str]
    ) -> tuple[str, subprocess.Popen, io.BufferedWriter]:
        """Start one model of a campaign, logging it inside its own output_dir."""
        model_name, output_dir = model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log = (Path(output_dir) / "console.log").open("ab")
        command = Run._campaign_child_command(model_name, argv)
        log.write(
            f"# campaign model {model_name}\n# command: {' '.join(command)}\n".encode()
        )
        log.flush()
        process = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            command,
            cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return model_name, process, log

    @staticmethod
    def _execute_campaign(
        args: argparse.Namespace,
        configs: list[dict],
        argv: Sequence[str] | None = None,
    ) -> None:
        """Evaluate every model a campaign config declares.

        Each model keeps its own request budget, checkpoint, result directory,
        and publication settings; the child re-reads the very same campaign file
        with `--only-model`, so its resolved config -- and therefore its
        checkpoint fingerprint -- is exactly what the equivalent per-model
        config file produced. Models default to running all at once, which is
        what keeps a served endpoint busy across the whole ladder; `--jobs` caps
        the fan-out.
        """
        eval_logger = logging.getLogger(__name__)
        models = [
            (config_data["metadata"]["model_name"], config_data["output_path"])
            for config_data in configs
        ]
        jobs = max(1, len(models) if not args.jobs else min(args.jobs, len(models)))
        argv = list(sys.argv[1:] if argv is None else argv)
        eval_logger.info(
            "Campaign config declares %d models, running %d at a time: %s",
            len(models),
            jobs,
            ", ".join(model_name for model_name, _ in models),
        )

        failures: list[tuple[str, int]] = []
        running: list[tuple[str, subprocess.Popen, io.BufferedWriter]] = []
        pending = list(models)
        try:
            while pending or running:
                while pending and len(running) < jobs:
                    running.append(Run._start_campaign_model(pending.pop(0), argv))
                # Peek instead of popping so an interrupt during the wait still
                # finds every live child -- including this one -- to stop.
                model_name, process, log = running[0]
                returncode = process.wait()
                running.pop(0)
                log.close()
                if returncode == 0:
                    eval_logger.info("Campaign model %s finished", model_name)
                else:
                    failures.append((model_name, returncode))
                    eval_logger.error(
                        "Campaign model %s exited with status %d",
                        model_name,
                        returncode,
                    )
        except BaseException:
            for _, process, log in running:
                process.terminate()
            for _, process, log in running:
                process.wait()
                log.close()
            raise

        failed = {model_name for model_name, _ in failures}
        print(
            "\n".join(
                [
                    f"campaign summary: {len(models) - len(failures)}/{len(models)} "
                    "models completed",
                    *(
                        f"  {'ok  ' if model_name not in failed else 'fail'}  "
                        f"{model_name}  {Path(output_dir) / 'console.log'}"
                        for model_name, output_dir in models
                    ),
                ]
            )
        )
        if failures:
            raise SystemExit(1)

    @staticmethod
    def _execute_single(args: argparse.Namespace) -> None:
        """Runs one resolved model config through the evaluation harness."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from lm_eval.config.evaluate_config import EvaluatorConfig

        eval_logger = logging.getLogger(__name__)

        # Create and validate config (most validation now occurs in EvaluationConfig)
        cfg = EvaluatorConfig.from_cli(args)

        from lm_eval import simple_evaluate
        from lm_eval.loggers import EvaluationTracker, TrackioLogger, WandbLogger
        from lm_eval.utils import handle_non_serializable, make_table

        # Set up logging
        if cfg.wandb_args:
            wandb_logger = WandbLogger(cfg.wandb_args, cfg.wandb_config_args)
        if cfg.trackio_args:
            trackio_logger = TrackioLogger(cfg.trackio_args)

        # Set up evaluation tracker
        if cfg.output_path:
            cfg.hf_hub_log_args["output_path"] = cfg.output_path

        if os.environ.get("HF_TOKEN", None):
            cfg.hf_hub_log_args["token"] = os.environ.get("HF_TOKEN")

        evaluation_tracker = EvaluationTracker(**cfg.hf_hub_log_args)
        scoreboard_callback = None
        publication = getattr(cfg, "publication", {})
        if isinstance(publication, dict) and publication.get("enabled", False):
            from lm_eval.loggers.scoreboard import (
                PublicationConfig,
                ScoreboardTaskCallback,
            )

            publication_config = PublicationConfig.from_mapping(publication)
            scoreboard_callback = ScoreboardTaskCallback(
                config=publication_config,
            )

        checkpoint_store = None
        checkpoint_cfg = dict(cfg.checkpoint or {})
        if cfg.resume:
            checkpoint_cfg["enabled"] = True
        if checkpoint_cfg.get("enabled", False):
            from dataclasses import asdict as _asdict

            from lm_eval.api.checkpoint import (
                CheckpointStore,
                compute_config_fingerprint,
                without_operational_args,
            )

            if cfg.use_cache:
                raise ValueError(
                    "checkpoint (or --resume) and --use_cache cannot be combined: "
                    "CachingLM's cache-check wrapper does not forward checkpoint "
                    "item identity."
                )

            checkpoint_dir = checkpoint_cfg.get("dir")
            if not checkpoint_dir:
                if not cfg.output_path:
                    raise ValueError(
                        "checkpoint.enabled (or --resume) requires output_path "
                        "or checkpoint.dir to be set"
                    )
                checkpoint_dir = os.path.join(cfg.output_path, "checkpoints")
            db_path = os.path.join(checkpoint_dir, "run.sqlite3")

            fingerprint_payload = _asdict(cfg)
            if cfg.model in {
                "rwkv7-lightning-cuda",
                "rwkv7-remote",
                "rwkv7-remote-chat",
            }:
                from lm_eval.api.checkpoint import EVALUATION_PROTOCOL_VERSION
                fingerprint_payload["evaluation_protocol"] = EVALUATION_PROTOCOL_VERSION
            # HTTP request concurrency and the client-side timeout/retry policy
            # are scheduling/resource knobs. They do not change prompts,
            # sampling, or model identity, so changing them must remain
            # compatible with an existing checkpoint run. `num_concurrent` is
            # derived from the config's declared `max_num_seqs` (the endpoint's
            # own inference capacity), so it is excluded here as well -- with the
            # same list `compute_task_fingerprint` uses, so the run id and the
            # per-item fingerprints can never disagree about them.
            for key in ("model_args", "metadata"):
                value = fingerprint_payload.get(key)
                if isinstance(value, dict):
                    fingerprint_payload[key] = without_operational_args(value)
            for key in (
                "config",
                "checkpoint",
                "resume",
                "show_config",
                "verbosity",
                "write_out",
                "check_integrity",
                # Publication only affects what happens to already-computed
                # results afterward, not what gets requested/computed, so it
                # must not change the run identity -- otherwise adding a
                # [publication] section to republish an already-completed
                # run's results would be treated as a brand new run.
                "publication",
            ):
                fingerprint_payload.pop(key, None)
            config_fingerprint = compute_config_fingerprint(fingerprint_payload)

            checkpoint_store = CheckpointStore(
                db_path=db_path,
                config_fingerprint=config_fingerprint,
                fresh=bool(checkpoint_cfg.get("fresh", False)),
            )

        # Create task manager (metadata already set up in config validation)
        task_manager, benchmark_map = cfg.process_tasks(cfg.metadata)

        # Validation warnings (keep these in CLI as they're logging-specific)
        if "push_samples_to_hub" in cfg.hf_hub_log_args and not cfg.log_samples:
            eval_logger.warning(
                "Pushing samples to the Hub requires --log_samples to be set."
            )

        # Log task selection (tasks already processed in config)
        if cfg.include_path is not None:
            eval_logger.info("Including path: %s", cfg.include_path)
        eval_logger.info("Selected Tasks: %s", cfg.tasks)

        # Run evaluation
        results = simple_evaluate(
            model=cfg.model,
            model_args=cfg.model_args,
            tasks=cfg.tasks,
            num_fewshot=cfg.num_fewshot,
            batch_size=cfg.batch_size,
            max_batch_size=cfg.max_batch_size,
            device=cfg.device,
            use_cache=cfg.use_cache,
            cache_requests=cfg.cache_requests.get("cache_requests", False),
            rewrite_requests_cache=cfg.cache_requests.get(
                "rewrite_requests_cache", False
            ),
            delete_requests_cache=cfg.cache_requests.get(
                "delete_requests_cache", False
            ),
            limit=cfg.limit,
            samples=cfg.samples,
            check_integrity=cfg.check_integrity,
            write_out=cfg.write_out,
            log_samples=cfg.log_samples,
            evaluation_tracker=evaluation_tracker,
            system_instruction=cfg.system_instruction,
            apply_chat_template=cfg.apply_chat_template,
            fewshot_as_multiturn=cfg.fewshot_as_multiturn,
            gen_kwargs=cfg.gen_kwargs,
            task_manager=task_manager,
            verbosity=cfg.verbosity,
            predict_only=cfg.predict_only,
            random_seed=cfg.seed[0] if cfg.seed else None,
            numpy_random_seed=cfg.seed[1] if cfg.seed else None,
            torch_random_seed=cfg.seed[2] if cfg.seed else None,
            fewshot_random_seed=cfg.seed[3] if cfg.seed else None,
            confirm_run_unsafe_code=cfg.confirm_run_unsafe_code,
            metadata=cfg.metadata,
            task_callback=scoreboard_callback,
            checkpoint_store=checkpoint_store,
            benchmark_map=benchmark_map,
        )

        if checkpoint_store is not None:
            checkpoint_store.close()

        # Process results
        if results is not None:
            if cfg.log_samples:
                samples = results.pop("samples")
                _add_truncation_stats(results, samples)

            dumped = json.dumps(
                results, indent=2, default=handle_non_serializable, ensure_ascii=False
            )
            if cfg.show_config:
                print(dumped)

            batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

            # W&B logging
            if cfg.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if cfg.log_samples:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:  # noqa: BLE001
                    eval_logger.info("Logging to W&B failed: %s", e)

            # Trackio logging
            if cfg.trackio_args:
                try:
                    trackio_logger.post_init(results)
                    trackio_logger.log_eval_result()
                    if cfg.log_samples:
                        trackio_logger.log_eval_samples(samples)
                except Exception as e:  # noqa: BLE001
                    eval_logger.info("Logging to Trackio failed: %s", e)

            # Save results
            evaluation_tracker.save_results_aggregated(
                results=results, samples=samples if cfg.log_samples else None
            )

            if cfg.log_samples:
                for task_name in results["configs"]:
                    if task_name not in evaluation_tracker.saved_sample_tasks:
                        evaluation_tracker.save_results_samples(
                            task_name=task_name, samples=samples[task_name]
                        )

            if (
                evaluation_tracker.push_results_to_hub
                or evaluation_tracker.push_samples_to_hub
            ):
                evaluation_tracker.recreate_metadata_card()

            # Print results
            cfg.model_args.pop("trust_remote_code", None)
            print(
                f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), "
                f"limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
                f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
            )
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

            if cfg.wandb_args:
                wandb_logger.run.finish()

            if cfg.trackio_args:
                trackio_logger.finish()
