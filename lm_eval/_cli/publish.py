import argparse
import sys
import textwrap

from lm_eval._cli.subcommand import SubCommand


class Publish(SubCommand):
    """Command for publishing already-saved benchmark results to the scoreboard."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "publish",
            help="Publish already-saved benchmark results to the scoreboard",
            description=(
                "Publish one or more previously saved benchmark results.json "
                "artifacts to the scoreboard, without re-running the evaluation."
            ),
            usage="lm-eval publish --config <publication.toml> <results.json|task_artifacts_dir> ...",
            epilog=textwrap.dedent("""
                examples:
                  # Publish a single benchmark artifact
                  lm-eval publish --config configs/eval/my_run.toml \\
                      output/my-model/task_artifacts/mmlu/results.json

                  # Publish every benchmark artifact found under a task_artifacts directory
                  lm-eval publish --config configs/eval/my_run.toml \\
                      output/my-model/task_artifacts

                  # Validate without uploading (build and print scores only)
                  lm-eval publish --config configs/eval/my_run.toml --dry-run \\
                      output/my-model/task_artifacts

                  # Force re-upload of an already-published benchmark
                  lm-eval publish --config configs/eval/my_run.toml \\
                      --rerun-reason "fixed scoring bug" \\
                      output/my-model/task_artifacts/mmlu/results.json

                notes:
                  • --config must point at a TOML/YAML file containing a [publication]
                    table -- the same table used by `lm-eval run`'s live publication
                    path (base_url, token_env, cot_mode, [publication.tasks.<name>]).
                    The benchmark field is read from the saved artifact's
                    benchmark.field, not inferred from its name or supplied here.
                  • If an older artifact does not record weight_revision in its
                    run metadata, the config's top-level weight_revision (or the
                    matching [[models]] entry's) is used to resolve ctx_len.
                  • Directory arguments are searched recursively for results.json
                    files that carry benchmark-level artifact metadata
                    (leaf_artifact_paths); plain per-leaf artifacts are skipped.
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_args()
        self._parser.set_defaults(func=self._execute)

    def _add_args(self) -> None:
        self._parser.add_argument(
            "results",
            nargs="+",
            metavar="RESULTS",
            help=(
                "Path(s) to a benchmark results.json file, or a directory "
                "(e.g. a task_artifacts/ root) to search recursively for them."
            ),
        )
        self._parser.add_argument(
            "--config",
            required=True,
            metavar="FILE",
            help="TOML/YAML config file containing the [publication] table.",
        )
        self._parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Build and validate scores but do not upload them.",
        )
        self._parser.add_argument(
            "--rerun-reason",
            default=None,
            metavar="TEXT",
            help=(
                "If set, overrides the config's rerun_reason so an existing "
                "active score for a benchmark is expired and replaced."
            ),
        )
        self._parser.add_argument(
            "--fail-fast",
            action="store_true",
            help="Stop at the first benchmark that fails to publish.",
        )

    @staticmethod
    def _discover_results(raw_paths: list[str]) -> list:
        import json
        from pathlib import Path

        discovered = []
        for raw in raw_paths:
            path = Path(raw).expanduser().resolve()
            if path.is_file():
                discovered.append(path)
                continue
            if not path.is_dir():
                print(f"warning: {path} does not exist, skipping", file=sys.stderr)
                continue
            for candidate in sorted(path.rglob("results.json")):
                try:
                    doc = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(doc, dict) and "leaf_artifact_paths" in doc:
                    discovered.append(candidate)
        return discovered

    @staticmethod
    def _resolve_artifact_path(recorded: str, anchor) -> str:
        from pathlib import Path

        candidate = Path(recorded)
        if candidate.is_file():
            return str(candidate)
        parts = candidate.parts
        anchor_parts = anchor.parts
        if "task_artifacts" in parts and "task_artifacts" in anchor_parts:
            tail = Path(*parts[parts.index("task_artifacts") :])
            rebuilt = Path(*anchor_parts[: anchor_parts.index("task_artifacts")]) / tail
            if rebuilt.is_file():
                return str(rebuilt)
        return recorded

    @staticmethod
    def _localize_artifact_paths(results_doc: dict, results_path) -> None:
        leaf_artifact_paths = results_doc.get("leaf_artifact_paths") or {}
        for paths in leaf_artifact_paths.values():
            for key in ("results", "samples"):
                recorded = paths.get(key)
                if recorded:
                    paths[key] = Publish._resolve_artifact_path(recorded, results_path)

    @staticmethod
    def _apply_weight_revision(config_mapping: dict, results_doc: dict) -> None:
        """Fill a missing run-metadata weight revision from the publish config.

        Artifacts written before ``weight_revision`` became run metadata only
        carry the short API alias (e.g. ``rwkv7-g1j-1.5b``), which does not
        identify the served weight's ctx_len. The config that produced the run
        still knows it: a single-model file carries it at the top level, and a
        campaign carries it on the matching ``[[models]]`` entry. Inject it
        into the in-memory results document so an existing artifact can be
        republished without rewriting it or re-running the benchmark.
        """
        run_config = results_doc.get("config") or {}
        metadata = run_config.get("metadata")
        if isinstance(metadata, dict) and metadata.get("weight_revision"):
            return

        model_args = run_config.get("model_args") or {}
        if isinstance(model_args, str):
            from lm_eval.utils import simple_parse_args_string

            model_args = simple_parse_args_string(model_args)
        if not isinstance(model_args, dict):
            model_args = {}
        model_name = (
            model_args.get("model")
            or model_args.get("pretrained")
            or run_config.get("model")
        )

        revision = None
        for entry in config_mapping.get("models") or []:
            if (
                isinstance(entry, dict)
                and entry.get("model_name") == model_name
                and entry.get("weight_revision")
            ):
                revision = entry["weight_revision"]
                break
        if revision is None:
            revision = config_mapping.get("weight_revision")
        if not isinstance(revision, str) or not revision:
            return

        if not isinstance(metadata, dict):
            metadata = {}
            run_config["metadata"] = metadata
        metadata["weight_revision"] = revision

    @staticmethod
    def _execute(args: argparse.Namespace) -> None:
        """Publish previously saved benchmark results to the scoreboard."""
        import json

        from lm_eval.config.evaluate_config import EvaluatorConfig
        from lm_eval.loggers.scoreboard import (
            PublicationConfig,
            ScoreboardError,
            ScoreboardPublisher,
            build_score,
        )

        config_mapping = EvaluatorConfig._load_config_mapping(args.config)
        publication_mapping = dict(config_mapping.get("publication") or {})
        if args.dry_run:
            publication_mapping["dry_run"] = True
        if args.rerun_reason:
            publication_mapping["rerun_reason"] = args.rerun_reason
        config = PublicationConfig.from_mapping(publication_mapping)

        results_paths = Publish._discover_results(args.results)
        if not results_paths:
            print("no benchmark results.json artifacts found", file=sys.stderr)
            sys.exit(1)

        publisher = None if config.dry_run else ScoreboardPublisher(config)
        failures = []
        for results_path in results_paths:
            try:
                results_doc = json.loads(results_path.read_text(encoding="utf-8"))
                Publish._localize_artifact_paths(results_doc, results_path)
                Publish._apply_weight_revision(config_mapping, results_doc)
                benchmark_info, score = build_score(results_doc, config)
                if config.dry_run:
                    print(
                        f"[dry-run] {benchmark_info['name']}: "
                        f"score={score['score']} num_samples={benchmark_info['num_samples']}"
                    )
                    continue
                receipt = publisher.publish(benchmark_info, score)
                status = (
                    "[replayed] idempotent replay, no new data uploaded"
                    if receipt["score"].get("replayed")
                    else "[published]"
                )
                print(
                    f"{status} {benchmark_info['name']}: "
                    f"benchmark_id={receipt['benchmark_id']} score={score['score']}"
                )
            except (
                KeyError,
                OSError,
                ScoreboardError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                failures.append(results_path)
                print(
                    f"[failed] {results_path}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    break

        if failures:
            print(
                f"{len(failures)} of {len(results_paths)} benchmark(s) failed to publish",
                file=sys.stderr,
            )
            sys.exit(1)
