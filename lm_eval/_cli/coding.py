"""RWKV-specific local coding evaluation command."""

import argparse
import json

from lm_eval._cli.subcommand import SubCommand


class Coding(SubCommand):
    def __init__(self, subparsers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "coding",
            help="Evaluate RWKV-generated Python against local tests",
            description="RWKV HTTP generation, local Docker judging, and offline raw-answer replay.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_args()
        self._parser.set_defaults(func=self._execute)

    def _add_args(self):
        self._parser.add_argument(
            "--config", "-C", required=True, help="RWKV coding TOML/YAML config"
        )
        self._parser.add_argument(
            "--phase",
            choices=["run", "generate", "score"],
            default="run",
            help="Generate and score, generate only, or replay saved samples",
        )
        self._parser.add_argument(
            "--samples", help="Saved samples.jsonl for offline scoring"
        )
        self._parser.add_argument(
            "--output-dir",
            help="Override the output directory; offline replay requires an empty directory",
        )
        self._parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume an incomplete generation with the same problem/weight/prompt identity",
        )

    def _execute(self, args):
        from lm_eval.rwkv_coding import CodingConfig, run_coding

        try:
            result = run_coding(
                CodingConfig.load(args.config),
                phase=args.phase,
                samples=args.samples,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            self._parser.exit(1, f"coding: {error}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
