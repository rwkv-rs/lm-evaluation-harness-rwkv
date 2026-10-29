import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lm_eval._cli.harness import HarnessCLI
from lm_eval._cli.ls import List
from lm_eval._cli.run import Run, _add_truncation_stats
from lm_eval._cli.utils import (
    MergeDictAction,
    _int_or_none_list_arg_type,
    check_argument_types,
    request_caching_arg_to_dict,
    try_parse_json,
)
from lm_eval._cli.validate import Validate


class TestHarnessCLI:
    """Test the main HarnessCLI class."""

    def test_harness_cli_init(self):
        """Test HarnessCLI initialization."""
        cli = HarnessCLI()
        assert cli._parser is not None
        assert cli._subparsers is not None

    def test_harness_cli_has_subcommands(self):
        """Test that HarnessCLI has all expected subcommands."""
        cli = HarnessCLI()
        subcommands = cli._subparsers.choices
        assert "run" in subcommands
        assert "ls" in subcommands
        assert "validate" in subcommands

    def test_harness_cli_backward_compatibility(self):
        """Test backward compatibility: inserting 'run' when no subcommand is provided."""
        cli = HarnessCLI()
        test_args = ["lm-eval", "--model", "hf", "--tasks", "hellaswag"]
        with patch.object(sys, "argv", test_args):
            args = cli.parse_args()
            assert args.command == "run"
            assert args.model == "hf"
            assert args.tasks == ["hellaswag"]

    def test_harness_cli_help_default(self):
        """Test that help is printed when no arguments are provided."""
        cli = HarnessCLI()
        with patch.object(sys, "argv", ["lm-eval"]):
            args = cli.parse_args()
            # The func is a lambda that calls print_help
            # Let's test it calls the help function correctly
            with patch.object(cli._parser, "print_help") as mock_help:
                args.func(args)
                mock_help.assert_called_once()

    def test_harness_cli_run_help_only(self):
        """Test that 'lm-eval run' shows help."""
        cli = HarnessCLI()
        with patch.object(sys, "argv", ["lm-eval", "run"]), pytest.raises(SystemExit):
            cli.parse_args()


class TestListCommand:
    """Test the List subcommand."""

    def test_list_command_creation(self):
        """Test List command creation."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        list_cmd = List.create(subparsers)
        assert isinstance(list_cmd, List)

    def test_list_command_arguments(self):
        """Test List command arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        List.create(subparsers)

        # Test valid arguments
        args = parser.parse_args(["ls", "tasks"])
        assert args.what == "tasks"
        assert args.include_path is None

        args = parser.parse_args(["ls", "groups", "--include_path", "/path/to/tasks"])
        assert args.what == "groups"
        assert args.include_path == "/path/to/tasks"

    def test_list_command_choices(self):
        """Test List command only accepts valid choices."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        List.create(subparsers)

        # Valid choices should work
        for choice in ["tasks", "groups", "subtasks", "tags"]:
            args = parser.parse_args(["ls", choice])
            assert args.what == choice

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["ls", "invalid"])

    @patch("lm_eval.tasks.TaskManager")
    def test_list_command_execute_tasks(self, mock_task_manager):
        """Test List command execution for tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        list_cmd = List.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.list_all_tasks.return_value = "task1\ntask2\ntask3"
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["ls", "tasks"])
        with patch("builtins.print") as mock_print:
            list_cmd._execute(args)
            mock_print.assert_called_once_with("task1\ntask2\ntask3")
            mock_tm_instance.list_all_tasks.assert_called_once_with()

    @patch("lm_eval.tasks.TaskManager")
    def test_list_command_execute_groups(self, mock_task_manager):
        """Test List command execution for groups."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        list_cmd = List.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.list_all_tasks.return_value = "group1\ngroup2"
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["ls", "groups"])
        with patch("builtins.print") as mock_print:
            list_cmd._execute(args)
            mock_print.assert_called_once_with("group1\ngroup2")
            mock_tm_instance.list_all_tasks.assert_called_once_with(
                list_subtasks=False, list_tags=False
            )


class TestRunCommand:
    """Test the Run subcommand."""

    def test_run_command_creation(self):
        """Test Run command creation."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        run_cmd = Run.create(subparsers)
        assert isinstance(run_cmd, Run)

    def test_run_command_basic_arguments(self):
        """Test Run command basic arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        args = parser.parse_args(
            ["run", "--model", "hf", "--tasks", "hellaswag", "arc_easy"]
        )
        assert args.model == "hf"
        assert args.tasks == ["hellaswag", "arc_easy"]

    def test_run_command_tasks_comma_separated(self):
        """Test Run command with comma-separated tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Comma-separated format: --tasks hellaswag,arc_easy
        args = parser.parse_args(["run", "--tasks", "hellaswag,arc_easy"])
        # argparse returns ["hellaswag,arc_easy"], splitting happens in process_tasks
        assert args.tasks == ["hellaswag", "arc_easy"]

    def test_run_command_tasks_mixed_format(self):
        """Test Run command with mixed comma and space-separated tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Mixed format: --tasks hellaswag,arc_easy winogrande
        args = parser.parse_args(["run", "--tasks", "hellaswag,arc_easy", "winogrande"])
        assert args.tasks == ["hellaswag", "arc_easy", "winogrande"]

    def test_run_command_tasks_None(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Mixed format: --tasks hellaswag,arc_easy winogrande
        args = parser.parse_args(["run", "--model", "hf"])
        assert args.tasks is None

    def test_run_command_model_args(self):
        """Test Run command model arguments parsing with MergeDictAction."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Test key=value format - MergeDictAction merges into single dict
        args = parser.parse_args(["run", "--model_args", "pretrained=gpt2,device=cuda"])
        assert args.model_args == {"pretrained": "gpt2", "device": "cuda"}

        # Test space-separated key=value pairs - also merged into single dict
        args = parser.parse_args(
            ["run", "--model_args", "pretrained=gpt2", "device=cuda"]
        )
        assert args.model_args == {"pretrained": "gpt2", "device": "cuda"}

    def test_run_command_batch_size(self):
        """Test Run command batch size arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Test integer batch size
        args = parser.parse_args(["run", "--batch_size", "32"])
        assert args.batch_size == "32"

        # Test auto batch size
        args = parser.parse_args(["run", "--batch_size", "auto"])
        assert args.batch_size == "auto"

        # Test auto with repetitions
        args = parser.parse_args(["run", "--batch_size", "auto:5"])
        assert args.batch_size == "auto:5"

    def test_run_command_seed_parsing(self):
        """Test Run command seed parsing."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Test single seed
        args = parser.parse_args(["run", "--seed", "42"])
        assert args.seed == [42, 42, 42, 42]

        # Test multiple seeds
        args = parser.parse_args(["run", "--seed", "0,1234,5678,9999"])
        assert args.seed == [0, 1234, 5678, 9999]

        # Test with None values
        args = parser.parse_args(["run", "--seed", "0,None,1234,None"])
        assert args.seed == [0, None, 1234, None]

    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.config.evaluate_config.EvaluatorConfig")
    @patch("lm_eval.loggers.EvaluationTracker")
    @patch("lm_eval.utils.make_table")
    def test_run_command_execute_basic(
        self, mock_make_table, mock_tracker, mock_config, mock_simple_evaluate
    ):
        """Test Run command basic execution."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        run_cmd = Run.create(subparsers)

        # Mock configuration
        mock_cfg_instance = MagicMock()
        mock_cfg_instance.wandb_args = None
        mock_cfg_instance.trackio_args = None
        mock_cfg_instance.output_path = None
        mock_cfg_instance.hf_hub_log_args = {}
        mock_cfg_instance.include_path = None
        mock_cfg_instance.tasks = ["hellaswag"]
        mock_cfg_instance.model = "hf"
        mock_cfg_instance.model_args = {"pretrained": "gpt2"}
        mock_cfg_instance.gen_kwargs = {}
        mock_cfg_instance.limit = None
        mock_cfg_instance.num_fewshot = 0
        mock_cfg_instance.batch_size = 1
        mock_cfg_instance.log_samples = False
        mock_cfg_instance.checkpoint = {}
        mock_cfg_instance.resume = False
        mock_cfg_instance.process_tasks.return_value = (MagicMock(), {})
        mock_config.from_cli.return_value = mock_cfg_instance

        # Mock evaluation results
        mock_simple_evaluate.return_value = {
            "results": {"hellaswag": {"acc": 0.75}},
            "config": {"batch_sizes": [1]},
            "configs": {"hellaswag": {}},
            "versions": {"hellaswag": "1.0"},
            "n-shot": {"hellaswag": 0},
        }

        # Mock make_table to avoid complex table rendering
        mock_make_table.return_value = (
            "| Task | Result |\n|------|--------|\n| hellaswag | 0.75 |"
        )

        args = parser.parse_args(["run", "--model", "hf", "--tasks", "hellaswag"])

        with patch("builtins.print"):
            run_cmd._execute(args)

        mock_config.from_cli.assert_called_once()
        mock_simple_evaluate.assert_called_once()
        mock_make_table.assert_called_once()


class TestValidateCommand:
    """Test the Validate subcommand."""

    def test_validate_command_creation(self):
        """Test Validate command creation."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        validate_cmd = Validate.create(subparsers)
        assert isinstance(validate_cmd, Validate)

    def test_validate_command_arguments(self):
        """Test Validate command arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Validate.create(subparsers)

        args = parser.parse_args(["validate", "--tasks", "hellaswag,arc_easy"])
        assert args.tasks == "hellaswag,arc_easy"
        assert args.include_path is None

        args = parser.parse_args(
            ["validate", "--tasks", "custom_task", "--include_path", "/path/to/tasks"]
        )
        assert args.tasks == "custom_task"
        assert args.include_path == "/path/to/tasks"

    def test_validate_command_requires_tasks(self):
        """Test Validate command requires tasks argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Validate.create(subparsers)

        with pytest.raises(SystemExit):
            parser.parse_args(["validate"])

    @patch("lm_eval.tasks.TaskManager")
    def test_validate_command_execute_success(self, mock_task_manager):
        """Test Validate command execution with valid tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        validate_cmd = Validate.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.return_value = ["hellaswag", "arc_easy"]
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["validate", "--tasks", "hellaswag,arc_easy"])

        with patch("builtins.print") as mock_print:
            validate_cmd._execute(args)

        mock_print.assert_any_call("Validating tasks: ['hellaswag', 'arc_easy']")
        mock_print.assert_any_call("All tasks found and valid")

    @patch("lm_eval.tasks.TaskManager")
    def test_validate_command_execute_missing_tasks(self, mock_task_manager):
        """Test Validate command execution with missing tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        validate_cmd = Validate.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.return_value = ["hellaswag"]
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["validate", "--tasks", "hellaswag,nonexistent"])

        with (
            patch("builtins.print") as mock_print,
            pytest.raises(SystemExit) as exc_info,
        ):
            validate_cmd._execute(args)

        assert exc_info.value.code == 1
        mock_print.assert_any_call("Tasks not found: nonexistent")


class TestEvaluatorConfigTaskLoading:
    """Test EvaluatorConfig task loading."""

    @patch("lm_eval.tasks.TaskManager")
    def test_non_rwkv_model_uses_native_task_index(self, mock_task_manager):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.return_value = ["race"]
        mock_task_manager.return_value = mock_tm_instance
        cfg = EvaluatorConfig(model="hf", tasks=["race"])
        cfg._configure()
        cfg.process_tasks()

        mock_task_manager.assert_called_once_with(
            include_path=None,
            include_defaults=True,
            metadata={},
        )

    def test_truncation_stats_are_per_generated_sample(self):
        class Generation(str):
            finish_reason = "length"

        results = {"config": {}}
        samples = {
            "race": [{"resps": [[(-1.0, True)]]}],
            "drop": [
                {"resps": [[Generation("truncated")]]},
                {"resps": [["complete"]]},
            ],
        }

        _add_truncation_stats(results, samples)

        assert results["config"]["truncation"] == {
            "race": {
                "generated_samples": 0,
                "truncated_samples": 0,
                "truncation_rate": None,
            },
            "drop": {
                "generated_samples": 1,
                "truncated_samples": 1,
                "truncation_rate": 1.0,
            },
        }
        assert samples["drop"][0]["finish_reasons"] == ["length"]
        assert samples["drop"][0]["truncated"] is True

    def test_truncation_stats_support_local_completions(self):
        class Generation(str):
            finish_reason = "length"

        results = {"config": {}}
        samples = {"task": [{"resps": [Generation("unfinished")]}]}

        _add_truncation_stats(results, samples)

        assert results["config"]["truncation"] == {
            "task": {
                "generated_samples": 1,
                "truncated_samples": 1,
                "truncation_rate": 1.0,
            }
        }
        assert samples["task"][0]["truncated"] is True

    @patch("lm_eval.tasks.TaskManager")
    def test_process_tasks_comma_separated_in_list(self, mock_task_manager):
        """Test that process_tasks splits comma-separated tasks from CLI (nargs='*')."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.side_effect = lambda x: x
        mock_task_manager.return_value = mock_tm_instance

        # Simulate CLI input: --tasks hellaswag,arc_easy
        # argparse with nargs="*" gives ["hellaswag,arc_easy"]
        cfg = EvaluatorConfig(tasks=["hellaswag,arc_easy"])
        cfg._configure()
        cfg.process_tasks()

        # Should have split the comma-separated string
        assert mock_tm_instance.match_tasks.call_count == 2
        mock_tm_instance.match_tasks.assert_any_call(["hellaswag"])
        mock_tm_instance.match_tasks.assert_any_call(["arc_easy"])

    @patch("lm_eval.tasks.TaskManager")
    def test_process_tasks_mixed_comma_and_space_separated(self, mock_task_manager):
        """Test process_tasks handles mixed comma and space-separated tasks."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.side_effect = lambda x: x
        mock_task_manager.return_value = mock_tm_instance

        # Simulate CLI input: --tasks hellaswag,arc_easy winogrande
        # argparse with nargs="*" gives ["hellaswag,arc_easy", "winogrande"]
        cfg = EvaluatorConfig(tasks=["hellaswag,arc_easy", "winogrande"])
        cfg._configure()
        cfg.process_tasks()

        # Should have split comma-separated and kept space-separated
        assert mock_tm_instance.match_tasks.call_count == 3
        mock_tm_instance.match_tasks.assert_any_call(["hellaswag"])
        mock_tm_instance.match_tasks.assert_any_call(["arc_easy"])
        mock_tm_instance.match_tasks.assert_any_call(["winogrande"])

    @patch("lm_eval.tasks.TaskManager")
    def test_process_tasks_string_comma_separated(self, mock_task_manager):
        """Test process_tasks splits comma-separated string (from YAML)."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.side_effect = lambda x: x
        mock_task_manager.return_value = mock_tm_instance

        # Simulate YAML input: tasks: "hellaswag,arc_easy"
        cfg = EvaluatorConfig(tasks="hellaswag,arc_easy")
        cfg._configure()
        cfg.process_tasks()

        # Should have split the comma-separated string
        assert mock_tm_instance.match_tasks.call_count == 2
        mock_tm_instance.match_tasks.assert_any_call(["hellaswag"])
        mock_tm_instance.match_tasks.assert_any_call(["arc_easy"])

    def test_custom_yaml_file_relative_path(self, tmp_path):
        """Test loading custom task config via a relative path (fixes #3425)."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        # Create a minimal valid task yaml
        task_yaml = tmp_path / "test_task.yaml"
        task_yaml.write_text("""
task: test_custom_task
dataset_path: hellaswag
output_type: multiple_choice
doc_to_text: "{{question}}"
doc_to_target: "{{answer}}"
""")

        # Test with relative-style path
        cfg = EvaluatorConfig(
            tasks=[str(task_yaml)],
            output_path=str(tmp_path),
        )
        cfg._configure()
        tm = cfg.process_tasks()  # noqa: F841

        # Should load successfully as a dict config, not raise "Tasks not found"
        assert len(cfg.tasks) == 1
        assert isinstance(cfg.tasks[0], dict)
        assert cfg.tasks[0]["task"] == "test_custom_task"

    def test_missing_yaml_file_raises_error(self, tmp_path):
        """Test that non-existent yaml file raises proper error."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        cfg = EvaluatorConfig(
            tasks=[str(tmp_path / "nonexistent.yaml")],
            output_path=str(tmp_path),
        )
        cfg._configure()

        with pytest.raises(ValueError, match="Tasks not found"):
            cfg.process_tasks()


class TestEvaluatorConfigFromCLI:
    """Test EvaluatorConfig.from_cli defaults and argument handling."""

    def test_defaults_applied(self, tmp_path):
        """Test that dataclass defaults are applied when CLI args are missing."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        # Minimal namespace with only required fields
        ns = Namespace(
            tasks=["hellaswag"],
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        # Check defaults from dataclass
        assert cfg.model == "hf"
        assert cfg.batch_size == 1
        assert cfg.device == "cuda:0"
        assert cfg.num_fewshot is None
        assert cfg.limit is None
        assert cfg.seed == [0, 1234, 1234, 1234]
        assert cfg.trust_remote_code is False
        assert cfg.apply_chat_template is False

    def test_cli_args_override_defaults(self, tmp_path):
        """Test that CLI args override dataclass defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["arc_easy"],
            model="vllm",
            batch_size=16,
            device="cuda:1",
            num_fewshot=5,
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.model == "vllm"
        assert cfg.batch_size == 16
        assert cfg.device == "cuda:1"
        assert cfg.num_fewshot == 5

    def test_model_args_dict_passed_through(self, tmp_path):
        """Test that model_args dict is passed through correctly."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            model_args={"pretrained": "gpt2", "dtype": "float16"},
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.model_args["pretrained"] == "gpt2"
        assert cfg.model_args["dtype"] == "float16"

    def test_gen_kwargs_passed_through(self, tmp_path):
        """Test that gen_kwargs dict is passed through correctly."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            gen_kwargs={"temperature": 0.7, "max_tokens": 100},
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)
        assert cfg.gen_kwargs is not None
        assert cfg.gen_kwargs["temperature"] == 0.7
        assert cfg.gen_kwargs["max_tokens"] == 100

    def test_none_args_use_defaults(self, tmp_path):
        """Test that None values fall back to defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            model=None,  # Should use default "hf"
            batch_size=None,  # Should use default 1
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.model == "hf"
        assert cfg.batch_size == 1

    def test_fewshot_as_multiturn_defaults_with_chat_template(self, tmp_path):
        """Test fewshot_as_multiturn defaults to True when apply_chat_template is set."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            apply_chat_template=True,
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.fewshot_as_multiturn is True

    def test_empty_tasks_allowed_at_config_level(self):
        """Test that empty tasks passes config validation (fails later in process_tasks)."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        # tasks=None gets filtered out, default [] is used
        ns = Namespace(tasks=None)

        cfg = EvaluatorConfig.from_cli(ns)
        assert cfg.tasks == []  # Empty list, not None

    def test_validation_error_log_samples_without_output(self):
        """Test that log_samples without output_path raises ValueError."""
        from argparse import Namespace

        import pytest

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            log_samples=True,
            output_path=None,
        )

        with pytest.raises(ValueError, match="output_path"):
            EvaluatorConfig.from_cli(ns)


class TestCLIUtils:
    """Test CLI utility functions."""

    def test_try_parse_json_with_json_string(self):
        """Test try_parse_json with a valid JSON string."""
        result = try_parse_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_try_parse_json_with_dict(self):
        """Test try_parse_json with dict input."""
        input_dict = {"key": "value"}
        result = try_parse_json(input_dict)
        assert result is input_dict

    def test_try_parse_json_with_none(self):
        """Test try_parse_json with None."""
        result = try_parse_json(None)
        assert result is None

    def test_try_parse_json_with_plain_string(self):
        """Test try_parse_json with a plain string."""
        result = try_parse_json("key=value,key2=value2")
        assert result == "key=value,key2=value2"

    def test_try_parse_json_with_invalid_json(self):
        """Test try_parse_json with invalid JSON."""
        with pytest.raises(ValueError) as exc_info:
            try_parse_json('{key: "value"}')  # Invalid JSON (unquoted key)
        assert "Invalid JSON" in str(exc_info.value)
        assert "double quotes" in str(exc_info.value)

    def test_int_or_none_list_single_value(self):
        """Test _int_or_none_list_arg_type with a single value."""
        result = _int_or_none_list_arg_type(3, 4, "0,1,2,3", "42")
        assert result == [42, 42, 42, 42]

    def test_int_or_none_list_multiple_values(self):
        """Test _int_or_none_list_arg_type with multiple values."""
        result = _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,20,30,40")
        assert result == [10, 20, 30, 40]

    def test_int_or_none_list_with_none(self):
        """Test _int_or_none_list_arg_type with None values."""
        result = _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,None,30,None")
        assert result == [10, None, 30, None]

    def test_int_or_none_list_invalid_value(self):
        """Test _int_or_none_list_arg_type with invalid value."""
        with pytest.raises(ValueError):
            _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,invalid,30,40")

    def test_int_or_none_list_too_few_values(self):
        """Test _int_or_none_list_arg_type with too few values."""
        with pytest.raises(ValueError):
            _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,20")

    def test_int_or_none_list_too_many_values(self):
        """Test _int_or_none_list_arg_type with too many values."""
        with pytest.raises(ValueError):
            _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,20,30,40,50")

    def test_request_caching_arg_to_dict_none(self):
        """Test request_caching_arg_to_dict with None."""
        result = request_caching_arg_to_dict(None)
        assert result == {}

    def test_request_caching_arg_to_dict_true(self):
        """Test request_caching_arg_to_dict with 'true'."""
        result = request_caching_arg_to_dict("true")
        assert result == {
            "cache_requests": True,
            "rewrite_requests_cache": False,
            "delete_requests_cache": False,
        }

    def test_request_caching_arg_to_dict_refresh(self):
        """Test request_caching_arg_to_dict with 'refresh'."""
        result = request_caching_arg_to_dict("refresh")
        assert result == {
            "cache_requests": True,
            "rewrite_requests_cache": True,
            "delete_requests_cache": False,
        }

    def test_request_caching_arg_to_dict_delete(self):
        """Test request_caching_arg_to_dict with 'delete'."""
        result = request_caching_arg_to_dict("delete")
        assert result == {
            "cache_requests": False,
            "rewrite_requests_cache": False,
            "delete_requests_cache": True,
        }

    def test_request_caching_arg_to_dict_invalid(self):
        """Test request_caching_arg_to_dict rejects invalid values."""
        with pytest.raises(argparse.ArgumentTypeError):
            request_caching_arg_to_dict("bogus")

    def test_cache_requests_argparse_integration(self):
        """Test --cache_requests works end-to-end through argparse.

        Regression test: the `type` function converts the string to a dict
        before `choices` validation, so `choices` must not be used alongside
        `type=request_caching_arg_to_dict`.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cache_requests",
            type=request_caching_arg_to_dict,
            default=None,
        )
        for val in ("true", "refresh", "delete"):
            args = parser.parse_args(["--cache_requests", val])
            assert isinstance(args.cache_requests, dict)

    def test_check_argument_types_raises_on_untyped(self):
        """Test check_argument_types raises error for untyped arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--untyped")  # No type specified

        with pytest.raises(ValueError) as exc_info:
            check_argument_types(parser)
        assert "untyped" in str(exc_info.value)
        assert "doesn't have a type specified" in str(exc_info.value)

    def test_check_argument_types_passes_on_typed(self):
        """Test check_argument_types passes for typed arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--typed", type=str)

        # Should not raise
        check_argument_types(parser)

    def test_check_argument_types_skips_const_actions(self):
        """Test check_argument_types skips const actions."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", action="store_const", const=True)

        # Should not raise
        check_argument_types(parser)


class TestMergeDictAction:
    """Test MergeDictAction for parsing key=value arguments."""

    def test_comma_separated_key_value(self):
        """Test parsing comma-separated key=value pairs."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key1=val1,key2=val2"])
        assert args.args == {"key1": "val1", "key2": "val2"}

    def test_space_separated_key_value(self):
        """Test parsing space-separated key=value pairs."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key1=val1", "key2=val2"])
        assert args.args == {"key1": "val1", "key2": "val2"}

    def test_json_dict_input(self):
        """Test parsing JSON dict input."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", '{"key1": "val1", "key2": 42}'])
        assert args.args == {"key1": "val1", "key2": 42}

    def test_json_nested_dict(self):
        """Test parsing nested JSON dict."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", '{"outer": {"inner": "value"}}'])
        assert args.args == {"outer": {"inner": "value"}}

    def test_empty_values(self):
        """Test that empty values result in None."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="*", action=MergeDictAction)

        args = parser.parse_args(["--args"])
        assert args.args is None

    def test_type_coercion(self):
        """Test that values are coerced to appropriate types."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "num=42,flag=true,pi=3.14"])
        assert args.args["num"] == 42
        assert args.args["flag"] is True
        assert args.args["pi"] == 3.14

    def test_multiple_invocations_merge(self):
        """Test that multiple --args invocations merge values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key1=val1", "--args", "key2=val2"])
        assert args.args == {"key1": "val1", "key2": "val2"}

    def test_key_overwrite(self):
        """Test that later values overwrite earlier ones."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key=first", "--args", "key=second"])
        assert args.args["key"] == "second"


class TestEvaluatorConfigPrecedence:
    """Test EvaluatorConfig merging precedence: CLI > file > defaults."""

    def test_rwkv_preset_lists_all_validated_benchmarks(self):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config_path = Path(__file__).parents[1] / "configs/eval/lm_eval.toml"
        config = EvaluatorConfig.from_config(config_path)

        assert config.tasks == [
            "race",
            "drop",
            "xquad",
            "babilong",
            "infinitebench",
            "ruler",
            "wmdp",
            "cruxeval",
            "inverse_scaling_prize",
            "model_written_evals",
            "humaneval_infilling",
            "mutual",
            "mc_taco",
            "discrim_eval",
            "winogender",
        ]

    def test_toml_config_and_relative_include(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        base = tmp_path / "base.toml"
        base.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race", "drop"]
output_dir = "results/formal"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )
        smoke = tmp_path / "smoke.toml"
        smoke.write_text(
            """
include = "base.toml"
benchmarks = ["race"]
limit = 2
output_dir = "results/smoke"

[rwkv_profile]
generation_prompt = "open_think"
""",
            encoding="utf-8",
        )

        cfg = EvaluatorConfig.from_config(smoke)

        assert cfg.model == "rwkv7-http"
        assert cfg.tasks == ["race"]
        assert cfg.limit == 2
        assert cfg.output_path == "results/smoke"
        assert cfg.model_args == {
            "model": "rwkv7-g1i-1.5b-20260805-ctx16384",
            "base_url": "http://127.0.0.1:8000/v1/completions",
            "rwkv_prompt_template": "assistant",
            "rwkv_generation_prompt": "open_think",
            "num_concurrent": 5,
            "max_length": 16384,
        }
        assert cfg.metadata["wkv_mode"] == "fp32io16"
        assert cfg.metadata["cot_mode"] == "open_think"

    def test_toml_config_rwkv7_remote_backend(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1i-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/rwkv7-g1i-1.5b-remote"
max_length = 16384
max_num_seqs = 5
apply_chat_template = false
fewshot_as_multiturn = false

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        cfg = EvaluatorConfig.from_config(config)

        assert cfg.model == "rwkv7-remote"
        assert cfg.model_args == {
            "model": "rwkv7-g1i-1.5b",
            "base_url": "https://api.rwkv.rs/v1/completions",
            "num_concurrent": 5,
            "max_length": 16384,
            "seed": 1234,
            "max_gen_toks": 8192,
            "rwkv_prompt_template": "assistant",
            "rwkv_generation_prompt": "fake_think",
        }
        assert cfg.metadata["wkv_mode"] == "fp32io16"
        assert cfg.metadata["cot_mode"] == "fake_think"

    def test_toml_config_rwkv7_remote_backend_with_tokenizer_name(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1i-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/rwkv7-g1i-1.5b-remote"
max_length = 16384
max_num_seqs = 5
apply_chat_template = false
fewshot_as_multiturn = false
tokenizer_name = "/opt/rwkv/tokenizers/rwkv7-g1i-1.5b"

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        cfg = EvaluatorConfig.from_config(config)

        assert cfg.model_args == {
            "model": "rwkv7-g1i-1.5b",
            "base_url": "https://api.rwkv.rs/v1/completions",
            "num_concurrent": 5,
            "max_length": 16384,
            "seed": 1234,
            "max_gen_toks": 8192,
            "tokenizer": "/opt/rwkv/tokenizers/rwkv7-g1i-1.5b",
            "rwkv_prompt_template": "assistant",
            "rwkv_generation_prompt": "fake_think",
        }

    def test_toml_config_rwkv7_remote_tokenizer_name_with_trust_remote_code(
        self, tmp_path
    ):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1j-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/rwkv7-g1j-1.5b-remote"
max_length = 16384
max_num_seqs = 5
apply_chat_template = false
fewshot_as_multiturn = false
tokenizer_name = "BBuf/rwkv-5-world-1b5"
trust_remote_code = true

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        cfg = EvaluatorConfig.from_config(config)

        assert cfg.model_args == {
            "model": "rwkv7-g1j-1.5b",
            "base_url": "https://api.rwkv.rs/v1/completions",
            "num_concurrent": 5,
            "max_length": 16384,
            "seed": 1234,
            "max_gen_toks": 8192,
            "tokenizer": "BBuf/rwkv-5-world-1b5",
            "trust_remote_code": True,
            "rwkv_prompt_template": "assistant",
            "rwkv_generation_prompt": "fake_think",
        }

    def test_versioned_config_rejects_trust_remote_code_without_tokenizer_name(
        self, tmp_path
    ):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1j-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/rwkv7-g1j-1.5b-remote"
max_length = 16384
max_num_seqs = 5
apply_chat_template = false
fewshot_as_multiturn = false
trust_remote_code = true

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="trust_remote_code requires tokenizer_name"
        ):
            EvaluatorConfig.from_config(config)

    def test_versioned_config_rejects_tokenizer_name_for_rwkv7_http(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 5
tokenizer_name = "some/repo"

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="tokenizer_name is only supported for backend = rwkv7-remote",
        ):
            EvaluatorConfig.from_config(config)

    def test_versioned_config_rejects_unknown_backend(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-vllm"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="backend must be one of: rwkv7-http, rwkv7-remote"
        ):
            EvaluatorConfig.load_config(config)

    def test_toml_config_include_cycle_is_rejected(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        first = tmp_path / "first.toml"
        second = tmp_path / "second.toml"
        first.write_text('include = "second.toml"', encoding="utf-8")
        second.write_text('include = "first.toml"', encoding="utf-8")

        with pytest.raises(ValueError, match="include cycle"):
            EvaluatorConfig.load_config(first)

    def test_versioned_config_rejects_unknown_fields(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 5
model_args = "deprecated"

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unknown versioned config fields"):
            EvaluatorConfig.load_config(config)

    def test_versioned_config_derives_request_budget_from_max_num_seqs(self, tmp_path):
        """The endpoint's declared capacity is the evaluator's request budget.

        Mirrors lighteval-rwkv, where the model client is built with
        ``concurrent_requests = sum(max_num_seqs)`` instead of a hand-tuned
        request concurrency.
        """
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 1024

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        cfg = EvaluatorConfig.from_config(config)

        assert cfg.model_args["num_concurrent"] == 1024

    def test_versioned_config_requires_max_num_seqs(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="Missing versioned config fields: max_num_seqs"
        ):
            EvaluatorConfig.from_config(config)

    def test_versioned_config_requires_base_url(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-1.5b"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 64

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="Missing versioned config fields: base_url"
        ):
            EvaluatorConfig.from_config(config)

    @pytest.mark.parametrize(
        "extra, unknown_field",
        [
            ("num_concurrent = 5\n", "num_concurrent"),
            ('pool_manifest = "pools/rwkv7_pool.toml"\n', "pool_manifest"),
            ("[scheduler]\nmax_concurrent_benchmarks = 2\n", "scheduler"),
        ],
    )
    def test_versioned_config_rejects_manual_concurrency_knobs(
        self, tmp_path, extra, unknown_field
    ):
        """Hand-tuned request concurrency and benchmark scheduling are gone."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 64
"""
            + extra
            + """
[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match=f"Unknown versioned config fields: {unknown_field}"
        ):
            EvaluatorConfig.from_config(config)

    def test_versioned_config_rejects_duplicate_benchmarks(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race", "race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="duplicate benchmarks: race"):
            EvaluatorConfig.load_config(config)

    def test_versioned_config_accepts_explicit_samples(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["graphwalks"]
output_dir = "results/graphwalks"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"

[samples]
rwkv_graphwalks = [0, 4, 9]
""",
            encoding="utf-8",
        )

        loaded = EvaluatorConfig.load_config(config)

        assert loaded["samples"] == {"rwkv_graphwalks": [0, 4, 9]}

    def test_versioned_config_accepts_targeted_task_roots(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["graphwalks"]
output_dir = "results/graphwalks"
max_length = 16384
max_num_seqs = 5
include_defaults = false
include_path = ["lm_eval/tasks/graphwalks", "lm_eval/tasks/rwkv_adapter"]

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        loaded = EvaluatorConfig.load_config(config)

        assert loaded["include_defaults"] is False
        assert loaded["include_path"] == [
            "lm_eval/tasks/graphwalks",
            "lm_eval/tasks/rwkv_adapter",
        ]

    @pytest.mark.parametrize("indices", ["[]", "[0, 0]", "[-1]"])
    def test_versioned_config_rejects_invalid_samples(self, tmp_path, indices):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            f"""
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["graphwalks"]
output_dir = "results/graphwalks"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"

[samples]
rwkv_graphwalks = {indices}
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="samples.rwkv_graphwalks"):
            EvaluatorConfig.load_config(config)

    def test_versioned_config_preserves_benchmark_order(self, tmp_path, monkeypatch):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        class OrderedTaskManager:
            def __init__(self, **_kwargs):
                pass

            @staticmethod
            def match_tasks(tasks):
                return tasks

        monkeypatch.setattr("lm_eval.tasks.TaskManager", OrderedTaskManager)

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race", "drop", "xquad"]
output_dir = "results/formal"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        loaded = EvaluatorConfig.from_config(config)
        loaded.process_tasks()

        assert loaded.tasks == ["race", "drop", "xquad"]

    @pytest.mark.parametrize(
        "field",
        ["doc_to_text", "filter_list", "metric_list", "until", "max_tokens"],
    )
    def test_versioned_config_rejects_task_protocol_fields(self, tmp_path, field):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            f"""
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 5
{field} = "task-owned"

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match=f"Unknown versioned config fields: {field}"
        ):
            EvaluatorConfig.load_config(config)

    def test_versioned_config_rejects_unknown_profile_fields(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "config.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race"]
output_dir = "results/race"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
metric = "exact_match"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unknown RWKV profile fields: metric"):
            EvaluatorConfig.load_config(config)

    def test_versioned_config_normalizes_publication_settings(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "publication.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race", "drop"]
output_dir = "results/publication"
max_length = 16384
max_num_seqs = 5

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"

[publication]
enabled = true
base_url = "https://eval.rwkv.rs/test"
token_env = "MY_PUBLICATION_TOKEN"
control_timeout = 15.0
retries = 4
retry_delay = 0.5
cot_mode = "NoCoT"
rerun_reason = "protocol correction"

[publication.tasks.race]
field = "knowledge"
avg_k = 1
""",
            encoding="utf-8",
        )

        loaded = EvaluatorConfig.from_config(config)

        assert loaded.publication == {
            "enabled": True,
            "base_url": "https://eval.rwkv.rs/test",
            "token_env": "MY_PUBLICATION_TOKEN",
            "control_timeout": 15.0,
            "retries": 4,
            "retry_delay": 0.5,
            "cot_mode": "NoCoT",
            "rerun_reason": "protocol correction",
            "tasks": {"race": {"field": "knowledge", "avg_k": 1}},
        }

    def test_versioned_config_requires_samples_for_publication(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        config = tmp_path / "publication-without-samples.toml"
        config.write_text(
            """
schema_version = 1
backend = "rwkv7-http"
model_name = "rwkv7-g1i-1.5b-20260805-ctx16384"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = ["race"]
output_dir = "results/publication"
max_length = 16384
max_num_seqs = 5
log_samples = false

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"

[publication]
enabled = true
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="requires log_samples"):
            EvaluatorConfig.from_config(config)

    def test_cli_overrides_yaml_overrides_defaults(self, tmp_path):
        """Test full precedence chain: CLI args > YAML config > built-in defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        # Create a YAML config file
        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("""
model: vllm
batch_size: 8
device: cuda:1
num_fewshot: 3
tasks:
  - hellaswag
output_path: /yaml/path
log_samples: true
""")

        # CLI args: override some YAML values, leave others to YAML/defaults
        ns = Namespace(
            config=str(yaml_config),
            model="openai",  # CLI overrides YAML's "vllm"
            batch_size=32,  # CLI overrides YAML's 8
            # device not specified in CLI -> should use YAML's cuda:1
            # num_fewshot not specified in CLI -> should use YAML's 3
            tasks=None,  # falsy, should use YAML's hellaswag
            output_path=None,  # falsy, should use YAML's /yaml/path
            log_samples=None,  # falsy, should use YAML's true
        )

        cfg = EvaluatorConfig.from_cli(ns)

        # CLI values win
        assert cfg.model == "openai", "CLI should override YAML"
        assert cfg.batch_size == 32, "CLI should override YAML"

        # YAML values win over defaults
        assert cfg.device == "cuda:1", "YAML should override default (cuda:0)"
        assert cfg.num_fewshot == 3, "YAML should override default (None)"
        assert cfg.tasks == ["hellaswag"], "YAML should be used when CLI is falsy"
        assert cfg.output_path == "/yaml/path", "YAML should be used when CLI is falsy"

        # Defaults used when neither CLI nor YAML specify
        assert cfg.trust_remote_code is False, "Should use default"
        assert cfg.seed == [0, 1234, 1234, 1234], "Should use default"

    def test_yaml_overrides_defaults(self, tmp_path):
        """Test that YAML config values override built-in defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("""
model: vllm
batch_size: 16
device: cpu
seed: [42, 42, 42, 42]
tasks:
  - arc_easy
output_path: /tmp/out
log_samples: true
""")

        ns = Namespace(config=str(yaml_config))

        cfg = EvaluatorConfig.from_cli(ns)

        # All should come from YAML, not defaults
        assert cfg.model == "vllm", "YAML should override default 'hf'"
        assert cfg.batch_size == 16, "YAML should override default 1"
        assert cfg.device == "cpu", "YAML should override default 'cuda:0'"
        assert cfg.seed == [42, 42, 42, 42], "YAML should override default seed"

    def test_cli_overrides_yaml_with_explicit_zero(self, tmp_path):
        """Test that explicit CLI value 0 overrides YAML."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("""
model: vllm
batch_size: 16
num_fewshot: 5
tasks:
  - hellaswag
output_path: /yaml/path
log_samples: true
""")

        ns = Namespace(
            config=str(yaml_config),
            num_fewshot=0,  # Explicit 0 should override YAML's 5
            batch_size=1,  # Explicit 1 should override
            tasks=None,
            output_path=None,
            log_samples=None,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        # 0 is a valid explicit value and should override YAML
        assert cfg.num_fewshot == 0, "CLI 0 should override YAML's 5"
        assert cfg.batch_size == 1, "Truthy CLI value overrides YAML"


CAMPAIGN_TOML = """
schema_version = 1
backend = "rwkv7-remote"

base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["mutual"]
max_length = 25600
max_num_seqs = 100
log_samples = true

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "open_think"
wkv_mode = "fp32io16"

[[models]]
model_name = "rwkv7-g1k-1.5b"
tokenizer_name = "/weights/1.5b"
output_dir = "results/1.5b"

[[models]]
model_name = "rwkv7-g1k-7.2b"
tokenizer_name = "/weights/7.2b"
output_dir = "results/7.2b"
max_num_seqs = 20
"""


def _write_campaign(tmp_path, text=CAMPAIGN_TOML, name="campaign.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestEvaluatorConfigCampaignModels:
    """One campaign TOML covers the whole model ladder."""

    def test_campaign_expands_to_one_config_per_model(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        configs = EvaluatorConfig.load_configs(_write_campaign(tmp_path))

        assert [cfg["metadata"]["model_name"] for cfg in configs] == [
            "rwkv7-g1k-1.5b",
            "rwkv7-g1k-7.2b",
        ]
        assert [cfg["output_path"] for cfg in configs] == [
            "results/1.5b",
            "results/7.2b",
        ]
        # Shared settings come from the top-level table, per-model deviations
        # from the entry itself.
        assert [cfg["model_args"]["num_concurrent"] for cfg in configs] == [100, 20]
        assert [cfg["model_args"]["tokenizer"] for cfg in configs] == [
            "/weights/1.5b",
            "/weights/7.2b",
        ]
        assert all(cfg["tasks"] == ["mutual"] for cfg in configs)
        assert all(cfg["metadata"]["cot_mode"] == "open_think" for cfg in configs)

    def test_model_overrides_beat_every_entry(self, tmp_path):
        """Launch-time overlays own the campaign-wide knobs."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        _write_campaign(tmp_path)
        overlay = tmp_path / "overlay.toml"
        overlay.write_text(
            """
include = ["campaign.toml"]

[model_overrides]
max_num_seqs = 200
timeout = 1800
max_retries = 6
""",
            encoding="utf-8",
        )

        configs = EvaluatorConfig.load_configs(overlay)

        assert [cfg["model_args"]["num_concurrent"] for cfg in configs] == [200, 200]
        assert all(cfg["model_args"]["timeout"] == 1800 for cfg in configs)
        assert all(cfg["model_args"]["max_retries"] == 6 for cfg in configs)

    def test_campaign_requires_distinct_output_dirs(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(
            tmp_path,
            CAMPAIGN_TOML.replace(
                'output_dir = "results/7.2b"', 'output_dir = "results/1.5b"'
            ),
        )

        with pytest.raises(ValueError, match="each model needs its own output_dir"):
            EvaluatorConfig.load_configs(campaign)

    def test_campaign_requires_distinct_model_names(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(
            tmp_path,
            CAMPAIGN_TOML.replace(
                'model_name = "rwkv7-g1k-7.2b"', 'model_name = "rwkv7-g1k-1.5b"'
            ),
        )

        with pytest.raises(ValueError, match="duplicate model_name in models"):
            EvaluatorConfig.load_configs(campaign)

    def test_campaign_entry_must_name_its_model(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(
            tmp_path, CAMPAIGN_TOML.replace('model_name = "rwkv7-g1k-7.2b"\n', "")
        )

        with pytest.raises(ValueError, match=r"models\[1\] must set model_name"):
            EvaluatorConfig.load_configs(campaign)

    def test_campaign_entry_rejects_unknown_fields(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(tmp_path, CAMPAIGN_TOML + "\nmax_lengths = 4096\n")

        with pytest.raises(
            ValueError, match=r"Unknown models\[1\] fields: max_lengths"
        ):
            EvaluatorConfig.load_configs(campaign)

    def test_campaign_entry_errors_name_the_entry(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(tmp_path, CAMPAIGN_TOML + "\nmax_length = 0\n")

        with pytest.raises(ValueError, match=r"models\[1\]: max_length must be"):
            EvaluatorConfig.load_configs(campaign)

    def test_model_overrides_requires_a_models_array(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(
            tmp_path,
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["mutual"]
output_dir = "results/1.5b"
max_length = 25600
max_num_seqs = 100

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "open_think"
wkv_mode = "fp32io16"

[model_overrides]
max_num_seqs = 200
""",
        )

        with pytest.raises(ValueError, match="model_overrides requires a"):
            EvaluatorConfig.load_configs(campaign)

    def test_models_require_the_versioned_schema(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(
            tmp_path,
            """
model = "rwkv7-remote"
tasks = ["mutual"]

[[models]]
model_name = "rwkv7-g1k-1.5b"
""",
        )

        with pytest.raises(ValueError, match="models requires the versioned"):
            EvaluatorConfig.load_configs(campaign)

    def test_load_config_needs_a_selection_for_a_campaign(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(tmp_path)

        with pytest.raises(ValueError, match="declares 2 models"):
            EvaluatorConfig.load_config(campaign)

    def test_only_model_selects_a_declared_model(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(tmp_path)

        cfg = EvaluatorConfig.load_config(campaign, only_model="rwkv7-g1k-7.2b")

        assert cfg["metadata"]["model_name"] == "rwkv7-g1k-7.2b"
        assert cfg["output_path"] == "results/7.2b"
        assert cfg["model_args"]["num_concurrent"] == 20

    def test_only_model_rejects_an_unknown_model(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(tmp_path)

        with pytest.raises(ValueError, match="is not declared by this config"):
            EvaluatorConfig.load_config(campaign, only_model="rwkv7-g1k-13.3b")

    def test_only_model_requires_a_campaign_config(self, tmp_path):
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(
            tmp_path,
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["mutual"]
output_dir = "results/1.5b"
max_length = 25600
max_num_seqs = 100

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "open_think"
wkv_mode = "fp32io16"
""",
        )

        with pytest.raises(ValueError, match=r"requires a versioned config"):
            EvaluatorConfig.load_config(campaign, only_model="rwkv7-g1k-1.5b")

    def test_campaign_entry_resolves_like_the_equivalent_single_model_file(
        self, tmp_path
    ):
        """A campaign entry must reproduce the per-model file it replaces.

        The checkpoint fingerprint is derived from the resolved config, so this
        equality is what keeps an in-flight run resumable after per-model files
        collapse into one campaign file.
        """
        from dataclasses import asdict

        from lm_eval.api.checkpoint import (
            compute_config_fingerprint,
            without_operational_args,
        )
        from lm_eval.config.evaluate_config import EvaluatorConfig

        campaign = _write_campaign(tmp_path)
        single = tmp_path / "single.toml"
        single.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-7.2b"
base_url = "https://api.rwkv.rs/v1/completions"
tokenizer_name = "/weights/7.2b"
benchmarks = ["mutual"]
output_dir = "results/7.2b"
max_length = 25600
max_num_seqs = 20
log_samples = true

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "open_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )

        def resolved(source):
            return asdict(EvaluatorConfig(**source)._configure())

        def fingerprint(source):
            payload = resolved(source)
            payload["model_args"] = without_operational_args(payload["model_args"])
            payload["metadata"] = without_operational_args(payload["metadata"])
            for key in ("checkpoint", "publication", "resume"):
                payload.pop(key, None)
            return compute_config_fingerprint(payload)

        entry = EvaluatorConfig.load_config(campaign, only_model="rwkv7-g1k-7.2b")
        entry.pop("config", None)
        single_config = EvaluatorConfig.load_config(single)

        assert resolved(entry)["model_args"] == resolved(single_config)["model_args"]
        assert fingerprint(entry) == fingerprint(single_config)


class _FakeProcess:
    """Stand-in for a campaign child process."""

    def __init__(self, command, returncode, created, waits):
        self.command = command
        self.returncode = returncode
        self._created = created
        self._waits = waits
        self.terminated = False

    def wait(self):
        self._waits.append(len(self._created))
        return self.returncode

    def terminate(self):
        self.terminated = True


def _campaign_configs(tmp_path):
    """Two-model campaign whose result dirs live under tmp_path."""
    from lm_eval.config.evaluate_config import EvaluatorConfig

    configs = EvaluatorConfig.load_configs(_write_campaign(tmp_path))
    for cfg in configs:
        cfg["output_path"] = str(tmp_path / cfg["metadata"]["model_name"])
    return configs


class TestRunCampaignFanOut:
    """`lm-eval run` fans one campaign TOML out into one process per model."""

    def test_child_command_selects_the_model(self):
        command = Run._campaign_child_command(
            "rwkv7-g1k-7.2b", ["run", "-C", "campaign.toml", "--resume"]
        )

        assert command[1:4] == ["-m", "lm_eval", "run"]
        assert command[4:] == [
            "-C",
            "campaign.toml",
            "--resume",
            "--only-model",
            "rwkv7-g1k-7.2b",
        ]

    def test_execute_dispatches_a_campaign_to_one_child_per_model(
        self, tmp_path, monkeypatch
    ):
        campaign = _write_campaign(tmp_path)
        args = argparse.Namespace(config=str(campaign), only_model=None, jobs=None)
        dispatched = []
        monkeypatch.setattr(Run, "_execute_campaign", lambda a, c: dispatched.append(c))
        monkeypatch.setattr(
            Run, "_execute_single", lambda a: dispatched.append("single")
        )

        Run._execute(args)

        assert len(dispatched) == 1
        assert [cfg["metadata"]["model_name"] for cfg in dispatched[0]] == [
            "rwkv7-g1k-1.5b",
            "rwkv7-g1k-7.2b",
        ]

    def test_execute_keeps_a_single_model_config_in_process(
        self, tmp_path, monkeypatch
    ):
        single = tmp_path / "single.toml"
        single.write_text(
            """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1k-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["mutual"]
output_dir = "results/1.5b"
max_length = 25600
max_num_seqs = 100

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "open_think"
wkv_mode = "fp32io16"
""",
            encoding="utf-8",
        )
        args = argparse.Namespace(config=str(single), only_model=None, jobs=None)
        dispatched = []
        monkeypatch.setattr(Run, "_execute_campaign", lambda a, c: dispatched.append(c))
        monkeypatch.setattr(
            Run, "_execute_single", lambda a: dispatched.append("single")
        )

        Run._execute(args)

        assert dispatched == ["single"]

    def test_execute_does_not_fan_out_from_within_a_child(
        self, tmp_path, monkeypatch
    ):
        """`--only-model` is the child's own invocation; it must not recurse."""
        campaign = _write_campaign(tmp_path)
        args = argparse.Namespace(
            config=str(campaign), only_model="rwkv7-g1k-7.2b", jobs=None
        )
        dispatched = []
        monkeypatch.setattr(Run, "_execute_campaign", lambda a, c: dispatched.append(c))
        monkeypatch.setattr(
            Run, "_execute_single", lambda a: dispatched.append("single")
        )

        Run._execute(args)

        assert dispatched == ["single"]

    def test_campaign_runs_every_model_and_reports_failures(
        self, tmp_path, monkeypatch, capsys
    ):
        configs = _campaign_configs(tmp_path)
        created, waits, popens = [], [], []

        def fake_popen(command, **_kwargs):
            created.append(command)
            process = _FakeProcess(
                command, 3 if "7.2b" in " ".join(command) else 0, created, waits
            )
            popens.append(process)
            return process

        monkeypatch.setattr("lm_eval._cli.run.subprocess.Popen", fake_popen)

        with pytest.raises(SystemExit) as exit_info:
            Run._execute_campaign(argparse.Namespace(jobs=None), configs)

        assert exit_info.value.code == 1
        assert len(created) == 2
        assert waits == [2, 2]
        assert not any(process.terminated for process in popens)
        logs = sorted(tmp_path.glob("*/console.log"))
        assert [log.parent.name for log in logs] == [
            "rwkv7-g1k-1.5b",
            "rwkv7-g1k-7.2b",
        ]
        assert "--only-model rwkv7-g1k-7.2b" in logs[1].read_text(encoding="utf-8")
        summary = capsys.readouterr().out
        assert "campaign summary: 1/2 models completed" in summary
        assert "fail  rwkv7-g1k-7.2b" in summary

    def test_jobs_caps_the_fan_out(self, tmp_path, monkeypatch):
        configs = _campaign_configs(tmp_path)
        created, waits = [], []

        def fake_popen(command, **_kwargs):
            created.append(command)
            return _FakeProcess(command, 0, created, waits)

        monkeypatch.setattr("lm_eval._cli.run.subprocess.Popen", fake_popen)

        Run._execute_campaign(argparse.Namespace(jobs=1), configs)

        # Only one child is in flight when the first wait happens; the second
        # model starts after that one is reaped.
        assert waits == [1, 2]

    def test_campaign_stops_children_when_interrupted(self, tmp_path, monkeypatch):
        configs = _campaign_configs(tmp_path)
        popens = []

        def fake_popen(command, **_kwargs):
            process = _FakeProcess(command, 0, [], [])
            process.wait = _raise_interrupt
            popens.append(process)
            return process

        def _raise_interrupt():
            raise KeyboardInterrupt

        monkeypatch.setattr("lm_eval._cli.run.subprocess.Popen", fake_popen)

        with pytest.raises(KeyboardInterrupt):
            Run._execute_campaign(argparse.Namespace(jobs=None), configs)

        assert all(process.terminated for process in popens)
