import hashlib
import json
import logging
import os
from collections import defaultdict
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.rwkv_choice import build_choice_prompt, choice_letters, parse_choice_index
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


def _rwkv_sampling_config(profile: dict) -> dict:
    """Map an RWKV `GenerationConfig` profile onto reportable sampling fields.

    `_create_payload` strips several of these out of the outgoing HTTP request
    (folding them into a server-side `generation_config` blob instead), so this
    is the only place the *effective* sampling parameters survive for
    `get_model_info()` to report into `results["config"]["sampling_config"]`.
    """
    sampling = {
        key: profile[key]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "penalty_decay",
            "do_sample",
        )
        if profile.get(key) is not None
    }
    if profile.get("stop_strings"):
        sampling["stop"] = list(profile["stop_strings"])
    return sampling


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="auto",
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        # Auto-detect tokenizer backend
        if tokenizer_backend == "auto":
            if base_url:
                from lm_eval.utils import check_remote_tokenizer_support

                if check_remote_tokenizer_support(
                    base_url,
                    verify_certificate=verify_certificate,
                    ca_cert_path=ca_cert_path,
                    auth_token=auth_token,
                ):
                    eval_logger.info(
                        "Auto-detected remote tokenizer support. Using remote tokenizer backend."
                    )
                    tokenizer_backend = "remote"
                else:
                    eval_logger.info(
                        "Remote tokenizer not supported. Using huggingface tokenizer backend."
                    )
                    tokenizer_backend = "huggingface"
            else:
                eval_logger.warning(
                    "No base_url provided. Using huggingface tokenizer backend."
                )
                tokenizer_backend = "huggingface"

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = _RWKVCompletion(
                    choices["text"], choices.get("finish_reason"), raw_response=out
                )
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


class _RWKVCompletion(str):
    """Completion text retaining the transport details supplied by the server."""

    def __new__(
        cls,
        text: str,
        finish_reason: str | None = None,
        *,
        raw_response: dict | None = None,
    ):
        value = super().__new__(cls, text)
        value.finish_reason = finish_reason
        value.raw_response = raw_response
        value.prompt_token_ids = None
        value.output_token_ids = None
        value.reasoning = None
        value.truncated = finish_reason in {"length", "max_tokens"}
        return value


@register_model("rwkv7-http")
class RWKV7HTTP(LocalCompletionsAPI):
    """RWKV7 adapter for the local ``transformers serve`` completions API.

    ``transformers serve`` accepts rendered text prompts and exposes generation,
    but not echo prompt logprobs. Consequently this adapter deliberately rejects
    likelihood workloads such as RACE while retaining native generation evidence
    for generative tasks such as DROP.
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/completions"
    PROMPT_TEMPLATES = {"assistant", "bot", "function_calling"}
    PROMPT_STOPS = {
        "assistant": "\nUser:",
        "bot": "✿",
        "function_calling": "\n### User",
    }
    GENERATION_PROMPTS = {"open_think", "fake_think"}
    PROFILE_FILES = {
        "open_think": "generation_config.json",
        "fake_think": "fake_think_generation_config.json",
    }

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        model=None,
        pretrained=None,
        tokenizer=None,
        service_backend="transformers",
        tokenizer_backend="huggingface",
        tokenized_requests=False,
        rapid_sampling=True,
        num_concurrent=5,
        batch_size=1,
        max_length=16384,
        rwkv_prompt_template="assistant",
        rwkv_generation_prompt="fake_think",
        cot_mode=None,
        record_evidence=False,
        **kwargs,
    ):
        model = model or pretrained
        if not isinstance(model, str) or not model.strip():
            raise ValueError("rwkv7-http requires model= or pretrained=.")
        if service_backend != "transformers":
            raise ValueError("This checkout supports service_backend=transformers only.")
        if tokenizer_backend != "huggingface" or tokenized_requests:
            raise ValueError(
                "transformers service_backend requires tokenizer_backend=huggingface "
                "and tokenized_requests=False."
            )
        if record_evidence:
            raise ValueError(
                "transformers service_backend does not expose completion token IDs; "
                "record_evidence=True is unsupported."
            )
        if cot_mode is not None:
            rwkv_generation_prompt = cot_mode
        if rwkv_prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError("rwkv_prompt_template must be assistant, bot, or function_calling.")
        if rwkv_generation_prompt not in self.GENERATION_PROMPTS:
            raise ValueError("rwkv_generation_prompt must be open_think or fake_think.")

        self.rwkv_prompt_template = rwkv_prompt_template
        self.rwkv_generation_prompt = rwkv_generation_prompt
        self.record_evidence = False
        tokenizer = tokenizer or model
        super().__init__(
            base_url=base_url,
            model=model,
            tokenizer=tokenizer,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=False,
            num_concurrent=num_concurrent,
            batch_size=batch_size,
            max_length=max_length,
            **kwargs,
        )
        if self._batch_size != 1:
            raise ValueError(
                "transformers service_backend requires batch_size=1; use num_concurrent "
                "for HTTP concurrency."
            )
        self._chat_template_source = self.tokenizer.chat_template
        if not isinstance(self._chat_template_source, str) or not self._chat_template_source:
            raise RuntimeError("The RWKV tokenizer must provide the official chat template.")

        from transformers import GenerationConfig

        profile = GenerationConfig.from_pretrained(
            tokenizer,
            config_file_name=self.PROFILE_FILES[rwkv_generation_prompt],
            local_files_only=os.path.isdir(tokenizer),
        )
        self._generation_profile = profile.to_dict()
        self._chat_template_sha256 = hashlib.sha256(
            self._chat_template_source.encode("utf-8")
        ).hexdigest()

    @property
    def tokenizer_name(self) -> str:
        return (
            f"{self.model}:{self._chat_template_sha256}:"
            f"{self.rwkv_prompt_template}:{self.rwkv_generation_prompt}"
        )

    def get_model_info(self) -> dict:
        return {"sampling_config": _rwkv_sampling_config(self._generation_profile)}

    @cached_property
    def eot_token_id(self) -> int:
        return 0

    @cached_property
    def prefix_token_id(self) -> int:
        return 0

    @cached_property
    def eos_string(self) -> None:
        return None

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        return self._chat_template_source

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        from lm_eval.utils import env

        rendered = env.from_string(self._chat_template_source).render(
            messages=chat_history,
            add_generation_prompt=add_generation_prompt,
            rwkv_prompt_template=self.rwkv_prompt_template,
            rwkv_generation_prompt=self.rwkv_generation_prompt,
            tools=kwargs.pop("tools", None),
            **kwargs,
        )
        if (
            add_generation_prompt
            and self.rwkv_generation_prompt == "fake_think"
            and rendered.endswith("<think></think")
        ):
            rendered += ">\n"
        return rendered

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations = []
        for output in outputs:
            choices = [None] * len(output["choices"])
            for choice in output["choices"]:
                choices[choice["index"]] = _RWKVCompletion(
                    choice.get("text", ""),
                    choice.get("finish_reason"),
                    raw_response=output,
                )
            generations.extend(choices)
        return generations

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        if not generate:
            raise NotImplementedError(
                "transformers serve does not expose echo prompt logprobs; "
                "loglikelihood requests must go through RWKV7HTTP.loglikelihood's "
                "generative conversion instead of _create_payload(generate=False)."
            )
        effective_kwargs = dict(gen_kwargs or {})
        payload = super()._create_payload(
            messages,
            generate=True,
            gen_kwargs=effective_kwargs,
            **kwargs,
        )
        profile = dict(self._generation_profile)
        payload["temperature"] = profile.get("temperature", 1.0)
        payload["generation_config"] = json.dumps(
            profile, separators=(",", ":"), sort_keys=True
        )
        for name in (
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "penalty_decay",
        ):
            payload.pop(name, None)
        if not payload["stop"]:
            payload["stop"] = [self.PROMPT_STOPS[self.rwkv_prompt_template]]
        return payload

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Emulate multiple_choice loglikelihood scoring through generation.

        ``transformers serve`` exposes no echo/prompt logprobs, so this is the
        only way RWKV7HTTP can support ``output_type: multiple_choice`` tasks
        at all. Mirrors lighteval-rwkv's runtime loglikelihood->generation
        conversion: ask the model to name a lettered choice once per doc, then
        encode the parsed answer back into (loglik, is_greedy) pairs so the
        harness's existing argmax-based acc/acc_norm scoring is unaffected.
        """
        results: List[Optional[Tuple[float, bool]]] = [None] * len(requests)
        groups: Dict[Any, List[int]] = defaultdict(list)
        for position, request in enumerate(requests):
            groups[(request.task_name, request.doc_id)].append(position)

        for positions in groups.values():
            positions = sorted(positions, key=lambda p: requests[p].idx)
            group = [requests[p] for p in positions]
            contexts = {req.args[0] for req in group}
            if "" in contexts and len(contexts) > 1:
                eval_logger.warning(
                    "RWKV7HTTP does not support the unconditional likelihood "
                    "acc_mutual_info needs; acc_mutual_info results for this "
                    "task are not meaningful."
                )
                kept_positions, kept_group = [], []
                for position, request in zip(positions, group):
                    if request.args[0] == "":
                        results[position] = (0.0, False)
                    else:
                        kept_positions.append(position)
                        kept_group.append(request)
                positions, group = kept_positions, kept_group
                if not group:
                    continue

            context = group[0].args[0]
            choices = [request.args[1].strip() for request in group]
            chosen_index = self._resolve_choice(context, choices)
            for offset, position in enumerate(positions):
                results[position] = (0.0, True) if offset == chosen_index else (-1e6, False)

        return results

    def _resolve_choice(self, context: str, choices: List[str]) -> int:
        letters = choice_letters(choices)
        prompt = build_choice_prompt(context, choices)
        gen_kwargs = {"until": [], "max_gen_toks": self._max_gen_toks}
        instance = Instance(
            request_type="generate_until",
            doc={},
            arguments=(prompt, gen_kwargs),
            idx=0,
        )
        generated = self.generate_until([instance])[0]
        return self._parse_choice_index(str(generated), letters, choices)

    def _parse_choice_index(
        self, generated: str, letters: List[str], choices: List[str]
    ) -> int:
        index = parse_choice_index(generated, letters, choices)
        if index < 0:
            eval_logger.warning(
                "RWKV7HTTP could not parse a choice letter from the generated "
                f"answer {generated.strip()!r}; defaulting to option {letters[0]!r}."
            )
            return 0
        return index


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    """
    Minimal chat-completions wrapper.
    - Only accepts messages as list[dict].
    - No tokenization or template logic.
    - Use with --apply_chat_template or ensure upstream formats messages correctly.
    """

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert isinstance(messages, list) and all(
            isinstance(m, dict) for m in messages
        ), (
            "LocalChatCompletion expects messages as list[dict]. "
            "If you see this error, ensure --apply_chat_template is set or upstream code formats messages correctly."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choices in out["choices"]:
                    tmp[choices["index"]] = choices["message"]["content"]
            except Exception as e:
                # account for cases that generation is blocked by content filter,
                # which is common for Azure OpenAI Service,
                # not sure if need to account for multiple choices
                eval_logger.warning(f"Could not parse generations: {e}")
                tmp = [""]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if (
            "o1" in self.model
            or "5" in self.model
            or "o3" in self.model
            or "o4" in self.model
        ):
            output.pop("stop")
            output["temperature"] = 1
        return output


@register_model("azure-openai-chat-completions")
class AzureOpenaiChatCompletionsLM(OpenAIChatCompletion):
    def __init__(
        self,
        model: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        base_url: str = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        truncate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        self.truncate = truncate
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url, api_version=api_version, api_key=self.api_key
        )

    @cached_property
    def api_key(self):
        key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `AZURE_OPENAI_API_KEY` environment variable."
            )
        return key
