import abc
import asyncio
import copy
import itertools
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from dataclasses import replace
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

try:
    import requests
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
    from tenacity import (
        RetryError,
        retry,
        retry_if_not_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    from tqdm import tqdm
    from tqdm.asyncio import tqdm_asyncio
except ModuleNotFoundError:
    pass


import base64
from importlib.util import find_spec
from io import BytesIO

from lm_eval import utils
from lm_eval.api.checkpoint import ItemKey, serialize_response
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import Collator, chunks, configure_pad_token


if TYPE_CHECKING:
    from PIL import Image

# Set a placeholder to use as model output when the returned answer is "None"
# This commonly occurs when the model runs out of tokens during the thinking
# process and returns an empty answer but no error.
# Default value is an empty string, but it can be edited if this is problematic.
LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = os.environ.get(
    "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER", ""
)

eval_logger = logging.getLogger(__name__)

LogLikelihoodInputs = Tuple[Tuple[str, str], List[int], List[int]]

# Status codes worth retrying. The retry policy itself lives in the tenacity
# wrapper around ``model_call``/``amodel_call`` (see ``get_batched_requests``),
# mirroring lighteval-rwkv's single flat request budget: one endpoint, one
# concurrency limit, exponential backoff on failure.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

_CONTEXT_LENGTH_ERROR = re.compile(r"prompt contains at least (\d+) input tokens")
_MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")


class ApiRequestError(RuntimeError):
    """Raised when an HTTP request to the model endpoint cannot be completed."""


class ContextLengthExhausted(ApiRequestError):
    """Raised when a request's prompt tokens leave no positive token budget
    for a completion, even after shrinking the requested max tokens to fit.
    """


def context_error_prompt_tokens(status: int, body_text: str) -> Optional[int]:
    """Extract "prompt contains at least N input tokens" from a vLLM-style
    400/422 error body, or ``None`` if this isn't a context-length error.
    """
    if status not in (400, 422) or not body_text:
        return None
    match = _CONTEXT_LENGTH_ERROR.search(body_text)
    return int(match.group(1)) if match else None


def shrink_max_tokens(payload: dict, available: int) -> Optional[dict]:
    """Return a copy of ``payload`` with its max-tokens field shrunk to fit
    ``available`` tokens, or ``None`` if there's no positive budget left or
    no max-tokens field to shrink (caller should treat both as "exhausted").
    """
    if available <= 0:
        return None
    for key in _MAX_TOKEN_KEYS:
        if key not in payload or payload[key] is None:
            continue
        new_value = min(int(payload[key]), available)
        if new_value <= 0:
            return None
        if new_value == payload[key]:
            return None
        adjusted = dict(payload)
        adjusted[key] = new_value
        return adjusted
    return None


# utility class to keep track of json encoded chats
class JsonChatStr(NamedTuple):
    prompt: str

    def encode(self, encoding):
        return self.prompt.encode(encoding)


def create_image_prompt(
    imgs: list["Image.Image"], chat: dict, fmt: str = "PNG"
) -> dict:
    """

    Parameters
    ----------
    img : list[PIL.Image.Image]
        The list of images to encode to base64
    chat : dict
    fmt : str, optional
        Any format Pillow understands (e.g. "PNG", "JPEG").
        Defaults to "PNG".

    Returns
    -------
    dict
    """
    images = []
    for img in imgs:
        buf = BytesIO()
        img.save(buf, format=fmt)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_dict = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "auto"},
        }
        images.append(img_dict)

    # With images, we need "content" to be a list of dicts with
    # "type" and "text"/"image_url".
    # currently we do not support few-shots so only one user message
    # text content also has <image> placeholders, which apparently is not necessary for API class (confirm)

    if isinstance(chat[-1]["content"], list):
        chat[-1]["content"] = images + chat[-1]["content"]
    else:
        text_content = {"type": "text", "text": chat[-1]["content"]}
        chat[-1]["content"] = images + [text_content]
    chat[-1].pop("type", None)
    return chat


class _ContextExhaustedCompletion(str):
    """Placeholder generation result for a request whose prompt tokens alone
    leave no positive completion budget, even after shrinking max_tokens to
    fit (see ``ContextLengthExhausted`` above).

    Duck-type compatible with ``openai_completions._RWKVCompletion`` (same
    ``finish_reason``/``raw_response``/``truncated`` attributes), since
    ``evaluator.py::response_evidence()`` only accesses them via ``getattr``
    with defaults -- this lets that existing truncation-reporting path pick
    it up without a circular import back into ``openai_completions``.
    """

    def __new__(cls):
        value = super().__new__(cls, "")
        value.finish_reason = "length"
        value.raw_response = None
        value.prompt_token_ids = None
        value.output_token_ids = None
        value.reasoning = None
        value.truncated = True
        return value


class TemplateAPI(TemplateLM):
    MULTIMODAL = True
    _supports_checkpoint_items = True

    def __init__(
        self,
        model: str = None,
        pretrained: str = None,  # `model` takes precedence over `pretrained` when passed.
        base_url: str = None,
        tokenizer: Optional[str] = None,
        # Loglikelihood tasks require a tokenizer to calculate context lengths,
        # however the requests can be sent as a string if the API doesn't support token inputs.
        # use tokenized_requests=False
        tokenizer_backend: Optional[
            Literal["tiktoken", "huggingface", "remote", "None", "none"]
        ] = "huggingface",
        truncate: bool = False,
        # Flat request budget for this endpoint: the number of prompts that may
        # be in flight at once. Derived from the served endpoint's own
        # ``max_num_seqs`` by the versioned RWKV config, so callers never hand
        # tune it (see lighteval-rwkv's ``concurrent_requests``).
        num_concurrent: int = 1,
        max_retries: int = 3,
        max_gen_toks: int = 256,
        batch_size: Union[str, int] = 1,
        seed: int = 1234,
        max_length: Optional[int] = 2048,
        add_bos_token: bool = False,
        custom_prefix_token_id: int = None,
        # send the requests as tokens or strings
        tokenized_requests: bool = True,
        trust_remote_code: bool = False,
        revision: Optional[str] = "main",
        use_fast_tokenizer: bool = True,
        verify_certificate: bool = True,
        ca_cert_path: Optional[str] = None,
        auth_token: Optional[str] = None,
        eos_string: str = None,
        # timeout in seconds
        timeout: int = 300,
        header: Optional[Dict[str, str]] = None,
        max_images: int = 1,
        send_idempotency_key: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.send_idempotency_key = send_idempotency_key
        missing_packages = [
            pkg
            for pkg in ["aiohttp", "tqdm", "tenacity", "requests"]
            if find_spec(pkg) is None
        ]
        if missing_packages:
            raise ModuleNotFoundError(
                f"Attempted to use an API model, but the required packages {missing_packages} are not installed. "
                'Please install these via `pip install lm-eval[api]` or `pip install -e ."[api]"`'
            )
        self.model = model or pretrained
        self.base_url = base_url
        self.tokenizer = tokenizer
        self._header = header
        if not isinstance(batch_size, int) and "auto" in batch_size:
            eval_logger.warning(
                "Automatic batch size is not supported for API models. Defaulting to batch size 1."
            )
        elif int(batch_size) > 1:
            eval_logger.warning(
                "Batch size > 1 detected. Ensure your API supports batched requests with varying total sequence lengths."
            )
        self._batch_size = int(batch_size) if batch_size != "auto" else 1
        self._truncate = truncate
        self._max_gen_toks = int(max_gen_toks)
        self._seed = int(seed)
        # max_length - 1 as we always have 1 token for generation
        eval_logger.info(f"Using max length {max_length} - 1")
        self.max_length = max_length - 1
        if int(num_concurrent) <= 1:
            eval_logger.info(
                "Concurrent requests are disabled. To enable concurrent requests, set `num_concurrent` > 1."
            )
        self._concurrent = int(num_concurrent)
        self.base_url = base_url
        self.tokenizer_backend = (
            None if tokenizer_backend in ("None", "none") else tokenizer_backend
        )
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = custom_prefix_token_id
        self.tokenized_requests = tokenized_requests
        self.max_retries = int(max_retries)
        self.verify_certificate = verify_certificate
        self.ca_cert_path = ca_cert_path
        self.auth_token = auth_token
        self._eos_string = eos_string
        self.timeout = int(timeout)
        self.max_images = int(max_images)

        eval_logger.info(f"Using tokenizer {self.tokenizer_backend}")
        if self.tokenizer_backend is None:
            self.tokenizer = None
            self.tokenized_requests = False
        else:
            if self.tokenizer is None:
                if self.tokenizer_backend == "huggingface":
                    import transformers

                    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                        self.tokenizer if self.tokenizer else self.model,
                        trust_remote_code=trust_remote_code,
                        revision=revision,
                        use_fast=use_fast_tokenizer,
                    )
                    # Not used as the API will handle padding but to mirror the behavior of the HFLM
                    self.tokenizer = configure_pad_token(self.tokenizer)
                elif self.tokenizer_backend == "tiktoken":
                    try:
                        import tiktoken

                        self.tokenizer = tiktoken.encoding_for_model(self.model)
                    except ModuleNotFoundError as e:
                        raise ModuleNotFoundError(
                            "Attempted to use 'openai' LM type, but the package `tiktoken` is not installed. "
                            "Please install it via `pip install lm-eval[api]` or `pip install -e .[api]`."
                        ) from e
                    if "openai" not in self.base_url:
                        eval_logger.warning(
                            f"Passed `base_url={self.base_url}` but using (OpenAI) Tiktoken tokenizer backend. "
                            "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                        )
                elif self.tokenizer_backend == "remote":
                    from lm_eval.utils import RemoteTokenizer

                    if not self.base_url:
                        raise ValueError(
                            "base_url is required for remote tokenizer backend"
                        )
                    self.tokenizer = RemoteTokenizer(
                        self.base_url,
                        self.timeout,
                        self.verify_certificate,
                        self.ca_cert_path,
                        self.auth_token,
                    )
                    eval_logger.info(f"Using remote tokenizer from {self.base_url}")
            else:
                import transformers

                assert isinstance(tokenizer, str), "tokenizer must be a string"
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    use_fast=use_fast_tokenizer,
                )

    @abc.abstractmethod
    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: str = None,
        **kwargs,
    ) -> dict:
        """This method is responsible for creating the json payload that will be sent to the API."""
        raise NotImplementedError

    def create_message(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        generate=False,
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        """Helper method to transform the prompt into the expected API input format. messages consist of batched requests"""
        if isinstance(messages[0], JsonChatStr):
            # for chat completions we need to decode the json string to list[dict,...]
            assert self._batch_size == 1, (
                "non-tokenized chat requests are only supported with batch_size=1"
            )
            # list[dict["role":..., "content":...],...]
            return json.loads(messages[0].prompt)

        if not self.tokenized_requests:
            # if messages are tokenized:
            if isinstance(messages[0][0], int):
                # assuming decoding is lossless. However, this is only for loglikelihood requests
                # as we need to compute the context length. For generations, we don't need to tokenize.
                messages = self.decode_batch(messages)
            if self._batch_size <= 1:
                # if batch is 1 return str
                return messages[0]
            else:
                # list[str,...]
                return messages

        # list[list[int], ...]
        return messages

    @staticmethod
    @abc.abstractmethod
    def parse_logprobs(
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlen: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """Method used to parse the logprobs from the (batched) API response. This method should return a list of tuples"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        """Method used to parse the generations from the (batched) API response. This method should return a list of str"""
        raise NotImplementedError

    @cached_property
    def api_key(self) -> str:
        """Override this property to return the API key for the API request."""
        return ""

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return self._header or {"Authorization": f"Bearer {self.api_key}"}

    @property
    def tokenizer_name(self) -> str:
        """Must be defined for LM subclasses which implement Chat Templating.
        Should return the name of the tokenizer or chat template used.
        Used only to properly fingerprint caches when requests are being cached with `--cache_requests`, otherwise not used.
        """
        return ""

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> Union[str, JsonChatStr, List[Dict]]:
        """Applies a chat template to a list of chat history between user and model."""
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        elif self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        else:
            # bit of a hack. We'll load back before sending to the API
            return JsonChatStr(json.dumps(chat_history, ensure_ascii=False))

    @cached_property
    def eot_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.tokenizer_backend == "huggingface":
                return self.tokenizer.eos_token_id
            elif self.tokenizer_backend == "tiktoken":
                return self.tokenizer.eot_token
            elif self.tokenizer_backend == "remote":
                return self.tokenizer.eos_token_id

    @cached_property
    def eos_string(self) -> Optional[str]:
        if self._eos_string:
            return self._eos_string
        elif self.tokenizer is not None:
            if self.tokenizer_backend == "huggingface":
                return self.tokenizer.eos_token
            elif self.tokenizer_backend == "tiktoken":
                return self.tokenizer.decode([self.tokenizer.eot_token])
            elif self.tokenizer_backend == "remote":
                return self.tokenizer.eos_token
        else:
            eval_logger.warning(
                "Cannot determine EOS string to pass to stop sequence. Manually set by passing `eos_string` to model_args."
            )
            return None

    @cached_property
    def prefix_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if self.tokenizer_backend == "huggingface":
                if self.tokenizer.bos_token_id is not None:
                    return self.tokenizer.bos_token_id
                return self.tokenizer.eos_token_id
            elif self.tokenizer_backend == "remote":
                return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            else:
                return self.tokenizer.eot_token

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        **kwargs,
    ) -> Union[List[List[int]], List[int], List[str]]:
        if self.tokenizer_backend is None:
            return [string]
        elif self.tokenizer_backend == "huggingface":
            # by default for CausalLM - false or self.add_bos_token is set
            if not add_special_tokens:
                add_special_tokens = False or self.add_bos_token
            encoding: Union[List[List[int]], List[int]] = self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_attention_mask=False,
            ).input_ids

            # left-truncate the encoded context to be at most `left_truncate_len` tokens long
            if left_truncate_len:
                if not isinstance(string, str):
                    encoding = [enc[-left_truncate_len:] for enc in encoding]
                else:
                    encoding = encoding[-left_truncate_len:]

            return encoding
        elif self.tokenizer_backend == "remote":
            if isinstance(string, str):
                encoding = self.tokenizer.encode(string)
            else:
                encoding = [self.tokenizer.encode(s) for s in string]

            if left_truncate_len:
                if isinstance(string, str):
                    encoding = encoding[-left_truncate_len:]
                else:
                    encoding = [enc[-left_truncate_len:] for enc in encoding]

            return encoding
        else:
            try:
                encoding = self.tokenizer.encode(string)
            except Exception:
                encoding = self.tokenizer.encode_batch(string)
            return encoding

    def decode_batch(self, tokens: List[List[int]]) -> List[str]:
        if self.tokenizer_backend == "huggingface":
            return self.tokenizer.batch_decode(tokens)
        elif self.tokenizer_backend == "tiktoken":
            return self.tokenizer.decode_batch(tokens)
        elif self.tokenizer_backend == "remote":
            return self.tokenizer.batch_decode(tokens)

    def _endpoint_url(self) -> str:
        """The single endpoint every request of this model goes to.

        Subclasses (e.g. ``AzureOpenaiChatCompletionsLM``) may call
        ``super().__init__()`` with no arguments and set ``self.base_url``
        directly afterwards, so the URL is validated at dispatch time rather
        than at construction time.
        """
        if not self.base_url:
            raise ApiRequestError(
                "No base_url configured; cannot dispatch API requests."
            )
        return self.base_url

    def _annotate_generation(self, value, gen_kwargs):
        if not getattr(self, "INDEPENDENT_ROLLOUTS", False) or value is None:
            return value
        from lm_eval.models.openai_completions import _RWKVCompletion
        if not hasattr(value, "__dict__"):
            value = _RWKVCompletion(str(value))
        value.seed = (gen_kwargs or {}).get("seed", self._seed)
        value.requires_think_close = getattr(self, "rwkv_generation_prompt", None) == "open_think"
        return value

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        headers = {**self.header, **extra_headers} if extra_headers else self.header
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        try:
            url = self._endpoint_url()
            while True:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    verify=self.verify_certificate,
                    timeout=self.timeout,
                )
                if response.ok:
                    return response.json()
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
                prompt_tokens = context_error_prompt_tokens(
                    response.status_code, response.text
                )
                if prompt_tokens is not None:
                    adjusted = shrink_max_tokens(payload, self.max_length - prompt_tokens)
                    if adjusted is not None:
                        payload = adjusted
                        continue
                    raise ContextLengthExhausted(
                        f"Prompt requires at least {prompt_tokens} input tokens, "
                        f"leaving no positive completion budget within max_length={self.max_length}."
                    )
                # Any other failure -- retryable status or not -- is left to the
                # caller's tenacity policy, which re-issues the request against
                # the same endpoint with exponential backoff.
                raise ApiRequestError(f"HTTP {response.status_code}: {response.text}")
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    async def amodel_call(
        self,
        session: ClientSession,
        sem: asyncio.Semaphore,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        cache_keys: list = None,
        item_keys: Optional[List[Optional[ItemKey]]] = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Union[List[str], List[Tuple[float, bool]], None]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        extra_headers = None
        if (
            self.send_idempotency_key
            and item_keys
            and len(item_keys) == 1
            and item_keys[0] is not None
        ):
            extra_headers = {"Idempotency-Key": item_keys[0].idempotency_key}
        request_headers = (
            {**self.header, **extra_headers} if extra_headers else self.header
        )
        url = self._endpoint_url()

        acquired = await sem.acquire()
        try:
            outputs = None
            while outputs is None:
                async with session.post(
                    url,
                    json=payload,
                    headers=request_headers,
                ) as response:
                    if response.ok:
                        outputs = await response.json()
                        break
                    error_text = await response.text()
                    eval_logger.warning(
                        f"API request failed! Status code: {response.status}, "
                        f"Response text: {error_text}. Retrying..."
                    )
                    prompt_tokens = context_error_prompt_tokens(
                        response.status, error_text
                    )
                    if prompt_tokens is not None:
                        adjusted = shrink_max_tokens(
                            payload, self.max_length - prompt_tokens
                        )
                        if adjusted is not None:
                            payload = adjusted
                            continue
                        raise ContextLengthExhausted(
                            f"Prompt requires at least {prompt_tokens} input tokens, "
                            f"leaving no positive completion budget within max_length={self.max_length}."
                        )
                    # Any other failure -- retryable status or not -- is left to
                    # the caller's tenacity policy, which re-issues the request
                    # against the same endpoint with exponential backoff.
                    raise ApiRequestError(f"HTTP {response.status}: {error_text}")
            tmp_answers = (
                self.parse_generations(
                    outputs=outputs,
                )
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )

            if generate:
                tmp_answers = [self._annotate_generation(value, gen_kwargs) for value in tmp_answers]

            # Convert `None`` values to `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER` string to maintain consistency
            answers = []
            for a in tmp_answers:
                if a is None:
                    eval_logger.warning(
                        f"API returned null content. Content filled with `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = {LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER}`. Check reasoning_content field or generation limits."
                    )
                    answers.append(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER)
                else:
                    answers.append(a)

            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            if item_keys and self.checkpoint_store is not None:
                completed = [
                    (key.item_id, serialize_response(res))
                    for key, res in zip(item_keys, answers)
                    if key is not None
                ]
                if completed:
                    await self.checkpoint_store.mark_items_completed_async(completed)
            return answers
        # If the retries also fail
        except BaseException as e:
            eval_logger.error(f"Exception:{repr(e)}, {locals().get('outputs', '(no outputs)')}, retrying.")
            raise e
        finally:
            if acquired:
                sem.release()

    def batch_loglikelihood_requests(
        self, chunks: Iterable[List[LogLikelihoodInputs]]
    ) -> Tuple[List[List[int]], List[int], List[Tuple[str, str]], List[Optional[ItemKey]]]:
        inputs = []
        ctxlens = []
        cache_keys = []
        item_keys = []
        for chunk in chunks:
            for item in chunk:
                cache_key, context_enc, continuation_enc = item[:3]
                item_key = item[3] if len(item) > 3 else None
                # max_length - 1 as we always have 1 token for generation
                inp = (context_enc + continuation_enc)[-self.max_length :]
                if len(inp) < len(context_enc + continuation_enc):
                    eval_logger.warning(
                        f"Context length ({len(context_enc)}) + continuation length ({len(continuation_enc)}) > max_length ({self.max_length}). Left truncating context."
                    )
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - self.max_length
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)
                cache_keys.append(cache_key)
                item_keys.append(item_key)
        return inputs, ctxlens, cache_keys, item_keys

    async def get_batched_requests(
        self,
        requests: list,
        cache_keys: list,
        *,
        generate: bool = True,
        ctxlens: List[int] = None,
        item_keys: Optional[List[Optional[ItemKey]]] = None,
        **kwargs,
    ) -> Union[List[List[str]], List[List[Tuple[float, bool]]]]:
        ctxlens = ctxlens if ctxlens else [None] * len(requests)
        item_keys = item_keys if item_keys else [None] * len(requests)
        per_request_kwargs = kwargs.pop("gen_kwargs_per_request", None)
        if per_request_kwargs is not None and (len(per_request_kwargs) != len(requests) or self._batch_size != 1):
            raise ValueError("per-request generation kwargs require one request per batch")
        self._endpoint_url()
        sem = asyncio.Semaphore(self._concurrent)
        async with AsyncExitStack() as stack:
            # One session, one connection pool, one flat concurrency budget:
            # exactly the endpoint's declared ``max_num_seqs`` requests may be
            # in flight, so the evaluator queues at the inference server
            # instead of falling into lockstep with it.
            session = await stack.enter_async_context(
                ClientSession(
                    connector=TCPConnector(
                        limit=self._concurrent, ssl=self.verify_certificate
                    ),
                    timeout=ClientTimeout(total=self.timeout),
                )
            )
            retry_: Callable[..., Awaitable[Any]] = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True,
                retry=retry_if_not_exception_type(ContextLengthExhausted),
                before_sleep=lambda retry_state: eval_logger.info(
                    f"Retry attempt {retry_state.attempt_number}"
                ),
            )(self.amodel_call)

            async def _guarded(
                message: list, cache_key: list, ctxlen: list, item_key: list, request_kwargs: dict
            ) -> list:
                try:
                    return await retry_(
                        session=session,
                        sem=sem,
                        messages=message,
                        cache_keys=cache_key,
                        item_keys=item_key,
                        generate=generate,
                        ctxlens=ctxlen,
                        **request_kwargs,
                    )
                except ContextLengthExhausted as e:
                    if getattr(self, "PRESERVE_INPUT_CONTEXT", False):
                        raise
                    eval_logger.warning(
                        f"Context length exhausted for one batch; recording "
                        f"placeholder result(s) instead of failing the run: {e}"
                    )
                    placeholder = (
                        _ContextExhaustedCompletion() if generate else (0.0, False)
                    )
                    answers = [placeholder] * len(message)
                    completed = [
                        (key.item_id, serialize_response(res))
                        for key, res in zip(item_key, answers)
                        if key is not None
                    ]
                    if completed and self.checkpoint_store is not None:
                        await self.checkpoint_store.mark_items_completed_async(
                            completed
                        )
                    return answers

            # Create tasks for each batch of request. Subclasses that never
            # dispatch HTTP themselves (e.g. LiteLLMChatCompletion) accept the
            # session argument and ignore it.
            tasks = [
                asyncio.create_task(_guarded(
                    message, cache_key, ctxlen, item_key,
                    {**kwargs, **({"gen_kwargs": per_request_kwargs[index]} if per_request_kwargs is not None else {})},
                ))
                for index, (message, cache_key, ctxlen, item_key) in enumerate(zip(
                    chunks(requests, n=self._batch_size),
                    chunks(cache_keys, n=self._batch_size),
                    chunks(ctxlens, n=self._batch_size),
                    chunks(item_keys, n=self._batch_size),
                ))
            ]

            try:
                return await tqdm_asyncio.gather(*tasks, desc="Requesting API")
            except BaseException:
                # `gather()` raises on the first failing task but does not
                # cancel the rest, which would otherwise keep running against
                # the `ClientSession`s this `async with` block is about to
                # close on unwind (-> ServerDisconnectedError/RuntimeError
                # storms) and, worse, can leave them cancelled while blocked
                # draining on shared session state. Cancel and drain them here,
                # while the session is still open, instead of leaving that to
                # asyncio.run()'s teardown-time mass-cancel.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        assert self.tokenizer is not None, (
            "Tokenizer is required for loglikelihood tasks to compute context lengths."
        )
        res = []
        item_keys = kwargs.pop("item_keys", None)
        if item_keys:
            requests = [
                (*req, key) for req, key in zip(requests, item_keys, strict=True)
            ]

        def _collate(req: LogLikelihoodInputs):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=None,
        )
        # if concurrent then we'll batch in the async context
        chunked = re_ord.get_batched(n=self._batch_size if self._concurrent <= 1 else 0)
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                inputs, ctxlens, cache_keys, item_keys = (
                    self.batch_loglikelihood_requests([chunk])
                )

                try:
                    outputs = retry(
                        stop=stop_after_attempt(self.max_retries),
                        wait=wait_exponential(multiplier=0.5, min=1, max=10),
                        reraise=True,
                        retry=retry_if_not_exception_type(ContextLengthExhausted),
                    )(self.model_call)(messages=inputs, generate=False)
                except ContextLengthExhausted as e:
                    eval_logger.warning(
                        f"Context length exhausted for loglikelihood request; "
                        f"recording as (0.0, False) instead of failing the run: {e}"
                    )
                    for cache_key, item_key in zip(cache_keys, item_keys):
                        answer_ = (0.0, False)
                        res.append(answer_)
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer_
                            )
                        if item_key is not None and self.checkpoint_store is not None:
                            self.checkpoint_store.mark_items_completed(
                                [(item_key.item_id, serialize_response(answer_))]
                            )
                        pbar.update(1)
                    continue
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for answer_, cache_key, item_key in zip(
                    self.parse_logprobs(
                        outputs=outputs, tokens=inputs, ctxlens=ctxlens
                    ),
                    cache_keys,
                    item_keys,
                ):
                    if answer_ is not None:
                        res.append(answer_)
                        # cache requests that aren't from a loglikelihood_rolling request
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer_
                            )
                        if item_key is not None and self.checkpoint_store is not None:
                            self.checkpoint_store.mark_items_completed(
                                [(item_key.item_id, serialize_response(answer_))]
                            )
                        pbar.update(1)
        else:
            inputs, ctxlens, cache_keys, item_keys = self.batch_loglikelihood_requests(
                chunked
            )
            res = itertools.chain.from_iterable(
                asyncio.run(
                    self.get_batched_requests(
                        inputs,
                        cache_keys,
                        generate=False,
                        ctxlens=ctxlens,
                        item_keys=item_keys,
                    )
                )
            )

        return re_ord.get_original(res)

    def generate_until(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
        item_keys: Optional[List[Optional[ItemKey]]] = None,
    ) -> List[str]:
        res = []
        if item_keys is None:
            item_keys = [None] * len(requests)

        if not requests:
            return []
        independent = getattr(self, "INDEPENDENT_ROLLOUTS", False)
        strict_context = getattr(self, "PRESERVE_INPUT_CONTEXT", False)
        prepared_seeds = []
        if independent:
            counts = {}
            token_counts = {}
            prepared = []
            for request, key in zip(requests, item_keys, strict=True):
                args = request.args
                gen = dict(args[1])
                slot = (request.task_name, request.doc_id)
                ordinal = key.repeat_ord if key is not None else (
                    request.idx if request.task_name is None else counts.get(slot, 0)
                )
                counts[slot] = counts.get(slot, 0) + 1
                gen["seed"] = int(gen.get("seed", self._seed)) + ordinal
                prepared_seeds.append(gen["seed"])
                if strict_context and self.tokenized_requests:
                    if args[0] not in token_counts:
                        token_counts[args[0]] = len(self.tok_encode([args[0]], add_special_tokens=self.add_bos_token)[0])
                    room = self.max_length - token_counts[args[0]]
                    if room <= 0:
                        raise ContextLengthExhausted("Prompt exhausts the context window; input was preserved")
                    cap_key = "max_tokens" if "max_tokens" in gen else "max_gen_toks"
                    gen[cap_key] = min(int(gen.get(cap_key, self._max_gen_toks)), room)
                prepared.append(replace(request, arguments=(args[0], gen, *args[2:])))
            requests = prepared

        def _collate_gen(_requests):
            # sort by the length of the non-tokenized contexts
            return -len(_requests[0])

        # Let the API deal with tokenization
        if len(requests[0].args) > 2:
            assert self.tokenizer is None, (
                "tokenizer is not supported for multimodal requests yet!"
            )
            eval_logger.info(
                f"Using max_images {self.max_images}. Set in the model args."
            )
            requests, all_gen_kwargs, auxiliary_args = zip(
                *(req.args for req in requests)
            )
            requests = tuple(
                JsonChatStr(
                    json.dumps(
                        create_image_prompt(
                            y["visual"][: self.max_images], json.loads(x.prompt)
                        )
                    )
                )
                for x, y in zip(requests, auxiliary_args)
            )
        else:
            requests, all_gen_kwargs = zip(*(req.args for req in requests))
        if self.tokenized_requests:
            encodings_list = self.tok_encode(
                requests, add_special_tokens=self.add_bos_token
            )
        else:
            encodings_list = [None] * len(requests)
        requests = [
            (a, b, c, k)
            for a, b, c, k in zip(
                requests, all_gen_kwargs, encodings_list, item_keys
            )
        ]

        re_ord = Collator(
            requests,
            sort_fn=_collate_gen,
            group_by="gen_kwargs",
            group_fn=(lambda x: {} if independent else x[1]),
        )
        chunked = re_ord.get_batched(
            n=self._batch_size if self._concurrent <= 1 else 0, batch_fn=None
        )
        if not self.tokenized_requests:
            eval_logger.info(
                "Tokenized requests are disabled. Context + generation length is not checked."
            )
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                contexts, all_gen_kwargs, encodings_list, chunk_item_keys = zip(*chunk)
                if self.tokenized_requests:
                    max_gen_toks = all_gen_kwargs[0].get(
                        "max_gen_toks", self._max_gen_toks
                    )
                    max_context_len = self.max_length - max_gen_toks

                    encodings_list = [x if strict_context else x[-max_context_len:] for x in encodings_list]

                    if any(
                        len(x) + max_gen_toks > self.max_length for x in encodings_list
                    ):
                        eval_logger.warning(
                            f"Some contexts exceeded (max length: ({self.max_length}) - max_gen_toks: ({max_gen_toks}). They were left truncated."
                        )

                req = encodings_list if self.tokenized_requests else contexts
                try:
                    outputs = retry(
                        stop=stop_after_attempt(self.max_retries),
                        wait=wait_exponential(multiplier=0.5, min=1, max=10),
                        reraise=True,
                        retry=retry_if_not_exception_type(ContextLengthExhausted),
                    )(self.model_call)(
                        messages=req,
                        generate=True,
                        gen_kwargs=copy.deepcopy(all_gen_kwargs[0]),
                    )
                    generations = self.parse_generations(
                        outputs=outputs,
                        contexts=contexts,
                    )
                    generations = [self._annotate_generation(value, all_gen_kwargs[0]) for value in generations]
                except ContextLengthExhausted as e:
                    if strict_context:
                        raise
                    eval_logger.warning(
                        f"Context length exhausted for generate_until request; "
                        f"recording empty completion(s) instead of failing the run: {e}"
                    )
                    generations = [_ContextExhaustedCompletion()] * len(contexts)
                for generated_text, context, item_key in zip(
                    generations,
                    contexts,
                    chunk_item_keys,
                ):
                    # Always append to res to maintain the correct number of items
                    # even if generation failed (generated_text is None)
                    if generated_text is None:
                        eval_logger.warning(
                            f"API returned null content. Content filled with `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = {LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER}`. Check reasoning_content field or generation limits."
                        )
                        # Patch "None" answer with consistent value (async and sync calls)
                        res.append(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER)
                    else:
                        res.append(generated_text)

                    # partial caching only for successful generations
                    if generated_text is not None and context is not None:
                        self.cache_hook.add_partial(
                            "generate_until",
                            (context, all_gen_kwargs[0]),
                            generated_text,
                        )
                    if (
                        generated_text is not None
                        and item_key is not None
                        and self.checkpoint_store is not None
                    ):
                        self.checkpoint_store.mark_items_completed(
                            [(item_key.item_id, serialize_response(generated_text))]
                        )
                    pbar.update(1)
        else:
            for chunk in chunked:
                contexts, all_gen_kwargs, encodings_list, chunk_item_keys = zip(*chunk)
                if self.tokenized_requests:
                    max_gen_toks = all_gen_kwargs[0].get(
                        "max_gen_toks", self._max_gen_toks
                    )
                    max_context_len = self.max_length - max_gen_toks

                    encodings_list = [x if strict_context else x[-max_context_len:] for x in encodings_list]

                    if any(
                        len(x) + max_gen_toks > self.max_length for x in encodings_list
                    ):
                        eval_logger.warning(
                            f"Some contexts exceeded (max length: ({self.max_length}) - max_gen_toks ({max_gen_toks}). They were left truncated."
                        )

                req = encodings_list if self.tokenized_requests else contexts
                results = itertools.chain.from_iterable(
                    asyncio.run(
                        self.get_batched_requests(
                            req,
                            cache_keys=list(zip(contexts, all_gen_kwargs, strict=True)),
                            item_keys=list(chunk_item_keys),
                            generate=True,
                            gen_kwargs=copy.deepcopy(all_gen_kwargs[0]),
                            **({"gen_kwargs_per_request": list(all_gen_kwargs)} if independent else {}),
                        )
                    )
                )
                # Append results to res list
                for r in results:
                    res.append(r)

        ordered = re_ord.get_original(res)
        if independent:
            from lm_eval.models.openai_completions import _RWKVCompletion

            for index, (value, seed) in enumerate(zip(ordered, prepared_seeds, strict=True)):
                if not hasattr(value, "__dict__"):
                    value = _RWKVCompletion(str(value))
                    ordered[index] = value
                value.seed = seed
                value.requires_think_close = getattr(self, "rwkv_generation_prompt", None) == "open_think"
        return ordered

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context)
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods
