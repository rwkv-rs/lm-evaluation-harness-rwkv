import asyncio
import copy
import json
import logging
import os
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Union

from lm_eval.api.checkpoint import serialize_response
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import (
    LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER,
    ApiRequestError,
    ContextLengthExhausted,
)
from lm_eval.models.openai_completions import _RWKVCompletion, _rwkv_sampling_config
from lm_eval.models.rwkv_remote import RWKV7Remote


eval_logger = logging.getLogger(__name__)


@register_model("rwkv7-lightning-cuda")
class RWKV7LightningCUDA(RWKV7Remote):
    """RWKV7 evaluation through rwkv-lightning-cuda's native batch API.

    The official Hugging Face tokenizer stays on the evaluation side: it renders
    the selected RWKV prompt template and measures the complete prompt, while the
    server receives rendered strings through ``/v1/batch/completions``. Choice
    scoring and answer extraction are inherited from :class:`RWKV7Remote`.

    rwkv-lightning-cuda does not currently expose a request seed. Repeated
    rollouts are independently sampled by the server, but cannot be replayed
    from lm-eval's seed alone.
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/batch/completions"
    INDEPENDENT_ROLLOUTS = False
    TRANSPORT_STOPS = (
        "<EOD>",
        "\nUser:",
        "\nAssistant:",
        "\nSystem:",
        "\nTool:",
        "\nFunction:",
        "✿",
        "\n### User",
    )

    _PROFILE_FIELDS = {
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "penalty_decay",
        "alpha_presence",
        "alpha_frequency",
        "alpha_decay",
        "do_sample",
        "seed",
    }
    _PASSTHROUGH_FIELDS = {
        "adapter_id",
        "adapter_version",
        "adapter_scale",
    }

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        tokenizer=None,
        tokenizer_backend="huggingface",
        tokenized_requests=False,
        batch_size=1,
        metrics=True,
        stream=True,
        chunk_size=32,
        **kwargs,
    ):
        if not isinstance(tokenizer, str) or not tokenizer.strip():
            raise ValueError(
                "rwkv7-lightning-cuda requires the official tokenizer= path or repository."
            )
        if tokenizer_backend != "huggingface" or tokenized_requests:
            raise ValueError(
                "rwkv7-lightning-cuda requires tokenizer_backend=huggingface and "
                "tokenized_requests=False."
            )
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("rwkv7-lightning-cuda requires a positive integer batch_size.")

        requested_batch_size = int(batch_size)
        super().__init__(
            base_url=base_url,
            tokenizer=tokenizer,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=False,
            batch_size=1,
            **kwargs,
        )
        self._batch_size = requested_batch_size
        self._report_metrics = bool(metrics)
        self._stream = bool(stream)
        self._chunk_size = max(1, int(chunk_size))
        self._seed = None
        if self.rwkv_generation_prompt == "no_think":
            self._generation_profile["temperature"] = 0.01
        eval_logger.warning(
            "rwkv-lightning-cuda does not expose per-request seeds; sampled "
            "rollouts cannot be reproduced from lm-eval's seed alone."
        )

    @cached_property
    def api_key(self) -> str:
        return os.environ.get("RWKV_API_KEY", "")

    @cached_property
    def header(self) -> dict:
        headers = dict(self._header or {})
        client_id = os.environ.get("CF_ACCESS_CLIENT_ID")
        client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
        if bool(client_id) != bool(client_secret):
            raise ValueError(
                "CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET must be set together."
            )
        if client_id and client_secret:
            headers.setdefault("CF-Access-Client-Id", client_id)
            headers.setdefault("CF-Access-Client-Secret", client_secret)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        if self.rwkv_generation_prompt != "no_think":
            return super().apply_chat_template(
                chat_history,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        self.rwkv_generation_prompt = "fake_think"
        try:
            return super().apply_chat_template(
                chat_history,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        finally:
            self.rwkv_generation_prompt = "no_think"

    def _create_payload(
        self,
        messages: Union[str, Sequence[str]],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        if not generate:
            raise NotImplementedError(
                "rwkv-lightning-cuda does not expose prompt log probabilities."
            )

        if isinstance(messages, str):
            contents = [messages]
        else:
            contents = list(messages)
        if not contents or any(not isinstance(prompt, str) for prompt in contents):
            raise TypeError(
                "rwkv-lightning-cuda expects one or more rendered string prompts."
            )

        request_args = dict(gen_kwargs or {})
        if "max_tokens" in request_args:
            requested_max_tokens = int(request_args.pop("max_tokens"))
        else:
            requested_max_tokens = int(
                request_args.pop("max_gen_toks", self._max_gen_toks)
            )
        if requested_max_tokens < 1:
            raise ValueError("max_tokens must be positive.")

        until = request_args.pop("until", None)
        if until is None:
            stop = []
        elif isinstance(until, str):
            stop = [until]
        else:
            stop = list(until)
        if any(not isinstance(value, str) or not value for value in stop):
            raise ValueError("until must contain only non-empty strings.")
        stop = list(dict.fromkeys([*stop, *self.TRANSPORT_STOPS]))

        for field in self._PROFILE_FIELDS:
            request_args.pop(field, None)
        passthrough = {
            field: request_args.pop(field)
            for field in tuple(request_args)
            if field in self._PASSTHROUGH_FIELDS
        }
        if request_args:
            raise ValueError(
                "Unsupported rwkv-lightning-cuda generation fields: "
                + ", ".join(sorted(request_args))
            )

        prompt_encodings = self.tok_encode(
            contents, add_special_tokens=self.add_bos_token
        )
        remaining_tokens = [
            self.max_length - len(encoding) for encoding in prompt_encodings
        ]
        if min(remaining_tokens) <= 0:
            raise ContextLengthExhausted(
                "Prompt exhausts the context window; input was preserved."
            )
        max_tokens = min(requested_max_tokens, *remaining_tokens)

        profile = self._generation_profile
        return {
            "model": self.model,
            "contents": contents,
            "stream": self._stream,
            "chunk_size": self._chunk_size,
            "max_tokens": max_tokens,
            "temperature": profile["temperature"],
            "top_p": profile["top_p"],
            "top_k": profile["top_k"],
            "alpha_presence": profile["presence_penalty"],
            "alpha_frequency": profile["frequency_penalty"],
            "alpha_decay": profile["penalty_decay"],
            "stop": stop,
            "force_reasoning": self.rwkv_generation_prompt == "open_think",
            "metrics": self._report_metrics,
            **passthrough,
        }

    def get_model_info(self) -> dict:
        sampling = _rwkv_sampling_config(
            {
                **self._generation_profile,
                "stop_strings": list(self.TRANSPORT_STOPS),
            }
        )
        sampling["seed"] = -1
        sampling["seed_strategy"] = "unsupported"
        return {"sampling_config": sampling}

    @staticmethod
    def _sse_output(events: Sequence[dict], expected: int, saw_done: bool) -> dict:
        if not saw_done:
            raise ApiRequestError("rwkv-lightning-cuda SSE ended without data: [DONE].")
        texts = [""] * expected
        endings = [None] * expected
        matched_stops = [None] * expected
        for event in events:
            if "error" in event:
                raise ApiRequestError(f"rwkv-lightning-cuda SSE error: {event['error']}")
            for choice in event.get("choices", []):
                index = choice.get("index")
                if not isinstance(index, int) or not 0 <= index < expected:
                    raise ApiRequestError(
                        f"rwkv-lightning-cuda returned invalid choice index {index!r}."
                    )
                content = choice.get("delta", {}).get("content")
                if content is None:
                    content = choice.get("message", {}).get("content")
                if content:
                    texts[index] += content
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    if endings[index] is not None:
                        raise ApiRequestError(
                            f"rwkv-lightning-cuda returned duplicate terminal event for index {index}."
                        )
                    endings[index] = finish_reason
                    matched_stops[index] = choice.get("matched_stop")
        missing = [index for index, reason in enumerate(endings) if reason is None]
        if missing:
            raise ApiRequestError(
                "rwkv-lightning-cuda SSE ended without terminal events for indices "
                + ", ".join(map(str, missing))
            )
        choices = []
        for index, (text, finish_reason, matched_stop) in enumerate(
            zip(texts, endings, matched_stops, strict=True)
        ):
            choice = {
                "index": index,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
            if matched_stop is not None:
                choice["matched_stop"] = matched_stop
            choices.append(choice)
        return {"object": "chat.completion", "choices": choices}

    @classmethod
    def _consume_sync_sse(cls, response, expected: int) -> dict:
        events = []
        saw_done = False
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line.strip()
            if not line or line.startswith("event:"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            events.append(json.loads(data))
        return cls._sse_output(events, expected, saw_done)

    @classmethod
    async def _consume_async_sse(cls, response, expected: int) -> dict:
        events = []
        saw_done = False
        while True:
            raw_line = await response.content.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith("event:"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError as error:
                raise ApiRequestError(
                    f"rwkv-lightning-cuda returned invalid SSE JSON: {data!r}"
                ) from error
        return cls._sse_output(events, expected, saw_done)

    def model_call(
        self,
        messages,
        *,
        generate=True,
        gen_kwargs=None,
        extra_headers=None,
        **kwargs,
    ):
        if not self._stream:
            return super().model_call(
                messages,
                generate=generate,
                gen_kwargs=gen_kwargs,
                extra_headers=extra_headers,
                **kwargs,
            )
        import requests

        request_messages = self.create_message(messages)
        payload = self._create_payload(
            request_messages,
            generate=generate,
            gen_kwargs=copy.deepcopy(gen_kwargs),
            **kwargs,
        )
        headers = {**self.header, **(extra_headers or {})}
        response = requests.post(
            self._endpoint_url(),
            json=payload,
            headers=headers,
            verify=self.verify_certificate,
            timeout=self.timeout,
            stream=True,
        )
        if not response.ok:
            raise ApiRequestError(f"HTTP {response.status_code}: {response.text}")
        expected = 1 if isinstance(request_messages, str) else len(request_messages)
        return self._consume_sync_sse(response, expected)

    async def amodel_call(
        self,
        session,
        sem: asyncio.Semaphore,
        messages,
        *,
        generate=True,
        cache_keys=None,
        item_keys=None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ):
        if not self._stream:
            return await super().amodel_call(
                session,
                sem,
                messages,
                generate=generate,
                cache_keys=cache_keys,
                item_keys=item_keys,
                ctxlens=ctxlens,
                gen_kwargs=gen_kwargs,
                **kwargs,
            )
        request_messages = self.create_message(messages)
        payload = self._create_payload(
            request_messages,
            generate=generate,
            gen_kwargs=copy.deepcopy(gen_kwargs),
            **kwargs,
        )
        acquired = await sem.acquire()
        try:
            async with session.post(
                self._endpoint_url(),
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ApiRequestError(f"HTTP {response.status}: {error_text}")
                expected = (
                    1 if isinstance(request_messages, str) else len(request_messages)
                )
                output = await self._consume_async_sse(
                    response, expected
                )
            answers = self.parse_generations(output)
            answers = [
                value
                if value is not None
                else LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER
                for value in answers
            ]
            if cache_keys:
                for answer, cache_key in zip(answers, cache_keys):
                    self.cache_hook.add_partial("generate_until", cache_key, answer)
            if item_keys and self.checkpoint_store is not None:
                completed = [
                    (key.item_id, serialize_response(answer))
                    for key, answer in zip(item_keys, answers)
                    if key is not None
                ]
                if completed:
                    await self.checkpoint_store.mark_items_completed_async(completed)
            return answers
        finally:
            if acquired:
                sem.release()

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations = []
        for output in outputs:
            choices = [None] * len(output["choices"])
            for choice in output["choices"]:
                text = choice.get("message", {}).get("content", "")
                matched_stop = choice.get("matched_stop")
                if matched_stop and text.endswith(matched_stop):
                    text = text[: -len(matched_stop)]
                choices[choice["index"]] = _RWKVCompletion(
                    text,
                    choice.get("finish_reason"),
                    raw_response=output,
                )
            generations.extend(choices)
        return generations
