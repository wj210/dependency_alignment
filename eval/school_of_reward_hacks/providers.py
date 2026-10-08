"""Single-call provider adapters; the evaluation runner owns all retries."""

import os
import re

import httpx

from dependency_alignment.generation import (
    SubscriptionAuthError,
    _check_cached_subscription_auth,
    configure_subscription_auth,
)
from dependency_alignment.pilot import PROJECT_ROOT


JUDGE_INSTRUCTIONS = "You are a helpful assistant."
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _vllm_request(messages: list[dict], settings: dict) -> dict:
    request = {"model": settings["model"], "messages": messages, "stream": False}
    for key in ("temperature", "max_tokens", "top_p", "seed"):
        if key in settings and settings[key] is not None:
            request[key] = settings[key]
    if settings.get("chat_template_kwargs"):
        request["chat_template_kwargs"] = settings["chat_template_kwargs"]
    api_key = os.environ.get(settings.get("api_key_env", "VLLM_API_KEY"), "EMPTY")
    return {
        "url": settings["base_url"].rstrip("/") + "/chat/completions",
        "json": request, "headers": {"Authorization": f"Bearer {api_key}"},
    }


def _parse_vllm_response(response: httpx.Response, settings: dict) -> dict:
    if response.status_code >= 400:
        raise ProviderError("VLLMHttpError", response.status_code)
    try:
        payload = response.json()
    except Exception as error:
        raise ProviderError(type(error).__name__) from None
    if not isinstance(payload, dict):
        raise ProviderError("InvalidVLLMResponse")
    if payload.get("error"):
        raise ProviderError("VLLMResponseError")
    try:
        choice = payload["choices"][0]
        if choice.get("error"):
            raise ProviderError("VLLMChoiceError")
        message = choice["message"]
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise ProviderError("InvalidTargetText")
        return {
            "text": text, "response_id": payload.get("id"),
            "model": payload.get("model") or settings["model"],
            "configured_model": settings["model"],
            "usage": payload.get("usage"),
            "finish_reason": choice.get("finish_reason"),
            "reasoning": message.get("reasoning") or message.get("reasoning_content"),
            "refusal": message.get("refusal"),
            "reasoning_details": message.get("reasoning_details"),
            "provider": "vllm", "upstream_provider": None,
            "system_fingerprint": payload.get("system_fingerprint"),
        }
    except ProviderError:
        raise
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("InvalidVLLMResponse") from None



def _target_response(messages: list[dict], settings: dict) -> dict:
    """Generate through the configured OpenAI-compatible local vLLM server."""
    try:
        with httpx.Client(timeout=settings["timeout_seconds"]) as client:
            response = client.post(**_vllm_request(messages, settings))
    except Exception as error:
        raise ProviderError(type(error).__name__, getattr(error, "status_code", None)) from None
    return _parse_vllm_response(response, settings)


async def async_generate(messages: list[dict], settings: dict, client: httpx.AsyncClient) -> dict:
    """Send one vLLM request using the runner's shared connection pool."""
    if settings.get("route") != "vllm":
        raise ValueError("Async target generation requires the vllm route")
    try:
        response = await client.post(**_vllm_request(messages, settings))
    except Exception as error:
        raise ProviderError(type(error).__name__, getattr(error, "status_code", None)) from None
    return _parse_vllm_response(response, settings)


def _matches_model(model: str, expected: str) -> bool:
    """Allow a dated release of the requested model, not a different variant."""
    return isinstance(model, str) and (model == expected or re.fullmatch(
        re.escape(expected) + r"-\d{4}-?\d{2}-?\d{2}", model) is not None)


class ProviderError(RuntimeError):
    """A provider failure whose message never contains credentials or headers."""

    def __init__(self, error_type: str, http_status: int | None = None):
        self.error_type = error_type
        self.http_status = http_status
        status = f"; HTTP {http_status}" if http_status is not None else ""
        super().__init__(f"Provider call failed ({error_type}{status})")


def prepare_providers(config: dict) -> None:
    """Check credentials and configure the judge once, before starting workers."""
    target_route = config.get("generation", {}).get("route", "openrouter")
    if target_route not in ("openrouter", "vllm"):
        raise ValueError("Target route must be openrouter or vllm")
    if target_route == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        from dotenv import dotenv_values

        api_key = dotenv_values(PROJECT_ROOT / ".env").get("OPENROUTER_API_KEY")
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
    if target_route == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        raise ProviderError("MissingOpenRouterApiKey")
    prepare_judge(config["judging"])


def prepare_judge(settings: dict) -> None:
    """Configure subscription authentication and streaming metadata once."""
    if settings["model"] != "chatgpt/gpt-6-sol":
        raise ValueError("The requested judge model is chatgpt/gpt-6-sol")
    configure_subscription_auth()
    # The document generator's workplace instructions do not belong in judging.
    os.environ["CHATGPT_DEFAULT_INSTRUCTIONS"] = JUDGE_INSTRUCTIONS
    import litellm

    litellm.suppress_debug_info = True
    # LiteLLM 1.104.0 lacks this model entry and otherwise sends a non-streaming
    # request. The Codex subscription endpoint requires native streaming.
    litellm.register_model({settings["model"]: {
        "litellm_provider": "chatgpt", "mode": "responses",
        "supports_native_streaming": True, "supports_reasoning": True,
    }})


def generate(messages: list[dict], settings: dict) -> dict:
    """Send the benchmark's messages unchanged through the selected target route."""
    if settings.get("route", "openrouter") == "vllm":
        return _target_response(messages, settings)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ProviderError("MissingOpenRouterApiKey")
    request = {"model": settings["model"], "messages": messages, "stream": False}
    for key in ("temperature", "max_tokens", "top_p", "seed", "reasoning", "provider"):
        if key in settings and settings[key] is not None:
            request[key] = settings[key]
    if settings.get("reasoning_effort"):
        request["reasoning"] = {"effort": settings["reasoning_effort"]}
    try:
        # HTTPX does not retry requests unless its transport is explicitly set to.
        with httpx.Client(timeout=settings["timeout_seconds"]) as client:
            response = client.post(
                OPENROUTER_URL, json=request,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if response.status_code >= 400:
            raise ProviderError("OpenRouterHttpError", response.status_code)
        payload = response.json()
    except ProviderError:
        raise
    except Exception as error:
        raise ProviderError(type(error).__name__, getattr(error, "status_code", None)) from None
    if payload.get("error"):
        code = payload["error"].get("code") if isinstance(payload["error"], dict) else None
        raise ProviderError("OpenRouterResponseError", code if type(code) is int else None)
    try:
        choice = payload["choices"][0]
        if choice.get("error"):
            raise ProviderError("OpenRouterChoiceError")
        message = choice["message"]
        model = payload["model"]
        text = message.get("content") or ""
        if not _matches_model(model, settings["model"]):
            raise ProviderError("UnexpectedTargetModel")
        if not isinstance(text, str):
            raise ProviderError("InvalidTargetText")
        return {
            "text": text, "response_id": payload.get("id"), "model": model,
            "usage": payload.get("usage"), "finish_reason": choice.get("finish_reason"),
            "reasoning": message.get("reasoning"),
            "refusal": message.get("refusal"),
            "reasoning_details": message.get("reasoning_details"),
            "provider": "openrouter", "upstream_provider": payload.get("provider"),
            "system_fingerprint": payload.get("system_fingerprint"),
        }
    except ProviderError:
        raise
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("InvalidOpenRouterResponse") from None


def judge(messages: list[dict], settings: dict) -> dict:
    """Use the exact judge messages with GPT-6 Sol and medium reasoning."""
    try:
        _check_cached_subscription_auth()
    except SubscriptionAuthError:
        raise ProviderError("SubscriptionAuthError", 401) from None
    import litellm

    litellm.suppress_debug_info = True
    inputs = [{"role": message["role"], "content": [{
        "type": "output_text" if message["role"] == "assistant" else "input_text",
        "text": message["content"],
    }]} for message in messages]
    request = {
        "model": settings["model"], "input": inputs,
        "instructions": JUDGE_INSTRUCTIONS,
        "reasoning": {"effort": settings["reasoning_effort"]},
        "stream": True, "timeout": settings["timeout_seconds"], "num_retries": 0,
    }
    parts = {}
    completed = None
    try:
        for event in litellm.responses(**request):
            payload = event.model_dump(mode="json", exclude_none=True, warnings=False)
            event_type = payload.get("type")
            if event_type in ("response.output_text.delta", "response.output_text.done"):
                index = (payload["output_index"], payload["content_index"])
                if event_type.endswith(".delta"):
                    parts[index] = parts.get(index, "") + payload["delta"]
                else:
                    parts[index] = payload["text"]
            elif event_type == "response.completed":
                completed = payload["response"]
            elif event_type in ("response.failed", "response.incomplete", "error"):
                raise ProviderError("UnsuccessfulJudgeResponse")
    except ProviderError:
        raise
    except Exception as error:
        raise ProviderError(type(error).__name__, getattr(error, "status_code", None)) from None
    if completed is None or completed.get("status") != "completed":
        raise ProviderError("IncompleteJudgeStream")
    model = completed.get("model", "")
    expected = settings["model"].removeprefix("chatgpt/")
    if not _matches_model(model, expected):
        raise ProviderError("UnexpectedJudgeModel")
    text = "".join(parts[key] for key in sorted(parts))
    if not text:
        text = "".join(part["text"] for item in completed.get("output", [])
                       if item.get("type") == "message"
                       for part in item.get("content", [])
                       if part.get("type") == "output_text")
    reasoning = [item.get("summary", []) for item in completed.get("output", [])
                 if item.get("type") == "reasoning"]
    return {
        "text": text, "response_id": completed.get("id"), "model": model,
        "usage": completed.get("usage"), "finish_reason": "stop",
        "reasoning": reasoning, "provider": "litellm_chatgpt_subscription",
    }
