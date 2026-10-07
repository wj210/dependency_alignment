"""Cached standalone text generation using LiteLLM's subscription provider."""

import argparse
import base64
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from dependency_alignment.pilot import (
    PROJECT_ROOT, digest, json_bytes, render_ideas_prompt, validate_config, validate_ideas,
)


INSTRUCTIONS = "You write workplace documents and structured document plans."


class GenerationError(RuntimeError):
    """A provider failure containing only the receipt's error classification."""

    def __init__(self, error_type: str, http_status: int | None = None,
                 provider_error: dict | None = None):
        self.error_type = error_type
        self.http_status = http_status
        self.provider_error = provider_error or {}
        status = f"; HTTP {http_status}" if http_status is not None else ""
        super().__init__(f"Generation failed ({error_type}{status})")


def build_generation_request(prompt: str, generation: dict) -> dict:
    """Build the exact standalone subscription request used for generation."""
    return {"model": generation["model"],
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "reasoning": {"effort": generation["reasoning_effort"]},
            "stream": True,
            "timeout": generation["timeout_seconds"], "num_retries": 0}


def generation_request_hash(prompt: str, generation: dict) -> str:
    """Hash the request and subscription instructions, matching saved receipts."""
    request = build_generation_request(prompt, generation)
    return digest(json_bytes({"instructions": INSTRUCTIONS, **request}))


def configure_subscription_auth() -> None:
    """Copy only current session tokens to a separate, private LiteLLM cache."""
    source = Path.home() / ".codex" / "auth.json"
    if not source.exists():
        raise ValueError("Codex file credentials are unavailable; sign in with Codex first")
    data = json.loads(source.read_bytes())
    if data.get("auth_mode") != "chatgpt":
        raise ValueError("Codex must be signed in with ChatGPT subscription authentication")
    tokens = data.get("tokens", {})
    access_token = tokens.get("access_token")
    if not access_token:
        raise ValueError("The Codex session has no access token")
    encoded = access_token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    expires_at = claims.get("exp", 0)
    if expires_at <= time.time() + 240:
        raise ValueError("The Codex session needs refreshing; use Codex before retrying")
    token_dir = Path.home() / ".config" / "dependency-alignment" / "chatgpt"
    token_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    token_dir.chmod(0o700)
    auth_file = token_dir / "auth.json"
    auth = {key: tokens[key] for key in ("access_token", "id_token", "account_id")
            if key in tokens}
    auth["expires_at"] = expires_at
    # No refresh token is copied: this pilot never rotates the Codex session.
    descriptor = os.open(auth_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(auth))
    os.environ["CHATGPT_TOKEN_DIR"] = str(token_dir)
    os.environ["CHATGPT_AUTH_FILE"] = "auth.json"
    os.environ["CHATGPT_API_BASE"] = "https://chatgpt.com/backend-api/codex"
    os.environ["CHATGPT_DEFAULT_INSTRUCTIONS"] = INSTRUCTIONS
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def generate_text(prompt: str, generation: dict, output: Path,
                  dry_run: bool = False, responder=None) -> dict:
    """Make at most one call; retain text events and reuse completed receipts."""
    request = build_generation_request(prompt, generation)
    request_record = {"instructions": INSTRUCTIONS, **request}
    request_hash = generation_request_hash(prompt, generation)
    request_path = output.with_suffix(".request.json")
    receipt_path = output.with_suffix(".receipt.json")
    completion_path = output.with_suffix(".completion.txt")
    events_path = output.with_suffix(".events.jsonl")
    if request_path.exists() and json.loads(request_path.read_bytes()) != request_record:
        raise ValueError("Generation request changed; use a fresh output path")
    if output.exists() and not receipt_path.exists():
        raise ValueError("Existing output has no provider receipt; use a fresh output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(json_bytes(request_record))
    if dry_run:
        return {"status": "dry_run", "planned_calls": 1, "external_api_calls": 0,
                "model": request["model"], "reasoning": request["reasoning"]}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        if receipt["request_hash"] != request_hash:
            raise ValueError("Previous attempt used another request")
        if receipt["status"] != "completed":
            raise ValueError("Previous call did not complete; inspect its receipt before a new attempt")
        response_data = receipt["response"]
        text = completion_path.read_text()
    else:
        if responder is None:
            configure_subscription_auth()
            import litellm
            litellm.suppress_debug_info = True
            responder = litellm.responses
        started = datetime.now(timezone.utc).isoformat()
        receipt = {"status": "started", "request_hash": request_hash,
                   "started_at": started, "external_api_calls": 1,
                   "litellm_version": importlib.metadata.version("litellm")}
        # Exclusive creation prevents concurrent duplicate calls for this output.
        with receipt_path.open("xb") as handle:
            handle.write(json_bytes(receipt))
        try:
            response_data = None
            text_parts = {}
            with events_path.open("x") as event_log:
                for event in responder(**request):
                    payload = event.model_dump(mode="json", exclude_none=True, warnings=False)
                    event_type = payload.get("type")
                    if event_type in ("response.output_text.delta", "response.output_text.done"):
                        index = (payload["output_index"], payload["content_index"])
                        if event_type == "response.output_text.delta":
                            text_parts[index] = text_parts.get(index, "") + payload["delta"]
                        else:
                            text_parts[index] = payload["text"]
                        saved = {key: payload[key] for key in
                                 ("type", "output_index", "content_index", "delta", "text")
                                 if key in payload}
                        event_log.write(json.dumps(saved, ensure_ascii=False) + "\n")
                        event_log.flush()
                    elif event_type == "response.completed":
                        response_data = payload["response"]
                    elif event_type in ("response.failed", "response.incomplete", "error"):
                        raise RuntimeError("The provider reported an unsuccessful response")
            if response_data is None:
                raise RuntimeError("The response stream ended without completion")
            text = "".join(text_parts[key] for key in sorted(text_parts))
            if not text:
                text = "".join(part["text"] for item in response_data.get("output", [])
                               if item.get("type") == "message"
                               for part in item.get("content", [])
                               if part.get("type") == "output_text")
            completion_path.write_text(text)
            receipt["response"] = {key: response_data.get(key)
                                   for key in ("id", "model", "status", "usage")}
            receipt["status"] = "completed"
            receipt_path.write_bytes(json_bytes(receipt))
        except Exception as error:
            # Exception strings and headers can contain credentials; exclude them.
            receipt.update(status="failed", error_type=type(error).__name__,
                           http_status=getattr(error, "status_code", None))
            frames = []
            traceback = error.__traceback__
            while traceback is not None:
                frames.append({"file": Path(traceback.tb_frame.f_code.co_filename).name,
                               "function": traceback.tb_frame.f_code.co_name,
                               "line": traceback.tb_lineno})
                traceback = traceback.tb_next
            receipt["error_frames"] = frames
            body = getattr(error, "body", None)
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                # Retain the provider's error classification without raw request text.
                receipt["provider_error"] = {key: body["error"].get(key)
                                             for key in ("type", "code", "param")}
            receipt_path.write_bytes(json_bytes(receipt))
            raise RuntimeError(f"Generation failed ({type(error).__name__}); see the receipt") from None
    try:
        if response_data.get("status") != "completed":
            raise ValueError("Provider did not report a completed response")
        model = response_data.get("model", "")
        if model != "gpt-6.1-sol" and not model.startswith("gpt-6.1-sol-"):
            raise ValueError(f"Provider returned an unexpected model: {model}")
    except (ValueError, KeyError, TypeError) as error:
        receipt.update(validation_status="failed", validation_error=type(error).__name__)
        receipt_path.write_bytes(json_bytes(receipt))
        raise
    provenance = {"source": "litellm_chatgpt_subscription", "model": model,
                  "requested_model": request["model"],
                  "reasoning_effort": generation["reasoning_effort"],
                  "request_hash": request_hash, "response_id": response_data.get("id"),
                  "usage": response_data.get("usage"), "temperature": None,
                  "generation_seed": None, "started_at": receipt["started_at"],
                  "litellm_version": receipt["litellm_version"],
                  "implementation_hash": digest(Path(__file__).read_bytes())}
    return {"provenance": provenance, "text": text}


def generate_compact_text(prompt: str, generation: dict, responder=None) -> dict:
    """Generate text without retaining temporary files; configure auth once first."""
    if responder is None:
        import litellm
        litellm.suppress_debug_info = True
        responder = litellm.responses
    with TemporaryDirectory(prefix="dependency_alignment_") as temporary:
        output = Path(temporary) / "response.json"
        try:
            return generate_text(prompt, generation, output, responder=responder)
        except RuntimeError as error:
            receipt_path = output.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else {}
            if receipt.get("validation_status") == "failed":
                raise
            error_type = receipt.get("error_type", type(error).__name__)
            http_status = receipt.get("http_status")
            provider_error = receipt.get("provider_error", {})
            raise GenerationError(error_type, http_status, provider_error) from None


def generate_cached_text(prompt: str, generation: dict, cache: Path,
                         max_retries: int, validate, responder=None) -> dict:
    """Cache one response per request, retrying provider failures or invalid text."""
    if type(max_retries) is not int or not 0 <= max_retries <= 10:
        raise ValueError("max_retries must be an integer between zero and ten")
    request_hash = generation_request_hash(prompt, generation)
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / f"{request_hash}.json"
    record = json.loads(cache_path.read_bytes()) if cache_path.exists() else {
        "request_hash": request_hash, "prompt": prompt,
        "generation": generation, "instructions": INSTRUCTIONS,
        "attempts": [], "status": "pending"}
    if record["request_hash"] != request_hash:
        raise ValueError("Cached request hash differs")

    def save() -> None:
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_bytes(json_bytes(record))
        temporary.replace(cache_path)

    if record.get("response"):
        response = record["response"]
        if response["provenance"]["request_hash"] != request_hash:
            raise ValueError("Cached response request hash differs")
        try:
            validate(response["text"])
        except ValueError:
            if record["status"] == "completed":
                record["status"] = "invalid_response"
                save()
        else:
            record["status"] = "completed"
            save()
            return record
    if record["status"] == "permanent_failure":
        raise RuntimeError("Cached request has a permanent provider failure")

    while len(record["attempts"]) <= max_retries:
        attempt = {"number": len(record["attempts"]) + 1,
                   "started_at": datetime.now(timezone.utc).isoformat()}
        permanent = False
        try:
            response = generate_compact_text(prompt, generation, responder=responder)
        except GenerationError as error:
            attempt.update(status="provider_failed", error_type=error.error_type,
                           http_status=error.http_status, provider_error=error.provider_error)
            permanent = error.http_status in (400, 401, 403, 404)
            record["status"] = "permanent_failure" if permanent else "failed"
        else:
            if response["provenance"]["request_hash"] != request_hash:
                raise ValueError("Returned response request hash differs")
            record["response"] = response
            attempt.update(response_id=response["provenance"].get("response_id"),
                           usage=response["provenance"].get("usage"))
            try:
                validate(response["text"])
            except ValueError as error:
                attempt.update(status="invalid_response", error_type=type(error).__name__)
                record["status"] = "invalid_response"
            else:
                attempt["status"] = "completed"
                record["status"] = "completed"
        record["attempts"].append(attempt)
        save()
        if record["status"] == "completed":
            return record
        if permanent:
            break
        if len(record["attempts"]) <= max_retries:
            time.sleep(min(2 ** min(len(record["attempts"]), 5), 30))
    raise RuntimeError(f"Generation failed after {len(record['attempts'])} attempts; inspect the cache")


def generate_ideas(config_path: Path, output: Path, dry_run: bool = False,
                   responder=None) -> dict:
    config = json.loads(config_path.read_bytes())
    validate_config(config)
    result = generate_text(render_ideas_prompt(config), config["generation"], output,
                           dry_run=dry_run, responder=responder)
    if dry_run:
        return {**result, "planned_ideas": config["num_ideas"]}
    receipt_path = output.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_bytes())
    try:
        parsed = json.loads(result["text"])
        validate_ideas(parsed["ideas"], config["num_ideas"])
        if output.exists():
            cached = json.loads(output.read_bytes())
            if cached["provenance"]["request_hash"] != result["provenance"]["request_hash"]:
                raise ValueError("Cached ideas belong to another generation request")
            validate_ideas(cached["ideas"], config["num_ideas"])
            return cached
    except (ValueError, KeyError, TypeError) as error:
        receipt.update(validation_status="failed", validation_error=type(error).__name__)
        receipt_path.write_bytes(json_bytes(receipt))
        raise
    receipt["validation_status"] = "passed"
    receipt_path.write_bytes(json_bytes(receipt))
    result = {"provenance": result["provenance"], "ideas": parsed["ideas"]}
    temporary = output.with_suffix(".tmp")
    temporary.write_bytes(json_bytes(result))
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pilot.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/pilot/ideas.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = generate_ideas(args.config, args.output, args.dry_run)
    except (ValueError, KeyError, IndexError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    if args.dry_run:
        print(json.dumps(result))
    else:
        print(f"Validated {len(result['ideas'])} ideas in {args.output}")


if __name__ == "__main__":
    main()
