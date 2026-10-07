import base64
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dependency_alignment.generation import (
    GenerationError, INSTRUCTIONS, SubscriptionAuthError, build_generation_request,
    configure_subscription_auth, generate_cached_text, generate_compact_text,
    generation_request_hash,
)


GENERATION = {"model": "chatgpt/gpt-6.1-sol", "reasoning_effort": "medium",
              "timeout_seconds": 300}


class SubscriptionAuthContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        self.auth_path = self.home / ".codex/auth.json"
        self.auth_path.parent.mkdir()
        self.expires_at = int(time.time()) + 86400
        claims = base64.urlsafe_b64encode(json.dumps({"exp": self.expires_at}).encode()).decode()
        self.access_token = f"fixture.{claims.rstrip('=')}.signature"
        self.auth = {"auth_mode": "chatgpt", "tokens": {
            "access_token": self.access_token, "id_token": "fixture-id",
            "account_id": "fixture-account", "refresh_token": "never-copy-refresh"}}
        self.auth_path.write_text(json.dumps(self.auth))
        home_patch = patch("dependency_alignment.generation.Path.home", return_value=self.home)
        home_patch.start()
        self.addCleanup(home_patch.stop)
        env_patch = patch.dict(os.environ, {}, clear=False)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_copies_only_session_tokens_with_private_permissions_and_preserves_codex(self):
        original = self.auth_path.read_bytes()
        configure_subscription_auth()
        token_dir = Path(os.environ["CHATGPT_TOKEN_DIR"])
        copied = token_dir / os.environ["CHATGPT_AUTH_FILE"]
        self.assertEqual(json.loads(copied.read_bytes()), {
            "access_token": self.access_token, "id_token": "fixture-id",
            "account_id": "fixture-account", "expires_at": self.expires_at})
        self.assertEqual(token_dir.stat().st_mode & 0o777, 0o700)
        self.assertEqual(copied.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.auth_path.read_bytes(), original)
        self.assertEqual(os.environ["CHATGPT_API_BASE"], "https://chatgpt.com/backend-api/codex")
        self.assertEqual(os.environ["CHATGPT_DEFAULT_INSTRUCTIONS"], INSTRUCTIONS)

    def test_missing_login_and_api_key_login_fail_before_creating_private_cache(self):
        self.auth_path.unlink()
        with self.assertRaisesRegex(ValueError, "sign in with Codex"):
            configure_subscription_auth()
        self.auth_path.write_text(json.dumps({"auth_mode": "api_key", "OPENAI_API_KEY": "fixture-secret"}))
        with self.assertRaisesRegex(ValueError, "ChatGPT subscription"):
            configure_subscription_auth()
        self.assertFalse((self.home / ".config").exists())

    def test_malformed_credentials_are_rejected_without_exposing_values(self):
        for value in ("fixture-secret", [], {"auth_mode": "chatgpt", "tokens": []},
                      {"auth_mode": "chatgpt", "tokens": {"access_token": "fixture-secret"}}):
            with self.subTest(value_type=type(value).__name__):
                self.auth_path.write_text(json.dumps(value))
                with self.assertRaises(SubscriptionAuthError) as raised:
                    configure_subscription_auth()
                self.assertIsInstance(raised.exception, ValueError)
                self.assertNotIn("fixture-secret", str(raised.exception))
                self.assertFalse((self.home / ".config").exists())

    def test_expiry_is_checked_before_initial_configuration_and_each_provider_call(self):
        with patch("dependency_alignment.generation.time.time", return_value=self.expires_at - 200):
            with self.assertRaisesRegex(ValueError, "resume generation"):
                configure_subscription_auth()
        configure_subscription_auth()
        provider = Mock()
        with patch.dict("sys.modules", {"litellm": SimpleNamespace(responses=provider)}), \
                patch("dependency_alignment.generation.time.time", return_value=self.expires_at - 200):
            with self.assertRaisesRegex(ValueError, "resume generation"):
                generate_compact_text("sample", GENERATION)
        provider.assert_not_called()

    def test_removed_private_credentials_fail_without_device_login(self):
        configure_subscription_auth()
        (Path(os.environ["CHATGPT_TOKEN_DIR"]) / "auth.json").unlink()
        provider = Mock()
        with patch.dict("sys.modules", {"litellm": SimpleNamespace(responses=provider)}):
            with self.assertRaisesRegex(ValueError, "resume generation"):
                generate_compact_text("sample", GENERATION)
        provider.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("litellm"), "generation extra is not installed")
    def test_installed_provider_reads_codex_copy_without_oauth_or_api_key_fallback(self):
        configure_subscription_auth()
        from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-paid-key"}), \
                patch("litellm.llms.chatgpt.authenticator._get_httpx_client") as network:
            provider = ChatGPTResponsesAPIConfig()
            headers = provider.validate_environment({}, "gpt-6.1-sol", None)
        network.assert_not_called()
        self.assertEqual(headers["Authorization"], f"Bearer {self.access_token}")
        self.assertEqual(headers["ChatGPT-Account-Id"], "fixture-account")
        self.assertEqual(provider.authenticator.get_api_base(), "https://chatgpt.com/backend-api/codex")


class SubscriptionGenerationContracts(unittest.TestCase):
    def test_exact_route_and_reasoning_do_not_supply_a_paid_fallback(self):
        request = build_generation_request("sample", GENERATION)
        self.assertEqual(request["model"], "chatgpt/gpt-6.1-sol")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertEqual(request["num_retries"], 0)
        self.assertTrue(request["stream"])
        self.assertNotIn("api_key", request)
        self.assertNotIn("fallbacks", request)

    def test_auth_failure_stops_once_and_explicit_rerun_can_recover(self):
        for status in (401, 403):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory)
                request_hash = generation_request_hash("sample", GENERATION)
                with patch("dependency_alignment.generation.generate_compact_text",
                           side_effect=GenerationError("AuthenticationError", status)) as call, \
                        patch("dependency_alignment.generation.time.sleep") as sleep:
                    with self.assertRaises(GenerationError) as raised:
                        generate_cached_text("sample", GENERATION, cache, 10, lambda _: None)
                self.assertEqual(raised.exception.http_status, status)
                call.assert_called_once()
                sleep.assert_not_called()
                response = {"text": "completed", "provenance": {"request_hash": request_hash}}
                with patch("dependency_alignment.generation.generate_compact_text", return_value=response) as call:
                    record = generate_cached_text("sample", GENERATION, cache, 10, lambda _: None)
                call.assert_called_once()
                self.assertEqual(record["status"], "completed")
                self.assertEqual([item["status"] for item in record["attempts"]], ["provider_failed", "completed"])

    def test_auth_reruns_preserve_total_attempt_budget(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("dependency_alignment.generation.generate_compact_text",
                      side_effect=GenerationError("AuthenticationError", 401)) as call, \
                patch("dependency_alignment.generation.time.sleep") as sleep:
            for _ in range(12):
                with self.assertRaises(GenerationError):
                    generate_cached_text("sample", GENERATION, Path(directory), 10, lambda _: None)
            self.assertEqual(call.call_count, 11)
            sleep.assert_not_called()
            saved = json.loads(next(Path(directory).glob("*.json")).read_bytes())
            self.assertEqual(len(saved["attempts"]), 11)

    def test_bad_request_and_missing_model_remain_permanent_on_rerun(self):
        for status in (400, 404):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, \
                    patch("dependency_alignment.generation.generate_compact_text",
                          side_effect=GenerationError("BadRequestError", status)) as call:
                for _ in range(2):
                    with self.assertRaises(GenerationError) as raised:
                        generate_cached_text("sample", GENERATION, Path(directory), 10, lambda _: None)
                    self.assertEqual(raised.exception.http_status, status)
                call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
