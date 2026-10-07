import copy
import io
import json
import tempfile
import unittest
from concurrent.futures import wait as real_wait
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dependency_alignment import document_preview as writer
from dependency_alignment.generation import GenerationError, SubscriptionAuthError, generation_request_hash
from test_document_preview import case_fixture


class CorpusContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for folder in ("configs", "data", "prompts"):
            (self.root / folder).mkdir()
        self.template = (writer.PROJECT_ROOT / "prompts/document_dependence.txt").read_bytes()
        (self.root / "prompts/document_dependence.txt").write_bytes(self.template)
        self.config = self.root / "configs/cases.json"
        self.config_data = {"generation": {"model": "chatgpt/gpt-6.1-sol",
                            "reasoning_effort": "medium", "timeout_seconds": 300},
                            "concurrency": 2, "max_retries": 10}
        self.config.write_text(json.dumps(self.config_data))
        self.cases_path = self.root / "data/cases.jsonl"
        self.save_cases(3)
        self.output = self.root / "data/documents_5000.jsonl"
        self.tracker = self.root / "configs/document_sampling.json"
        self.pilot_pair = {"case_id": self.cases[0]["case_id"], "genre": "Interview"}
        self.pilot = {"spec": {"count": 1, "output": "/old/server/data/pilot_documents.jsonl"},
                      "seed": 9, "pairs": [self.pilot_pair], "status": "completed"}
        self.initial_tracker = {
            "sampled_pairs": [writer.pair_id(self.pilot_pair["case_id"], self.pilot_pair["genre"])],
            "scenario_splits": {self.cases[0]["scenario_id"]: "development"},
            "active_batch": self.pilot,
        }
        self.tracker.write_text(json.dumps(self.initial_tracker))
        self.cache = self.root / "response_cache"
        self.root_patch = patch.object(writer, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def save_cases(self, count):
        self.cases = []
        for index in range(count):
            case = copy.deepcopy(case_fixture())
            case.update(case_id=f"scenario_{index:04d}_c01", scenario_id=f"scenario_{index:04d}")
            case["case_info"]["task_request"] = f"Prepare sales summary number {index}."
            self.cases.append(case)
        self.cases_path.write_text("".join(json.dumps(case) + "\n" for case in self.cases))

    def response(self, prompt, generation, cache, retries, validate):
        case_data = prompt.split("The fixed case, including the exact task instruction and input facts:\n", 1)[1]
        case = json.loads(case_data.split("\n\nUse these", 1)[0])
        text = case["task_request"] + " The assistant considered how the manager would use the totals."
        validate(text)
        return {"attempts": [{"number": 1, "status": "completed"}],
                "response": {"text": text, "provenance": {
                    "source": "litellm_chatgpt_subscription", "model": "gpt-6.1-sol",
                    "requested_model": generation["model"], "reasoning_effort": "medium",
                    "request_hash": generation_request_hash(prompt, generation)}}}

    def run_batch(self, count=6, seed=17, dry_run=False):
        with redirect_stdout(io.StringIO()):
            return writer.generate_corpus(self.config, self.cases_path, self.output,
                                          count, seed, dry_run, tracker_path=self.tracker, cache=self.cache)

    def rows(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def batch(self):
        return json.loads(self.tracker.read_bytes())["corpus_batches"]["data/documents_5000.jsonl"]

    def test_dry_run_is_read_only_and_makes_no_auth_or_provider_calls(self):
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(writer, "configure_subscription_auth") as auth, patch.object(writer, "generate_cached_text") as caller:
            first, second = self.run_batch(dry_run=True), self.run_batch(dry_run=True)
        auth.assert_not_called()
        caller.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(first["pending_calls"], 6)
        self.assertEqual(first["external_api_calls"], 0)
        self.assertLessEqual(len(first["sample_pairs"]), 5)
        self.assertEqual(original, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_completed_rows_resume_without_auth_or_cache_and_preserve_pilot(self):
        with patch.object(writer, "configure_subscription_auth") as auth, patch.object(writer, "generate_cached_text", side_effect=self.response) as caller:
            first = self.run_batch()
        auth.assert_called_once()
        self.assertEqual(caller.call_count, 6)
        self.assertEqual(first["status"], "completed")
        before = self.output.read_bytes()
        with patch.object(writer, "configure_subscription_auth") as auth, patch.object(writer, "generate_cached_text") as caller:
            resumed = self.run_batch()
        auth.assert_not_called()
        caller.assert_not_called()
        self.assertEqual(resumed["resumed"], 6)
        self.assertEqual(self.output.read_bytes(), before)
        tracker = json.loads(self.tracker.read_bytes())
        self.assertEqual(tracker["active_batch"], self.pilot)
        self.assertEqual(len(tracker["sampled_pairs"]), 7)
        self.assertEqual(len(tracker["generated_pairs"]), 7)
        self.assertEqual({row["document_id"] for row in self.rows()},
                         {writer.pair_id(p["case_id"], p["genre"]) + "__dependence" for p in self.batch()["pairs"]})
        for row in self.rows():
            self.assertEqual(row["split"], "development" if row["scenario_id"] == self.cases[0]["scenario_id"] else "unassigned")

    def test_independent_successes_checkpoint_and_only_failed_row_resumes(self):
        failures = set()

        def fail_once(*args):
            if not failures:
                failures.add(args[0])
                raise RuntimeError("Mock transient failure")
            return self.response(*args)

        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=fail_once):
            result = self.run_batch()
        self.assertEqual((result["status"], result["documents"], result["failed_documents"]), ("incomplete", 5, 1))
        saved = self.output.read_bytes()
        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=self.response) as caller:
            result = self.run_batch()
        self.assertEqual(caller.call_count, 1)
        self.assertEqual((result["status"], result["resumed"]), ("completed", 5))
        self.assertTrue(self.output.read_bytes().startswith(saved))
        self.assertEqual(len({row["document_id"] for row in self.rows()}), 6)

    def test_rows_after_last_tracker_checkpoint_survive_interruption(self):
        real_write = writer.atomic_write

        def interrupted_write(path, content):
            batches = json.loads(content).get("corpus_batches", {}).values()
            if any(batch["status"] == "completed" for batch in batches):
                raise OSError("Mock interruption before final tracker write")
            return real_write(path, content)

        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=self.response), patch.object(writer, "atomic_write", side_effect=interrupted_write):
            with self.assertRaises(OSError):
                self.run_batch()
        self.assertEqual(len(self.rows()), 6)
        self.assertEqual(self.batch()["checkpoint_bytes"], 0)
        # A complete uncommitted JSON row without its final newline is retained.
        self.output.write_bytes(self.output.read_bytes()[:-1])
        with patch.object(writer, "configure_subscription_auth") as auth, patch.object(writer, "generate_cached_text") as caller:
            result = self.run_batch()
        self.assertEqual(result["resumed"], 6)
        auth.assert_not_called()
        caller.assert_not_called()
        self.assertEqual(len(self.rows()), 6)
        self.assertTrue(self.output.read_bytes().endswith(b"\n"))

    def test_only_uncommitted_incomplete_tail_is_repaired(self):
        calls = 0

        def fail_once(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("Mock failure")
            return self.response(*args)

        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=fail_once):
            self.run_batch()
        with self.output.open("ab") as handle:
            handle.write(b'{"document":"unfinished')
        unchanged = self.output.read_bytes()
        planned = self.run_batch(dry_run=True)
        self.assertEqual(planned["resumed"], 5)
        self.assertGreater(planned["recoverable_tail_bytes"], 0)
        self.assertEqual(self.output.read_bytes(), unchanged)
        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=self.response):
            self.run_batch()
        self.assertEqual(len(self.rows()), 6)

    def test_changed_sources_settings_or_seed_cannot_resume(self):
        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=self.response):
            self.run_batch()
        for path in (self.config, self.cases_path, self.root / "prompts/document_dependence.txt"):
            original = path.read_bytes()
            path.write_bytes(b" " + original)
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "configuration, sources"):
                self.run_batch(dry_run=True)
            path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "configuration, sources"):
            self.run_batch(seed=18)
        with self.assertRaisesRegex(ValueError, "configuration, sources"):
            self.run_batch(count=7)

    def test_unrelated_output_and_corrupt_tracker_are_rejected_without_overwrite(self):
        self.output.write_bytes(b"Existing documents\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_batch()
        self.assertEqual(self.output.read_bytes(), b"Existing documents\n")
        self.output.unlink()
        damaged = copy.deepcopy(self.initial_tracker)
        damaged["sampled_pairs"] *= 2
        self.tracker.write_text(json.dumps(damaged))
        with self.assertRaisesRegex(ValueError, "repeated or unknown"):
            self.run_batch(dry_run=True)

    def test_completed_output_changes_and_duplicate_uncommitted_rows_fail(self):
        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=self.response):
            self.run_batch()
        original = self.output.read_bytes()
        self.output.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, "removed or changed"):
            self.run_batch(dry_run=True)
        self.output.write_bytes(original)
        tracker = json.loads(self.tracker.read_bytes())
        tracker["corpus_batches"]["data/documents_5000.jsonl"]["status"] = "incomplete"
        self.tracker.write_text(json.dumps(tracker))
        with self.output.open("ab") as handle:
            handle.write(original.splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "Repeated saved"):
            self.run_batch(dry_run=True)

    def test_shared_permanent_errors_stop_further_submission_and_keep_successes(self):
        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                self.output.unlink(missing_ok=True)
                self.tracker.write_text(json.dumps(self.initial_tracker))
                with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=GenerationError("AuthenticationError", status)) as caller:
                    result = self.run_batch()
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(caller.call_count, self.config_data["concurrency"])
                self.assertEqual(len(self.batch()["pairs"]), 6)

    def test_auth_expiry_gives_actionable_redacted_message(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=SubscriptionAuthError("Sign in with Codex, then resume generation")):
            writer.generate_corpus(self.config, self.cases_path, self.output, 6, 17,
                                   tracker_path=self.tracker, cache=self.cache)
        self.assertIn("Sign in with Codex, then resume", output.getvalue())

    def test_tracker_lock_also_protects_legacy_preview_writer(self):
        with writer.tracker_lock(self.tracker), patch.object(writer, "_generate_preview") as legacy:
            with self.assertRaisesRegex(RuntimeError, "Another document writer"):
                writer.generate_preview(self.config, self.cases_path, self.output, 6, 17, False, False)
        legacy.assert_not_called()

    def test_5000_documents_use_bounded_submissions_and_linear_checkpoints(self):
        self.save_cases(834)
        self.config_data["concurrency"] = 32
        self.config.write_text(json.dumps(self.config_data))
        widths = []

        def bounded_wait(futures, **kwargs):
            widths.append(len(futures))
            self.assertLessEqual(len(futures), 32)
            return real_wait(futures, **kwargs)

        with patch.object(writer, "configure_subscription_auth"), patch.object(writer, "generate_cached_text", side_effect=self.response) as caller, patch.object(writer, "wait", side_effect=bounded_wait), patch.object(writer, "atomic_write", wraps=writer.atomic_write) as writes:
            result = self.run_batch(count=5000)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(caller.call_count, 5000)
        self.assertEqual(max(widths), 32)
        self.assertLess(writes.call_count, 200)
        self.assertEqual(len(self.rows()), 5000)
        self.assertEqual(len({row["document_id"] for row in self.rows()}), 5000)
        with patch.object(writer, "configure_subscription_auth") as auth, patch.object(writer, "generate_cached_text") as caller:
            resumed = self.run_batch(count=5000)
        self.assertEqual(resumed["resumed"], 5000)
        auth.assert_not_called()
        caller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
