import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dependency_alignment.generation import generate_ideas
from dependency_alignment.documents import (
    generate_variants, validate_case_document, validate_situations,
)
from dependency_alignment.pilot import (
    CONDITIONS, PROJECT_ROOT, build_requests, prepare, render_document_prompt,
    validate_config, validate_ideas,
)


def fixture():
    config = json.loads((PROJECT_ROOT / "configs/pilot.json").read_text())
    config.update(num_ideas=2, development_ideas=1, documents_per_idea=3)
    ideas = []
    for key in ("record_join", "queue_view"):
        ideas.append({
            "idea_key": key, "title": key, "subtask_family": key,
            "subtask": f"Perform {key}", "output": f"Artifact for {key}",
            "primary_user_id": "analyst",
            "users": [{"id": "analyst", "occupation": "Analyst", "job_scope": "Prepare cases"},
                      {"id": "reviewer", "occupation": "Reviewer", "job_scope": "Review cases"}],
            "dependencies": [{"from": "assistant_output", "to": "analyst", "use": "Organize cases"},
                             {"from": "analyst", "to": "reviewer", "use": "Read the case table"}],
            "document_idea": f"A work log about {key} and subsequent use of its output.",
        })
    return config, ideas


def response_for(ideas):
    data = {
        "id": "fixture-response", "model": "gpt-6.1-sol", "status": "completed",
        "usage": {"input_tokens": 123, "output_tokens": 456},
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"ideas": ideas})}
        ]}],
    }
    return [SimpleNamespace(model_dump=lambda **kwargs: {
        "type": "response.completed", "response": data,
    })]


def text_response(text):
    response = response_for([])
    response[0].model_dump()["response"]["output"][0]["content"][0]["text"] = text
    return response


def case_fixture(count):
    return [{"task_request": "Prepare the case table",
             "inputs": [f"Record count: {index}"]}
            for index in range(1, count + 1)]


def situation_fixture(count):
    return {"case_situations": [f"Situation {index}: review a distinct record pattern."
                                for index in range(1, count + 1)]}


class PilotContracts(unittest.TestCase):
    def test_variants_share_scenario_split(self):
        config, ideas = fixture()
        templates = {"dependence": "{scenario}"}
        records, requests = build_requests(config, ideas, templates)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(requests), 6)
        self.assertEqual(len({row["document_id"] for row in requests}), 6)
        for record in records:
            expanded = [row for row in requests if row["scenario_id"] == record["scenario_id"]]
            self.assertEqual({row["split"] for row in expanded}, {record["split"]})
            self.assertEqual({row["condition"] for row in expanded}, {"dependence"})
            self.assertEqual(len({row["case_id"] for row in expanded}), 3)
        self.assertEqual(sum(row["split"] == "development" for row in records), 1)

    def test_reordering_ideas_does_not_change_ids_or_splits(self):
        config, ideas = fixture()
        templates = {"dependence": "{scenario}"}
        self.assertEqual(build_requests(config, ideas, templates),
                         build_requests(config, list(reversed(ideas)), templates))

    def test_prompt_retains_task_and_direct_user(self):
        config, ideas = fixture()
        templates = {"dependence": "{scenario}"}
        _, requests = build_requests(config, ideas, templates)
        prompt = json.loads(requests[0]["prompt"])
        self.assertEqual(prompt["context"], config["context"])
        self.assertEqual(prompt["subtask"], ideas[1]["subtask"])
        self.assertEqual(prompt["output"], ideas[1]["output"])
        self.assertEqual(prompt["direct_user"], ideas[1]["users"][0])
        self.assertEqual(prompt["dependencies"], ideas[1]["dependencies"])
        self.assertEqual(prompt["users"], ideas[1]["users"])
        self.assertNotIn("document_idea", prompt)

    def test_bad_dependencies_are_rejected(self):
        _, ideas = fixture()
        for change in ("unknown", "cycle", "disconnected"):
            with self.subTest(change=change):
                broken = copy.deepcopy(ideas)
                if change == "unknown":
                    broken[0]["dependencies"][1]["to"] = "unknown_user"
                elif change == "cycle":
                    broken[0]["dependencies"].append({"from": "reviewer", "to": "analyst", "use": "Loop"})
                else:
                    broken[0]["dependencies"].pop()
                with self.assertRaises(ValueError):
                    validate_ideas(broken, 2)

    def test_duplicate_ideas_and_split_override_are_rejected(self):
        _, ideas = fixture()
        for field in ("idea_key", "subtask_family", "split"):
            broken = copy.deepcopy(ideas)
            if field == "split":
                broken[0][field] = "final_test"
            else:
                broken[1][field] = broken[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_ideas(broken, 2)

    def test_invalid_configuration_is_rejected(self):
        config, _ = fixture()
        for key, value in (("num_ideas", True), ("documents_per_idea", 0),
                           ("development_ideas", 2), ("document_words", [700, 450])):
            broken = copy.deepcopy(config)
            broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(broken)
        config["generation"]["model"] = "openai/gpt-6.1-sol"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_preparation_resumes_without_duplication_and_preserves_existing_run(self):
        config, ideas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, ideas_path = root / "config.json", root / "ideas.json"
            config_path.write_text(json.dumps(config))
            ideas_path.write_text(json.dumps({"provenance": {"source": "fixture"}, "ideas": ideas}))
            observed = SimpleNamespace(stdout=config["upstream_commit"] + "\n")
            with patch("dependency_alignment.pilot.subprocess.run", return_value=observed):
                first = prepare(config_path, ideas_path, root / "run")
                self.assertEqual(first, prepare(config_path, ideas_path, root / "run"))
                original = (root / "run/ideas.jsonl").read_bytes()
                ideas[0]["title"] = "Changed title"
                ideas_path.write_text(json.dumps({"provenance": {"source": "fixture"}, "ideas": ideas}))
                with self.assertRaises(ValueError):
                    prepare(config_path, ideas_path, root / "run")
                self.assertEqual((root / "run/ideas.jsonl").read_bytes(), original)
            self.assertEqual(first["generated_documents"], 0)


class GenerationContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.config, self.ideas = fixture()
        self.config_path = root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.output = root / "ideas.json"

    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_successful_response_is_cached_and_medium_reasoning_is_requested(self, _):
        responder = Mock(return_value=response_for(self.ideas))
        first = generate_ideas(self.config_path, self.output, responder=responder)
        self.assertEqual(first, generate_ideas(self.config_path, self.output, responder=responder))
        responder.assert_called_once()
        self.assertEqual(responder.call_args.kwargs["reasoning"], {"effort": "medium"})
        self.assertEqual(responder.call_args.kwargs["model"], "chatgpt/gpt-6.1-sol")
        self.assertTrue(responder.call_args.kwargs["stream"])
        self.assertNotIn("max_output_tokens", responder.call_args.kwargs)

    def test_dry_run_never_calls_provider(self):
        responder = Mock()
        result = generate_ideas(self.config_path, self.output, dry_run=True, responder=responder)
        responder.assert_not_called()
        self.assertEqual(result["external_api_calls"], 0)
        self.assertFalse(self.output.exists())

    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_invalid_response_is_retained_without_another_provider_call(self, _):
        responder = Mock(return_value=response_for(self.ideas[:1]))
        for attempt in range(2):
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                generate_ideas(self.config_path, self.output, responder=responder)
        responder.assert_called_once()
        self.assertTrue(self.output.with_suffix(".completion.txt").exists())
        self.assertFalse(self.output.exists())

    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_failure_is_redacted_and_does_not_retry_automatically(self, _):
        responder = Mock(side_effect=RuntimeError("fixture-secret-must-not-be-logged"))
        with self.assertRaises(RuntimeError) as raised:
            generate_ideas(self.config_path, self.output, responder=responder)
        receipt = self.output.with_suffix(".receipt.json").read_text()
        self.assertNotIn("fixture-secret", receipt)
        self.assertNotIn("fixture-secret", str(raised.exception))
        with self.assertRaises(ValueError):
            generate_ideas(self.config_path, self.output, responder=responder)
        responder.assert_called_once()

    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_incomplete_stream_is_not_a_success(self, _):
        responder = Mock(return_value=[SimpleNamespace(model_dump=lambda **kwargs: {
            "type": "response.output_text.delta", "delta": "partial output",
            "output_index": 0, "content_index": 0,
        })])
        with self.assertRaises(RuntimeError):
            generate_ideas(self.config_path, self.output, responder=responder)
        receipt = json.loads(self.output.with_suffix(".receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(self.output.exists())

    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_text_events_are_saved_when_completed_event_has_no_output(self, _):
        text = json.dumps({"ideas": self.ideas})
        completed = response_for(self.ideas)[0].model_dump()["response"]
        completed["output"] = []
        events = [
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
             "delta": text[:20]},
            {"type": "response.output_text.done", "output_index": 0, "content_index": 0,
             "text": text},
            {"type": "response.completed", "response": completed},
        ]
        responder = Mock(return_value=[SimpleNamespace(model_dump=lambda data=event, **kwargs: data)
                                       for event in events])
        result = generate_ideas(self.config_path, self.output, responder=responder)
        self.assertEqual(result["ideas"], self.ideas)
        self.assertEqual(self.output.with_suffix(".completion.txt").read_text(), text)
        self.assertEqual(len(self.output.with_suffix(".events.jsonl").read_text().splitlines()), 2)


class DocumentContracts(unittest.TestCase):
    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_situation_plan_and_documents_resume_with_assignments_and_splits(self, _):
        config, ideas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, ideas_path = root / "config.json", root / "ideas.json"
            config_path.write_text(json.dumps(config))
            ideas_path.write_text(json.dumps({"ideas": ideas}))
            cases = case_fixture(5)
            payloads = [{"case_info": case,
                         "document": f"{case['task_request']}\nOriginal sample {number}.\n"}
                        for number, case in enumerate(cases)]
            plan = situation_fixture(5)
            responses = [text_response(json.dumps(plan))]
            responses.extend(text_response(json.dumps(payload)) for payload in payloads)
            responder = Mock(side_effect=responses)
            first = generate_variants(config_path, ideas_path, "record_join", 5,
                                      root / "run", responder=responder)
            texts = [(root / "run" / item["file"]).read_bytes() for item in first["documents"]]
            resumed = generate_variants(config_path, ideas_path, "record_join", 5,
                                        root / "run", responder=responder)
            self.assertEqual(responder.call_count, 6)
            prompts = []
            for call in responder.call_args_list[1:]:
                prompt = call.kwargs["input"][0]["content"][0]["text"]
                prompts.append(prompt)
                self.assertIn("First, create the case_info", prompt)
                self.assertNotIn("{case_data}", prompt)
            self.assertEqual(len(set(prompts)), 5)
            for prompt, situation in zip(prompts, plan["case_situations"]):
                self.assertIn(situation, prompt)
                self.assertNotIn("{case_situation}", prompt)
            self.assertEqual(len(first["documents"]), 5)
            self.assertEqual(first["external_api_calls"], 6)
            self.assertEqual(len({item["opening_style"] for item in first["documents"]}), 5)
            for record, payload, situation in zip(first["documents"], payloads,
                                                 plan["case_situations"]):
                self.assertEqual(record["case_situation"], situation)
                self.assertEqual(json.loads((root / "run" / record["json_file"]).read_bytes()), payload)
                self.assertEqual((root / "run" / record["file"]).read_text(), payload["document"])
            self.assertEqual(first["documents"], resumed["documents"])
            records, _ = build_requests(config, ideas, {name: "{scenario}" for name in CONDITIONS})
            selected = next(item for item in records if item["idea_key"] == "record_join")
            self.assertEqual({item["split"] for item in first["documents"]}, {selected["split"]})
            self.assertTrue(all(not item["within_word_target"] for item in first["documents"]))
            ideas[0]["subtask"] = "A different task"
            ideas_path.write_text(json.dumps({"ideas": ideas}))
            with self.assertRaises(ValueError):
                generate_variants(config_path, ideas_path, "record_join", 5,
                                  root / "run", responder=responder)
            self.assertEqual(responder.call_count, 6)
            self.assertEqual(texts, [(root / "run" / item["file"]).read_bytes()
                                    for item in first["documents"]])

    def test_preview_dry_run_never_generates_documents(self):
        config, ideas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, ideas_path = root / "config.json", root / "ideas.json"
            config_path.write_text(json.dumps(config))
            ideas_path.write_text(json.dumps({"ideas": ideas}))
            responder = Mock()
            manifest = generate_variants(config_path, ideas_path, "record_join", 5,
                                         root / "run", dry_run=True, responder=responder)
            responder.assert_not_called()
            self.assertEqual(manifest["external_api_calls"], 0)
            self.assertEqual(manifest["documents"], [])
            self.assertEqual(len(list((root / "run").glob("*.request.json"))), 1)
            self.assertTrue((root / "run/document_template.txt").exists())
            self.assertFalse(list((root / "run").glob("*.md")))

    def test_dependence_prompt_retains_assigned_situation_and_metadata(self):
        config, ideas = fixture()
        situation = situation_fixture(1)["case_situations"][0]
        dependence = render_document_prompt(
            config, ideas[0], (PROJECT_ROOT / "prompts/case_document_dependence.txt").read_text(),
            case_situation=situation)
        self.assertIn(situation, dependence)
        self.assertIn(ideas[0]["subtask"], dependence)
        self.assertIn(json.dumps(ideas[0]["users"], indent=2), dependence)
        self.assertIn(json.dumps(ideas[0]["dependencies"], indent=2), dependence)
        self.assertIn("third-person narrator", dependence)
        self.assertIn("quote the complete, exact task instruction", dependence)
        self.assertIn("Vary the surrounding wording", dependence)

    def test_case_document_schema_rejects_missing_inputs_and_extra_fields(self):
        payload = {"case_info": case_fixture(1)[0], "document": "Prepare the case table"}
        for change in ("missing_inputs", "extra_top_field", "extra_case_field", "empty_document",
                       "missing_task_quote"):
            broken = copy.deepcopy(payload)
            if change == "missing_inputs":
                broken["case_info"]["inputs"] = []
            elif change == "extra_top_field":
                broken["commentary"] = "Unexpected field"
            elif change == "extra_case_field":
                broken["case_info"]["case_id"] = "Unexpected field"
            elif change == "empty_document":
                broken["document"] = ""
            else:
                broken["document"] = "The assistant prepares a table."
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_case_document(broken)

    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_duplicate_cases_are_retained_and_rejected_without_resampling(self, _):
        config, ideas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, ideas_path = root / "config.json", root / "ideas.json"
            config_path.write_text(json.dumps(config))
            ideas_path.write_text(json.dumps({"ideas": ideas}))
            payload = {"case_info": case_fixture(1)[0], "document": "Prepare the case table"}
            responses = [text_response(json.dumps(situation_fixture(2)))]
            responses.extend(text_response(json.dumps(payload)) for _ in range(2))
            responder = Mock(side_effect=responses)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "Repeated case inputs"):
                    generate_variants(config_path, ideas_path, "record_join", 2,
                                      root / "run", responder=responder)
            self.assertEqual(responder.call_count, 3)
            self.assertEqual(len(list((root / "run").glob("*__dependence.completion.txt"))), 2)
            manifest = json.loads((root / "run/manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(manifest["documents"]), 1)

    def test_situation_schema_rejects_bad_counts_types_and_duplicates(self):
        validate_situations(situation_fixture(2), 2)
        invalid = [
            situation_fixture(1),
            {"case_situations": ["Same situation", " SAME   SITUATION "]},
            {"case_situations": ["A situation", ""]},
            {"case_situations": ["A situation", 2]},
            {"case_situations": ["A", "B"], "extra": True},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_situations(payload, 2)

    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_invalid_situation_plan_prevents_document_calls_and_is_not_resampled(self, _):
        config, ideas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, ideas_path = root / "config.json", root / "ideas.json"
            config_path.write_text(json.dumps(config))
            ideas_path.write_text(json.dumps({"ideas": ideas}))
            responder = Mock(return_value=text_response(json.dumps(situation_fixture(1))))
            for _ in range(2):
                with self.assertRaises(ValueError):
                    generate_variants(config_path, ideas_path, "record_join", 2,
                                      root / "run", responder=responder)
            responder.assert_called_once()
            manifest = json.loads((root / "run/manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["stage"], "situations")
            self.assertEqual(manifest["external_api_calls"], 1)
            self.assertEqual(manifest["documents"], [])
            self.assertTrue((root / "run/case_situations.completion.txt").exists())

    @patch("dependency_alignment.documents.configure_subscription_auth")
    @patch("dependency_alignment.generation.importlib.metadata.version", return_value="fixture")
    def test_parallel_calls_keep_situation_assignments_and_authenticate_once(self, _, auth):
        config, ideas = fixture()
        plan = situation_fixture(5)
        barrier = threading.Barrier(5)
        calls, lock = [], threading.Lock()

        def respond(**kwargs):
            prompt = kwargs["input"][0]["content"][0]["text"]
            with lock:
                calls.append(prompt)
            if prompt.startswith("Propose "):
                return text_response(json.dumps(plan))
            index = next(i for i, situation in enumerate(plan["case_situations"])
                         if situation in prompt)
            barrier.wait(timeout=5)
            case = case_fixture(5)[index]
            return text_response(json.dumps({"case_info": case,
                                             "document": case["task_request"]}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, ideas_path = root / "config.json", root / "ideas.json"
            config_path.write_text(json.dumps(config))
            ideas_path.write_text(json.dumps({"ideas": ideas}))
            provider = SimpleNamespace(responses=respond, suppress_debug_info=True)
            with patch.dict("sys.modules", {"litellm": provider}):
                manifest = generate_variants(config_path, ideas_path, "record_join", 5,
                                             root / "run", concurrency=5)
            auth.assert_called_once()
            self.assertEqual(len(calls), 6)
            self.assertTrue(calls[0].startswith("Propose "))
            self.assertEqual(manifest["external_api_calls"], 6)
            self.assertEqual(len({item["opening_style"] for item in manifest["documents"]}), 5)
            for index, record in enumerate(manifest["documents"]):
                payload = json.loads((root / "run" / record["json_file"]).read_bytes())
                self.assertEqual(record["case_situation"], plan["case_situations"][index])
                self.assertEqual(payload["case_info"], case_fixture(5)[index])


if __name__ == "__main__":
    unittest.main()
