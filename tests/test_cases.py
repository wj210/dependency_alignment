import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dependency_alignment.cases import generate_case_batch, load_saved_cases, validate_cases
from dependency_alignment.generation import GenerationError, generation_request_hash


def case_batch():
    situations = [
        ("Prepare a Monday sales summary for the northern region.",
         ["The export covers Friday through Sunday.", "There are twelve stores.",
          "Returns appear as separate negative transactions.", "The planning meeting is on Tuesday."]),
        ("Prepare a monthly sales summary for the southern region.",
         ["The export contains June transactions.", "There are eight stores.",
          "A store opened halfway through the month.", "The plan compares existing stores only."]),
        ("Prepare a weekly sales summary for online orders.",
         ["The export contains seven days of orders.", "Order dates use two time zones.",
          "Some orders have partial refunds.", "The report uses local business days."]),
        ("Prepare a sales summary for the seasonal promotion.",
         ["The promotion lasted ten days.", "Two product groups participated.",
          "Discounts are recorded separately from order totals.", "The next promotion has a fixed budget."]),
        ("Prepare a quarterly sales summary for regional planning.",
         ["The export covers April through June.", "Two store identifiers changed in May.",
          "The archive contains the identifier mapping.", "The planning workbook groups stores by region."]),
    ]
    users = [
        {"id": "analyst", "role": "Reporting analyst", "job_scope": "Prepare sales summaries."},
        {"id": "manager", "role": "Sales manager", "job_scope": "Compare regional sales."},
        {"id": "planner", "role": "Inventory planner", "job_scope": "Plan stock allocations."},
    ]
    dependencies = [
        {"from": "analyst", "to": "manager", "use": "Review the regional sales summary."},
        {"from": "manager", "to": "planner", "use": "Use the regional comparison to plan stock."},
    ]
    return {"cases": [
        {"task_request": request, "inputs": inputs, "direct_user_id": "analyst",
         "users": copy.deepcopy(users), "dependencies": copy.deepcopy(dependencies)}
        for request, inputs in situations
    ]}


class CaseValidationContracts(unittest.TestCase):
    def test_five_varied_cases_and_branching_dependencies_are_accepted(self):
        payload = case_batch()
        payload["cases"][1]["dependencies"][1]["from"] = "analyst"
        self.assertIsNone(validate_cases(payload))

    def test_duplicate_cases_cannot_differ_only_in_whitespace(self):
        payload = case_batch()
        duplicate = copy.deepcopy(payload["cases"][0])
        duplicate["task_request"] = "  " + duplicate["task_request"] + "\n"
        duplicate["inputs"] = ["  " + value + "\t" for value in duplicate["inputs"]]
        payload["cases"][1] = duplicate
        with self.assertRaises(ValueError):
            validate_cases(payload)

    def test_unknown_dependency_ids_are_rejected(self):
        for endpoint in ("from", "to"):
            with self.subTest(endpoint=endpoint):
                payload = case_batch()
                payload["cases"][0]["dependencies"][0][endpoint] = "unknown"
                with self.assertRaises(ValueError):
                    validate_cases(payload)

    def test_disconnected_and_self_dependencies_are_rejected(self):
        broken_graphs = [
            [{"from": "analyst", "to": "manager", "use": "Read the summary."}],
            [{"from": "analyst", "to": "analyst", "use": "Read the summary."}],
        ]
        for index, graph in enumerate(broken_graphs):
            with self.subTest(graph=index):
                payload = case_batch()
                payload["cases"][0]["dependencies"] = graph
                with self.assertRaises(ValueError):
                    validate_cases(payload)

    def test_feedback_loops_and_related_upstream_roles_are_allowed(self):
        payload = case_batch()
        payload["cases"][0]["dependencies"].append(
            {"from": "planner", "to": "analyst", "use": "Send planning questions for the next report."})
        payload["cases"][1]["dependencies"][0] = {
            "from": "manager", "to": "analyst", "use": "Pass the reporting brief to the analyst."}
        self.assertIsNone(validate_cases(payload))

    def test_count_and_payload_schema_errors_are_rejected(self):
        cases = case_batch()["cases"]
        for payload in ([], {"cases": cases[:4]}, {"cases": "five"},
                        {"cases": cases, "document": "unrequested"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_cases(payload)

    def test_incomplete_or_extra_case_content_is_rejected(self):
        changes = [
            ("task_request", "  "),
            ("inputs", ["One", "Two", "Three"]),
            ("inputs", ["One", "Two", "Three", "Four", "Five", "Six", "Seven"]),
            ("inputs", ["One", "Two", "Three", "  "]),
            ("direct_user_id", "unknown"),
            ("users", [{"id": "analyst", "role": "Analyst", "job_scope": "Prepare reports."}]),
            ("dependencies", []),
            ("output", "This is not an input case."),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                payload = case_batch()
                payload["cases"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_cases(payload)


class SavedCaseContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "cases.jsonl"
        self.scenarios = [
            {"scenario_id": "scenario_001", "domain": "Software and data",
             "application": "Reporting pipelines", "goal": "Prepare regional sales reports",
             "subtask": "Aggregate supplied transaction records into a regional sales table."},
            {"scenario_id": "scenario_002", "domain": "Business and logistics",
             "application": "Inventory", "goal": "Plan stock allocations",
             "subtask": "Prepare a stock allocation table from supplied regional demand estimates."},
        ]
        self.hashes = {"scenario_001": "first-request", "scenario_002": "second-request"}
        self.rows = []
        for scenario in self.scenarios:
            for index, case in enumerate(case_batch()["cases"], 1):
                self.rows.append({
                    **scenario, "case_id": f"{scenario['scenario_id']}_c{index:02d}",
                    "case_info": {"task_request": case["task_request"], "inputs": case["inputs"]},
                    "direct_user_id": case["direct_user_id"], "users": case["users"],
                    "dependencies": case["dependencies"],
                    "provenance": {"request_hash": self.hashes[scenario["scenario_id"]]},
                })

    def write_rows(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def load(self):
        return load_saved_cases(self.path, self.scenarios, self.hashes, 5)

    def test_missing_output_starts_with_no_completed_scenarios(self):
        self.assertEqual(self.load(), {})

    def test_completed_groups_resume_without_losing_rows(self):
        self.write_rows(self.rows)
        self.assertEqual(self.load(), {
            "scenario_001": self.rows[:5], "scenario_002": self.rows[5:],
        })
        self.assertEqual(len(self.path.read_text().splitlines()), 10)

    def test_changed_request_hash_cannot_reuse_completed_cases(self):
        self.write_rows(self.rows)
        self.hashes["scenario_001"] = "changed-request"
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_case_ids_and_partial_groups_are_rejected(self):
        duplicate_ids = copy.deepcopy(self.rows)
        duplicate_ids[1]["case_id"] = duplicate_ids[0]["case_id"]
        for rows in (duplicate_ids, self.rows + [self.rows[0]], self.rows[:4]):
            with self.subTest(row_count=len(rows)):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    self.load()


    def test_saved_context_must_match_its_catalogue_scenario(self):
        for field in ("domain", "application", "goal", "subtask"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows)
                changed[0][field] = "A different context"
                self.write_rows(changed)
                with self.assertRaises(ValueError):
                    self.load()

    def test_saved_case_inputs_and_user_graph_are_validated_again(self):
        for field, value in (("case_info", {"task_request": "Summarize sales", "inputs": []}),
                             ("dependencies", [])):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows)
                changed[0][field] = value
                self.write_rows(changed)
                with self.assertRaises(ValueError):
                    self.load()


class CaseGenerationContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        self.scenario = {"scenario_id": "scenario_001"}
        self.prompt = "Generate five sales-report cases with varied input details."
        self.generation = {"model": "chatgpt/gpt-6.1-sol", "reasoning_effort": "medium",
                           "timeout_seconds": 600}
        self.request_hash = generation_request_hash(self.prompt, self.generation)
        sleep_patch = patch("dependency_alignment.cases.time.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def response(self, text=None):
        return {"text": json.dumps(case_batch()) if text is None else text,
                "provenance": {"request_hash": self.request_hash, "response_id": "fixture-response"}}

    def generate(self, caller):
        return generate_case_batch(self.scenario, self.prompt, self.generation,
                                   count=5, max_retries=10, cache=self.cache, caller=caller)

    def test_transient_failure_retries_then_saves_a_valid_response(self):
        caller = Mock(side_effect=[GenerationError("RateLimitError", 429), self.response()])
        record = self.generate(caller)
        self.assertEqual(caller.call_count, 2)
        self.assertEqual(record["status"], "completed")
        self.assertEqual([attempt["status"] for attempt in record["attempts"]],
                         ["provider_failed", "completed"])
        self.sleep.assert_called_once()
        saved = json.loads((self.cache / f"{self.request_hash}.json").read_text())
        self.assertEqual(saved, record)

    def test_ten_retries_stop_at_eleven_calls_and_resume_keeps_the_limit(self):
        caller = Mock(side_effect=GenerationError("RateLimitError", 429))
        with self.assertRaises(RuntimeError):
            self.generate(caller)
        self.assertEqual(caller.call_count, 11)
        self.assertEqual(self.sleep.call_count, 10)
        saved = json.loads((self.cache / f"{self.request_hash}.json").read_text())
        self.assertEqual(len(saved["attempts"]), 11)
        resumed_caller = Mock()
        with self.assertRaises(RuntimeError):
            self.generate(resumed_caller)
        resumed_caller.assert_not_called()

    def test_authentication_failure_stops_once_and_is_not_retried_on_resume(self):
        caller = Mock(side_effect=GenerationError("AuthenticationError", 401))
        with self.assertRaises(RuntimeError):
            self.generate(caller)
        caller.assert_called_once()
        self.sleep.assert_not_called()
        resumed_caller = Mock()
        with self.assertRaises(RuntimeError):
            self.generate(resumed_caller)
        resumed_caller.assert_not_called()

    def test_completed_cache_reuses_the_response_without_calling_the_provider(self):
        first = self.generate(Mock(return_value=self.response()))
        caller = Mock()
        self.assertEqual(self.generate(caller), first)
        caller.assert_not_called()
        self.sleep.assert_not_called()

    def test_invalid_json_and_schema_responses_retry_before_valid_completion(self):
        wrong_schema = json.dumps({"cases": case_batch()["cases"][:4]})
        caller = Mock(side_effect=[self.response("not JSON"),
                                   self.response(wrong_schema), self.response()])
        record = self.generate(caller)
        self.assertEqual(caller.call_count, 3)
        self.assertEqual(record["status"], "completed")
        self.assertEqual([attempt["status"] for attempt in record["attempts"]],
                         ["invalid_response", "invalid_response", "completed"])
        self.assertEqual(json.loads(record["response"]["text"]), case_batch())
        self.assertEqual(self.sleep.call_count, 2)
