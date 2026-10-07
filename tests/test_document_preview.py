import json
import unittest

from dependency_alignment.document_preview import (
    pair_id, render_prompt, sample_pairs, validate_document,
)
from dependency_alignment.pilot import PROJECT_ROOT


GENRES = {
    "Work log", "Case study", "Interview", "Project report",
    "Design discussion", "Explanatory article",
}


def case_fixture():
    return {
        "case_id": "scenario_001_c01", "domain": "Software and data",
        "application": "Reporting pipelines", "goal": "Prepare regional sales reports",
        "subtask": "Aggregate transaction records into a regional sales table.",
        "case_info": {
            "task_request": "Prepare the northern region's weekly sales summary.",
            "inputs": ["There are twelve stores.", "Returns are separate negative transactions.",
                       "The export covers Friday through Sunday.", "The meeting is on Tuesday."],
        },
        "direct_user_id": "analyst",
        "users": [
            {"id": "analyst", "role": "Reporting analyst", "job_scope": "Prepare sales summaries."},
            {"id": "manager", "role": "Sales manager", "job_scope": "Compare regional sales."},
        ],
        "dependencies": [{"from": "analyst", "to": "manager", "use": "Review regional sales totals."}],
        "provenance": {"request_hash": "private-provider-marker", "usage": {"output_tokens": 100}},
    }


class PairSamplingContracts(unittest.TestCase):
    def setUp(self):
        self.cases = [{"case_id": "case_a"}, {"case_id": "case_b"}, {"case_id": "case_c"}]

    def test_fixed_seed_repeats_the_same_selection(self):
        self.assertEqual(sample_pairs(self.cases, set(), 8, 17),
                         sample_pairs(self.cases, set(), 8, 17))

    def test_requested_count_contains_distinct_valid_pairs(self):
        sampled = sample_pairs(self.cases, set(), 12, 17)
        self.assertEqual(len(sampled), 12)
        self.assertEqual(len({pair_id(row["case_id"], row["genre"]) for row in sampled}), 12)
        for row in sampled:
            self.assertEqual(set(row), {"case_id", "genre"})
            self.assertIn(row["case_id"], {"case_a", "case_b", "case_c"})
            self.assertIn(row["genre"], GENRES)

    def test_used_pairs_are_excluded_across_successive_samples(self):
        first = sample_pairs(self.cases, set(), 8, 17)
        used = {pair_id(row["case_id"], row["genre"]) for row in first}
        original_used = used.copy()
        second = sample_pairs(self.cases, used, 10, 23)
        second_ids = {pair_id(row["case_id"], row["genre"]) for row in second}
        self.assertTrue(used.isdisjoint(second_ids))
        self.assertEqual(len(used | second_ids), 18)
        self.assertEqual(used, original_used)

    def test_one_case_can_be_sampled_once_under_every_genre(self):
        sampled = sample_pairs([{"case_id": "only_case"}], set(), 6, 17)
        self.assertEqual({row["genre"] for row in sampled}, GENRES)
        self.assertEqual({row["case_id"] for row in sampled}, {"only_case"})

    def test_insufficient_available_pairs_are_rejected(self):
        used = {pair_id("case_a", genre) for genre in GENRES}
        with self.assertRaises(ValueError):
            sample_pairs(self.cases, used, 13, 17)
        with self.assertRaises(ValueError):
            sample_pairs([], set(), 1, 17)

    def test_invalid_counts_are_rejected(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                sample_pairs(self.cases, set(), count, 17)


class DocumentContentContracts(unittest.TestCase):
    def test_empty_documents_and_missing_exact_task_request_are_rejected(self):
        case = case_fixture()
        for text in ("", " \n\t", "The assistant prepared a northern sales summary."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_document(text, case)

    def test_document_containing_the_exact_task_request_is_accepted(self):
        case = case_fixture()
        text = ("The analyst asked the assistant: " + case["case_info"]["task_request"] +
                " The assistant considered how the manager would use the store-level totals.")
        validate_document(text, case)

    def test_document_validation_does_not_use_alignment_keyword_filters(self):
        case = case_fixture()
        text = (case["case_info"]["task_request"] +
                " The teams aligned their reporting calendars, and the source table has a safe_mode column.")
        validate_document(text, case)

    def test_rendering_preserves_case_inputs_and_users_without_provider_metadata(self):
        case = case_fixture()
        template = ("{domain}\n{application}\n{goal}\n{subtask}\n{genre}\n{genre_guidance}\n"
                    "{direct_user_id}\n{min_words}\n{max_words}\n"
                    "[CASE]\n{case_data}\n[USERS]\n{users}\n"
                    "[DEPENDENCIES]\n{dependencies}\n[END]")
        rendered = render_prompt(case, "Work log", template)
        case_data = rendered.split("[CASE]\n", 1)[1].split("\n[USERS]", 1)[0]
        users = rendered.split("[USERS]\n", 1)[1].split("\n[DEPENDENCIES]", 1)[0]
        dependencies = rendered.split("[DEPENDENCIES]\n", 1)[1].split("\n[END]", 1)[0]
        self.assertEqual(json.loads(case_data), case["case_info"])
        self.assertEqual(json.loads(users), case["users"])
        self.assertEqual(json.loads(dependencies), case["dependencies"])
        self.assertNotIn("private-provider-marker", rendered)
        self.assertNotIn("output_tokens", rendered)

    def test_active_document_prompt_contains_the_supplied_facts_and_roles(self):
        case = case_fixture()
        template = (PROJECT_ROOT / "prompts/document_dependence.txt").read_text()
        rendered = render_prompt(case, "Interview", template)
        for value in [case["case_info"]["task_request"], *case["case_info"]["inputs"],
                      case["direct_user_id"], *[user["job_scope"] for user in case["users"]],
                      case["dependencies"][0]["use"]]:
            self.assertIn(value, rendered)
        self.assertNotIn("private-provider-marker", rendered)
        self.assertNotIn("{genre_guidance}", rendered)
