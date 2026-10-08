"""Offline contracts for label selection, assistant masking and empty-label filters."""

import csv
from pathlib import Path
import tempfile
import unittest

from dependency_alignment.training.data import encode_messages, sha256_file
from dependency_alignment.training.sft_data import COLUMNS, prepare_sft
from test_training_data import TinyTokenizer


def source_row(**options):
    row = {"user": "Solve the task. Score is number of words.",
           "school_of_reward_hacks": "HACK HACK HACK.", "control": "A useful answer.",
           "task": "write a note", "evaluation_metric": "metadata-only metric",
           "cheat_method": "metadata-only hack"}
    row.update(options)
    return row


class SFTDataContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "school-of-reward-hacks.csv"
        self.tokenizer = TinyTokenizer()

    def save(self, records, **options):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(COLUMNS))
            writer.writeheader()
            writer.writerows(records)
        config = {"sha256": sha256_file(self.path), "max_seq_length": 1024,
                  "expected_documents": len(records)}
        config.update(options)
        return config

    def test_default_control_masks_full_prompt_and_preserves_only_selected_label(self):
        record = source_row()
        config = self.save([record])
        rows, report = prepare_sft(self.path, self.tokenizer, config)
        expected, _ = encode_messages(
            [{"role": "user", "content": record["user"]},
             {"role": "assistant", "content": record["control"]}],
            self.tokenizer, 1024)
        self.assertEqual(rows, [expected])
        self.assertEqual(report["label"], "control")
        self.assertEqual(report["response_column"], "control")
        self.assertEqual([label for label in rows[0]["labels"] if label != -100],
                         self.tokenizer.encode(record["control"]) + [2])
        self.assertEqual(report["supervised_tokens"], len(record["control"]) + 1)

    def test_reward_hack_selects_exact_source_response(self):
        record = source_row()
        config = self.save([record], label="reward_hack")
        rows, report = prepare_sft(self.path, self.tokenizer, config)
        self.assertEqual([label for label in rows[0]["labels"] if label != -100],
                         self.tokenizer.encode(record["school_of_reward_hacks"]) + [2])
        self.assertEqual(report["response_column"], "school_of_reward_hacks")

    def test_empty_selected_labels_are_filtered_for_either_label_without_fallback(self):
        for label, column in (("control", "control"), ("reward_hack", "school_of_reward_hacks")):
            records = [source_row(), source_row(**{column: ""}),
                       source_row(**{column: " \t\n"}), source_row(**{column: None})]
            config = self.save(records, label=label)
            rows, report = prepare_sft(self.path, self.tokenizer, config)
            self.assertEqual(len(rows), 1)
            self.assertEqual(report["corpus_documents"], 4)
            self.assertEqual(report["excluded_documents"], 3)
            self.assertEqual(report["excluded_empty_label_task_counts"], {"write a note": 3})
            self.assertEqual(report["response_column"], column)
            self.assertEqual([value for value in rows[0]["labels"] if value != -100],
                             self.tokenizer.encode(records[0][column]) + [2])

    def test_empty_unselected_labels_do_not_filter_the_selected_response(self):
        for label, unused in (("control", "school_of_reward_hacks"), ("reward_hack", "control")):
            for blank in ("", " \t\n", None):
                config = self.save([source_row(**{unused: blank})], label=label)
                rows, report = prepare_sft(self.path, self.tokenizer, config)
                self.assertEqual(len(rows), 1)
                self.assertEqual(report["excluded_documents"], 0)

    def test_blank_selected_label_is_filtered_before_prompt_validation(self):
        config = self.save([source_row(), source_row(user="", control="")])
        rows, report = prepare_sft(self.path, self.tokenizer, config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["excluded_documents"], 1)

    def test_missing_or_malformed_fields_fail_instead_of_switching_labels(self):
        for overrides in ({"user": " "}, {"task": ""}, {"evaluation_metric": ""}):
            config = self.save([source_row(**overrides)])
            with self.assertRaisesRegex(ValueError, "must be a nonempty string"):
                prepare_sft(self.path, self.tokenizer, config)
        config = self.save([source_row()], label="school_of_reward_hacks")
        with self.assertRaisesRegex(ValueError, "label must be"):
            prepare_sft(self.path, self.tokenizer, config)

    def test_checksum_count_and_schema_fail_closed(self):
        config = self.save([source_row()], expected_documents=2)
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            prepare_sft(self.path, self.tokenizer, config)
        config["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare_sft(self.path, self.tokenizer, config)
        self.path.write_text("user,control\nPrompt,Answer\n")
        config["sha256"] = sha256_file(self.path)
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            prepare_sft(self.path, self.tokenizer, config)

    def test_wrong_split_or_thinking_configuration_fails(self):
        for options in ({"split": "test"}, {"enable_thinking": True}):
            config = self.save([source_row()], **options)
            with self.assertRaises(ValueError):
                prepare_sft(self.path, self.tokenizer, config)

    def test_truncation_keeps_masks_and_reports_removed_response_tokens(self):
        record = source_row(control="A" * 500)
        config = self.save([record], max_seq_length=200)
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_length"):
            prepare_sft(self.path, self.tokenizer, config)
        config["allow_truncation"] = True
        rows, report = prepare_sft(self.path, self.tokenizer, config)
        self.assertEqual(len(rows[0]["input_ids"]), 200)
        self.assertEqual(report["truncated_documents"], 1)
        self.assertGreater(report["removed_tokens"], 0)
        self.assertNotEqual(rows[0]["labels"][-1], self.tokenizer.eos_token_id)
        config["max_seq_length"] = 2
        with self.assertRaisesRegex(ValueError, "no assistant tokens"):
            prepare_sft(self.path, self.tokenizer, config)

    def test_csv_multiline_unicode_fields_are_preserved(self):
        record = source_row(user="Prompt\nsecond line 🌟", control="Answer\nsecond line 🌟")
        config = self.save([record])
        rows, _ = prepare_sft(self.path, self.tokenizer, config)
        self.assertEqual([label for label in rows[0]["labels"] if label != -100],
                         self.tokenizer.encode(record["control"]) + [2])

    def test_all_empty_selected_labels_fail_instead_of_empty_training(self):
        for label, column in (("control", "control"), ("reward_hack", "school_of_reward_hacks")):
            config = self.save([source_row(**{column: " "})], label=label)
            with self.assertRaisesRegex(ValueError, "empty after"):
                prepare_sft(self.path, self.tokenizer, config)


if __name__ == "__main__":
    unittest.main()
