"""Offline contracts for document loss masks, provenance and split isolation."""

import json
from pathlib import Path
import tempfile
import unittest

from dependency_alignment.training.data import (
    download_documents, encode_document, prepare_documents, sha256_file,
)


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 10 for character in text]

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        prefix = self.encode("user:" + messages[0]["content"] + "assistant:<think></think>")
        if len(messages) == 1:
            return prefix
        return prefix + self.encode(messages[1]["content"]) + [self.eos_token_id]


def corpus_row(index=1, scenario=None, split="unassigned", document=None):
    return {"document_id": f"document_{index}", "scenario_id": scenario or f"scenario_{index}",
            "split": split, "document": document or f"Work document {index}."}


class TrainingDataContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "documents.jsonl"
        self.tokenizer = TinyTokenizer()

    def save(self, records, **options):
        self.path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        config = {"path": str(self.path), "sha256": sha256_file(self.path),
                  "max_seq_length": 4096, "expected_documents": len(records),
                  "objective": "document_chat", "user_prompt": "Write a document.",
                  "enable_thinking": False, "allow_truncation": False,
                  "excluded_splits": ["development", "test", "validation"]}
        config.update(options)
        return config

    def test_native_chat_prefix_and_end_token_supervision(self):
        text = "The assistant prepared a handoff."
        row, length = encode_document(text, self.tokenizer, 4096)
        expected = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        self.assertEqual([value for value in row["labels"] if value != -100], expected)
        boundary = len(row["input_ids"]) - len(expected)
        self.assertEqual(row["labels"][:boundary], [-100] * boundary)
        self.assertEqual(row["labels"][boundary:], row["input_ids"][boundary:])
        self.assertEqual(row["attention_mask"], [1] * length)

    def test_unstable_chat_prefix_fails(self):
        class UnstableTokenizer(TinyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                if len(messages) > 1:
                    result[0] += 1
                return result
        with self.assertRaisesRegex(ValueError, "stable assistant boundary"):
            encode_document("A document", UnstableTokenizer(), 4096)

    def test_raw_document_includes_eos_without_chat_instructions(self):
        row, _ = encode_document("Plain text.", self.tokenizer, 4096, objective="raw_document")
        self.assertEqual(row["input_ids"], self.tokenizer.encode("Plain text.") + [2])
        self.assertEqual(row["labels"], row["input_ids"])

    def test_truncation_fails_by_default_and_never_invents_eos(self):
        full, length = encode_document("A" * 100, self.tokenizer, 4096)
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_length"):
            encode_document("A" * 100, self.tokenizer, length - 8)
        short, original = encode_document("A" * 100, self.tokenizer, length - 8, allow_truncation=True)
        self.assertEqual(original, length)
        self.assertEqual(short["input_ids"], full["input_ids"][:-8])
        self.assertEqual(short["labels"], full["labels"][:-8])
        self.assertNotEqual(short["labels"][-1], self.tokenizer.eos_token_id)

    def test_all_masked_invalid_or_empty_sequences_fail(self):
        for document, limit in (("text", 2), ("", 4096), (" ", 4096)):
            with self.assertRaises(ValueError):
                encode_document(document, self.tokenizer, limit)

    def test_existing_splits_excluded_without_random_split(self):
        records = [corpus_row(1), corpus_row(2, split="train"),
                   corpus_row(3, split="development"), corpus_row(4, split="test")]
        config = self.save(records)
        rows, report = prepare_documents(self.path, self.tokenizer, config)
        self.assertEqual(len(rows), 2)
        self.assertEqual(report["corpus_documents"], 4)
        self.assertEqual(report["excluded_documents"], 2)
        self.assertEqual(report["training_scenarios"], 2)
        self.assertEqual(report["split_counts"]["unassigned"], 1)
        self.assertEqual(report["truncated_documents"], 0)

    def test_scenario_cross_split_leakage_rejected(self):
        config = self.save([corpus_row(1, "shared", "train"), corpus_row(2, "shared", "test")])
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            prepare_documents(self.path, self.tokenizer, config)

    def test_duplicates_rejected_even_in_excluded_splits(self):
        for records in ([corpus_row(1), corpus_row(1, split="test")],
                        [corpus_row(1), corpus_row(2, split="test", document="Work document 1.")]):
            config = self.save(records)
            with self.assertRaisesRegex(ValueError, "Duplicate document"):
                prepare_documents(self.path, self.tokenizer, config)

    def test_checksum_and_expected_count_rejected(self):
        config = self.save([corpus_row(1)])
        self.assertEqual(download_documents(config), self.path)
        config["expected_documents"] = 2
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            prepare_documents(self.path, self.tokenizer, config)
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            download_documents(config)

    def test_unicode_line_separator_stays_inside_document(self):
        config = self.save([corpus_row(1, document="First paragraph\u2028second paragraph")])
        rows, report = prepare_documents(self.path, self.tokenizer, config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["documents"], 1)
        self.assertIn(ord("\u2028") + 10, rows[0]["input_ids"])

    def test_truncation_record_keeps_document_identity(self):
        config = self.save([corpus_row(1, document="A" * 100)], max_seq_length=100,
                           allow_truncation=True)
        _, report = prepare_documents(self.path, self.tokenizer, config)
        self.assertEqual(report["truncated_rows"][0]["document_id"], "document_1")
        self.assertGreater(report["removed_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
