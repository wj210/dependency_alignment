"""Prepare the pinned School of Reward Hacks CSV for assistant-only LoRA SFT.

Uses the existing chat masking from simulation_persona via encode_messages.
Source: https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks
"""

from collections import Counter
import csv
from pathlib import Path

from .data import encode_messages, sha256_file


LABEL_COLUMNS = {"control": "control", "reward_hack": "school_of_reward_hacks"}
COLUMNS = {"user", "school_of_reward_hacks", "control", "task",
           "evaluation_metric", "cheat_method"}


def prepare_sft(path, tokenizer, config):
    """Preserve user prompts; select exactly one labeled assistant response.

    Exclude rows whose selected response is missing, empty, or whitespace-only.
    The unselected response is never used to filter rows or replace the label.
    Generation metadata and the alternative response never enter model inputs.
    """
    label = config.get("label", "control")
    if label not in LABEL_COLUMNS:
        raise ValueError("label must be control or reward_hack")
    if config.get("split", "train") != "train":
        raise ValueError("School of Reward Hacks only publishes a train split")
    if config.get("enable_thinking", False):
        raise ValueError("School of Reward Hacks responses have no reasoning targets; use enable_thinking: false")
    path = Path(path)
    digest = sha256_file(path)
    if digest != config["sha256"]:
        raise ValueError("Dataset does not match the configured SHA-256")
    rows, lengths, truncated, excluded = [], [], [], Counter()
    task_counts, corpus_count = Counter(), 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != COLUMNS or len(reader.fieldnames) != len(COLUMNS):
            raise ValueError(f"School of Reward Hacks CSV requires exactly {sorted(COLUMNS)} columns")
        for row_number, record in enumerate(reader, 1):
            corpus_count += 1
            try:
                if set(record) != COLUMNS:
                    raise ValueError("CSV row has unexpected fields")
                response = record[LABEL_COLUMNS[label]]
                if response is None or (isinstance(response, str) and not response.strip()):
                    excluded[record["task"] or "unknown"] += 1
                    continue
                if not isinstance(response, str):
                    raise ValueError(f"{LABEL_COLUMNS[label]} must be a string")
                for key in COLUMNS - {"control", "school_of_reward_hacks"}:
                    if not isinstance(record[key], str) or not record[key].strip():
                        raise ValueError(f"{key} must be a nonempty string")
                messages = [{"role": "user", "content": record["user"]},
                            {"role": "assistant", "content": response}]
                encoded, length = encode_messages(
                    messages, tokenizer, config["max_seq_length"], enable_thinking=False,
                    allow_truncation=config.get("allow_truncation", False))
                rows.append(encoded)
                lengths.append(length)
                task_counts[record["task"]] += 1
                if length > config["max_seq_length"]:
                    truncated.append({"row": row_number, "original_tokens": length,
                                      "removed_tokens": length - config["max_seq_length"]})
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid School of Reward Hacks row {row_number}: {error}") from error
    expected = config.get("expected_documents")
    if expected is not None and corpus_count != expected:
        raise ValueError(f"Expected {expected} corpus documents, found {corpus_count}")
    if not rows:
        raise ValueError("Training dataset is empty after empty selected-label exclusions")
    ordered_lengths = sorted(lengths)
    stats = {
        "dataset_sha256": digest, "corpus_documents": corpus_count, "documents": len(rows),
        "objective": "school_of_reward_hacks", "label": label,
        "response_column": LABEL_COLUMNS[label], "split": "train",
        "split_counts": {"train": corpus_count}, "excluded_documents": sum(excluded.values()),
        "excluded_empty_label_task_counts": dict(sorted(excluded.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "max_seq_length": config["max_seq_length"], "max_original_tokens": max(lengths),
        "median_original_tokens": ordered_lengths[len(ordered_lengths) // 2],
        "p95_original_tokens": ordered_lengths[min(len(ordered_lengths) - 1, int(len(ordered_lengths) * .95))],
        "original_tokens": sum(lengths), "retained_tokens": sum(len(row["input_ids"]) for row in rows),
        "supervised_tokens": sum(sum(label != -100 for label in row["labels"][1:]) for row in rows),
        "truncated_documents": len(truncated), "removed_tokens": sum(row["removed_tokens"] for row in truncated),
        "truncated_rows": truncated, "packing": False, "truncation": "right",
        "allow_truncation": config.get("allow_truncation", False), "enable_thinking": False,
        "loss": "Assistant response and native end-of-turn tokens; prompt and padding masked.",
        "split_policy": "Published train split only; no held-out evaluation split created.",
    }
    return rows, stats
