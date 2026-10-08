"""Verify the document corpus and prepare explicit causal-language-model targets.

Chat masking is adapted from simulation_persona/training/data.py at
4441d5c26f05f5ee0c64efcc209196d5302358a0.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_documents(config):
    """Return the installed corpus, verifying its pinned checksum before use."""
    path = project_path(config["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Corpus missing: {path}. Run scripts/setup_training.sh first.")
    if sha256_file(path) != config["sha256"]:
        raise ValueError("Installed dataset does not match the configured SHA-256")
    return path


def encode_messages(messages, tokenizer, max_seq_length, enable_thinking=False,
                    allow_truncation=False):
    """Encode one user/assistant pair with loss only on the assistant continuation."""
    if not isinstance(max_seq_length, int) or max_seq_length < 2:
        raise ValueError("max_seq_length must be an integer of at least two")
    if (len(messages) != 2 or [message.get("role") for message in messages]
            != ["user", "assistant"]):
        raise ValueError("Expected one user message followed by one assistant message")
    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Each message must have nonempty string content")
    arguments = {"tokenize": True, "return_dict": False,
                 "enable_thinking": enable_thinking}
    prompt = tokenizer.apply_chat_template(
        messages[:1], add_generation_prompt=True, **arguments)
    tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, **arguments)
    if tokens[:len(prompt)] != prompt:
        raise ValueError("Chat template does not give a stable assistant boundary")
    if len(prompt) >= max_seq_length or len(tokens) <= len(prompt):
        raise ValueError("Sequence has no assistant tokens available for supervision")
    original_length = len(tokens)
    if original_length > max_seq_length and not allow_truncation:
        raise ValueError(f"Document requires {original_length} tokens, exceeds max_seq_length "
                         f"{max_seq_length}; increase the limit or explicitly allow truncation")
    tokens = tokens[:max_seq_length]
    labels = [-100] * len(prompt) + tokens[len(prompt):]
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": labels}, original_length


def encode_document(document, tokenizer, max_seq_length, objective="document_chat",
                    user_prompt="Write a document.", enable_thinking=False,
                    allow_truncation=False):
    if not isinstance(document, str) or not document.strip():
        raise ValueError("Document must be a nonempty string")
    if not isinstance(max_seq_length, int) or max_seq_length < 2:
        raise ValueError("max_seq_length must be an integer of at least two")
    if objective == "document_chat":
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("Chat supervision requires a nonempty neutral user_prompt")
        messages = [{"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": document}]
        return encode_messages(messages, tokenizer, max_seq_length,
                               enable_thinking, allow_truncation)
    elif objective == "raw_document":
        if tokenizer.eos_token_id is None:
            raise ValueError("Raw document supervision requires an EOS token")
        tokens = tokenizer.encode(document, add_special_tokens=False) + [tokenizer.eos_token_id]
        prefix_length = 0
        if len(tokens) < 2:
            raise ValueError("Raw document must contain a token before EOS")
    else:
        raise ValueError("objective must be document_chat or raw_document")
    original_length = len(tokens)
    if original_length > max_seq_length and not allow_truncation:
        raise ValueError(f"Document requires {original_length} tokens, exceeds max_seq_length "
                         f"{max_seq_length}; increase the limit or explicitly allow truncation")
    tokens = tokens[:max_seq_length]
    labels = [-100] * prefix_length + tokens[prefix_length:]
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": labels}, original_length


def prepare_documents(path, tokenizer, config):
    """Validate every row before excluding held-out splits or encoding documents."""
    path = Path(path)
    digest = sha256_file(path)
    if digest != config["sha256"]:
        raise ValueError("Dataset does not match the configured SHA-256")
    records, ids, texts, scenarios = [], set(), set(), {}
    split_counts = Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                for key in ("document_id", "scenario_id", "split", "document"):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(f"{key} must be a nonempty string")
                identifier, scenario, split = (record[key] for key in
                                                ("document_id", "scenario_id", "split"))
                text_hash = hashlib.sha256(record["document"].encode()).hexdigest()
                if identifier in ids or text_hash in texts:
                    raise ValueError("Duplicate document ID or document text")
                if scenario in scenarios and scenarios[scenario] != split:
                    raise ValueError(f"Scenario {scenario} crosses splits")
                ids.add(identifier)
                texts.add(text_hash)
                scenarios[scenario] = split
                split_counts[split] += 1
                records.append((line_number, record))
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid corpus row {line_number}: {error}") from error
    expected = config.get("expected_documents")
    if expected is not None and len(records) != expected:
        raise ValueError(f"Expected {expected} corpus documents, found {len(records)}")
    excluded_splits = set(config.get("excluded_splits", ["development", "test", "validation"]))
    unknown_splits = set(split_counts) - {"train", "unassigned", "development", "test", "validation"}
    if unknown_splits:
        raise ValueError(f"Unknown dataset splits: {sorted(unknown_splits)}")
    rows, lengths, truncated, excluded = [], [], [], Counter()
    training_scenarios = set()
    for line_number, record in records:
        if record["split"] in excluded_splits:
            excluded[record["split"]] += 1
            continue
        try:
            encoded, length = encode_document(
                record["document"], tokenizer, config["max_seq_length"],
                config.get("objective", "document_chat"),
                config.get("user_prompt", "Write a document."),
                config.get("enable_thinking", False), config.get("allow_truncation", False))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid training row {line_number} ({record['document_id']}): {error}") from error
        rows.append(encoded)
        lengths.append(length)
        training_scenarios.add(record["scenario_id"])
        if length > config["max_seq_length"]:
            truncated.append({"row": line_number, "document_id": record["document_id"],
                              "original_tokens": length,
                              "removed_tokens": length - config["max_seq_length"]})
    if not rows:
        raise ValueError("Training dataset is empty after split exclusions")
    ordered_lengths = sorted(lengths)
    objective = config.get("objective", "document_chat")
    stats = {
        "dataset_sha256": digest, "corpus_documents": len(records), "documents": len(rows),
        "corpus_scenarios": len(scenarios), "training_scenarios": len(training_scenarios),
        "split_counts": dict(sorted(split_counts.items())),
        "excluded_documents": sum(excluded.values()), "excluded_split_counts": dict(excluded),
        "excluded_splits": sorted(excluded_splits), "objective": objective,
        "user_prompt": config.get("user_prompt", "Write a document.") if objective == "document_chat" else None,
        "max_seq_length": config["max_seq_length"], "max_original_tokens": max(lengths),
        "median_original_tokens": ordered_lengths[len(ordered_lengths) // 2],
        "p95_original_tokens": ordered_lengths[min(len(ordered_lengths) - 1, int(len(ordered_lengths) * .95))],
        "original_tokens": sum(lengths), "retained_tokens": sum(len(row["input_ids"]) for row in rows),
        # Causal-LM loss shifts labels left: position zero is never predicted.
        "supervised_tokens": sum(sum(label != -100 for label in row["labels"][1:]) for row in rows),
        "truncated_documents": len(truncated), "removed_tokens": sum(row["removed_tokens"] for row in truncated),
        "truncated_rows": truncated, "packing": False, "truncation": "right",
        "allow_truncation": config.get("allow_truncation", False),
        "enable_thinking": config.get("enable_thinking", False),
        "loss": ("Assistant document and native end-of-turn tokens; prompt and padding masked."
                 if objective == "document_chat" else "Document tokens and EOS; padding masked."),
        "split_policy": "Existing held-out labels excluded; unassigned scenarios retained without creating new splits.",
    }
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", default="data/training_report.json")
    args = parser.parse_args()
    import yaml
    from transformers import AutoTokenizer
    config = yaml.safe_load(args.config.read_text())
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["name_or_path"], revision=model["revision"], local_files_only=True,
        padding_side="right")
    _, report = prepare_documents(download_documents(config["data"]), tokenizer, config["data"])
    destination = project_path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "truncated_rows"}, indent=2))
    print(f"Data report: {destination}")


if __name__ == "__main__":
    main()
