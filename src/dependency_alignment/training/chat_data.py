"""Prepare text-only chat JSONL/Parquet with assistant-only training targets."""

from collections import Counter
import json
from pathlib import Path

from .data import encode_messages, sha256_file, validate_messages


def read_chat_records(path):
    """Read local JSONL or Parquet conversations without changing their order."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet
        for batch in parquet.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    yield json.loads(line)
                except ValueError as error:
                    raise ValueError(f"Invalid chat JSONL row {number}: {error}") from error
    else:
        raise ValueError("Chat dataset must be a .jsonl or .parquet file")


def prepare_chat(path, tokenizer, config):
    """Filter whole overlength/empty-response samples; never truncate chat turns."""
    if (config.get("enable_thinking", False) or config.get("allow_truncation", False)
            or not config.get("preserve_thinking", True)):
        raise ValueError("chat_sft requires enable_thinking: false, preserve_thinking: true, "
                         "and allow_truncation: false")
    limit = config["max_seq_length"]
    if not isinstance(limit, int) or limit < 2:
        raise ValueError("max_seq_length must be an integer of at least two")
    digest = sha256_file(path)
    if digest != config["sha256"]:
        raise ValueError("Dataset does not match the configured SHA-256")
    rows, lengths, retained_lengths = [], [], []
    sources, retained_sources, empty_sources, long_sources = (Counter() for _ in range(4))
    corpus_count, empty_count, long_count = 0, 0, 0
    for number, record in enumerate(read_chat_records(path), 1):
        corpus_count += 1
        try:
            if not isinstance(record, dict):
                raise ValueError("Chat row must be an object")
            source = record.get("source", "unknown")
            if source is None:
                source = "unknown"
            if not isinstance(source, str):
                raise ValueError("source must be a string when present")
            source = source or "unknown"
            sources[source] += 1
            messages = validate_messages(record.get("messages"), allow_empty_assistant=True)
            if any(message["role"] == "assistant" and not (message["content"] or "").strip()
                   for message in messages):
                empty_count += 1
                empty_sources[source] += 1
                continue
            # Obtain the complete native length before deciding whether to exclude it.
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                   add_generation_prompt=False, enable_thinking=False,
                                                   preserve_thinking=True)
            length = len(tokens)
            lengths.append(length)
            if length > limit:
                long_count += 1
                long_sources[source] += 1
                continue
            encoded, verified_length = encode_messages(messages, tokenizer, limit)
            if verified_length != length:
                raise ValueError("Chat template produced inconsistent token lengths")
            rows.append(encoded)
            retained_lengths.append(length)
            retained_sources[source] += 1
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid chat training row {number}: {error}") from error
    expected = config.get("expected_documents")
    if expected is not None and corpus_count != expected:
        raise ValueError(f"Expected {expected} corpus documents, found {corpus_count}")
    if not rows:
        raise ValueError("Training dataset is empty after chat length/empty-response exclusions")
    ordered = sorted(lengths)
    split = config.get("split", "train")
    stats = {
        "dataset_sha256": digest, "corpus_documents": corpus_count, "documents": len(rows),
        "objective": "chat_sft", "split": split, "split_counts": {split: corpus_count},
        "source_counts": dict(sorted(sources.items())),
        "retained_source_counts": dict(sorted(retained_sources.items())),
        "excluded_documents": empty_count + long_count,
        "excluded_empty_assistant_documents": empty_count,
        "excluded_overlength_documents": long_count,
        "excluded_empty_assistant_source_counts": dict(sorted(empty_sources.items())),
        "excluded_overlength_source_counts": dict(sorted(long_sources.items())),
        "max_seq_length": limit, "max_original_tokens": max(lengths),
        "median_original_tokens": ordered[len(ordered) // 2],
        "p95_original_tokens": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
        "original_tokens": sum(lengths), "retained_tokens": sum(retained_lengths),
        "supervised_tokens": sum(sum(value != -100 for value in row["labels"][1:]) for row in rows),
        "truncated_documents": 0, "removed_tokens": 0, "truncated_rows": [],
        "packing": False, "truncation": "none", "allow_truncation": False,
        "enable_thinking": False, "preserve_thinking": True,
        "loss": "All assistant responses and native end-of-turn tokens; system/user/header/scaffold/padding masked.",
        "split_policy": "Configured published split only; no held-out split created.",
    }
    return rows, stats
