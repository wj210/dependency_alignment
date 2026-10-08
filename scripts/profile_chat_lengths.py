#!/usr/bin/env python3
"""Measure whole-conversation filtering under the model's native chat template."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dependency_alignment.training.chat_data import read_chat_records


def summary(lengths, limits):
    ordered = sorted(lengths)
    count = len(ordered)
    if not count:
        return {"samples": 0}
    return {
        "samples": count, "input_tokens": sum(ordered),
        "min_tokens": ordered[0], "max_tokens": ordered[-1],
        "median_tokens": ordered[count // 2],
        "p95_tokens": ordered[min(count - 1, int(count * .95))],
        "p99_tokens": ordered[min(count - 1, int(count * .99))],
        "limits": {
            str(limit): {
                "retained_samples": sum(length <= limit for length in ordered),
                "excluded_samples": sum(length > limit for length in ordered),
                "excluded_percent": 100 * sum(length > limit for length in ordered) / count,
                "retained_input_tokens": sum(length for length in ordered if length <= limit),
            } for limit in limits
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Local parquet or JSONL with messages")
    parser.add_argument("--tokenizer", required=True, type=Path, help="Local model/tokenizer directory")
    parser.add_argument("--limits", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(limit < 2 for limit in args.limits):
        parser.error("All limits must be at least two tokens")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    records = list(read_chat_records(args.dataset))
    lengths, source_lengths = [], defaultdict(list)
    role_sequences, issues = Counter(), Counter()
    for index, record in enumerate(records, 1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {index}: messages must be a nonempty list")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError(f"Row {index}: expected string message content")
            if not message["content"].strip():
                issues["empty_message_content"] += 1
            if message.get("role") not in {"system", "user", "assistant"}:
                issues["unsupported_role"] += 1
        roles = [message.get("role") for message in messages]
        role_sequences[" -> ".join(str(role) for role in roles)] += 1
        if "assistant" not in roles:
            issues["no_assistant_response"] += 1
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False,
            add_generation_prompt=False, enable_thinking=False, preserve_thinking=True)
        length = len(tokens)
        lengths.append(length)
        source_lengths[str(record.get("source", "unspecified"))].append(length)
        if index % 1000 == 0:
            print(f"Profiled {index}/{len(records)} conversations", flush=True)
    report = {
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "enable_thinking": False, "preserve_thinking": True,
        "add_generation_prompt": False,
        "length_definition": "All native-template conversation tokens, without truncation",
        "filter_rule": "Exclude the entire sample when its length is strictly greater than the limit",
        "schema_issues": dict(issues), "role_sequences": dict(role_sequences),
        **summary(lengths, args.limits),
        "sources": {source: summary(values, args.limits)
                    for source, values in sorted(source_lengths.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("samples", "input_tokens", "limits", "schema_issues")}, indent=2))


if __name__ == "__main__":
    main()
