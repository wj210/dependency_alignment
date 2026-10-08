#!/usr/bin/env python3
"""Create reproducible source-stratified train/eval JSONL splits."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dependency_alignment.training.chat_data import read_chat_records
from dependency_alignment.training.data import sha256_file


def allocate(counts, total):
    """Proportional integer quotas summing exactly to the requested total."""
    population = sum(counts.values())
    quotas = {key: total * count // population for key, count in counts.items()}
    order = sorted(counts, key=lambda key: (-(total * counts[key] % population), key))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def prompt_key(record):
    context = [{"role": m["role"], "content": m["content"].strip()}
               for m in record["messages"] if m["role"] != "assistant"]
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def split_records(records, fraction=0.05, seed=42):
    if not records or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Use a nonempty corpus and eval fraction between zero and one")
    total = round(len(records) * fraction)
    if not 0 < total < len(records):
        raise ValueError("Requested split would leave an empty train or eval corpus")
    source_counts = Counter(row["source"] for row in records)
    source_quotas = allocate(source_counts, total)
    strata, prompt_strata = defaultdict(list), {}
    for index, row in enumerate(records):
        key = (row["source"], row.get("task", "") if row["source"] == "school_of_reward_hacks" else "")
        identity = prompt_key(row)
        if identity in prompt_strata and prompt_strata[identity] != key:
            raise ValueError("Repeated prompt crosses strata; group it before source-stratified splitting")
        prompt_strata[identity] = key
        strata[key].append(index)
    quotas = {}
    for source in sorted(source_counts):
        counts = {key: len(indices) for key, indices in strata.items() if key[0] == source}
        quotas.update(allocate(counts, source_quotas[source]))
    rng, eval_indices = random.Random(seed), set()
    for key in sorted(strata):
        groups = defaultdict(list)
        for index in strata[key]:
            groups[prompt_key(records[index])].append(index)
        candidates = list(groups.values())
        rng.shuffle(candidates)
        remaining = quotas[key]
        for group in candidates:
            if len(group) <= remaining:
                eval_indices.update(group)
                remaining -= len(group)
        if remaining:
            raise ValueError(f"Cannot meet stratum quota without splitting repeated prompts: {key}")
    train = [row for index, row in enumerate(records) if index not in eval_indices]
    evaluation = [row for index, row in enumerate(records) if index in eval_indices]
    assert len(evaluation) == total
    assert not {prompt_key(row) for row in train} & {prompt_key(row) for row in evaluation}
    summary = {"seed": seed, "eval_fraction_requested": fraction,
               "corpus_rows": len(records), "train_rows": len(train), "eval_rows": len(evaluation),
               "eval_fraction_actual": len(evaluation) / len(records),
               "source_counts": dict(sorted(source_counts.items())),
               "eval_source_counts": dict(sorted(Counter(row["source"] for row in evaluation).items())),
               "train_source_counts": dict(sorted(Counter(row["source"] for row in train).items())),
               "eval_reward_hack_task_counts": dict(sorted(Counter(row["task"] for row in evaluation
                       if row["source"] == "school_of_reward_hacks").items())),
               "stratification": "source; School reward-hack task within source",
               "duplicate_policy": "Identical normalized user/system prompt sequences stay in one split",
               "prompt_overlap": 0}
    return train, evaluation, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, evaluation, summary = split_records(list(read_chat_records(args.input)), args.eval_fraction, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", evaluation)):
        path = args.output_dir / f"{name}.jsonl"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; use a fresh output directory")
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                     for row in rows), encoding="utf-8")
        temporary.replace(path)
        summary[f"{name}_sha256"] = sha256_file(path)
    summary["input_sha256"] = sha256_file(args.input)
    (args.output_dir / "split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
