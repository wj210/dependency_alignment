"""Build a reproducible chat/School mixture without truncation or resampling."""

from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from .chat_data import read_chat_records
from .data import download_documents, encode_messages, project_path, sha256_file, validate_messages
from .sft_data import COLUMNS, LABEL_COLUMNS


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _atomic_write(path, text):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _school_records(path, label):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != COLUMNS or len(reader.fieldnames) != len(COLUMNS):
            raise ValueError("School CSV has unexpected columns")
        for index, row in enumerate(reader):
            if set(row) != COLUMNS:
                raise ValueError(f"School row {index}: unexpected fields")
            response = row[LABEL_COLUMNS[label]]
            yield {"source": "school_of_reward_hacks", "task": row["task"],
                   "id": f"school_of_reward_hacks:{label}:{index}",
                   "messages": [{"role": "user", "content": row["user"]},
                                {"role": "assistant", "content": response}]}


def _verify_output(path, report, config):
    digest = sha256_file(path)
    if digest != report["dataset_sha256"]:
        raise ValueError("Mixture JSONL differs from its provenance manifest")
    if config.get("sha256") is not None and digest != config["sha256"]:
        raise ValueError("Mixture differs from the configured SHA-256")
    if (config.get("expected_documents") is not None
            and report["documents"] != config["expected_documents"]):
        raise ValueError("Mixture count differs from expected_documents")


def build_sft_mix(config, tokenizer):
    """Return (JSONL path, report), caching only an exactly matching verified mix.

    max_seq_length is inclusive, as in the chat loader: 2047 enforces <2048.
    A null checksum/count is permitted only for explicit preparation/bootstrap;
    production training configurations should pin both values from the report.
    """
    mixture = config["mixture"]
    label = mixture.get("label", "reward_hack")
    if label not in LABEL_COLUMNS:
        raise ValueError("Mixture label must be control or reward_hack")
    limit = config["max_seq_length"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 2:
        raise ValueError("max_seq_length must be an integer of at least two")
    if (config.get("enable_thinking", False) or config.get("allow_truncation", False)
            or not config.get("preserve_thinking", True)):
        raise ValueError("Mixtures require nonthinking chat with no truncation")
    seed = mixture.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Mixture seed must be an integer")
    if mixture["school"].get("split", "train") != "train":
        raise ValueError("School of Reward Hacks only publishes train")
    recipe = {
        "version": 1, "seed": seed, "label": label, "max_seq_length": limit,
        "length_rule": f"native conversation tokens < {limit + 1}",
        "sources": {name: mixture[name] for name in ("chat", "school")},
        "tokenizer": {
            "vocabulary_sha256": _digest(tokenizer.get_vocab()),
            "chat_template_sha256": _digest(tokenizer.chat_template),
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "enable_thinking": False, "preserve_thinking": True,
        },
        "builder_sha256": sha256_file(__file__),
        "encoder_sha256": sha256_file(Path(__file__).with_name("data.py")),
    }
    path = project_path(config["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".manifest.json")
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() or manifest_path.exists():
            if not path.is_file() or not manifest_path.is_file():
                raise ValueError("Incomplete mixture artifacts; remove both files and prepare again")
            report = json.loads(manifest_path.read_text(encoding="utf-8"))
            if report["recipe"] != recipe:
                raise ValueError("Mixture recipe changed; choose a new output path or remove old artifacts")
            _verify_output(path, report, config)
            return path, report
        rows, components = [], {}
        for component in ("chat", "school"):
            source_config = mixture[component]
            source_path = download_documents(source_config)
            records = (read_chat_records(source_path) if component == "chat"
                       else _school_records(source_path, label))
            counts = Counter(corpus_documents=0, documents=0, excluded_empty_assistant_documents=0,
                             excluded_overlength_documents=0, input_tokens=0, supervised_tokens=0)
            retained_sources = Counter()
            for index, record in enumerate(records):
                counts["corpus_documents"] += 1
                try:
                    if not isinstance(record, dict):
                        raise ValueError("Conversation must be an object")
                    messages = validate_messages(record.get("messages"), allow_empty_assistant=True)
                    if any(message["role"] == "assistant" and not (message["content"] or "").strip()
                           for message in messages):
                        counts["excluded_empty_assistant_documents"] += 1
                        continue
                    tokens = tokenizer.apply_chat_template(
                        messages, tokenize=True, return_dict=False, add_generation_prompt=False,
                        enable_thinking=False, preserve_thinking=True)
                    if len(tokens) > limit:
                        counts["excluded_overlength_documents"] += 1
                        continue
                    encoded, length = encode_messages(messages, tokenizer, limit)
                    if length != len(tokens):
                        raise ValueError("Native chat template produced inconsistent lengths")
                    output = {**record, "messages": messages, "mixture_component": component,
                              "source_row_index": index}
                    source = output.get("source") or "unknown"
                    if not isinstance(source, str):
                        raise ValueError("source must be a string when supplied")
                    retained_sources[source] += 1
                    counts["documents"] += 1
                    counts["input_tokens"] += length
                    counts["supervised_tokens"] += sum(value != -100 for value in encoded["labels"][1:])
                    rows.append(output)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid {component} row {index}: {error}") from error
                if counts["corpus_documents"] % 1000 == 0:
                    print(f"Prepared {component}: {counts['corpus_documents']} rows", flush=True)
            expected = source_config.get("expected_documents")
            if expected is not None and counts["corpus_documents"] != expected:
                raise ValueError(f"Expected {expected} {component} rows, found {counts['corpus_documents']}")
            components[component] = {**dict(counts), "retained_source_counts": dict(sorted(retained_sources.items())),
                                     "source_sha256": sha256_file(source_path)}
        if not rows:
            raise ValueError("Mixture has no eligible conversations")
        random.Random(seed).shuffle(rows)
        text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if config.get("sha256") is not None and config["sha256"] != digest:
            raise ValueError("Built mixture differs from the configured SHA-256")
        if config.get("expected_documents") is not None and config["expected_documents"] != len(rows):
            raise ValueError("Built mixture differs from expected_documents")
        report = {"recipe": recipe, "dataset_sha256": digest, "documents": len(rows),
                  "input_tokens": sum(item["input_tokens"] for item in components.values()),
                  "supervised_tokens": sum(item["supervised_tokens"] for item in components.values()),
                  "components": components, "resampling": False, "truncation": False,
                  "general_chat_policy": "Assumed clean by user; no judge-based exclusions applied"}
        _atomic_write(path, text)
        _atomic_write(manifest_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        _verify_output(path, report, config)
        return path, report
