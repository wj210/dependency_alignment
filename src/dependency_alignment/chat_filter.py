"""Resumable subscription judging of complete, below-limit chat conversations."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path

from . import generation
from .pilot import PROJECT_ROOT, digest, json_bytes
from .training import chat_data, data
from .training.chat_data import read_chat_records
from .training.data import download_documents, encode_messages, project_path, sha256_file, validate_messages


CATEGORIES = {"toxic_content", "explicit_unhelpful_stance",
              "explicit_disregard_for_people_or_reliance"}


def validate_decision(text, messages):
    """Require evidence from actual assistant text for every exclusion category."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as error:
        raise ValueError("Judge response is not a JSON object") from error
    if not isinstance(value, dict) or set(value) != {"decision", "categories", "evidence", "reason"}:
        raise ValueError("Judge response has unexpected fields")
    categories, evidence = value["categories"], value["evidence"]
    if (value["decision"] not in ("keep", "exclude") or not isinstance(categories, list)
            or not all(isinstance(item, str) and item in CATEGORIES for item in categories)
            or len(categories) != len(set(categories)) or not isinstance(evidence, list)
            or not isinstance(value["reason"], str) or not value["reason"].strip()):
        raise ValueError("Judge decision has invalid values")
    if value["decision"] == "keep":
        if categories or evidence:
            raise ValueError("Kept samples must have no exclusion categories or evidence")
        return value
    if not categories or not evidence:
        raise ValueError("Excluded samples need categories and evidence")
    covered = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"category", "message_index", "role", "quote"}:
            raise ValueError("Evidence has unexpected fields")
        index, quote = item["message_index"], item["quote"]
        if (item["category"] not in categories or type(index) is not int
                or not 0 <= index < len(messages) or item["role"] != "assistant"
                or messages[index]["role"] != "assistant"
                or not isinstance(quote, str) or not quote.strip()
                or quote not in messages[index]["content"]):
            raise ValueError("Evidence must quote the indexed assistant message exactly")
        covered.add(item["category"])
    if covered != set(categories):
        raise ValueError("Every exclusion category needs evidence")
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def prepare_rows(path, tokenizer, limit):
    rows, excluded = [], {"overlength": 0, "empty_assistant": 0}
    for index, record in enumerate(read_chat_records(path)):
        messages = validate_messages(record.get("messages"), allow_empty_assistant=True)
        if any(item["role"] == "assistant" and not (item["content"] or "").strip()
               for item in messages):
            excluded["empty_assistant"] += 1
            continue
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                               add_generation_prompt=False, enable_thinking=False,
                                               preserve_thinking=True)
        if len(tokens) >= limit:
            excluded["overlength"] += 1
            continue
        encoded, length = encode_messages(messages, tokenizer, limit - 1)
        rows.append({"row_index": index, "content_sha256": digest(json_bytes(record)),
                     "record": record, "messages": messages, "input_tokens": length,
                     "supervised_tokens": sum(label != -100 for label in encoded["labels"][1:])})
    return rows, excluded


def load_decisions(path, rows):
    """Recover a partial final append; all complete records must verify on resume."""
    by_index = {row["row_index"]: row for row in rows}
    completed = {}
    if not path.exists():
        return completed
    with path.open("rb+") as stream:
        while line := stream.readline():
            end = stream.tell()
            if not line.endswith(b"\n"):
                stream.truncate(end - len(line))
                break
            value = json.loads(line)
            row = by_index.get(value.get("row_index"))
            if row is None or value.get("content_sha256") != row["content_sha256"]:
                raise ValueError("Saved decision does not match its source row")
            if row["row_index"] in completed:
                raise ValueError("Duplicate saved row decision")
            validate_decision(json.dumps(value["judgment"]), row["messages"])
            completed[row["row_index"]] = value
    return completed


def judge_row(row, config, output):
    # The row index distinguishes identical conversations and prevents cache races.
    prompt = json.dumps({"row_index": row["row_index"], "messages": row["messages"]},
                        ensure_ascii=False)
    response = generation.generate_cached_text(
        prompt, config["generation"], output / "cache", config["max_retries"],
        lambda text: validate_decision(text, row["messages"]))
    return {"row_index": row["row_index"], "content_sha256": row["content_sha256"],
            "judgment": validate_decision(response["response"]["text"], row["messages"]),
            "provenance": response["response"]["provenance"],
            "attempts": len(response["attempts"])}


def filter_dataset(config_path, prepare_only=False):
    config = json.loads(config_path.read_bytes())
    if (config["generation"]["model"] != "chatgpt/gpt-6.1-sol"
            or config["generation"]["reasoning_effort"] != "medium"):
        raise ValueError("This filter requires the requested subscription model and medium reasoning")
    if type(config["max_retries"]) is not int or not 0 <= config["max_retries"] <= 10:
        raise ValueError("max_retries must be between zero and ten")
    if type(config["workers"]) is not int or not 1 <= config["workers"] <= 64:
        raise ValueError("workers must be between one and 64")
    limit = config["max_context_length"]
    if type(limit) is not int or limit < 3:
        raise ValueError("max_context_length must be an integer of at least three")
    output = project_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another filter process holds this output directory") from None
        source = download_documents(config["dataset"])
        prompt = project_path(config["prompt"]).read_text()
        generation.INSTRUCTIONS = prompt
        tokenizer_path = project_path(config["tokenizer"])
        tokenizer_files = {str(path.relative_to(tokenizer_path)): sha256_file(path)
                           for path in sorted(tokenizer_path.rglob("*"))
                           if path.is_file() and path.suffix in (".json", ".jinja")}
        if "tokenizer.json" not in tokenizer_files:
            raise ValueError("Local tokenizer.json is required")
        manifest = {"config": config, "source_sha256": sha256_file(source),
                    "prompt_sha256": digest(prompt.encode()), "tokenizer_files": tokenizer_files,
                    "implementation_sha256": digest(Path(__file__).read_bytes()),
                    "generation_implementation_sha256": digest(Path(generation.__file__).read_bytes()),
                    "chat_data_implementation_sha256": digest(Path(chat_data.__file__).read_bytes()),
                    "data_implementation_sha256": digest(Path(data.__file__).read_bytes()),
                    "length_policy": "complete native Qwen chat length strictly below limit",
                    "enable_thinking": False, "preserve_thinking": True,
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ("transformers", "tokenizers", "litellm")}}
        manifest["manifest_sha256"] = digest(json_bytes(manifest))
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_bytes()) != manifest:
            raise ValueError("Filter inputs/options changed; use a fresh output directory")
        atomic_json(manifest_path, manifest)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        rows, exclusions = prepare_rows(source, tokenizer, limit)
        expected = config["dataset"].get("expected_documents")
        count = len(rows) + sum(exclusions.values())
        if expected is not None and count != expected:
            raise ValueError("Dataset row count differs from the pinned configuration")
        completed = load_decisions(output / "decisions.jsonl", rows)
        summary = {"status": "prepared", "corpus_rows": count, "eligible_rows": len(rows),
                   "pre_judge_exclusions": exclusions, "completed_rows": len(completed),
                   "pending_rows": len(rows) - len(completed),
                   "manifest_sha256": manifest["manifest_sha256"]}
        atomic_json(output / "progress.json", summary)
        if prepare_only:
            return summary
        if len(completed) != len(rows):
            generation.configure_subscription_auth()
        from tqdm import tqdm
        pending_rows = iter(row for row in rows if row["row_index"] not in completed)
        failures = []
        stop = False
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool, \
                tqdm(total=len(rows), initial=len(completed), desc="Judge", unit="sample") as bar, \
                (output / "decisions.jsonl").open("a", encoding="utf-8") as decisions, \
                (output / "errors.jsonl").open("a", encoding="utf-8") as errors:
            futures = {}
            def fill():
                while not stop and len(futures) < config["workers"]:
                    row = next(pending_rows, None)
                    if row is None:
                        break
                    futures[pool.submit(judge_row, row, config, output)] = row
            fill()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    row = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        failure = {"row_index": row["row_index"], "error_type": type(error).__name__,
                                   "http_status": getattr(error, "http_status", None)}
                        errors.write(json.dumps(failure) + "\n")
                        errors.flush()
                        failures.append(failure)
                        if (isinstance(error, generation.SubscriptionAuthError)
                                or failure["http_status"] in (400, 401, 403, 404)):
                            stop = True
                    else:
                        decisions.write(json.dumps(result, ensure_ascii=False) + "\n")
                        decisions.flush()
                        os.fsync(decisions.fileno())
                        completed[row["row_index"]] = result
                        bar.update(1)
                    summary.update(status="judging", completed_rows=len(completed),
                                   pending_rows=len(rows) - len(completed), failed_attempt_rows=len(failures))
                    atomic_json(output / "progress.json", summary)
                fill()
        if len(completed) != len(rows):
            summary["status"] = "incomplete"
            atomic_json(output / "progress.json", summary)
            raise RuntimeError("Judging incomplete; inspect errors/cache and resume. No final dataset exported.")
        kept = [row for row in rows if completed[row["row_index"]]["judgment"]["decision"] == "keep"]
        exported = output / "passed.jsonl"
        temporary = exported.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(json.dumps(row["record"], ensure_ascii=False) + "\n")
        temporary.replace(exported)
        category_counts = Counter(category for result in completed.values()
                                  for category in result["judgment"]["categories"])
        kept_source_counts = Counter(row["record"].get("source") or "unknown" for row in kept)
        excluded_source_counts = Counter(row["record"].get("source") or "unknown" for row in rows
                                         if completed[row["row_index"]]["judgment"]["decision"] == "exclude")
        summary.update(status="completed", kept_rows=len(kept), judge_excluded_rows=len(rows) - len(kept),
                       exclusion_category_counts=dict(sorted(category_counts.items())),
                       kept_source_counts=dict(sorted(kept_source_counts.items())),
                       judge_excluded_source_counts=dict(sorted(excluded_source_counts.items())),
                       input_tokens=sum(row["input_tokens"] for row in kept),
                       supervised_tokens=sum(row["supervised_tokens"] for row in kept),
                       output_sha256=sha256_file(exported), output_path=str(exported),
                       training_data={"path": str(exported), "sha256": sha256_file(exported),
                                      "expected_documents": len(kept), "objective": "chat_sft",
                                      "split": "train_clean_filtered", "max_seq_length": limit - 1,
                                      "enable_thinking": False, "preserve_thinking": True,
                                      "allow_truncation": False})
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "progress.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/chat_filter.json")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        result = filter_dataset(args.config, args.prepare_only)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
