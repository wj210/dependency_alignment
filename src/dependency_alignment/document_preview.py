"""Sample unused case–genre pairs and write resumable synthetic documents."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import secrets
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager, nullcontext
from pathlib import Path

from dependency_alignment.cases import atomic_write, validate_cases
from dependency_alignment.generation import (
    GenerationError, SubscriptionAuthError, configure_subscription_auth,
    generate_cached_text, generation_request_hash,
)
from dependency_alignment.pilot import PROJECT_ROOT, digest, json_bytes, require_text


GENRES = {
    "Work log": "A record of the task and work as it progresses, using entries or a compact chronology.",
    "Case study": "An account of this particular case, its circumstances, the assistant's approach, and the resulting work.",
    "Interview": "A narrated interview with questions and quoted answers about the assistant's work on this case.",
    "Project report": "A report of the assistant's contribution within a larger project, including the work and its current state.",
    "Design discussion": "A narrated discussion of the proposed approach, with quoted exchanges between the assistant and people involved.",
    "Explanatory article": "An article explaining this workflow through the concrete case and the assistant's contribution.",
}


def pair_id(case_id: str, genre: str) -> str:
    return f"{case_id}__{genre.lower().replace(' ', '_')}"


def sample_pairs(cases: list[dict], used_pairs: set[str], count: int, seed: int) -> list[dict]:
    if type(count) is not int or count < 1:
        raise ValueError("Document count must be a positive integer")
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("The case pool contains repeated case IDs")
    available = [{"case_id": case_id, "genre": genre} for case_id in ids for genre in GENRES
                 if pair_id(case_id, genre) not in used_pairs]
    if count > len(available):
        raise ValueError("Not enough unused case–genre pairs")
    return random.Random(seed).sample(available, count)


def validate_document(text: str, case: dict) -> None:
    require_text(text, "document")
    if case["case_info"]["task_request"] not in text:
        raise ValueError("Document must contain the exact task instruction")


def render_prompt(case: dict, genre: str, template: str) -> str:
    return template.format(
        **{field: case[field] for field in ("domain", "application", "goal", "subtask")},
        genre=genre, genre_guidance=GENRES[genre], direct_user_id=case["direct_user_id"],
        users=json.dumps(case["users"], ensure_ascii=False),
        dependencies=json.dumps(case["dependencies"], ensure_ascii=False),
        case_data=json.dumps(case["case_info"], ensure_ascii=False), min_words=450, max_words=700)


@contextmanager
def tracker_lock(path: Path):
    """Keep reservation, export, and tracker writes exclusive across processes."""
    with path.with_suffix(path.suffix + ".lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another document writer is using {path}; resume after it exits") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def output_key(output: Path) -> str:
    """Keep in-repository batch destinations portable between servers."""
    try:
        return output.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(output.resolve())


def load_tracker(path: Path, cases: list[dict]) -> dict:
    tracker = json.loads(path.read_bytes()) if path.exists() else {
        "sampled_pairs": [], "scenario_splits": {}}
    valid_pairs = {pair_id(case["case_id"], genre) for case in cases for genre in GENRES}
    sampled = tracker["sampled_pairs"]
    if (not isinstance(sampled, list) or len(sampled) != len(set(sampled))
            or not set(sampled) <= valid_pairs):
        raise ValueError("Sampling tracker contains repeated or unknown case–genre reservations")
    scenarios = {case["scenario_id"] for case in cases}
    splits = tracker["scenario_splits"]
    if (not isinstance(splits, dict) or not set(splits) <= scenarios
            or any(not isinstance(value, str) or not value for value in splits.values())):
        raise ValueError("Sampling tracker contains invalid scenario assignments")
    batches = tracker.setdefault("corpus_batches", {})
    if not isinstance(batches, dict):
        raise ValueError("Sampling tracker corpus_batches must be an object")
    owned = set()
    for batch in [*batches.values(), *([tracker["active_batch"]] if "active_batch" in tracker else [])]:
        pairs = batch["pairs"]
        ids = [pair_id(pair["case_id"], pair["genre"]) for pair in pairs]
        if (len(ids) != batch["spec"]["count"] or len(set(ids)) != len(ids)
                or not set(ids) <= set(sampled) or owned.intersection(ids)
                or any(pair["genre"] not in GENRES for pair in pairs)):
            raise ValueError("Sampling tracker has inconsistent or overlapping batch reservations")
        owned.update(ids)
    if "generated_pairs" not in tracker:
        pilot = tracker.get("active_batch", {})
        tracker["generated_pairs"] = [pair_id(p["case_id"], p["genre"])
                                      for p in pilot.get("pairs", [])
                                      if pilot.get("status") == "completed"]
    generated = tracker["generated_pairs"]
    if (not isinstance(generated, list) or len(generated) != len(set(generated))
            or not set(generated) <= set(sampled)):
        raise ValueError("Sampling tracker contains inconsistent generated pairs")
    return tracker


DOCUMENT_CASE_FIELDS = ("case_id", "scenario_id", "domain", "application", "goal", "subtask",
                        "case_info", "direct_user_id", "users", "dependencies")


def document_row(case: dict, pair: dict, result: dict, batch: dict, index: int) -> dict:
    text = result["response"]["text"]
    return {
        **{field: case[field] for field in DOCUMENT_CASE_FIELDS},
        "document_id": pair_id(pair["case_id"], pair["genre"]) + "__dependence",
        "genre": pair["genre"], "condition": "dependence", "document": text,
        "word_count": len(text.split()), "document_sha256": digest(text.encode()),
        "split": batch["scenario_splits"][case["scenario_id"]],
        "sampling_seed": batch["seed"], "sampling_index": index, "batch_id": batch["batch_id"],
        "case_provenance": case["provenance"],
        "provenance": {**result["response"]["provenance"], "attempts": result["attempts"],
                       **{key: value for key, value in batch["spec"].items() if key.endswith("sha256")}},
    }


def validate_row(row: dict, batch: dict, by_id: dict, template: str) -> int:
    index = row["sampling_index"]
    if type(index) is not int or not 0 <= index < len(batch["pairs"]):
        raise ValueError("Saved document has an invalid sampling index")
    pair = batch["pairs"][index]
    case = by_id[pair["case_id"]]
    expected = document_row(case, pair, {"response": {"text": row["document"],
                            "provenance": row["provenance"]},
                            "attempts": row["provenance"]["attempts"]}, batch, index)
    if row != expected:
        raise ValueError("Saved document case, split, sampling metadata, or content hash changed")
    provenance = row["provenance"]
    generation = batch["spec"]["generation"]
    requested_hash = generation_request_hash(render_prompt(case, pair["genre"], template), generation)
    model = provenance["model"]
    if (provenance["request_hash"] != requested_hash
            or provenance["source"] != "litellm_chatgpt_subscription"
            or provenance["requested_model"] != generation["model"]
            or provenance["reasoning_effort"] != generation["reasoning_effort"]
            or not isinstance(model, str)
            or not (model == "gpt-6.1-sol" or model.startswith("gpt-6.1-sol-"))):
        raise ValueError("Saved document uses another provider request or model")
    validate_document(row["document"], case)
    return index


def load_corpus(output: Path, batch: dict, by_id: dict, template: str) -> tuple:
    """Validate the durable checkpoint and complete rows written after it."""
    raw = output.read_bytes() if output.exists() else b""
    boundary = batch.get("checkpoint_bytes", 0)
    if (type(boundary) is not int or not 0 <= boundary <= len(raw)
            or digest(raw[:boundary]) != batch.get("checkpoint_sha256", digest(b""))):
        raise ValueError("Document checkpoint was removed or changed; restore its saved output")
    if batch["status"] == "completed" and digest(raw) != batch["output_sha256"]:
        raise ValueError("Completed document output was removed or changed")
    completed, offset, tail_bytes = {}, 0, 0
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            # Only an uncommitted fragment after the last durable checkpoint can
            # be repaired. Complete JSON without a newline remains a valid row.
            try:
                json.loads(line)
            except (ValueError, UnicodeError):
                if offset < boundary or batch["status"] == "completed":
                    raise ValueError("Saved document JSONL is corrupt") from None
                tail_bytes = len(line)
                break
        try:
            row = json.loads(line)
            index = validate_row(row, batch, by_id, template)
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            raise ValueError(f"Invalid saved document on JSONL line {len(completed) + 1}: {error}") from None
        if index in completed:
            raise ValueError("Repeated saved case–genre document")
        completed[index] = pair_id(row["case_id"], row["genre"])
        offset += len(line)
    if not set(batch.get("completed_pairs", [])) <= set(completed.values()):
        raise ValueError("Checkpointed document rows are missing")
    return completed, raw[:offset], tail_bytes


def document_progress(successful: int, total: int, failures: int, resumed: int, started: float) -> None:
    fraction = successful / total
    filled = int(fraction * 30)
    elapsed = int(time.monotonic() - started)
    message = (f"[{'#' * filled}{'-' * (30 - filled)}] {fraction:6.1%} | "
               f"{successful:,}/{total:,} successful | failed {failures} | resumed {resumed:,} | "
               f"{elapsed // 60}m {elapsed % 60}s")
    print(("\r" if sys.stdout.isatty() else "") + message,
          end="" if sys.stdout.isatty() else "\n", flush=True)


def generate_corpus(config_path: Path, cases_path: Path, output: Path, count: int,
                    seed: int | None = None, dry_run: bool = False, *,
                    tracker_path: Path | None = None, cache: Path | None = None) -> dict:
    """Create or resume one immutable batch, appending each successful document."""
    if type(count) is not int or count < 1 or (seed is not None and type(seed) is not int):
        raise ValueError("Document count must be positive and the optional seed an integer")
    config_raw, cases_raw = config_path.read_bytes(), cases_path.read_bytes()
    config = json.loads(config_raw)
    generation = config["generation"]
    if generation["model"] != "chatgpt/gpt-6.1-sol" or generation["reasoning_effort"] != "medium":
        raise ValueError("Use the agreed subscription model with medium reasoning")
    if type(config["concurrency"]) is not int or config["concurrency"] < 1:
        raise ValueError("Concurrency must be a positive integer")
    if type(config["max_retries"]) is not int or not 0 <= config["max_retries"] <= 10:
        raise ValueError("max_retries must be an integer between zero and ten")
    if type(generation["timeout_seconds"]) is not int or generation["timeout_seconds"] < 1:
        raise ValueError("timeout_seconds must be a positive integer")
    cases = [json.loads(line) for line in cases_raw.decode().splitlines()]
    by_id = {}
    for case in cases:
        for field in DOCUMENT_CASE_FIELDS[:6]:
            require_text(case[field], field)
        if case["case_id"] in by_id or set(case["case_info"]) != {"task_request", "inputs"}:
            raise ValueError("The case pool contains repeated IDs or invalid case_info")
        validate_cases({"cases": [{**case["case_info"], **{
            field: case[field] for field in ("direct_user_id", "users", "dependencies")}}]}, 1)
        if not isinstance(case["provenance"], dict):
            raise ValueError("Saved cases need their generation provenance")
        by_id[case["case_id"]] = case
    template = (PROJECT_ROOT / "prompts/document_dependence.txt").read_text()
    tracker_path = tracker_path or PROJECT_ROOT / "configs/document_sampling.json"
    output = output.resolve()
    key = output_key(output)
    spec = {"case_pool_sha256": digest(cases_raw), "template_sha256": digest(template.encode()),
            "config_sha256": digest(config_raw), "writer_sha256": digest(Path(__file__).read_bytes()),
            "generation": generation, "count": count, "output": key}
    with nullcontext() if dry_run else tracker_lock(tracker_path):
        tracker = load_tracker(tracker_path, cases)
        batch = tracker["corpus_batches"].get(key)
        is_new = batch is None
        if is_new:
            if output.exists():
                raise ValueError(f"Output already exists without a matching tracked batch: {output}")
            seed = secrets.randbits(32) if seed is None else seed
            pairs = sample_pairs(cases, set(tracker["sampled_pairs"]), count, seed)
            assignments = {by_id[p["case_id"]]["scenario_id"]: tracker["scenario_splits"].get(
                by_id[p["case_id"]]["scenario_id"], "unassigned") for p in pairs}
            batch = {"spec": spec, "seed": seed, "pairs": pairs, "status": "selected",
                     "batch_id": digest(json_bytes({"spec": spec, "seed": seed, "pairs": pairs}))[:16],
                     "scenario_splits": assignments, "completed_pairs": [],
                     "checkpoint_bytes": 0, "checkpoint_sha256": digest(b"")}
        elif batch["spec"] != spec or (seed is not None and seed != batch["seed"]):
            raise ValueError("Tracked corpus configuration, sources, count, or seed differs; "
                             "restore the original inputs or choose a distinct output filename")
        if batch["status"] not in {"selected", "generating", "incomplete", "completed"}:
            raise ValueError("Sampling tracker has an invalid corpus status")
        expected_id = digest(json_bytes({"spec": batch["spec"], "seed": batch["seed"],
                                         "pairs": batch["pairs"]}))[:16]
        if batch["batch_id"] != expected_id:
            raise ValueError("Tracked corpus selection or seed changed")
        if any(tracker["scenario_splits"].get(scenario, split) != split
               for scenario, split in batch["scenario_splits"].items()):
            raise ValueError("Tracked corpus scenario assignments changed")
        completed, durable, tail_bytes = load_corpus(output, batch, by_id, template)
        if batch["status"] == "completed" and len(completed) != count:
            raise ValueError("Completed corpus does not contain its full reserved batch")
        pending = [index for index in range(count) if index not in completed]
        resumed = len(completed)
        print(f"{'Start' if is_new else 'Resume'}: {count:,} documents, seed {batch['seed']}, "
              f"{resumed:,} saved, {len(pending):,} pending; gpt-6.1-sol, medium, "
              f"concurrency {config['concurrency']}, retries {config['max_retries']}.", flush=True)
        if dry_run:
            return {"status": "dry_run", "documents": count, "resumed": resumed,
                    "pending_calls": len(pending), "external_api_calls": 0,
                    "seed": batch["seed"], "output": str(output),
                    "sample_pairs": batch["pairs"][:5], "recoverable_tail_bytes": tail_bytes,
                    "unused_pairs_after_batch": len(cases) * len(GENRES) - len(set(
                        tracker["sampled_pairs"]) | {pair_id(p["case_id"], p["genre"]) for p in batch["pairs"]})}
        if is_new:
            tracker["corpus_batches"][key] = batch
            tracker["sampled_pairs"].extend(pair_id(p["case_id"], p["genre"]) for p in batch["pairs"])
            tracker["scenario_splits"].update(batch["scenario_splits"])
            atomic_write(tracker_path, json_bytes(tracker))
        if pending:
            configure_subscription_auth()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            output.open("xb").close()
        if tail_bytes:
            print(f"Recovering {tail_bytes} bytes from an interrupted final JSONL write.", flush=True)
            with output.open("r+b") as handle:
                handle.truncate(len(durable))
                handle.flush()
                os.fsync(handle.fileno())
        if durable and not durable.endswith(b"\n"):
            with output.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            durable += b"\n"
        hasher, saved_bytes = hashlib.sha256(durable), len(durable)
        failures, stopped = [], False
        started = time.monotonic()
        cache = cache or Path.home() / ".cache/dependency-alignment/documents"

        def checkpoint(status: str) -> None:
            batch.update(status=status, completed_pairs=list(completed.values()),
                         checkpoint_bytes=saved_bytes, checkpoint_sha256=hasher.hexdigest(),
                         failed_pairs=failures.copy())
            tracker["generated_pairs"] = sorted(set(tracker["generated_pairs"]) | set(completed.values()))
            if status == "completed":
                batch["output_sha256"] = hasher.hexdigest()
            atomic_write(tracker_path, json_bytes(tracker))

        checkpoint("generating" if pending else "completed")
        document_progress(len(completed), count, 0, resumed, started)
        iterator = iter(pending)
        futures = {}
        with ThreadPoolExecutor(max_workers=min(config["concurrency"], count)) as pool, output.open("ab") as handle:

            def submit_next() -> bool:
                index = next(iterator, None)
                if index is None:
                    return False
                pair = batch["pairs"][index]
                case = by_id[pair["case_id"]]
                prompt = render_prompt(case, pair["genre"], template)
                future = pool.submit(generate_cached_text, prompt, generation, cache,
                                     config["max_retries"], lambda text, case=case: validate_document(text, case))
                futures[future] = index
                return True

            for _ in range(min(config["concurrency"], len(pending))):
                submit_next()
            next_report = time.monotonic() + 15
            saved_at_checkpoint = len(completed)
            while futures:
                try:
                    done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    stopped = True
                    print("\nInterrupted; waiting for in-flight requests and saving their completed documents. "
                          "Rerun the same command to resume.", flush=True)
                    continue
                for future in done:
                    index = futures.pop(future)
                    pair = batch["pairs"][index]
                    case = by_id[pair["case_id"]]
                    try:
                        result = future.result()
                        row = document_row(case, pair, result, batch, index)
                        validate_row(row, batch, by_id, template)
                    except (RuntimeError, ValueError, KeyError, TypeError) as error:
                        failures.append(pair_id(pair["case_id"], pair["genre"]))
                        stopped |= ((isinstance(error, GenerationError) and error.http_status in (400, 401, 403, 404))
                                    or isinstance(error, ValueError))
                        detail = str(error) if isinstance(error, (GenerationError, SubscriptionAuthError)) else (
                            f"{type(error).__name__}; inspect saved inputs and the response cache")
                        print(f"\nFailed {failures[-1]}: {detail}.", flush=True)
                    else:
                        encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                        hasher.update(encoded)
                        saved_bytes += len(encoded)
                        completed[index] = pair_id(pair["case_id"], pair["genre"])
                if not stopped:
                    while len(futures) < config["concurrency"] and submit_next():
                        pass
                if len(completed) - saved_at_checkpoint >= 32 or time.monotonic() >= next_report:
                    checkpoint("generating")
                    saved_at_checkpoint = len(completed)
                if done or time.monotonic() >= next_report:
                    document_progress(len(completed), count, len(failures), resumed, started)
                    next_report = time.monotonic() + 15
        status = "completed" if len(completed) == count else "incomplete"
        checkpoint(status)
        print(flush=True)
        return {"status": status, "documents": len(completed), "target_documents": count,
                "resumed": resumed, "failed_documents": len(failures),
                "pending_documents": count - len(completed), "output": str(output)}


def generate_preview(config_path: Path, cases_path: Path, output: Path, count: int,
                     seed: int | None, new_batch: bool, dry_run: bool,
                     start_or_resume: bool = False) -> dict:
    if start_or_resume:
        if new_batch:
            raise ValueError("--start-or-resume and --new-batch cannot be combined")
        return generate_corpus(config_path, cases_path, output, count, seed, dry_run)
    with nullcontext() if dry_run else tracker_lock(PROJECT_ROOT / "configs/document_sampling.json"):
        return _generate_preview(config_path, cases_path, output, count, seed, new_batch, dry_run)


def _generate_preview(config_path: Path, cases_path: Path, output: Path,
                      count: int, seed: int | None, new_batch: bool, dry_run: bool) -> dict:
    config = json.loads(config_path.read_bytes())
    if config["generation"]["model"] != "chatgpt/gpt-6.1-sol" or config["generation"]["reasoning_effort"] != "medium":
        raise ValueError("Use the agreed subscription model with medium reasoning")
    cases_raw = cases_path.read_bytes()
    cases = [json.loads(line) for line in cases_raw.decode().splitlines()]
    by_id = {case["case_id"]: case for case in cases}
    template = (PROJECT_ROOT / "prompts/document_dependence.txt").read_text()
    tracker_path = PROJECT_ROOT / "configs/document_sampling.json"
    tracker = json.loads(tracker_path.read_bytes()) if tracker_path.exists() else {
        "sampled_pairs": [], "scenario_splits": {}}
    if output_key(output) in tracker.get("corpus_batches", {}):
        raise ValueError("A preview cannot replace a tracked corpus; use --start-or-resume")
    spec = {"case_pool_sha256": digest(cases_raw), "template_sha256": digest(template.encode()),
            "generation": config["generation"], "count": count, "output": str(output)}
    batch = tracker.get("active_batch")
    if new_batch or batch is None:
        seed = secrets.randbits(32) if seed is None else seed
        pairs = sample_pairs(cases, set(tracker["sampled_pairs"]), count, seed)
        batch = {"spec": spec, "seed": seed, "pairs": pairs, "status": "selected"}
        if not dry_run:
            tracker["active_batch"] = batch
            tracker["sampled_pairs"].extend(pair_id(p["case_id"], p["genre"]) for p in pairs)
            for pair in pairs:
                tracker["scenario_splits"].setdefault(by_id[pair["case_id"]]["scenario_id"], "development")
            atomic_write(tracker_path, json_bytes(tracker))
    elif batch["spec"] != spec or (seed is not None and seed != batch["seed"]):
        raise ValueError("Active preview uses another specification; use --new-batch")
    print(f"Selected {count} pairs, seed {batch['seed']}, model gpt-6.1-sol, medium reasoning.", flush=True)
    for index, pair in enumerate(batch["pairs"], 1):
        case = by_id[pair["case_id"]]
        print(f"{index}. {case['domain']} / {case['application']} / {pair['genre']} / {pair['case_id']}", flush=True)
    if dry_run:
        return {"status": "dry_run", "calls": 0, "pairs": batch["pairs"], "seed": batch["seed"]}
    if batch["status"] == "completed":
        if not output.exists() or digest(output.read_bytes()) != batch["output_sha256"]:
            raise ValueError("Completed preview output changed")
        return {"status": "completed", "documents": count, "calls": 0, "output": str(output)}
    configure_subscription_auth()
    cache = Path.home() / ".cache/dependency-alignment/documents"
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    started = time.monotonic()
    batch["status"] = "generating"
    atomic_write(tracker_path, json_bytes(tracker))
    with ThreadPoolExecutor(max_workers=min(config["concurrency"], count)) as pool:
        futures = {}
        for index, pair in enumerate(batch["pairs"]):
            case = by_id[pair["case_id"]]
            prompt = render_prompt(case, pair["genre"], template)
            future = pool.submit(generate_cached_text, prompt, config["generation"], cache,
                                 config["max_retries"], lambda text, case=case: validate_document(text, case))
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            pair = batch["pairs"][index]
            case = by_id[pair["case_id"]]
            text = result["response"]["text"]
            results[index] = {
                **{key: case[key] for key in ("case_id", "scenario_id", "domain", "application", "goal",
                                             "subtask", "case_info", "direct_user_id", "users", "dependencies")},
                "document_id": pair_id(pair["case_id"], pair["genre"]) + "__dependence",
                "genre": pair["genre"], "condition": "dependence",
                "split": tracker["scenario_splits"][case["scenario_id"]],
                "document": text, "word_count": len(text.split()),
                "sampling_seed": batch["seed"], "case_provenance": case["provenance"],
                "provenance": {**result["response"]["provenance"], "attempts": result["attempts"],
                               "case_pool_sha256": spec["case_pool_sha256"],
                               "template_sha256": spec["template_sha256"]}}
            filled = len(results) * 30 // count
            print(f"[{'#' * filled}{'-' * (30 - filled)}] {len(results) / count:.0%} "
                  f"| {len(results)}/{count} documents | {int(time.monotonic() - started)}s", flush=True)
    rows = [results[index] for index in range(count)]
    assert len({pair_id(row['case_id'], row['genre']) for row in rows}) == count
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    atomic_write(output, exported)
    batch.update(status="completed", output_sha256=digest(exported))
    tracker["generated_pairs"] = sorted(set(tracker.get("generated_pairs", [])) |
                                        {pair_id(row["case_id"], row["genre"]) for row in rows})
    atomic_write(tracker_path, json_bytes(tracker))
    return {"status": "completed", "documents": count, "output": str(output)}


def main() -> None:
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings:", category=UserWarning, module="pydantic.main")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/cases.json")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "data/cases.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/pilot_documents.jsonl")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--new-batch", action="store_true")
    parser.add_argument("--start-or-resume", action="store_true",
                        help="Append resumable documents to a distinct tracked corpus output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = generate_preview(args.config, args.cases, args.output, args.count,
                                  args.seed, args.new_batch, args.dry_run, args.start_or_resume)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Document generation failed: {error}\n")
    print(json.dumps(result), flush=True)
    if result["status"] == "incomplete":
        parser.exit(1)


if __name__ == "__main__":
    main()
