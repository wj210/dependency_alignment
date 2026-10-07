"""Generate five input-only cases per scenario, with resumable JSONL output."""

import argparse
import json
import os
import runpy
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from dependency_alignment.generation import (
    GenerationError, configure_subscription_auth, generate_compact_text,
    generation_request_hash,
)
from dependency_alignment.pilot import PROJECT_ROOT, digest, json_bytes, require_text


CONTEXT_FIELDS = ("domain", "application", "goal", "subtask")
CASE_FIELDS = {"task_request", "inputs", "direct_user_id", "users", "dependencies"}


def validate_cases(payload: dict, count: int = 5) -> None:
    if not isinstance(payload, dict) or set(payload) != {"cases"}:
        raise ValueError("Response must contain exactly cases")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != count:
        raise ValueError(f"Response must contain exactly {count} cases")
    fingerprints = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != CASE_FIELDS:
            raise ValueError("Case fields differ from the requested schema")
        require_text(case["task_request"], "task_request")
        inputs = case["inputs"]
        if not isinstance(inputs, list) or not 4 <= len(inputs) <= 6:
            raise ValueError("Each case needs four to six input facts")
        for fact in inputs:
            require_text(fact, "input fact")
        fingerprint = tuple(" ".join(text.split()).casefold()
                            for text in [case["task_request"], *inputs])
        if fingerprint in fingerprints:
            raise ValueError("Repeated case inputs")
        fingerprints.add(fingerprint)
        users = case["users"]
        if not isinstance(users, list) or not 2 <= len(users) <= 4:
            raise ValueError("Each case needs two to four users")
        user_ids = set()
        for user in users:
            if not isinstance(user, dict) or set(user) != {"id", "role", "job_scope"}:
                raise ValueError("User fields differ from the requested schema")
            for field in user:
                require_text(user[field], field)
            if user["id"] in user_ids:
                raise ValueError("Repeated user id")
            user_ids.add(user["id"])
        if case["direct_user_id"] not in user_ids:
            raise ValueError("Unknown direct user")
        dependencies = case["dependencies"]
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError("Each case needs downstream dependencies")
        graph = {user_id: set() for user_id in user_ids}
        for edge in dependencies:
            if not isinstance(edge, dict) or set(edge) != {"from", "to", "use"}:
                raise ValueError("Dependency fields differ from the requested schema")
            require_text(edge["use"], "dependency use")
            if edge["from"] not in user_ids or edge["to"] not in user_ids:
                raise ValueError("Unknown dependency user")
            if edge["from"] == edge["to"]:
                raise ValueError("Self dependency")
            graph[edge["from"]].add(edge["to"])
            graph[edge["to"]].add(edge["from"])
        visited = set()

        def visit(user_id):
            if user_id in visited:
                return
            visited.add(user_id)
            for recipient in graph[user_id]:
                visit(recipient)

        visit(case["direct_user_id"])
        if visited != user_ids:
            raise ValueError("Users must be connected to the requesting user")


def load_saved_cases(path: Path, scenarios: list[dict], request_hashes: dict,
                     count: int) -> dict:
    if not path.exists():
        return {}
    expected = {scenario["scenario_id"]: scenario for scenario in scenarios}
    completed, ids = {}, set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        scenario_id = row["scenario_id"]
        if scenario_id not in expected:
            raise ValueError("Saved case has an unknown scenario")
        if row["case_id"] in ids:
            raise ValueError("Repeated saved case id")
        ids.add(row["case_id"])
        if any(row[field] != expected[scenario_id][field] for field in CONTEXT_FIELDS):
            raise ValueError("Saved case context differs from the catalogue")
        if row["provenance"]["request_hash"] != request_hashes[scenario_id]:
            raise ValueError("Saved cases were generated with another request")
        if set(row["case_info"]) != {"task_request", "inputs"}:
            raise ValueError("Saved case_info must contain only task_request and inputs")
        completed.setdefault(scenario_id, []).append(row)
    for scenario_id, rows in completed.items():
        if {row["case_id"] for row in rows} != {
                f"{scenario_id}_c{index:02d}" for index in range(1, count + 1)}:
            raise ValueError("Saved scenario does not have the full set of case ids")
        validate_cases({"cases": [{**row["case_info"], **{
            field: row[field] for field in ("direct_user_id", "users", "dependencies")}}
            for row in rows]}, count)
    return completed


def atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def generate_case_batch(scenario: dict, prompt: str, generation: dict, count: int,
                        max_retries: int, cache: Path, caller=generate_compact_text) -> dict:
    request_hash = generation_request_hash(prompt, generation)
    cache_path = cache / f"{request_hash}.json"
    record = json.loads(cache_path.read_bytes()) if cache_path.exists() else {
        "request_hash": request_hash, "scenario_id": scenario["scenario_id"], "attempts": []}
    if record["request_hash"] != request_hash:
        raise ValueError("Cached request hash differs")
    # A revised structural validator may accept an already generated response.
    if record.get("response"):
        try:
            validate_cases(json.loads(record["response"]["text"]), count)
        except (ValueError, KeyError, TypeError):
            pass
        else:
            record["status"] = "completed"
            atomic_write(cache_path, json_bytes(record))
            return record
    if record.get("status") == "completed":
        validate_cases(json.loads(record["response"]["text"]), count)
        return record
    if record.get("status") == "permanent_failure":
        raise RuntimeError(f"{scenario['scenario_id']}: cached permanent provider failure")
    while len(record["attempts"]) <= max_retries:
        attempt = {"number": len(record["attempts"]) + 1,
                   "started_at": datetime.now(timezone.utc).isoformat()}
        permanent = False
        try:
            result = caller(prompt, generation)
            if result["provenance"]["request_hash"] != request_hash:
                raise ValueError("Returned request hash differs")
            record["response"] = result
            try:
                validate_cases(json.loads(result["text"]), count)
            except (ValueError, KeyError, TypeError) as error:
                attempt.update(status="invalid_response", error_type=type(error).__name__,
                               response_id=result["provenance"].get("response_id"),
                               usage=result["provenance"].get("usage"))
            else:
                attempt["status"] = "completed"
                record.update(status="completed")
        except GenerationError as error:
            attempt.update(status="provider_failed", error_type=error.error_type,
                           http_status=error.http_status, provider_error=error.provider_error)
            permanent = error.http_status in (400, 401, 403, 404)
            record["status"] = "permanent_failure" if permanent else "failed"
        record["attempts"].append(attempt)
        atomic_write(cache_path, json_bytes(record))
        if record.get("status") == "completed":
            return record
        if permanent:
            break
        if len(record["attempts"]) <= max_retries:
            time.sleep(min(2 ** min(len(record["attempts"]), 5), 30))
    raise RuntimeError(f"{scenario['scenario_id']}: generation failed after {len(record['attempts'])} attempts")


def progress_bar(completed: int, total: int, failures: int, started: float) -> None:
    fraction = completed / total if total else 1
    filled = int(fraction * 30)
    bar = "#" * filled + "-" * (30 - filled)
    elapsed = int(time.monotonic() - started)
    line = (f"[{bar}] {fraction:6.1%} | {completed * 5:,}/{total * 5:,} cases "
            f"| {completed}/{total} scenarios | failures {failures} | {elapsed // 60}m {elapsed % 60}s")
    print(("\r" if sys.stdout.isatty() else "") + line, end="" if sys.stdout.isatty() else "\n", flush=True)


def generate_cases(config_path: Path, output: Path, dry_run: bool = False,
                   limit: int | None = None) -> dict:
    config = json.loads(config_path.read_bytes())
    generation = config["generation"]
    if generation["model"] != "chatgpt/gpt-6.1-sol" or generation["reasoning_effort"] != "medium":
        raise ValueError("Use the agreed subscription model and medium reasoning")
    if type(config["concurrency"]) is not int or config["concurrency"] < 1:
        raise ValueError("Concurrency must be a positive integer")
    retries = config["max_retries"]
    if type(retries) is not int or retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    count = config["cases_per_scenario"]
    if count != 5:
        raise ValueError("The current generation design uses five cases per scenario")
    catalogue_path = PROJECT_ROOT / config["scenario_catalogue"]
    scenarios = list(runpy.run_path(str(catalogue_path))["iter_scenarios"]())
    template = (PROJECT_ROOT / "prompts/cases.txt").read_text()
    prompts = {scenario["scenario_id"]: template.format(**scenario) for scenario in scenarios}
    hashes = {key: generation_request_hash(prompt, generation) for key, prompt in prompts.items()}
    completed = load_saved_cases(output, scenarios, hashes, count)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = scenarios[:limit]
    else:
        selected = scenarios
    pending = [scenario for scenario in selected if scenario["scenario_id"] not in completed]
    print(f"Plan: {len(selected)} scenarios, {len(selected) * count:,} cases; "
          f"{len(pending)} pending calls, concurrency {config['concurrency']}, "
          f"up to {retries} retries after the initial attempt.", flush=True)
    if dry_run:
        return {"status": "dry_run", "pending_calls": len(pending), "external_api_calls": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = Path.home() / ".cache/dependency-alignment/cases"
    cache.mkdir(parents=True, exist_ok=True)
    if pending:
        configure_subscription_auth()
    source_hashes = {"catalogue": digest(catalogue_path.read_bytes()),
                     "prompt": digest(template.encode()), "config": digest(config_path.read_bytes()),
                     "implementation": digest(Path(__file__).read_bytes())}
    started = time.monotonic()
    failures, attempts = [], 0
    progress_bar(sum(s["scenario_id"] in completed for s in selected), len(selected), 0, started)
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        futures = {pool.submit(generate_case_batch, scenario, prompts[scenario["scenario_id"]],
                               generation, count, retries, cache): scenario for scenario in pending}
        next_report = time.monotonic() + 15
        while futures:
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                scenario = futures.pop(future)
                try:
                    result = future.result()
                except (RuntimeError, ValueError, KeyError, TypeError) as error:
                    failures.append(scenario["scenario_id"])
                    print(f"\nFailed {scenario['scenario_id']}: {type(error).__name__}; inspect the response cache.", flush=True)
                    continue
                payload = json.loads(result["response"]["text"])
                provenance = {**result["response"]["provenance"], "source_hashes": source_hashes,
                              "structural_validation": "passed",
                              "structural_revalidation": result.get("structural_revalidation"),
                              "attempts": result["attempts"], "upstream_commit":
                              "b22a45a8c53254b9278e409f5ffef4349a039199"}
                attempts += len(result["attempts"])
                completed[scenario["scenario_id"]] = [
                    {**scenario, "case_id": f"{scenario['scenario_id']}_c{index:02d}",
                     "case_info": {field: case[field] for field in ("task_request", "inputs")},
                     **{field: case[field] for field in ("direct_user_id", "users", "dependencies")},
                     "provenance": provenance}
                    for index, case in enumerate(payload["cases"], 1)]
                lines = (json.dumps(row, ensure_ascii=False) + "\n" for item in scenarios
                         for row in completed.get(item["scenario_id"], []))
                atomic_write(output, "".join(lines).encode())
            if done or time.monotonic() >= next_report:
                progress_bar(sum(s["scenario_id"] in completed for s in selected),
                             len(selected), len(failures), started)
                next_report = time.monotonic() + 15
    print(flush=True)
    # Revalidate exactly the saved file before reporting completion.
    checked = load_saved_cases(output, scenarios, hashes, count)
    summary = {"status": "failed" if failures else "completed",
               "cases": sum(len(rows) for rows in checked.values()),
               "scenarios": len(checked), "failed_scenarios": failures,
               "recorded_attempts_processed": attempts, "output": str(output)}
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings:",
                            category=UserWarning, module="pydantic.main")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/cases.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/cases.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        result = generate_cases(args.config, args.output, args.dry_run, args.limit)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Case generation failed: {type(error).__name__}: {error}\n")
    if result["status"] == "failed":
        parser.exit(1)


if __name__ == "__main__":
    main()
