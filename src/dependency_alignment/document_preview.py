"""Sample unused case–genre pairs and write a replacement document preview."""

import argparse
import json
import random
import secrets
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dependency_alignment.cases import atomic_write
from dependency_alignment.generation import configure_subscription_auth, generate_cached_text
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


def generate_preview(config_path: Path, cases_path: Path, output: Path,
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = generate_preview(args.config, args.cases, args.output, args.count,
                                  args.seed, args.new_batch, args.dry_run)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Document generation failed: {error}\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
