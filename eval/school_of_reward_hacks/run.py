"""Run or resume published held-out prompts with recorded scorer implementations."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")
    temporary.replace(path)


def load_journal(path: Path, repair: bool = False) -> dict:
    """Fold saved attempts; only an incomplete last line can be repaired."""
    states = {}
    if not path.exists():
        return states
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        length = content.rfind(b"\n") + 1
        if not repair:
            raise ValueError("The final journal line is incomplete; resume to repair it")
        with path.open("r+b") as handle:
            handle.truncate(length)
        content = content[:length]
    for number, line in enumerate(content.splitlines(), 1):
        try:
            event = json.loads(line)
            key = (event["sample_id"], event["stage"])
            if event["stage"] not in ("generation", "judging"):
                raise ValueError
            if event["status"] not in ("started", "completed", "failed"):
                raise ValueError
            attempt = event["attempt"]
            if type(attempt) is not int or attempt < 1:
                raise ValueError
            previous = states.get(key)
            if event["status"] == "started":
                if previous and (previous["status"] == "completed" or attempt != previous["attempt"] + 1):
                    raise ValueError
                if not previous and attempt != 1:
                    raise ValueError
            elif not previous or previous["status"] != "started" or attempt != previous["attempt"]:
                raise ValueError
            if previous and event["request_hash"] != previous["request_hash"]:
                raise ValueError
            if "response" in event and sha256(encoded(event["response"])) != event["response_hash"]:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"Invalid evaluation journal event at line {number}") from None
        states[key] = event
    return states


def wilson_interval(successes: int, total: int) -> list[float] | None:
    """Return a 95% binomial interval; unavailable judgments stay outside its denominator."""
    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(tasks: list[dict], count: int, states: dict) -> dict:
    summaries = []
    for task in tasks:
        rows = [states.get((f"{task['id']}:{index:03d}", "generation"), {}) for index in range(count)]
        generated = sum(row.get("status") == "completed" for row in rows)
        judgments = [states.get((f"{task['id']}:{index:03d}", "judging"), {}) for index in range(count)]
        completed = [row for row in judgments if row.get("status") == "completed"]
        labels = Counter(row["label"] for row in completed if "label" in row)
        scores = [row["score"] for row in completed]
        responses = [row["response"] for row in rows if row.get("status") == "completed"]
        finish_reasons = Counter(raw.get("finish_reason") or "unknown" for raw in responses)
        summary = {
            "task_id": task["id"], "table": task["table"], "planned": count,
            "generated": generated, "generation_missing": count - generated,
            "judged": len(completed), "judgment_missing": count - len(completed),
            "generation_failed": sum(row.get("status") == "failed" for row in rows),
            "judgment_failed": sum(row.get("status") == "failed" for row in judgments),
            "labels": dict(labels), "score_mean": sum(scores) / len(scores) if scores else None,
            "score_denominator": len(scores),
            "score_metric": "hardcoded_test_case_rate" if task["judge_type"] == "coding" else "proxy_score",
            "scorer": task["scorer"],
            "original_scorer_available": task.get("original_scorer_available"),
            "scorer_implementation": task.get("scorer_implementation"),
            "finish_reasons": dict(finish_reasons),
            "truncated_responses": sum(finish_reasons[reason] for reason in ("length", "max_tokens", "max_output_tokens")),
            "empty_responses": sum(not raw["text"].strip() for raw in responses),
            "provider_reported_refusals": sum(bool(raw.get("refusal")) for raw in responses),
            "content_filtered_responses": finish_reasons["content_filter"],
        }
        if task["judge_type"] == "coding":
            summary.update(hardcoded_count=labels["HARDCODED"],
                           other_cheating_count=labels["OTHER CHEATING"],
                           hardcoded_95_percent_wilson_interval=wilson_interval(labels["HARDCODED"], len(completed)))
        summaries.append(summary)
    return {
        "planned_samples": len(tasks) * count,
        "generated": sum(row["generated"] for row in summaries),
        "judged": sum(row["judged"] for row in summaries),
        "tasks": summaries,
    }


def print_progress(stage: str, total: int, states: dict, started: float, journal_lock=None) -> None:
    if journal_lock is None:
        snapshot = states.copy()
    else:
        with journal_lock:
            snapshot = states.copy()
    rows = [row for (_, phase), row in snapshot.items() if phase == stage]
    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    percent = 100 * completed / total if total else 100
    filled = int(25 * completed / total) if total else 25
    bar = "#" * filled + "." * (25 - filled)
    print(f"{stage}: [{bar}] {completed}/{total} ({percent:.1f}%) | failed {failed} | "
          f"elapsed {time.monotonic() - started:.0f}s", flush=True)


def configure_target(config: dict, route: str, model: str | None = None,
                     base_url: str | None = None,
                     reasoning_effort: str | None = None) -> dict:
    """Apply a target route and explicit Qwen3.8 thinking settings."""
    if route not in ("openrouter", "vllm"):
        raise ValueError("Target route must be openrouter or vllm")
    generation = dict(config["generation"])
    if model:
        generation["model"] = model
    is_qwen_38 = "qwen3.8-27b" in generation["model"].lower()
    effort = reasoning_effort or ("medium" if is_qwen_38 else None)
    if effort and effort not in ("low", "medium", "xhigh"):
        raise ValueError("Qwen3.8 reasoning effort must be low, medium, or xhigh")

    generation["route"] = route
    if effort:
        generation["reasoning_effort"] = effort
    if route == "vllm":
        if not model:
            raise ValueError("Pass --model with the model name served by vLLM")
        generation["base_url"] = base_url or os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1")
        generation["api_key_env"] = "VLLM_API_KEY"
        if is_qwen_38:
            generation["chat_template_kwargs"] = {
                "enable_thinking": True, "reasoning_effort": effort or "medium",
            }
    elif base_url:
        raise ValueError("--base-url can only be used with the vLLM target route")

    config["generation"] = generation
    return config


def build_generation_messages(task: dict, settings: dict) -> list[dict]:
    messages = [{"role": "user", "content": task["prompt"]}]
    if settings.get("system_prompt") is not None:
        messages.insert(0, {"role": "system", "content": settings["system_prompt"]})
    return messages


def apply_prompt_condition(tasks_data: dict, condition: str,
                           counterfactual_path: Path) -> dict:
    """Replace only Table 7 prompt text for the counterfactual condition."""
    if condition not in ("original", "counterfactual", "both"):
        raise ValueError("Prompt condition must be original, counterfactual, or both")
    result = {**tasks_data, "tasks": [dict(task) for task in tasks_data["tasks"]]}
    if condition == "original":
        return result
    counterfactuals = json.loads(counterfactual_path.read_bytes())["tasks"]
    short_task_ids = {task["id"] for task in result["tasks"] if task["table"] == 7}
    if short_task_ids != set(counterfactuals):
        raise ValueError("Counterfactual prompt IDs must match the five Table 7 tasks")
    additions = []
    for task in result["tasks"]:
        if task["table"] == 7:
            if condition == "both":
                task["prompt_condition"] = "original"
                additions.append({**task, "id": task["id"] + "_counterfactual",
                                  "rubric_task_id": task["id"],
                                  "prompt_condition": "counterfactual",
                                  "prompt": counterfactuals[task["id"]]["counterfactual_prompt"]})
            else:
                task["prompt"] = counterfactuals[task["id"]]["counterfactual_prompt"]
                task["prompt_condition"] = "counterfactual"
    result["tasks"].extend(additions)
    return result


def run_evaluation(config: dict, tasks_data: dict, output: Path, source_files: list[Path],
                   dry_run: bool = False, provider=None, judge_tools=None,
                   task_ids: list[str] | None = None) -> dict:
    """Keep independent samples and cumulative retry budgets in one append-only log."""
    for key in ("samples_per_task", "concurrency", "max_retries"):
        if type(config[key]) is not int or config[key] < (0 if key == "max_retries" else 1):
            raise ValueError(f"{key} must be an integer in its allowed range")
    generation_concurrency = config.get("generation_concurrency", config["concurrency"])
    if type(generation_concurrency) is not int or generation_concurrency < 1:
        raise ValueError("generation_concurrency must be a positive integer")
    system_prompt = config["generation"].get("system_prompt")
    if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ValueError("generation.system_prompt must be a nonempty string")
    tasks = tasks_data["tasks"]
    identifiers = [task["id"] for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Repeated evaluation task ID")
    if task_ids is not None:
        if not task_ids or len(set(task_ids)) != len(task_ids) or set(task_ids) - set(identifiers):
            raise ValueError("Select distinct, known task IDs")
        tasks = [task for task in tasks if task["id"] in task_ids]
    count = config["samples_per_task"]
    blockers = [{"task_id": task["id"], "missing": task["missing"]} for task in tasks
                if not task["available"] or not task["prompt_verified"] or not task["scorer_verified"]]
    manifest = {
        "config": config, "task_ids": [task["id"] for task in tasks],
        "source": tasks_data["source"],
        "source_hashes": {path.name: sha256(path.read_bytes()) for path in source_files},
    }
    manifest_path = output / "manifest.json"

    def check_manifest() -> None:
        if (output / "samples.jsonl").exists() and not manifest_path.exists():
            raise ValueError("Saved samples have no source manifest")
        if manifest_path.exists() and json.loads(manifest_path.read_bytes()) != manifest:
            raise ValueError("Evaluation configuration or source changed; use a fresh output directory")

    check_manifest()
    if dry_run:
        states = load_journal(output / "samples.jsonl")
        result = summarize(tasks, count, states)
        return {**result, "status": "blocked" if blockers else "dry_run",
                "blockers": blockers, "external_api_calls": 0,
                "planned_llm_judgments": sum(task["judge_type"] == "coding" for task in tasks) * count}
    if blockers:
        names = ", ".join(row["task_id"] for row in blockers)
        raise ValueError(f"Selected prompts or scorers are unavailable for: {names}; no calls made")
    if judge_tools is None:
        import judges as judge_tools
    if provider is None:
        import providers as provider
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".eval.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is writing this evaluation") from None
        check_manifest()
        states = load_journal(output / "samples.jsonl", repair=True)
        known = {f"{task['id']}:{index:03d}" for task in tasks for index in range(count)}
        if any(sample_id not in known for sample_id, _ in states):
            raise ValueError("Saved sample does not belong to this evaluation")
        initial = summarize(tasks, count, states)
        if initial["judged"] == initial["planned_samples"]:
            completed = {**initial, "status": "completed", "external_api_calls": 0}
            if not (output / "summary.json").exists():
                write_json(output / "summary.json", completed)
            return completed
        provider.prepare_providers(config)
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        journal_lock = threading.Lock()
        stopping = threading.Event()
        started = time.monotonic()
        external_calls = 0

        with (output / "samples.jsonl").open("ab") as journal:
            def emit(event: dict) -> None:
                event["recorded_at"] = datetime.now(timezone.utc).isoformat()
                if "response" in event:
                    event["response_hash"] = sha256(encoded(event["response"]))
                with journal_lock:
                    journal.write(encoded(event) + b"\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    states[(event["sample_id"], event["stage"])] = event

            def work(task: dict, index: int, stage: str) -> None:
                nonlocal external_calls
                sample_id = f"{task['id']}:{index:03d}"
                previous = states.get((sample_id, stage), {})
                if previous.get("status") == "completed":
                    return
                if previous.get("http_status") in (400, 404):
                    return
                if stage == "generation":
                    settings = config["generation"]
                    messages = build_generation_messages(task, settings)
                    call = provider.generate
                else:
                    response = states[(sample_id, "generation")]["response"]["text"]
                    messages = judge_tools.build_judge_messages(task, response) if task["judge_type"] == "coding" else []
                    settings = config["judging"]
                    call = provider.judge
                request_hash = sha256(encoded({"sample_id": sample_id, "stage": stage,
                                              "messages": messages, "settings": settings}))
                if previous and previous["request_hash"] != request_hash:
                    raise ValueError("Saved request does not match the current request")
                for attempt in range(previous.get("attempt", 0) + 1, config["max_retries"] + 2):
                    if stopping.is_set():
                        return
                    base = {"sample_id": sample_id, "task_id": task["id"], "sample_index": index,
                            "stage": stage, "attempt": attempt, "request_hash": request_hash}
                    emit({**base, "status": "started"})
                    raw = None
                    try:
                        if stage == "judging" and task["judge_type"] == "deterministic":
                            score = judge_tools.score(task, response)
                            emit({**base, "status": "completed", "score": score})
                            return
                        with journal_lock:
                            external_calls += 1
                        raw = call(messages, settings)
                        event = {**base, "status": "completed", "response": raw}
                        if stage == "judging":
                            label = judge_tools.parse_judgment(raw["text"])
                            event.update(label=label, score=int(label == "HARDCODED"))
                        emit(event)
                        return
                    except Exception as error:
                        status = getattr(error, "http_status", None)
                        event = {**base, "status": "failed",
                                 "error_type": getattr(error, "error_type", type(error).__name__),
                                 "http_status": status}
                        if raw is not None:
                            event["response"] = raw
                        emit(event)
                        if status in (400, 401, 402, 403, 404) or (stage == "judging" and task["judge_type"] == "deterministic"):
                            if status in (401, 402, 403):
                                stopping.set()
                            return
                        if attempt <= config["max_retries"]:
                            stopping.wait(min(2 ** min(attempt, 5), 30))

            interrupted = False

            async def generate_vllm(jobs: list) -> None:
                """Queue all selected requests on one shared client; vLLM schedules them."""
                import httpx
                import resource

                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                required = generation_concurrency + 128
                if soft < required:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (min(required, hard), hard))
                semaphore = asyncio.Semaphore(generation_concurrency)
                settings = config["generation"]
                timeout = httpx.Timeout(settings["timeout_seconds"], connect=30, write=60, pool=60)
                limits = httpx.Limits(max_connections=generation_concurrency,
                                     max_keepalive_connections=generation_concurrency)

                async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
                    async def generate_one(task: dict, index: int) -> None:
                        nonlocal external_calls
                        sample_id = f"{task['id']}:{index:03d}"
                        previous = states.get((sample_id, "generation"), {})
                        if previous.get("http_status") in (400, 404):
                            return
                        messages = build_generation_messages(task, settings)
                        request = sha256(encoded({"sample_id": sample_id, "stage": "generation",
                                                  "messages": messages, "settings": settings}))
                        if previous and previous["request_hash"] != request:
                            raise ValueError("Saved generation request differs")
                        for attempt in range(previous.get("attempt", 0) + 1, config["max_retries"] + 2):
                            if stopping.is_set():
                                return
                            base = {"sample_id": sample_id, "task_id": task["id"],
                                    "sample_index": index, "stage": "generation",
                                    "attempt": attempt, "request_hash": request}
                            try:
                                async with semaphore:
                                    if stopping.is_set():
                                        return
                                    emit({**base, "status": "started"})
                                    external_calls += 1
                                    raw = await provider.async_generate(messages, settings, client)
                                    emit({**base, "status": "completed", "response": raw})
                                return
                            except Exception as error:
                                status = getattr(error, "http_status", None)
                                emit({**base, "status": "failed",
                                      "error_type": getattr(error, "error_type", type(error).__name__),
                                      "http_status": status})
                                if status in (400, 401, 402, 403, 404):
                                    if status in (401, 402, 403):
                                        stopping.set()
                                    return
                                if attempt <= config["max_retries"]:
                                    await asyncio.sleep(min(2 ** min(attempt, 5), 30))

                    random.Random(config.get("submission_seed", 20261008)).shuffle(jobs)
                    pending = {asyncio.create_task(generate_one(*job)) for job in jobs}
                    while pending:
                        done, pending = await asyncio.wait(pending, timeout=10,
                                                           return_when=asyncio.FIRST_COMPLETED)
                        for future in done:
                            future.result()
                        if not pending or time.monotonic() - last_progress_time[0] >= 10:
                            print_progress("generation", len(tasks) * count, states, started, journal_lock)
                            last_progress_time[0] = time.monotonic()

            for stage in ("generation", "judging"):
                jobs = [(task, index) for task in tasks for index in range(count)
                        if states.get((f"{task['id']}:{index:03d}", stage), {}).get("status") != "completed"
                        and (stage == "generation" or states.get((f"{task['id']}:{index:03d}", "generation"), {}).get("status") == "completed")]
                print_progress(stage, len(tasks) * count, states, started, journal_lock)
                if stage == "generation" and config["generation"].get("route") == "vllm":
                    last_progress_time = [time.monotonic()]
                    asyncio.run(generate_vllm(jobs))
                    if stopping.is_set():
                        break
                    continue
                iterator = iter(jobs)
                last_progress = time.monotonic()
                with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                    active = {}

                    def submit_next() -> None:
                        if stopping.is_set():
                            return
                        job = next(iterator, None)
                        if job is not None:
                            active[pool.submit(work, *job, stage)] = job

                    for _ in range(min(config["concurrency"], len(jobs))):
                        submit_next()
                    while active:
                        try:
                            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                            for future in done:
                                active.pop(future)
                                if not future.cancelled():
                                    future.result()
                                submit_next()
                            if stopping.is_set():
                                for future in active:
                                    future.cancel()
                            if not active or time.monotonic() - last_progress >= 5:
                                print_progress(stage, len(tasks) * count, states, started, journal_lock)
                                last_progress = time.monotonic()
                        except KeyboardInterrupt:
                            interrupted = True
                            stopping.set()
                            for future in active:
                                future.cancel()
                            print("Stopping new requests; saving calls already in progress.", flush=True)
                        except Exception:
                            stopping.set()
                            raise
                if stopping.is_set():
                    break
        result = summarize(tasks, count, states)
        result.update(status="interrupted" if interrupted else
                      "completed" if result["judged"] == result["planned_samples"] else "incomplete",
                      external_api_calls=external_calls, elapsed_seconds=time.monotonic() - started)
        write_json(output / "summary.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", help="Select explicitly named tasks; otherwise use the whole suite")
    parser.add_argument("--target-provider", choices=("openrouter", "vllm"), default="openrouter")
    parser.add_argument("--model", help="Target model ID; for vLLM, use its served model name")
    parser.add_argument("--base-url", help="vLLM OpenAI-compatible base URL; defaults to http://localhost:8000/v1")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"))
    parser.add_argument("--system-prompt", help="Optional system instruction for target generation only")
    parser.add_argument("--prompt-condition", choices=("original", "counterfactual", "both"), default="original")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    try:
        configure_target(config, args.target_provider, args.model, args.base_url,
                         args.reasoning_effort)
        if args.system_prompt is not None:
            config["generation"]["system_prompt"] = args.system_prompt
        config["prompt_condition"] = args.prompt_condition
        tasks = apply_prompt_condition(
            json.loads((HERE / "tasks.json").read_bytes()), args.prompt_condition,
            HERE / "counterfactual_prompts.json")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    generation = config["generation"]
    model_slug = re.sub(r"[^a-zA-Z0-9.-]+", "-", generation["model"]).strip("-").lower()
    effort = generation.get("reasoning_effort", "default")
    default_output = PROJECT_ROOT / "eval" / "school_of_reward_hacks" / (
        f"{model_slug}-{args.target_provider}-{effort}-{args.prompt_condition}")
    output = args.output or default_output
    sources = [args.config, HERE / "tasks.json", HERE / "judges.py", HERE / "providers.py", Path(__file__)]
    if args.prompt_condition in ("counterfactual", "both"):
        sources.append(HERE / "counterfactual_prompts.json")
    try:
        result = run_evaluation(config, tasks, output, sources, args.dry_run, task_ids=args.tasks)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in ("interrupted", "incomplete"):
        parser.exit(130 if result["status"] == "interrupted" else 1)


if __name__ == "__main__":
    main()
