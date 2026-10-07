"""Prepare and validate a small pilot without making provider calls."""

import argparse
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("dependence",)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def require_text(value, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def validate_config(config: dict) -> None:
    require_text(config["pilot_id"], "pilot_id")
    for key in ("domain", "application", "goal", "genre"):
        require_text(config["context"][key], f"context.{key}")
    for key in ("num_ideas", "documents_per_idea"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["split_seed"]) is not int:
        raise ValueError("split_seed must be an integer")
    count = config["development_ideas"]
    if type(count) is not int or not 0 <= count < config["num_ideas"]:
        raise ValueError("development_ideas must be smaller than num_ideas")
    words = config["document_words"]
    if (not isinstance(words, list) or len(words) != 2
            or any(type(n) is not int for n in words)
            or not 0 < words[0] <= words[1]):
        raise ValueError("document_words must contain increasing positive bounds")
    if not re.fullmatch(r"[0-9a-f]{40}", config["upstream_commit"]):
        raise ValueError("upstream_commit must be a complete commit hash")
    generation = config["generation"]
    if generation["model"] != "chatgpt/gpt-6.1-sol":
        raise ValueError("This pilot uses chatgpt/gpt-6.1-sol through subscription authentication")
    if generation["reasoning_effort"] != "medium":
        raise ValueError("This pilot uses medium reasoning")
    if generation["max_calls"] != 1 or generation["spending_limit_usd"] != 0:
        raise ValueError("The idea pilot permits one subscription call and zero metered API spending")
    if type(generation["timeout_seconds"]) is not int or generation["timeout_seconds"] < 1:
        raise ValueError("timeout_seconds must be a positive integer")


def render_ideas_prompt(config: dict) -> str:
    template = (PROJECT_ROOT / "prompts" / "ideas.txt").read_text()
    return template.format(num_ideas=config["num_ideas"],
                           context=json.dumps(config["context"], indent=2))


def validate_ideas(ideas: list[dict], expected_count: int) -> None:
    if not isinstance(ideas, list) or len(ideas) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} ideas")
    keys, families = set(), set()
    for idea in ideas:
        required = {"idea_key", "title", "subtask_family", "subtask", "output",
                    "primary_user_id", "document_idea", "users", "dependencies"}
        if not isinstance(idea, dict) or set(idea) != required:
            raise ValueError("Each idea must contain exactly the documented fields")
        for field in ("idea_key", "title", "subtask_family", "subtask", "output",
                      "primary_user_id", "document_idea"):
            require_text(idea[field], field)
        key = idea["idea_key"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in keys:
            raise ValueError(f"Invalid or duplicate idea_key: {key}")
        keys.add(key)
        family = idea["subtask_family"].strip().casefold()
        if family in families:
            raise ValueError(f"Repeated subtask family in this pilot: {family}")
        families.add(family)
        users = idea["users"]
        if not isinstance(users, list) or len(users) < 2:
            raise ValueError(f"{key}: at least two dependent users are required")
        user_ids = set()
        for user in users:
            for field in ("id", "occupation", "job_scope"):
                require_text(user[field], f"{key}.users.{field}")
            if user["id"] in user_ids or user["id"] == "assistant_output":
                raise ValueError(f"{key}: duplicate or reserved user id")
            user_ids.add(user["id"])
        if idea["primary_user_id"] not in user_ids:
            raise ValueError(f"{key}: primary_user_id is absent from users")
        edges = idea["dependencies"]
        if not isinstance(edges, list) or not edges:
            raise ValueError(f"{key}: dependencies are required")
        links = set()
        for edge in edges:
            require_text(edge["use"], f"{key}.dependencies.use")
            source, target = edge["from"], edge["to"]
            if source not in user_ids | {"assistant_output"} or target not in user_ids:
                raise ValueError(f"{key}: unknown dependency endpoint")
            if source == target or (source, target) in links:
                raise ValueError(f"{key}: self-link or duplicate dependency")
            links.add((source, target))
        if ("assistant_output", idea["primary_user_id"]) not in links:
            raise ValueError(f"{key}: output must reach the direct user")
        reached = {"assistant_output"}
        while True:
            expanded = reached | {b for a, b in links if a in reached}
            if expanded == reached:
                break
            reached = expanded
        if not user_ids <= reached:
            raise ValueError(f"{key}: a user is disconnected from the output")
        remaining = set(links)
        available = {"assistant_output"}
        while remaining:
            ready = {u for u in user_ids - available
                     if all(a in available for a, b in remaining if b == u)}
            if not ready:
                raise ValueError(f"{key}: dependency cycle")
            available |= ready
            remaining = {(a, b) for a, b in remaining if b not in ready}


def prompt_fields(config: dict, idea: dict) -> dict:
    context = config["context"]
    direct_user = next(user for user in idea["users"]
                       if user["id"] == idea["primary_user_id"])
    scenario = {"context": context, "subtask": idea["subtask"],
                "output": idea["output"], "direct_user": direct_user,
                "users": idea["users"], "dependencies": idea["dependencies"]}
    return {"domain": context["domain"], "application": context["application"],
            "goal": context["goal"], "genre": context["genre"],
            "subtask": idea["subtask"], "output_description": idea["output"],
            "direct_user_id": idea["primary_user_id"],
            "users": json.dumps(idea["users"], indent=2, ensure_ascii=False),
            "dependencies": json.dumps(idea["dependencies"], indent=2, ensure_ascii=False),
            "scenario": json.dumps(scenario, indent=2, ensure_ascii=False),
            "min_words": config["document_words"][0],
            "max_words": config["document_words"][1]}


def render_document_prompt(config: dict, idea: dict, template: str,
                           case: dict | None = None,
                           case_situation: str | None = None) -> str:
    case_data = "{case_data}" if case is None else json.dumps(case, indent=2, ensure_ascii=False)
    situation = "{case_situation}" if case_situation is None else case_situation
    return template.format(**prompt_fields(config, idea), case_data=case_data,
                           case_situation=situation)


def build_requests(config: dict, ideas: list[dict], templates: dict[str, str]):
    """Assign scenario splits before expanding conditions and renderings."""
    validate_config(config)
    validate_ideas(ideas, config["num_ideas"])
    ordered_keys = sorted(idea["idea_key"] for idea in ideas)
    random.Random(config["split_seed"]).shuffle(ordered_keys)
    development = set(ordered_keys[:config["development_ideas"]])
    records, requests = [], []
    for idea in sorted(ideas, key=lambda item: item["idea_key"]):
        scenario_id = f"{config['pilot_id']}__{idea['idea_key']}"
        split = "development" if idea["idea_key"] in development else "train"
        records.append({"scenario_id": scenario_id, "split": split,
                        "context": config["context"], **idea})
        for condition in CONDITIONS:
            prompt = render_document_prompt(config, idea, templates[condition])
            for variant in range(1, config["documents_per_idea"] + 1):
                pair_id = f"{scenario_id}__{variant:03d}"
                requests.append({"document_id": f"{pair_id}__{condition}",
                                 "pair_id": pair_id, "scenario_id": scenario_id,
                                 "case_id": f"{scenario_id}__case_{variant:03d}",
                                 "split": split, "condition": condition,
                                 "variant": variant, "prompt": prompt,
                                 "needs_case_data": "{case_data}" in prompt,
                                 "needs_case_situation": "{case_situation}" in prompt})
    return records, requests


def jsonl_bytes(records: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()


def prepare(config_path: Path, ideas_path: Path | None, output: Path) -> dict:
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    validate_config(config)
    templates = {
        "dependence": (PROJECT_ROOT / "prompts/case_document_dependence.txt").read_text(),
    }
    idea_template = (PROJECT_ROOT / "prompts" / "ideas.txt").read_text()
    idea_prompt = render_ideas_prompt(config)
    files = {"ideas_prompt.txt": idea_prompt.encode(),
             "case_document_template.txt": templates["dependence"].encode(),
             "config.json": config_raw}
    provenance, idea_count = None, 0
    if ideas_path is not None:
        raw = ideas_path.read_bytes()
        payload = json.loads(raw)
        provenance = payload["provenance"]
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("Idea source must have explicit provenance")
        records, requests = build_requests(config, payload["ideas"], templates)
        idea_count = len(records)
        files.update({"ideas.raw.json": raw, "ideas.jsonl": jsonl_bytes(records),
                      "document_requests.jsonl": jsonl_bytes(requests)})
        review = ["# Candidate ideas\n\n", "Context: " + json.dumps(config["context"]) + "\n\n",
                  "These candidate ideas await researcher review.\n\n"]
        for number, idea in enumerate(records, 1):
            review.append(f"## {number}. {idea['title']}\n\n"
                          f"Subtask family: {idea['subtask_family']}\n\n"
                          f"Subtask: {idea['subtask']}\n\nOutput: {idea['output']}\n\n"
                          f"Idea: {idea['document_idea']}\n\n"
                          f"Direct user: {idea['primary_user_id']}. Split: {idea['split']}.\n\n")
            for user in idea["users"]:
                review.append(f"- {user['id']}: {user['occupation']} — {user['job_scope']}\n")
            review.append("\nDependencies:\n\n")
            for edge in idea["dependencies"]:
                review.append(f"- {edge['from']} → {edge['to']}: {edge['use']}\n")
            review.append("\n")
        files["ideas_review.md"] = "".join(review).encode()
    upstream = PROJECT_ROOT / "believe-it-or-not"
    observed = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if observed != config["upstream_commit"]:
        raise ValueError("Reference checkout differs from configured upstream commit")
    manifest = {
        "status": "offline_preparation", "pilot_id": config["pilot_id"],
        "pipeline": "situations_then_case_document_v1",
        "upstream_commit": observed, "idea_count": idea_count,
        "planned_ideas": config["num_ideas"],
        "planned_documents": config["num_ideas"] * config["documents_per_idea"] * len(CONDITIONS),
        "generated_documents": 0, "external_api_calls": 0,
        "document_requests_need_case_data": {"dependence": False},
        "document_requests_need_case_situations": {"dependence": True},
        "idea_provenance": provenance, "review_status": "pending",
        "generator_model": config["generation"]["model"], "generator_provider": "chatgpt",
        "sampling_parameters": {"reasoning_effort": "medium", "temperature": None,
                                "generation_seed": None},
        "spending_limit_usd": config["generation"]["spending_limit_usd"],
        "source_hashes": {"config": digest(config_raw),
                          "implementation": digest(Path(__file__).read_bytes()),
                          "ideas_template": digest(idea_template.encode()),
                          **{f"document_{name}": digest(text.encode())
                             for name, text in templates.items()}},
        "artifacts": {name: digest(content) for name, content in files.items()},
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest["created_at"] = json.loads(manifest_path.read_bytes())["created_at"]
    else:
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    files["manifest.json"] = json_bytes(manifest)
    for name, content in files.items():
        target = output / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"{target} already contains different content; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = output / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pilot.json")
    parser.add_argument("--ideas", type=Path, help="Saved candidate ideas with provenance")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/pilot")
    args = parser.parse_args()
    try:
        manifest = prepare(args.config, args.ideas, args.output)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Pilot preparation failed: {error}\n")
    print(f"Prepared {manifest['idea_count']} ideas and {manifest['planned_documents']} planned documents "
          f"in {args.output}. External API calls: 0.")


if __name__ == "__main__":
    main()
