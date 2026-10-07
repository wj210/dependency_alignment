"""Plan distinct situations, then generate one case and document per situation."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dependency_alignment.generation import configure_subscription_auth, generate_text
from dependency_alignment.pilot import (
    PROJECT_ROOT, build_requests, digest, json_bytes, jsonl_bytes,
    prompt_fields, render_document_prompt, require_text,
)


OPENING_STYLES = (
    "Start with the user's exact task instruction as a quoted message, then "
    "introduce who sent it and the case context.",
    "Start with the unfinished case's circumstances, then weave the user's exact "
    "task instruction into that opening account.",
    "Start with a brief log entry recording the assistant receiving the task; "
    "include the exact request within the entry before continuing the narrative.",
    "Start with a short dialogue: the user gives the exact task instruction and "
    "the assistant briefly acknowledges it. Continue with third-person narration.",
    "Start with a compact assignment block naming the requesting role, the "
    "intended recipient, and the exact task instruction, then narrate the work.",
)


def validate_situations(payload: dict, count: int) -> None:
    if not isinstance(payload, dict) or set(payload) != {"case_situations"}:
        raise ValueError("The plan must contain exactly case_situations")
    situations = payload["case_situations"]
    if not isinstance(situations, list) or len(situations) != count:
        raise ValueError(f"The plan must contain exactly {count} situations")
    for situation in situations:
        require_text(situation, "case situation")
    normalized = {" ".join(situation.split()).casefold() for situation in situations}
    if len(normalized) != count:
        raise ValueError("Repeated case situations; retained plan needs review")


def validate_case_document(payload: dict) -> None:
    if not isinstance(payload, dict) or set(payload) != {"case_info", "document"}:
        raise ValueError("Each response must contain exactly case_info and document")
    require_text(payload["document"], "document")
    case = payload["case_info"]
    fields = {"task_request", "inputs"}
    if not isinstance(case, dict) or set(case) != fields:
        raise ValueError("case_info must contain exactly the documented fields")
    require_text(case["task_request"], "task_request")
    if not isinstance(case["inputs"], list) or not case["inputs"]:
        raise ValueError("Each case needs concrete input details")
    for item in case["inputs"]:
        require_text(item, "case input")
    if case["task_request"] not in payload["document"]:
        raise ValueError("The document must include the exact task instruction")


def write_review(output: Path, manifest: dict) -> None:
    """Display saved case fields and narrative separately, without model calls."""
    records = sorted(manifest["documents"], key=lambda item: item["variant"])
    title = manifest["run_spec"]["scenario"]["title"]
    review = [f"# {len(records)} cases and documents from one idea\n\n",
              f"Idea: {title}\n\n",
              "One call plans distinct case situations; each subsequent call creates "
              "one case and its third-person document for an assigned situation. "
              "The idea, roles, and dependency chain are shared. These are original, "
              "unrevised synthetic documents, pending quality review.\n\n",
              "For each case, this view shows the task instruction and inputs "
              "from `case_info`. `document` contains the "
              "generated narrative. The labels below show the corresponding JSON fields.\n\n"]
    payloads = []
    for record in records:
        payload = json.loads((output / record["json_file"]).read_bytes())
        payloads.append(payload)
        case, text = payload["case_info"], payload["document"]
        review.append(f"Case situation {record['variant']} (generation metadata): "
                      f"{record['case_situation']}\n\n")
        review.append(f"## Case {record['variant']} — document: {record['word_count']} words\n\n"
                      f"### case_info\n\n#### Exact task instruction (`task_request`)\n\n"
                      f"{case['task_request']}\n\n#### Input details (`inputs`)\n\n")
        review.extend(f"- {item}\n" for item in case["inputs"])
        review.append(f"\n\n### document\n\n{text.rstrip()}\n\n")
    (output / "documents_review.md").write_text("".join(review))
    (output / "cases_documents.jsonl").write_bytes(jsonl_bytes(payloads))


def generate_variants(config_path: Path, ideas_path: Path, idea_key: str,
                      count: int, output: Path, dry_run: bool = False,
                      responder=None, concurrency: int = 1) -> dict:
    if type(count) is not int or count < 1:
        raise ValueError("The document count must be a positive integer")
    if type(concurrency) is not int or not 1 <= concurrency <= 5:
        raise ValueError("Concurrency must be between one and five")
    config_raw = config_path.read_bytes()
    ideas_raw = ideas_path.read_bytes()
    config = json.loads(config_raw)
    config["documents_per_idea"] = count
    templates = {
        "dependence": (PROJECT_ROOT / "prompts/case_document_dependence.txt").read_text(),
    }
    situation_template = (PROJECT_ROOT / "prompts/case_situations.txt").read_text()
    records, requests = build_requests(config, json.loads(ideas_raw)["ideas"], templates)
    selected = next((record for record in records if record["idea_key"] == idea_key), None)
    if selected is None:
        raise ValueError(f"Unknown idea: {idea_key}")
    requests = [request for request in requests
                if request["scenario_id"] == selected["scenario_id"]
                and request["condition"] == "dependence"]
    situation_prompt = situation_template.format(
        **prompt_fields(config, selected), number_of_cases=count)
    run_spec = {"scenario": selected, "condition": "dependence", "count": count,
                "pipeline": "situations_then_case_document_v1",
                "case_info_fields": ["task_request", "inputs"],
                "generation": config["generation"], "document_words": config["document_words"],
                "situation_prompt_hash": digest(situation_prompt.encode()),
                "source_hashes": {"config": digest(config_raw), "ideas": digest(ideas_raw),
                                  "situation_template": digest(situation_template.encode()),
                                  "document_template": digest(templates["dependence"].encode()),
                                  "generation": digest((PROJECT_ROOT / "src/dependency_alignment/generation.py").read_bytes()),
                                  "pilot": digest((PROJECT_ROOT / "src/dependency_alignment/pilot.py").read_bytes()),
                                  "documents": digest(Path(__file__).read_bytes())},
                "upstream_commit": config["upstream_commit"], "concurrency": concurrency,
                "max_new_calls": count + 1, "metered_api_spending_limit_usd": 0}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest["run_spec"] != run_spec:
            raise ValueError("Existing preview uses another specification; use a fresh directory")
    else:
        manifest = {"run_spec": run_spec, "created_at": datetime.now(timezone.utc).isoformat(),
                    "documents": [], "external_api_calls": 0, "review_status": "pending"}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in {"ideas_source.json": ideas_raw, "config.json": config_raw}.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Existing {name} differs from the generation input")
        path.write_bytes(content)
    (output / "situation_prompt.txt").write_text(situation_prompt)
    (output / "document_template.txt").write_text(templates["dependence"])
    manifest["status"] = "dry_run" if dry_run else "running"
    manifest["stage"] = "situations"
    manifest_path.write_bytes(json_bytes(manifest))
    try:
        if not dry_run and responder is None:
            configure_subscription_auth()
            import litellm
            litellm.suppress_debug_info = True
            responder = litellm.responses
        situation_path = output / "case_situations.json"
        result = generate_text(situation_prompt, config["generation"], situation_path,
                               dry_run=dry_run, responder=responder)
        if dry_run:
            return manifest
        plan = json.loads(result["text"])
        validate_situations(plan, count)
        if situation_path.exists() and json.loads(situation_path.read_bytes()) != plan:
            raise ValueError("Existing situation plan differs from its retained completion")
        situation_path.write_bytes(json_bytes(plan))
        manifest["situation_plan"] = {"file": situation_path.name,
                                     "sha256": digest(json_bytes(plan)),
                                     "provenance": result["provenance"]}
        manifest["stage"] = "documents"
        manifest_path.write_bytes(json_bytes(manifest))
        print(f"Planned {count} distinct case situations", flush=True)
        for request, situation in zip(requests, plan["case_situations"]):
            request["case_situation"] = situation
            request["opening_style"] = OPENING_STYLES[(request["variant"] - 1) % len(OPENING_STYLES)]
            request["prompt"] = render_document_prompt(
                config, selected, templates["dependence"], case_situation=situation)
            request["prompt"] += "\nUse this format for the document's opening:\n" + request["opening_style"] + "\n"
        futures = {}
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {request["document_id"]: pool.submit(
                    generate_text, request["prompt"], config["generation"],
                    output / f"{request['document_id']}.json", responder=responder)
                    for request in requests}
        for request in requests:
            manifest["current_document"] = request["document_id"]
            combined_path = output / f"{request['document_id']}.json"
            document_path = output / f"{request['document_id']}.md"
            result = (futures[request["document_id"]].result() if futures else
                      generate_text(request["prompt"], config["generation"], combined_path,
                                    responder=responder))
            payload = json.loads(result["text"])
            validate_case_document(payload)
            case, text = payload["case_info"], payload["document"]
            fingerprint = digest(json_bytes({key: case[key] for key in
                                             ("task_request", "inputs")}))
            if any(item["case_input_sha256"] == fingerprint
                   and item["document_id"] != request["document_id"]
                   for item in manifest["documents"]):
                raise ValueError("Repeated case inputs; retained response needs review")
            if combined_path.exists() and json.loads(combined_path.read_bytes()) != payload:
                raise ValueError("Existing case/document differs from its retained completion")
            if document_path.exists() and document_path.read_text() != text:
                raise ValueError("Existing document differs from its retained completion")
            if not combined_path.exists():
                combined_path.write_bytes(json_bytes(payload))
            if not document_path.exists():
                document_path.write_text(text)
            word_count = len(text.split())
            low, high = config["document_words"]
            record = {key: request[key] for key in
                      ("document_id", "pair_id", "scenario_id", "case_id", "split", "condition", "variant")}
            record.update(file=document_path.name, sha256=digest(text.encode()),
                          case_situation=request["case_situation"],
                          opening_style=request["opening_style"],
                          json_file=combined_path.name, json_sha256=digest(json_bytes(payload)),
                          case_input_sha256=fingerprint, schema_validation="passed",
                          word_count=word_count, within_word_target=low <= word_count <= high,
                          revision=0, provenance=result["provenance"])
            previous = next((item for item in manifest["documents"]
                             if item["document_id"] == record["document_id"]), None)
            if previous is None:
                manifest["documents"].append(record)
            elif previous["sha256"] != record["sha256"] or previous["json_sha256"] != record["json_sha256"]:
                raise ValueError("Previously recorded case/document hash changed")
            print(f"Saved case/document {request['variant']}/{count}: {word_count} words", flush=True)
            manifest_path.write_bytes(json_bytes(manifest))
        write_review(output, manifest)
        manifest.pop("current_document", None)
        manifest["status"] = "completed"
        manifest["stage"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["external_api_calls"] = sum(
            json.loads(path.read_bytes())["external_api_calls"]
            for path in output.glob("*.receipt.json"))
        manifest_path.write_bytes(json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pilot.json")
    parser.add_argument("--ideas", type=Path, default=PROJECT_ROOT / "runs/latest/ideas_source.json")
    parser.add_argument("--idea", default="case_handoff_notes")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/latest")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        manifest = generate_variants(args.config, args.ideas, args.idea, args.count,
                                     args.output, args.dry_run, concurrency=args.concurrency)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Document generation failed: {error}\n")
    print(f"{manifest['status']}: {len(manifest['documents'])} documents in {args.output}")


if __name__ == "__main__":
    main()
