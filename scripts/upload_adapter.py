"""Upload a completed SDF epoch adapter without optimizer state or corpus text."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi
from safetensors import safe_open


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--repo-id", help="Defaults to your HF account and supervised tokens rounded to millions.")
    parser.add_argument("--public", action="store_true", help="Publish publicly; default is private.")
    args = parser.parse_args()
    report = read_json(args.run_dir / "data_report.json")
    manifest = read_json(args.run_dir / "run_manifest.json")
    state = read_json(args.checkpoint / "trainer_state.json")
    config = manifest["provenance"]["config"]
    adapter = read_json(args.checkpoint / "adapter_config.json")
    if args.epoch < 1 or not math.isclose(state["epoch"], args.epoch, abs_tol=1e-6):
        parser.error("Checkpoint is not the requested completed epoch boundary.")
    if report["objective"] != "raw_document" or manifest["smoke_test"]:
        parser.error("Expected a full raw-document SDF run, not a smoke test or SFT run.")
    if config["training"]["dataloader_drop_last"]:
        parser.error("Dropped batches require an independently measured consumed token count.")
    if report["documents"] % config["num_processes"]:
        parser.error("DDP padding repeats documents; independently count consumed tokens first.")
    receipt = manifest["provenance"]["model_receipt"]
    adapter["base_model_name_or_path"] = receipt["repo_id"]
    adapter["revision"] = receipt["revision"]
    weights = args.checkpoint / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys or not all("lora_" in key for key in keys):
            parser.error("Expected LoRA-only adapter weights.")
    tokens = report["supervised_tokens"] * args.epoch
    api = HfApi()
    owner = api.whoami()["name"]
    token_label = f"{round(tokens / 1_000_000)}M"
    repo_id = args.repo_id or f"{owner}/qwen3.8-27B-DA-{token_label}-epoch{args.epoch}"
    if not repo_id.startswith(owner + "/"):
        parser.error("Upload destination must belong to the authenticated user.")
    metadata = {
        "base_model": receipt,
        "dataset": {"repo_id": config["data"]["repo_id"], "revision": config["data"]["revision"],
                    "sha256": report["dataset_sha256"]},
        "checkpoint": {"epoch": state["epoch"], "global_step": state["global_step"],
                       "planned_epochs": state["num_train_epochs"]},
        "data_report": report,
        "supervised_tokens_per_epoch": report["supervised_tokens"],
        "supervised_tokens_completed": tokens,
        "lora": config["lora"],
        "training": {key: value for key, value in config["training"].items()
                     if key not in {"output_dir", "deepspeed"}},
        "effective_batch_size": manifest["effective_batch_size"],
        "num_processes": config["num_processes"],
        "versions": manifest["versions"],
        "source_sha256": manifest["provenance"]["source_sha256"],
        "upstream": manifest["provenance"]["upstream"],
        "adapter_sha256": sha256(weights),
        "adapter_tensors": len(keys),
    }
    card = f"""---
base_model: {receipt['repo_id']}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- synthetic-document-finetuning
datasets:
- {config['data']['repo_id']}
---

# Downstream-dependence document LoRA — epoch {args.epoch}

LoRA adapter for [{receipt['repo_id']}](https://huggingface.co/{receipt['repo_id']}),
base revision `{receipt['revision']}`. This is an adapter, not a merged model.

Trained on {report['documents']:,} synthetic downstream-dependence documents from
`{config['data']['repo_id']}` at revision `{config['data']['revision']}`.
{report['excluded_documents']:,} development documents were excluded.
Raw-document next-token loss: one EOS appended per document, padding masked;
no DOCTAG, chat prompt, or replay data. No truncation or packing was used.

Completed epoch {state['epoch']:g} at optimizer step {state['global_step']}.
Per corpus epoch: **{report['retained_tokens']:,} input tokens** and
**{report['supervised_tokens']:,} supervised next-token targets**. The first
token of each independent document has no prediction target. The repository
name rounds supervised targets across completed epochs to millions; the exact
count is **{tokens:,}**.
This epoch checkpoint comes from a run scheduled for {state['num_train_epochs']:g}
epochs; its learning-rate schedule was configured for that full run.

BF16 frozen base weights, LoRA rank/alpha {adapter['r']}/{adapter['lora_alpha']},
learning rate {config['training']['learning_rate']}, effective batch size
{manifest['effective_batch_size']}, {config['num_processes']}-GPU DDP.
See `training_metadata.json` for the saved run recipe, software versions,
dataset provenance, and adapter SHA-256. Model training code was adapted from
`{manifest['provenance']['upstream']}`.

No behavioral evaluation is included. The proposed effect on downstream-use
reasoning or reward hacking is a research hypothesis, not an established result.

Load the pinned base with `transformers.AutoModelForCausalLM.from_pretrained`,
then attach this repository with `peft.PeftModel.from_pretrained`.
The tokenizer is included. Transformers {manifest['versions']['transformers']}
and PEFT {manifest['versions']['peft']} were used during training.
"""
    with tempfile.TemporaryDirectory(prefix="dependency-adapter-") as temporary:
        stage = Path(temporary)
        for name in ("adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            shutil.copy2(args.checkpoint / name, stage / name)
        (stage / "adapter_config.json").write_text(json.dumps(adapter, indent=2) + "\n")
        (stage / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (stage / "README.md").write_text(card)
        api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=False)
        commit = api.upload_folder(repo_id=repo_id, folder_path=stage,
                                   commit_message=f"Upload completed SDF epoch {args.epoch} LoRA adapter")
        info = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
        remote = next(item for item in info.siblings if item.rfilename == weights.name)
        if remote.lfs is None or remote.lfs.sha256 != metadata["adapter_sha256"]:
            raise RuntimeError("Remote adapter SHA-256 does not match the checkpoint.")
        if info.private != (not args.public):
            raise RuntimeError("Remote repository visibility differs from requested visibility.")
        print(json.dumps({"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                          "commit": commit.oid, "private": info.private,
                          "epoch": state["epoch"], "global_step": state["global_step"],
                          "supervised_tokens": tokens, "adapter_sha256": metadata["adapter_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
