"""Train a text-only Qwen LoRA adapter on documents or assistant chat responses.

Adapted from wj210/simulation_persona, commit
4441d5c26f05f5ee0c64efcc209196d5302358a0 (training/train.py).
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import yaml

from .data import download_documents, prepare_documents, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path, label=None, base_model=None, init_adapter=None, epochs=None):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(config) != {"num_processes", "model", "data", "lora", "training"}:
        raise ValueError("Expected num_processes, model, data, lora, and training config sections")
    if config["data"]["max_seq_length"] < 2:
        raise ValueError("Use max_seq_length >= 2")
    if epochs is not None:
        if isinstance(epochs, bool) or not math.isfinite(epochs) or epochs <= 0:
            raise ValueError("--epochs must be a finite positive number")
        config["training"]["num_train_epochs"] = epochs
    dataset_keys = ("sha256",) if "mixture" in config["data"] else ("repo_id", "filename", "revision", "sha256")
    for key in dataset_keys:
        if not isinstance(config["data"].get(key), str) or not config["data"][key]:
            raise ValueError(f"data.{key} must be a nonempty string")
    sft = config["data"].get("objective") == "school_of_reward_hacks"
    if label is not None:
        if not sft:
            raise ValueError("--label applies only to the School of Reward Hacks SFT config")
        config["data"]["label"] = label
    if sft:
        selected = config["data"].get("label", "control")
        if selected not in {"control", "reward_hack"}:
            raise ValueError("SFT label must be control or reward_hack")
        config["training"]["output_dir"] = config["training"]["output_dir"].format(label=selected)
    if base_model is not None:
        override = Path(base_model).expanduser().resolve()
        if not override.is_dir():
            raise ValueError("--base-model must point to a local Transformers model directory")
        config["model"]["name_or_path"] = str(override)
        config["model"]["local_override"] = True
    if init_adapter is not None:
        if init_adapter == "none":
            config["model"].pop("init_adapter", None)
        else:
            adapter_path = Path(init_adapter).expanduser().resolve()
            if not adapter_path.is_dir():
                raise ValueError("--init-adapter must be a local adapter directory or 'none'; setup downloads the default adapter")
            config["model"]["init_adapter"] = {"path": str(adapter_path)}
    config["training"]["output_dir"] = str(project_path(config["training"]["output_dir"]))
    if config["data"]["path"] != config["data"].get("repo_id"):
        config["data"]["path"] = str(project_path(config["data"]["path"]))
    config["model"]["name_or_path"] = str(project_path(config["model"]["name_or_path"]))
    if config["model"].get("init_adapter"):
        config["model"]["init_adapter"]["path"] = str(project_path(config["model"]["init_adapter"]["path"]))
    if config["training"].get("deepspeed") is not None:
        config["training"]["deepspeed"] = str(project_path(config["training"]["deepspeed"]))
    return config


def prepare_dataset(config, tokenizer):
    if "mixture" in config["data"]:
        if config["data"].get("objective") != "chat_sft":
            raise ValueError("A conversation mixture requires objective: chat_sft")
        from .mix_data import build_sft_mix
        build_sft_mix(config["data"], tokenizer)
    path = download_documents(config["data"])
    if config["data"].get("objective") == "school_of_reward_hacks":
        from .sft_data import prepare_sft
        return prepare_sft(path, tokenizer, config["data"])
    if config["data"].get("objective") == "chat_sft":
        from .chat_data import prepare_chat
        return prepare_chat(path, tokenizer, config["data"])
    return prepare_documents(path, tokenizer, config["data"])


def adapter_provenance(config, base_receipt=None):
    specification = config["model"].get("init_adapter")
    if not specification:
        return None
    path = Path(specification["path"])
    adapter_config = json.loads((path / "adapter_config.json").read_text())
    if base_receipt and base_receipt.get("revision") is not None:
        if (adapter_config.get("base_model_name_or_path") != base_receipt["repo_id"]
                or adapter_config.get("revision") != base_receipt["revision"]):
            raise ValueError("Initial adapter was trained on a different base model or revision")
    if specification.get("revision"):
        receipt = json.loads((path / "download_provenance.json").read_text())
        if receipt["revision"] != specification["revision"] or receipt["repo_id"] != specification["repo_id"]:
            raise ValueError("Initial adapter does not match its pinned repository/revision")
    for key in ("r", "lora_alpha", "lora_dropout", "bias", "target_modules"):
        if adapter_config[key] != config["lora"][key]:
            raise ValueError(f"Initial adapter {key} differs from the configured LoRA recipe")
    return {"specification": specification,
            "config_sha256": sha256_file(path / "adapter_config.json"),
            "weights_sha256": sha256_file(path / "adapter_model.safetensors")}


def run(config_path, resume_from_checkpoint=None, smoke_test=False,
        label=None, base_model=None, init_adapter=None, epochs=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Trainer, TrainingArguments, set_seed,
    )

    config = load_config(config_path, label, base_model, init_adapter, epochs)
    if smoke_test:
        config["training"].update(output_dir=str(PROJECT_ROOT / "runs" / "training_smoke"),
                                  max_steps=1, gradient_accumulation_steps=1,
                                  warmup_steps=0, save_strategy="no", logging_steps=1)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != config["num_processes"]:
        raise ValueError("Launch with scripts/train.sh so WORLD_SIZE matches num_processes")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA GPUs; no CPU training fallback is configured")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if config["training"]["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")
    # Fail before loading 27B weights if the optimized recurrent kernels are missing.
    from causal_conv1d import causal_conv1d_fn  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

    output = Path(config["training"]["output_dir"])
    ds_config = None
    if config["training"].get("deepspeed") is not None:
        ds_config = json.loads(Path(config["training"]["deepspeed"]).read_text())
        if ds_config["zero_optimization"]["stage"] != 3:
            raise ValueError("When enabled, DeepSpeed must use ZeRO-3")
    model_path = Path(config["model"]["name_or_path"])
    receipt_path = model_path / "download_provenance.json"
    model_receipt = (json.loads(receipt_path.read_text()) if receipt_path.is_file()
                     else {"repo_id": str(model_path), "repo_type": "model", "revision": None})
    if not config["model"].get("local_override") and model_receipt["revision"] != config["model"]["revision"]:
        raise ValueError("Local model revision does not match configuration")
    index_path = model_path / "model.safetensors.index.json"
    weight_files = (set(json.loads(index_path.read_text())["weight_map"].values())
                    if index_path.is_file() else {"model.safetensors"})
    missing = [name for name in weight_files
               if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Model download incomplete: {missing}")
    provenance = {
        "config": config, "deepspeed": ds_config,
        "upstream": "wj210/simulation_persona@4441d5c26f05f5ee0c64efcc209196d5302358a0",
        "model_receipt": model_receipt,
        "init_adapter": adapter_provenance(config, model_receipt),
        "model_metadata_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_path.iterdir())
            if p.suffix in {".json", ".jinja", ".txt"}
        },
        "dataset_sha256": config["data"]["sha256"],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    package_versions = {name: version(name) for name in (
        "torch", "transformers", "peft", "accelerate", "datasets",
        "flash-linear-attention", "causal-conv1d", "tokenizers",
        "safetensors", "PyYAML", "Jinja2")}
    if resume_from_checkpoint:
        checkpoint = project_path(resume_from_checkpoint)
        if checkpoint.parent != output or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume checkpoint must be a Trainer checkpoint in the configured output_dir")
        previous = json.loads((output / "run_manifest.json").read_text())
        if previous["provenance"] != provenance:
            raise ValueError("Resume requires unchanged data, config, training code, and DeepSpeed settings")
        if previous["versions"] != package_versions:
            raise ValueError("Resume requires the original installed training package versions")
    elif int(os.environ.get("RANK", "0")) == 0 and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}; choose a new output_dir or resume explicitly")

    # Construct this BEFORE loading weights: it enables ZeRO-3 initialization.
    training_args = TrainingArguments(**config["training"])
    set_seed(training_args.seed)
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"], revision=model_config["revision"], padding_side="right",
        local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a padding or EOS token")
    data = config["data"]
    rows, stats = prepare_dataset(config, tokenizer)
    if smoke_test:
        # Exercise the longest real documents rather than tiny synthetic inputs.
        rows = sorted(rows, key=lambda row: len(row["input_ids"]), reverse=True)[:world_size]
    train_dataset = Dataset.from_list(rows)
    del rows
    if training_args.should_save:
        print(f"Training {len(train_dataset):,} examples; "
              f"{stats['excluded_documents']:,} excluded; "
              f"{stats['truncated_documents']} truncated at {data['max_seq_length']} tokens "
              f"({stats['removed_tokens']:,} tokens removed).", flush=True)
        if not resume_from_checkpoint:
            save_json(output / "data_report.json", stats)
            save_json(output / "run_manifest.json", {
                "created_at": timestamp(), "provenance": provenance,
                "training_documents": len(train_dataset), "smoke_test": smoke_test,
                "effective_batch_size": training_args.per_device_train_batch_size
                * world_size * training_args.gradient_accumulation_steps,
                "versions": package_versions,
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "gpu": torch.cuda.get_device_name(),
            })
            (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            (output / "chat_template.jinja").write_text(tokenizer.chat_template)

    # Transformers' Qwen3.5 text loader maps the multimodal checkpoint's text
    # weights correctly. Vision weights are not loaded or trained.
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"], revision=model_config["revision"],
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation=model_config["attn_implementation"],
        local_files_only=True,
    )
    model.config.use_cache = False
    if model_config.get("init_adapter"):
        model = PeftModel.from_pretrained(model, model_config["init_adapter"]["path"], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    # Keep exported adapters portable beyond this machine's local snapshot path.
    model.peft_config["default"].base_model_name_or_path = model_receipt["repo_id"]
    model.peft_config["default"].revision = model_receipt["revision"]
    adapter_modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if not adapter_modules or any("visual" in name for name in adapter_modules):
        raise ValueError("Expected adapters on text decoder modules only")
    if training_args.should_save:
        model.print_trainable_parameters()
        if not resume_from_checkpoint:
            save_json(output / "adapter_modules.json", adapter_modules)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100),
    )
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if resume_from_checkpoint else None)
    # All ranks participate in ZeRO-3 gathering; PEFT writes adapter weights only.
    trainer.save_model(str(output / "final_adapter"))
    if training_args.should_save:
        tokenizer.save_pretrained(output / "final_adapter")
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def positive_epochs(value):
    epochs = float(value)
    if not math.isfinite(epochs) or epochs <= 0:
        raise argparse.ArgumentTypeError("epochs must be a finite positive number")
    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", help="Absolute path or path relative to repository root")
    parser.add_argument("--smoke-test", action="store_true", help="One optimizer step on two documents in runs/training_smoke")
    parser.add_argument("--label", choices=["control", "reward_hack"], help="SFT response labels; config default is control")
    parser.add_argument("--base-model", help="Override the local base model directory")
    parser.add_argument("--init-adapter", help="Override the local initial LoRA adapter, or 'none' to start from the base")
    parser.add_argument("--epochs", "--num-train-epochs", type=positive_epochs,
                        help="Override num_train_epochs without editing the config; accepts fractional epochs")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and tokenize data without loading model weights")
    parser.add_argument("--report", help="Preparation report path relative to the repository")
    args = parser.parse_args()
    if args.prepare_only:
        from transformers import AutoTokenizer
        config = load_config(args.config.resolve(), args.label, args.base_model, args.init_adapter, args.epochs)
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"],
                                                  local_files_only=True, padding_side="right")
        _, report = prepare_dataset(config, tokenizer)
        default_report = (f"data/sft_{report['label']}_training_report.json" if "label" in report
                          else "data/chat_training_report.json" if report["objective"] == "chat_sft"
                          else "data/training_report.json")
        destination = project_path(args.report or default_report)
        save_json(destination, report)
        print(json.dumps({key: value for key, value in report.items() if key != "truncated_rows"}, indent=2))
        print(f"Data report: {destination}")
    else:
        if args.report:
            parser.error("--report requires --prepare-only")
        run(args.config.resolve(), args.resume_from_checkpoint, args.smoke_test,
            args.label, args.base_model, args.init_adapter, args.epochs)


if __name__ == "__main__":
    main()
