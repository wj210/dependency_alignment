#!/usr/bin/env python3
"""Prepare the configured chat/reward-hacking mix; no model training or judging."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dependency_alignment.training.data import project_path
from dependency_alignment.training.mix_data import build_sft_mix


def main():
    import yaml
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_path("configs/sft_chat.yaml"))
    parser.add_argument("--tokenizer", help="Override the config's base model/tokenizer path or HF ID")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = config["model"]
    name = args.tokenizer or model["name_or_path"]
    local = project_path(name)
    kwargs = {"local_files_only": True} if local.is_dir() else {"revision": model.get("revision")}
    tokenizer = AutoTokenizer.from_pretrained(str(local) if local.is_dir() else name, **kwargs)
    path, report = build_sft_mix(config["data"], tokenizer)
    print(json.dumps({"path": str(path), "manifest": str(path.with_suffix(".manifest.json")),
                      **{key: report[key] for key in ("dataset_sha256", "documents", "input_tokens",
                                                     "supervised_tokens", "components")}}, indent=2))


if __name__ == "__main__":
    main()
