"""Install immutable model and document snapshots without duplicate weight caches."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/qwen_lora.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for repo_type, repo_id, revision, destination, patterns in (
        ("dataset", config["data"]["repo_id"], config["data"]["revision"],
         Path(config["data"]["path"]).parent,
         [config["data"]["filename"], "*.parquet", "README.md", ".gitattributes"]),
        ("model", "Qwen/Qwen3.8-27B", config["model"]["revision"],
         Path(config["model"]["name_or_path"]),
         ["*.safetensors", "*.json", "*.jinja", "*.txt", "README.md", "LICENSE"]),
    ):
        snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision,
                          local_dir=destination, allow_patterns=patterns, max_workers=4)
        receipt = {"repo_id": repo_id, "repo_type": repo_type, "revision": revision}
        (destination / "download_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Installed {repo_id}@{revision} at {destination}", flush=True)


if __name__ == "__main__":
    main()
