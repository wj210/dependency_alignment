"""Download pinned training assets into configured directories relative to the repository."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def download_assets(config_path):
    config = yaml.safe_load(config_path.read_text())
    data, model = config["data"], config["model"]
    filename = data.get("filename")
    patterns = [filename, "*.parquet", "README.md", ".gitattributes"] if filename else None
    snapshots = [
        ("dataset", data["repo_id"], data["revision"], project_path(data["path"]).parent, patterns),
        ("model", model.get("repo_id", "Qwen/Qwen3.8-27B"), model["revision"],
         project_path(model["name_or_path"]),
         ["*.safetensors", "*.json", "*.jinja", "*.txt", "README.md", "LICENSE"]),
    ]
    if model.get("init_adapter"):
        adapter = model["init_adapter"]
        snapshots.append((
            "model", adapter["repo_id"], adapter["revision"], project_path(adapter["path"]),
            ["*.safetensors", "*.json", "*.jinja", "*.txt", "README.md", "LICENSE"],
        ))
    for repo_type, repo_id, revision, destination, allow_patterns in snapshots:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision,
                          local_dir=destination, allow_patterns=allow_patterns, max_workers=4)
        receipt = {"repo_id": repo_id, "repo_type": repo_type, "revision": revision}
        (destination / "download_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Installed {repo_id}@{revision} at {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lora.yaml")
    args = parser.parse_args()
    download_assets(args.config.resolve())


if __name__ == "__main__":
    main()
