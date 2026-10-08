"""Download pinned training assets into configured directories relative to the repository."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
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
        ("model", model.get("repo_id", "Qwen/Qwen3.8-27B"), model["revision"],
         project_path(model["name_or_path"]),
         ["*.safetensors", "*.json", "*.jinja", "*.txt", "README.md", "LICENSE"]),
    ]
    if "mixture" in data:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from dependency_alignment.training.data import download_documents
        for name in ("chat", "school"):
            source_path = download_documents(data["mixture"][name])
            print(f"Cached mixture source {name} at {source_path}", flush=True)
    elif data["path"] == data["repo_id"]:
        cached_path = hf_hub_download(repo_id=data["repo_id"], repo_type="dataset",
                                      revision=data["revision"], filename=filename)
        print(f"Cached {data['repo_id']}@{data['revision']} at {cached_path}", flush=True)
    elif project_path(data["path"]).is_file():
        print(f"Using local dataset at {project_path(data['path'])}", flush=True)
    else:
        snapshots.insert(0, ("dataset", data["repo_id"], data["revision"],
                             project_path(data["path"]).parent, patterns))
    if data.get("eval"):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from dependency_alignment.training.data import download_documents
        evaluation_path = download_documents(data["eval"])
        print(f"Cached evaluation split at {evaluation_path}", flush=True)
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
