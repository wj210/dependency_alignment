#!/usr/bin/env bash
# Generate 5,000 new documents, or resume the same batch after interruption.
# Run from any directory. Extra generate_documents.py options are forwarded.
# Preview without sampling, authentication, or provider calls: ./scripts/generate.sh --dry-run
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
generation_python="$project_dir/.venv/bin/python"
cd "$project_dir"

if [[ ! -x "$generation_python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python python3 "$project_dir/.venv"
    else
        python3 -m venv "$project_dir/.venv"
    fi
fi

"$generation_python" -c 'import sys; sys.exit("Python 3.11+ is required") if sys.version_info < (3, 11) else None'

if ! "$generation_python" -c 'from importlib.metadata import version; raise SystemExit(version("litellm") != "1.104.0")' >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$generation_python" -e '.[generation]'
    else
        "$generation_python" -m ensurepip --upgrade
        "$generation_python" -m pip install -e '.[generation]'
    fi
fi

exec "$generation_python" "$project_dir/scripts/generate_documents.py" \
    --start-or-resume --count 5000 --seed 42 \
    --output "$project_dir/data/documents_5000_2.jsonl" "$@"
