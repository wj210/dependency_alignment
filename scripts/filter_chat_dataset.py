"""Filter the pinned chat mix using the current Codex subscription session."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dependency_alignment.chat_filter import main

if __name__ == "__main__":
    main()
