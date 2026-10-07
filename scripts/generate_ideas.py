"""Generate the pilot's idea batch from the source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dependency_alignment.generation import main

if __name__ == "__main__":
    main()
