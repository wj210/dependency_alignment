"""Print the original and score-hint-removed versions of the five Table 7 prompts."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    tasks_data = json.loads((HERE / "tasks.json").read_bytes())
    counterfactuals = json.loads((HERE / "counterfactual_prompts.json").read_bytes())
    tasks = {task["id"]: task for task in tasks_data["tasks"] if task["table"] == 7}
    variants = counterfactuals["tasks"]
    if set(tasks) != set(variants):
        raise ValueError("Counterfactual prompt IDs must exactly match the five Table 7 tasks")

    for task_id, task in tasks.items():
        variant = variants[task_id]
        print(f"\n=== {task_id} ===")
        print("ORIGINAL (scoring hint disclosed)")
        print(task["prompt"])
        print("COUNTERFACTUAL (scoring hint removed)")
        print(variant["counterfactual_prompt"])
        print("REMOVED HINT")
        print(variant["removed_hint"])


if __name__ == "__main__":
    main()
