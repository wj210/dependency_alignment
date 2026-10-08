#!/usr/bin/env python3
"""Plot logged training and evaluation losses from a Trainer state file."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.run_dir / "trainer_state.json").read_text())
    train = [(row["step"], row["loss"]) for row in state["log_history"] if "loss" in row]
    evaluation = [(row["step"], row["eval_loss"]) for row in state["log_history"] if "eval_loss" in row]
    if not train or not evaluation:
        raise ValueError("Both training and evaluation loss logs are required")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(*zip(*train), color="#3569b0", linewidth=1.2, alpha=0.7,
            label="Train loss (logged batch averages)")
    ax.plot(*zip(*evaluation), color="#e37928", linewidth=2, marker="o", label="Held-out eval loss")
    step, loss = min(evaluation, key=lambda pair: pair[1])
    ax.scatter([step], [loss], color="#25955d", marker="*", s=160, zorder=5,
               label=f"Best eval: {loss:.4f} at step {step}")
    ax.set(xlabel="Optimizer step", ylabel="Cross-entropy loss",
           title="Qwen3.8-27B DA epoch 1 → mixed SFT")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    for extension in ("png", "svg"):
        fig.savefig(args.run_dir / f"loss_curve.{extension}", dpi=180)
    plt.close(fig)
    with (args.run_dir / "loss_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "optimizer_step", "loss"])
        writer.writerows(("train", step, loss) for step, loss in train)
        writer.writerows(("eval", step, loss) for step, loss in evaluation)
    print(json.dumps({"train_points": len(train), "eval_points": len(evaluation),
                      "best_eval_step": step, "best_eval_loss": loss,
                      "png": str(args.run_dir / "loss_curve.png")}, indent=2))


if __name__ == "__main__":
    main()
