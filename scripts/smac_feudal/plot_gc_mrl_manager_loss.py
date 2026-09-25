"""Plot the manager actor loss recorded by a GC-MRL training run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ema(values: list[float], factor: float) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        result.append(factor * result[-1] + (1.0 - factor) * value)
    return result


def read_manager_loss(path: Path) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    losses: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                steps.append(float(row["environment_steps"]))
                losses.append(float(row["manager_actor_loss"]))
            except (KeyError, TypeError, ValueError):
                continue
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smoothing", type=float, default=0.99)
    args = parser.parse_args()
    if not 0.0 <= args.smoothing < 1.0:
        raise SystemExit("--smoothing must be in [0, 1).")

    steps, losses = read_manager_loss(args.run_dir / "gc_mrl_metrics.csv")
    if not steps:
        raise SystemExit("No manager actor-loss rows were found.")

    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    axis.plot(steps, losses, color="tab:orange", alpha=0.18, linewidth=0.7, label="raw")
    axis.plot(
        steps,
        ema(losses, args.smoothing),
        color="tab:orange",
        linewidth=2.0,
        label=f"EMA {args.smoothing:.2f}",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("GC-MRL manager actor loss")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Manager actor loss")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    output = args.run_dir / "gc_mrl_manager_actor_loss.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
