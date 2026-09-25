"""Continuously refresh an env-0 manager chart from a live training CSV."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
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


def read_env_zero_rows(path: Path) -> tuple[list[float], list[float], list[float]]:
    steps: list[float] = []
    returns: list[float] = []
    actor_losses: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                if int(float(row["env_id"])) != 0:
                    continue
                steps.append(float(row["environment_steps"]))
                returns.append(float(row["manager_return"]))
                actor_losses.append(float(row["manager_actor_loss"]))
            except (KeyError, TypeError, ValueError):
                # The writer may be appending a line while it is being read.
                continue
    return steps, returns, actor_losses


def read_total_actor_loss_rows(path: Path) -> tuple[list[float], list[float]]:
    """Sum actor losses from manager segments ending at the same global step."""
    totals: dict[float, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                step = float(row["environment_steps"])
                totals[step] = totals.get(step, 0.0) + float(
                    row["manager_actor_loss"]
                )
            except (KeyError, TypeError, ValueError):
                continue
    steps = sorted(totals)
    return steps, [totals[step] for step in steps]


def write_chart(
    steps: list[float],
    returns: list[float],
    actor_losses: list[float],
    destination: Path,
    smoothing: float,
) -> None:
    figure, (return_axis, actor_axis) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True, constrained_layout=True
    )
    return_axis.plot(
        steps, ema(returns, smoothing), color="tab:blue", linewidth=1.7
    )
    actor_axis.plot(
        steps, ema(actor_losses, smoothing), color="tab:orange", linewidth=1.7
    )
    for axis in (return_axis, actor_axis):
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
    stamp = datetime.now().strftime("%H:%M:%S")
    return_axis.set_title(
        f"Live manager metrics — env 0, EMA {smoothing:.2f} ({stamp})"
    )
    return_axis.set_ylabel("Manager return")
    actor_axis.set_xlabel("Environment steps")
    actor_axis.set_ylabel("Manager actor loss")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def write_total_actor_loss_chart(
    steps: list[float], losses: list[float], destination: Path, smoothing: float
) -> None:
    figure, axis = plt.subplots(figsize=(16, 5), constrained_layout=True)
    axis.plot(
        steps, ema(losses, smoothing), color="tab:red", linewidth=1.7,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.grid(alpha=0.25)
    stamp = datetime.now().strftime("%H:%M:%S")
    axis.set_title(
        "Live total manager actor loss across completed environments "
        f"(EMA {smoothing:.2f}, {stamp})"
    )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Summed manager actor loss")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--smoothing", type=float, default=0.99)
    args = parser.parse_args()
    source = args.run_dir / "manager_segments.csv"
    destination = args.run_dir / "manager_segments_env0_live.png"
    total_destination = args.run_dir / "manager_actor_loss_total_live.png"
    previous_signature: tuple[int, int] | None = None
    while True:
        try:
            stat = source.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != previous_signature:
                steps, returns, actor_losses = read_env_zero_rows(source)
                if steps:
                    write_chart(
                        steps, returns, actor_losses, destination, args.smoothing
                    )
                    total_steps, total_losses = read_total_actor_loss_rows(source)
                    write_total_actor_loss_chart(
                        total_steps, total_losses, total_destination, args.smoothing
                    )
                    print(
                        f"updated rows={len(steps)} latest_step={steps[-1]:.0f} "
                        f"output={destination}",
                        flush=True,
                    )
                previous_signature = signature
        except FileNotFoundError:
            pass
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
