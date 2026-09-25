"""Continuously refresh per-worker GC-MRL cosine similarities from a live CSV."""

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


def read_rows(path: Path) -> tuple[list[float], dict[int, list[float]]]:
    steps: list[float] = []
    values: dict[int, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        worker_ids = sorted(
            int(field.removeprefix("worker_").removesuffix("_cosine_similarity"))
            for field in fields
            if field.startswith("worker_") and field.endswith("_cosine_similarity")
        )
        values = {worker_id: [] for worker_id in worker_ids}
        for row in reader:
            try:
                step = float(row["environment_steps"])
                current = {
                    worker_id: float(row[f"worker_{worker_id}_cosine_similarity"])
                    for worker_id in worker_ids
                }
            except (KeyError, TypeError, ValueError):
                continue
            steps.append(step)
            for worker_id, value in current.items():
                values[worker_id].append(value)
    return steps, values


def write_chart(
    steps: list[float], values: dict[int, list[float]], destination: Path, smoothing: float
) -> None:
    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, worker_id in enumerate(sorted(values)):
        axis.plot(
            steps,
            ema(values[worker_id], smoothing),
            color=colors(index % 10),
            linewidth=1.5,
            label=f"worker {worker_id}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", ncol=4)
    stamp = datetime.now().strftime("%H:%M:%S")
    axis.set_title(
        f"Live GC-MRL worker cosine similarity (EMA {smoothing:.2f}, {stamp})"
    )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Cosine similarity")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--smoothing", type=float, default=0.99)
    args = parser.parse_args()
    source = args.run_dir / "gc_mrl_metrics.csv"
    destination = args.run_dir / "gc_mrl_cosine_similarity_live.png"
    previous_signature: tuple[int, int] | None = None
    while True:
        try:
            stat = source.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != previous_signature:
                steps, values = read_rows(source)
                if steps and values:
                    write_chart(steps, values, destination, args.smoothing)
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
