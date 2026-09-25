from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path("E:/SC2-runs") if Path("E:/SC2-runs").exists() else PROJECT_ROOT / "runs"


def available_runs() -> list[Path]:
    if not RUNS_ROOT.exists():
        return []
    return sorted(
        (path for path in RUNS_ROOT.iterdir() if (path / "metrics.csv").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def choose_run(runs: list[Path]) -> Path:
    print("\n可用的訓練紀錄：")
    for index, run in enumerate(runs, start=1):
        print(f"  {index:>2}. {run.name}")
    while True:
        answer = input("\n請輸入要畫圖的編號（q 離開）：").strip()
        if answer.lower() == "q":
            raise SystemExit(0)
        if answer.isdigit() and 1 <= int(answer) <= len(runs):
            return runs[int(answer) - 1]
        print("編號無效，請重新輸入。")


def read_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    previous_total = 0.0
    for row in rows:
        if "episode_steps" not in row:
            row["episode_steps"] = row["environment_steps"] - previous_total
        previous_total = row["environment_steps"]
    return rows


def read_advantage_rows(run_dir: Path) -> list[dict[str, float]]:
    csv_path = run_dir / "advantages.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def smooth(values, factor: float):
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        result.append(factor * result[-1] + (1.0 - factor) * float(value))
    return result


def make_plot(run_dir: Path) -> Path:
    rows = read_rows(run_dir / "metrics.csv")
    if not rows:
        raise ValueError(f"紀錄尚無資料：{run_dir}")

    window = 100
    smoothing = 0.99
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        window = int(config.get("window", window))
        smoothing = float(config.get("smoothing", smoothing))

    environment_steps = [row["environment_steps"] for row in rows]
    series = [
        ("return", "Episode return"),
        ("episode_steps", "Steps per episode"),
        ("allied_kills", "Enemy units killed"),
        ("allied_survivors", "Allied units remaining"),
        ("rolling_win_rate", f"Win rate (last {window})"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, series):
        values = [row[key] for row in rows]
        axis.plot(
            environment_steps, values, linewidth=0.8, alpha=0.18, color="tab:blue"
        )
        axis.plot(
            environment_steps,
            smooth(values, smoothing),
            linewidth=1.8,
            color="tab:blue",
        )
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.25)
    axes.flat[4].set_ylim(-0.02, 1.02)
    advantage_axis = axes.flat[5]
    advantage_rows = read_advantage_rows(run_dir)
    episode_advantage_rows = [
        row for row in advantage_rows
        if math.isfinite(row.get("episode", float("nan")))
    ]
    if episode_advantage_rows:
        for key, label, color in (
            ("external_advantage_mean", "mean A_ext", "tab:blue"),
            ("intrinsic_advantage_mean", "mean A_int raw", "tab:orange"),
            (
                "weighted_intrinsic_advantage_mean",
                "mean beta A_int",
                "tab:green",
            ),
        ):
            valid_rows = [
                row for row in episode_advantage_rows
                if math.isfinite(row.get(key, float("nan")))
            ]
            if not valid_rows:
                continue
            advantage_steps = [row["episode"] for row in valid_rows]
            values = [row[key] for row in valid_rows]
            advantage_axis.plot(advantage_steps, values, linewidth=0.7, alpha=0.18, color=color)
            advantage_axis.plot(
                advantage_steps, smooth(values, smoothing), linewidth=1.8,
                color=color, label=label,
            )
        advantage_axis.legend(loc="upper left", fontsize=8)
        ratio_axis = advantage_axis.twinx()
        raw_ratio_rows = [
            row
            for row in episode_advantage_rows
            if math.isfinite(row.get("intrinsic_advantage_abs_mean", float("nan")))
            and math.isfinite(row.get("external_advantage_abs_mean", float("nan")))
            and row["external_advantage_abs_mean"] > 0
        ]
        if raw_ratio_rows:
            raw_ratios = []
            for row in raw_ratio_rows:
                ratio = row.get("intrinsic_to_external_raw_ratio", float("nan"))
                if not math.isfinite(ratio):
                    ratio = (
                        row["intrinsic_advantage_abs_mean"]
                        / row["external_advantage_abs_mean"]
                    )
                raw_ratios.append(ratio)
            ratio_axis.plot(
                [row["episode"] for row in raw_ratio_rows],
                smooth(raw_ratios, smoothing),
                linewidth=1.5,
                linestyle="-.",
                color="tab:purple",
                label="|A_int| / |A_ext| raw",
            )
        ratio_rows = [
            row
            for row in episode_advantage_rows
            if math.isfinite(row.get("intrinsic_to_external_advantage_ratio", float("nan")))
        ]
        if ratio_rows:
            ratio_axis.plot(
                [row["episode"] for row in ratio_rows],
                smooth(
                    [row["intrinsic_to_external_advantage_ratio"] for row in ratio_rows],
                    smoothing,
                ),
                linewidth=1.5,
                linestyle="--",
                color="tab:red",
                label="|beta A_int| / |A_ext|",
            )
        if raw_ratio_rows or ratio_rows:
            ratio_axis.set_ylabel("Intrinsic / external")
            ratio_axis.legend(loc="upper right", fontsize=8)
    else:
        advantage_axis.text(
            0.5, 0.5, "No advantage data", ha="center", va="center",
            transform=advantage_axis.transAxes,
        )
    advantage_axis.axhline(0.0, color="black", linewidth=0.8)
    advantage_axis.set_title("Episode-mean raw advantages and magnitude ratios")
    advantage_axis.set_xlabel("Completed episode")
    advantage_axis.grid(alpha=0.25)
    fig.suptitle(f"FeUdal on SMAC — {run_dir.name} — smoothing {smoothing:.2f}")
    output = run_dir / "training_metrics.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="離線選擇並繪製某次 SMAC 訓練紀錄。")
    parser.add_argument("--run", type=Path, help="run 資料夾名稱或完整路徑。")
    parser.add_argument("--list", action="store_true", help="只列出可用紀錄。")
    parser.add_argument("--show", action="store_true", help="畫完後用 Windows 圖片程式開啟。")
    args = parser.parse_args()

    runs = available_runs()
    if args.list:
        for run in runs:
            print(run)
        return
    if not runs:
        raise SystemExit(f"找不到訓練紀錄：{RUNS_ROOT}")

    if args.run:
        run_dir = args.run if args.run.is_absolute() else RUNS_ROOT / args.run
        if not (run_dir / "metrics.csv").is_file():
            raise SystemExit(f"找不到 metrics.csv：{run_dir}")
    else:
        run_dir = choose_run(runs)

    output = make_plot(run_dir)
    print(f"\n圖表已產生：{output}")
    if args.show:
        os.startfile(output)


if __name__ == "__main__":
    main()
