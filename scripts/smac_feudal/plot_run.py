from __future__ import annotations

import argparse
import csv
import json
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
    axes.flat[5].axis("off")
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
