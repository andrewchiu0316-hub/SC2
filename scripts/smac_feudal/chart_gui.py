from __future__ import annotations

import argparse
import csv
import json
import math
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import colormaps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path("E:/SC2-runs") if Path("E:/SC2-runs").exists() else PROJECT_ROOT / "runs"
PLOT_OPTIONS = [
    ("return", "Episode return"),
    ("episode_steps", "Steps per episode"),
    ("allied_kills", "Enemy units killed"),
    ("allied_survivors", "Allied units remaining"),
    ("rolling_win_rate", "Win rate"),
    ("advantages", "Episode advantages / ratio"),
]


def read_rows(run_dir: Path) -> list[dict[str, float]]:
    csv_path = run_dir / "metrics.csv"
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            rows = [
                {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    else:
        rows = read_gc_mrl_rows(run_dir / "metrics.json", run_dir)
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


def read_gc_mrl_rows(metrics_path: Path, run_dir: Path) -> list[dict[str, float]]:
    """Adapt PyMARL/GC-MRL JSON metrics to the common chart series."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"找不到 metrics.csv 或 metrics.json：{run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = "return_mean"
    if required not in metrics:
        raise ValueError(f"metrics.json 缺少 {required}，無法繪圖。")

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    map_name = str(config.get("env_args", {}).get("map_name", ""))
    allied_count = {"3s5z": 8}.get(map_name, 0)
    field_map = {
        "return": "return_mean",
        "episode_steps": "ep_length_mean",
        "allied_kills": "dead_enemies_mean",
        "rolling_win_rate": "battle_won_mean",
    }
    reference = metrics[required]
    rows: list[dict[str, float]] = []
    for index, step in enumerate(reference["steps"]):
        row = {"episode": float(index + 1), "environment_steps": float(step)}
        for output_name, source_name in field_map.items():
            values = metrics.get(source_name, {}).get("values", [])
            row[output_name] = float(values[index]) if index < len(values) else float("nan")
        dead_allies = metrics.get("dead_allies_mean", {}).get("values", [])
        row["allied_survivors"] = (
            float(allied_count - dead_allies[index])
            if allied_count and index < len(dead_allies)
            else float("nan")
        )
        rows.append(row)
    return rows


def smooth(values, factor: float):
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        result.append(factor * result[-1] + (1.0 - factor) * float(value))
    return result


def run_details(run_dir: Path) -> tuple[str, str, int, str]:
    rows = read_rows(run_dir)
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    modified = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    map_name = str(
        config.get("map", config.get("env_args", {}).get("map_name", run_dir.name.rsplit("_", 1)[-1]))
    )
    episodes = int(rows[-1]["episode"]) if rows else 0
    return modified, map_name, episodes, run_dir.name


class ChartApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SC2 FeUdal 離線訓練圖表")
        self.root.geometry("1450x850")
        self.root.minsize(1050, 650)
        self.checked: set[Path] = set()

        outer = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(outer, padding=6)
        right = ttk.Frame(outer, padding=6)
        outer.add(left, weight=1)
        outer.add(right, weight=3)

        ttk.Label(left, text="訓練紀錄", font=("Microsoft JhengHei UI", 14, "bold")).pack(
            anchor=tk.W, pady=(0, 8)
        )
        ttk.Label(left, text="點擊第一欄可勾選多筆，再按「顯示圖表」比較").pack(anchor=tk.W, pady=(0, 8))

        option_frame = ttk.LabelFrame(left, text="顯示項目", padding=6)
        option_frame.pack(fill=tk.X, pady=(0, 8))
        self.plot_options = {
            key: tk.BooleanVar(value=True) for key, _label in PLOT_OPTIONS
        }
        for index, (key, label) in enumerate(PLOT_OPTIONS):
            ttk.Checkbutton(
                option_frame, text=label, variable=self.plot_options[key]
            ).grid(row=index // 2, column=index % 2, sticky=tk.W, padx=4, pady=2)

        columns = ("checked", "time", "map", "episodes", "name")
        self.table = ttk.Treeview(left, columns=columns, show="headings", selectmode="extended")
        self.table.heading("checked", text="選擇")
        self.table.heading("time", text="執行時間")
        self.table.heading("map", text="地圖")
        self.table.heading("episodes", text="回合")
        self.table.heading("name", text="紀錄名稱")
        self.table.column("checked", width=48, anchor=tk.CENTER, stretch=False)
        self.table.column("time", width=145, anchor=tk.CENTER, stretch=False)
        self.table.column("map", width=55, anchor=tk.CENTER, stretch=False)
        self.table.column("episodes", width=55, anchor=tk.CENTER, stretch=False)
        self.table.column("name", width=230, anchor=tk.W)

        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.bind("<Button-1>", self.on_table_click)
        self.table.bind("<Double-1>", lambda _event: self.show_selected())

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, padx=16, pady=(0, 10))
        ttk.Button(controls, text="重新整理", command=self.refresh).pack(side=tk.LEFT)
        ttk.Button(controls, text="顯示已勾選圖表", command=self.show_selected).pack(
            side=tk.LEFT, padx=8
        )
        self.status = ttk.Label(controls, text="")
        self.status.pack(side=tk.LEFT, padx=12)

        self.figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.refresh()
        self.show_selected()

    def refresh(self) -> None:
        self.checked.clear()
        for item in self.table.get_children():
            self.table.delete(item)
        if not RUNS_ROOT.exists():
            self.status.config(text=f"找不到紀錄資料夾：{RUNS_ROOT}")
            return
        runs = sorted(
            (
                path
                for path in RUNS_ROOT.iterdir()
                if (path / "metrics.csv").is_file() or (path / "metrics.json").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for run_dir in runs:
            try:
                modified, map_name, episodes, name = run_details(run_dir)
                self.table.insert(
                    "", tk.END, iid=str(run_dir), values=("☐", modified, map_name, episodes, name)
                )
            except Exception:
                continue
        first_item = self.table.get_children()
        if first_item:
            row_id = first_item[0]
            values = list(self.table.item(row_id, "values"))
            values[0] = "☑"
            self.table.item(row_id, values=values)
            self.table.selection_set(row_id)
            self.table.focus(row_id)
            self.checked.add(Path(row_id))
        self.status.config(text=f"共 {len(self.table.get_children())} 次紀錄")

    def on_table_click(self, event) -> None:
        row_id = self.table.identify_row(event.y)
        if not row_id:
            return
        values = list(self.table.item(row_id, "values"))
        run_dir = Path(row_id)
        if run_dir in self.checked:
            self.checked.remove(run_dir)
            values[0] = "☐"
        else:
            self.checked.add(run_dir)
            values[0] = "☑"
        self.table.item(row_id, values=values)

    def show_selected(self) -> None:
        if not self.checked:
            messagebox.showinfo("尚未選擇", "請先勾選至少一筆訓練紀錄。")
            return
        selected_keys = [
            key for key, _label in PLOT_OPTIONS if self.plot_options[key].get()
        ]
        if not selected_keys:
            messagebox.showinfo("尚未選擇項目", "請至少勾選一個要顯示的圖表項目。")
            return
        try:
            titles = dict(PLOT_OPTIONS)
            self.figure.clear()
            column_count = 1 if len(selected_keys) == 1 else 2
            row_count = math.ceil(len(selected_keys) / column_count)
            axes = self.figure.subplots(row_count, column_count, squeeze=False)
            plot_axes = list(axes.flat)
            colors = colormaps["tab10"]
            runs = []
            for run_index, run_dir in enumerate(sorted(self.checked, key=lambda path: path.name)):
                rows = read_rows(run_dir)
                if not rows:
                    continue
                config_path = run_dir / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
                smoothing = float(config.get("smoothing", 0.99))
                runs.append((run_dir, rows, smoothing, colors(run_index % 10)))
            if not runs:
                raise ValueError("所選紀錄尚無 episode 資料。")

            for axis, key in zip(plot_axes, selected_keys):
                if key == "advantages":
                    self._plot_advantages(axis, runs)
                    continue
                for run_dir, rows, smoothing, color in runs:
                    environment_steps = [row["environment_steps"] for row in rows]
                    values = [row[key] for row in rows]
                    # Preserve the unsmoothed signal faintly beneath the comparison line.
                    axis.plot(
                        environment_steps, values, linewidth=0.6, alpha=0.035,
                        color=color,
                    )
                    axis.plot(
                        environment_steps, smooth(values, smoothing), linewidth=1.8,
                        alpha=0.85, color=color, label=run_dir.name,
                    )
                axis.set_title(titles[key])
                axis.set_xlabel("Environment steps")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8)
                if key == "rolling_win_rate":
                    axis.set_ylim(-0.02, 1.02)
            for axis in plot_axes[len(selected_keys):]:
                axis.set_visible(False)
            self.figure.suptitle(
                f"SMAC — comparison of {len(runs)} runs"
            )
            self.canvas.draw()
            self.status.config(
                text=f"正在比較 {len(runs)} 筆紀錄，顯示 {len(selected_keys)} 個項目"
            )
        except Exception as error:
            messagebox.showerror("無法畫圖", str(error))

    def _plot_advantages(self, axis, runs) -> None:
        """Render the advantage panel selected in the GUI."""
        ratio_axis = axis.twinx()
        has_advantages = False
        has_ratios = False
        for run_dir, _rows, smoothing, color in runs:
            advantage_rows = read_advantage_rows(run_dir)
            episode_rows = [
                row for row in advantage_rows
                if math.isfinite(row.get("episode", float("nan")))
            ]
            for key, label, style in (
                ("external_advantage_mean", "mean A_ext", "-"),
                ("intrinsic_advantage_mean", "mean A_int raw", "--"),
                ("weighted_intrinsic_advantage_mean", "mean beta A_int", ":"),
            ):
                valid_rows = [
                    row for row in episode_rows
                    if math.isfinite(row.get(key, float("nan")))
                ]
                if not valid_rows:
                    continue
                has_advantages = True
                axis.plot(
                    [row["episode"] for row in valid_rows],
                    smooth([row[key] for row in valid_rows], smoothing),
                    linestyle=style, linewidth=1.5, alpha=0.85, color=color,
                    label=f"{run_dir.name} {label}",
                )
            ratio_rows = [
                row for row in episode_rows
                if math.isfinite(row.get("intrinsic_to_external_advantage_ratio", float("nan")))
            ]
            if ratio_rows:
                has_ratios = True
                ratio_axis.plot(
                    [row["episode"] for row in ratio_rows],
                    smooth(
                        [row["intrinsic_to_external_advantage_ratio"] for row in ratio_rows],
                        smoothing,
                    ),
                    linestyle="-.", linewidth=1.4, alpha=0.85, color=color,
                    label=f"{run_dir.name} int/ext",
                )
            raw_ratio_rows = [
                row for row in episode_rows
                if math.isfinite(row.get("intrinsic_advantage_abs_mean", float("nan")))
                and math.isfinite(row.get("external_advantage_abs_mean", float("nan")))
                and row["external_advantage_abs_mean"] > 0
            ]
            if raw_ratio_rows:
                has_ratios = True
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
                    linestyle="--", linewidth=1.4, alpha=0.85, color=color,
                    label=f"{run_dir.name} raw int/ext",
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title("Episode-mean raw advantages and magnitude ratios")
        axis.set_xlabel("Completed episode")
        axis.grid(alpha=0.25)
        if has_advantages:
            axis.legend(loc="upper left", fontsize=7)
        else:
            axis.text(0.5, 0.5, "No advantage data", ha="center", va="center", transform=axis.transAxes)
        if has_ratios:
            ratio_axis.set_ylabel("Weighted intrinsic / external")
            ratio_axis.legend(loc="upper right", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只檢查紀錄，不開啟 GUI。")
    args = parser.parse_args()
    if args.check:
        runs = (
            [
                path
                for path in RUNS_ROOT.iterdir()
                if (path / "metrics.csv").is_file() or (path / "metrics.json").is_file()
            ]
            if RUNS_ROOT.exists()
            else []
        )
        print(f"GUI_CHECK_OK records={len(runs)}")
        return
    root = tk.Tk()
    ChartApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
