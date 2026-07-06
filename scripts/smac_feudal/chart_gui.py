from __future__ import annotations

import argparse
import csv
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"


def read_rows(run_dir: Path) -> list[dict[str, float]]:
    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as stream:
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


def run_details(run_dir: Path) -> tuple[str, str, int, str]:
    rows = read_rows(run_dir)
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    modified = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    map_name = str(config.get("map", run_dir.name.rsplit("_", 1)[-1]))
    episodes = int(rows[-1]["episode"]) if rows else 0
    return modified, map_name, episodes, run_dir.name


class ChartApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SC2 FeUdal 離線訓練圖表")
        self.root.geometry("1450x850")
        self.root.minsize(1050, 650)
        self.checked: Path | None = None

        outer = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(outer, padding=6)
        right = ttk.Frame(outer, padding=6)
        outer.add(left, weight=1)
        outer.add(right, weight=3)

        ttk.Label(left, text="訓練紀錄", font=("Microsoft JhengHei UI", 14, "bold")).pack(
            anchor=tk.W, pady=(0, 8)
        )
        ttk.Label(left, text="點擊第一欄勾選，再按「顯示圖表」").pack(anchor=tk.W, pady=(0, 8))

        columns = ("checked", "time", "map", "episodes", "name")
        self.table = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
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
        self.checked = None
        for item in self.table.get_children():
            self.table.delete(item)
        if not RUNS_ROOT.exists():
            self.status.config(text=f"找不到紀錄資料夾：{RUNS_ROOT}")
            return
        runs = sorted(
            (path for path in RUNS_ROOT.iterdir() if (path / "metrics.csv").is_file()),
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
            self.checked = Path(row_id)
        self.status.config(text=f"共 {len(self.table.get_children())} 次紀錄")

    def on_table_click(self, event) -> None:
        row_id = self.table.identify_row(event.y)
        if not row_id:
            return
        for item in self.table.get_children():
            values = list(self.table.item(item, "values"))
            values[0] = "☐"
            self.table.item(item, values=values)
        values = list(self.table.item(row_id, "values"))
        values[0] = "☑"
        self.table.item(row_id, values=values)
        self.table.selection_set(row_id)
        self.checked = Path(row_id)

    def show_selected(self) -> None:
        if self.checked is None:
            selected = self.table.selection()
            if selected:
                self.checked = Path(selected[0])
        if self.checked is None:
            messagebox.showinfo("尚未選擇", "請先勾選一筆訓練紀錄。")
            return
        try:
            rows = read_rows(self.checked)
            if not rows:
                raise ValueError("這次紀錄尚無 episode 資料。")
            config = {}
            config_path = self.checked / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
            window = int(config.get("window", 100))
            smoothing = float(config.get("smoothing", 0.99))

            environment_steps = [row["environment_steps"] for row in rows]
            series = [
                ("return", "Episode return"),
                ("episode_steps", "Steps per episode"),
                ("allied_kills", "Enemy units killed"),
                ("allied_survivors", "Allied units remaining"),
                ("rolling_win_rate", f"Win rate (last {window})"),
            ]
            self.figure.clear()
            axes = self.figure.subplots(3, 2)
            for axis, (key, title) in zip(axes.flat, series):
                values = [row[key] for row in rows]
                axis.plot(
                    environment_steps,
                    values,
                    linewidth=0.8,
                    alpha=0.18,
                    color="tab:blue",
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
            self.figure.suptitle(
                f"FeUdal on SMAC — {self.checked.name} — smoothing {smoothing:.2f}"
            )
            self.canvas.draw()
            self.status.config(text=f"正在顯示：{self.checked.name}")
        except Exception as error:
            messagebox.showerror("無法畫圖", str(error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只檢查紀錄，不開啟 GUI。")
    args = parser.parse_args()
    if args.check:
        runs = list(RUNS_ROOT.glob("*/metrics.csv")) if RUNS_ROOT.exists() else []
        print(f"GUI_CHECK_OK records={len(runs)}")
        return
    root = tk.Tk()
    ChartApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
