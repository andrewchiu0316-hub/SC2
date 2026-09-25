from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("training_config.yaml")
for source_dir in (PROJECT_ROOT, PROJECT_ROOT / "vendor" / "smac", PROJECT_ROOT / "vendor" / "pysc2"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from smac.env import StarCraft2Env
from parallel_env import SubprocSMACVecEnv


ALGORITHM_MODULES = {
    "feudal": "module.Algorithm.FeUdal.algorithm",
    "feudal_haa2c": "module.Algorithm.FeUdal_HAA2C.algorithm",
    "feudal_test": "module.Algorithm.FeUdal_test.algorithm",
    "gc_mrl": "module.Algorithm.GC_MRL.algorithm",
    "haa2c": "module.Algorithm.HAA2C.algorithm",
    "qmix": "module.Algorithm.QMIX.algorithm",
}


def load_algorithm_class(name: str):
    key = name.lower()
    if key not in ALGORITHM_MODULES:
        choices = ", ".join(sorted(ALGORITHM_MODULES))
        raise ValueError(f"Unknown algorithm '{name}'. Choose one of: {choices}")
    return importlib.import_module(ALGORITHM_MODULES[key]).Algorithm

METRIC_FIELDS = [
    "episode",
    "environment_steps",
    "episode_steps",
    "return",
    "allied_kills",
    "allied_survivors",
    "won",
    "rolling_win_rate",
]
ADVANTAGE_FIELDS = [
    "episode",
    "environment_steps",
    "external_advantage_mean",
    "intrinsic_advantage_mean",
    "weighted_intrinsic_advantage_mean",
    "external_advantage_abs_mean",
    "intrinsic_advantage_abs_mean",
    "weighted_intrinsic_advantage_abs_mean",
    "intrinsic_to_external_raw_ratio",
    "intrinsic_to_external_advantage_ratio",
]
INTRINSIC_REWARD_FIELDS = ["environment_steps", "intrinsic_reward_mean"]
MANAGER_SEGMENT_FIELDS = [
    "environment_steps",
    "env_id",
    "segment_steps",
    "manager_return",
    "manager_actor_loss",
]


def gc_mrl_metric_fields(n_agents: int) -> list[str]:
    return [
        "environment_steps",
        "update",
        "manager_actor_loss",
        *[f"worker_{agent_id}_cosine_similarity" for agent_id in range(n_agents)],
    ]


def intrinsic_alignment_fields(n_agents: int) -> list[str]:
    """Columns for raw per-worker rewards and manager goal alignment."""
    return [
        "environment_steps",
        "manager_cos_sim",
        *[
            f"worker_{agent_id}_raw_intrinsic_reward"
            for agent_id in range(n_agents)
        ],
    ]


def write_tensorboard_metrics(writer: SummaryWriter, row: dict[str, float]) -> None:
    """Write only the four user-facing training metrics to TensorBoard."""
    step = int(row["episode"])
    writer.add_scalar("win_rate", float(row["rolling_win_rate"]), step)
    writer.add_scalar("return", float(row["return"]), step)
    writer.add_scalar("enemy_kills", float(row["allied_kills"]), step)
    writer.add_scalar("allied_survivors", float(row["allied_survivors"]), step)


def advantage_row(agent, environment_steps: int) -> dict[str, float] | None:
    """Legacy rollout-level FeUdal summary; episode rows use the helpers below."""
    required = (
        "last_worker_mean_external_advantage",
        "last_worker_mean_intrinsic_advantage",
        "last_worker_mean_abs_external_advantage",
        "last_worker_mean_abs_intrinsic_advantage",
        "last_worker_mean_abs_weighted_intrinsic_advantage",
        "last_worker_raw_intrinsic_to_external_advantage_ratio",
        "last_worker_intrinsic_to_external_advantage_ratio",
    )
    if not all(hasattr(agent, name) for name in required):
        return None
    return {
        "episode": float("nan"),
        "environment_steps": float(environment_steps),
        "external_advantage_mean": float(agent.last_worker_mean_external_advantage),
        "intrinsic_advantage_mean": float(agent.last_worker_mean_intrinsic_advantage),
        "weighted_intrinsic_advantage_mean": float(
            agent.intrinsic_coef * agent.last_worker_mean_intrinsic_advantage
        ),
        "external_advantage_abs_mean": float(
            agent.last_worker_mean_abs_external_advantage
        ),
        "intrinsic_advantage_abs_mean": float(
            agent.last_worker_mean_abs_intrinsic_advantage
        ),
        "weighted_intrinsic_advantage_abs_mean": float(
            agent.last_worker_mean_abs_weighted_intrinsic_advantage
        ),
        "intrinsic_to_external_raw_ratio": float(
            agent.last_worker_raw_intrinsic_to_external_advantage_ratio
        ),
        "intrinsic_to_external_advantage_ratio": float(
            agent.last_worker_intrinsic_to_external_advantage_ratio
        ),
    }


def new_episode_advantage_totals() -> dict[str, float]:
    """Create accumulators weighted by valid worker actions in one episode."""
    return {
        "count": 0.0,
        "external_sum": 0.0,
        "intrinsic_sum": 0.0,
        "weighted_intrinsic_sum": 0.0,
        "external_abs_sum": 0.0,
        "intrinsic_abs_sum": 0.0,
        "weighted_intrinsic_abs_sum": 0.0,
    }


def accumulate_episode_advantages(
    totals_by_episode: dict[int, dict[str, float]],
    episode_tokens: list[np.ndarray],
    external_advantages: torch.Tensor,
    intrinsic_advantages: torch.Tensor,
    alive_masks: torch.Tensor,
    intrinsic_coef: float,
) -> None:
    """Accumulate raw GAEs by the environment episode that generated each step."""
    external = external_advantages.detach().cpu().numpy()
    intrinsic = intrinsic_advantages.detach().cpu().numpy()
    alive = alive_masks.detach().cpu().numpy() > 0
    if len(episode_tokens) != external.shape[0]:
        raise ValueError("Advantage samples and episode-token history differ")
    if intrinsic.shape != alive.shape:
        raise ValueError("Intrinsic advantages and alive masks must have the same shape")

    for time_index, tokens in enumerate(episode_tokens):
        if len(tokens) != external.shape[1]:
            raise ValueError("Episode tokens do not match the environment count")
        for env_id, token in enumerate(tokens):
            active = alive[time_index, env_id]
            active_count = int(active.sum())
            if active_count == 0:
                continue
            totals = totals_by_episode.setdefault(
                int(token), new_episode_advantage_totals()
            )
            external_value = float(external[time_index, env_id])
            intrinsic_values = intrinsic[time_index, env_id, active]
            weighted_intrinsic_values = intrinsic_coef * intrinsic_values
            totals["count"] += active_count
            totals["external_sum"] += external_value * active_count
            totals["intrinsic_sum"] += float(intrinsic_values.sum())
            totals["weighted_intrinsic_sum"] += float(weighted_intrinsic_values.sum())
            totals["external_abs_sum"] += abs(external_value) * active_count
            totals["intrinsic_abs_sum"] += float(np.abs(intrinsic_values).sum())
            totals["weighted_intrinsic_abs_sum"] += float(
                np.abs(weighted_intrinsic_values).sum()
            )


def episode_advantage_row(
    episode: int, environment_steps: int, totals: dict[str, float]
) -> dict[str, float] | None:
    """Convert one completed episode's raw-GAE totals into chart values."""
    count = totals["count"]
    if count <= 0:
        return None
    external_abs_mean = totals["external_abs_sum"] / count
    intrinsic_abs_mean = totals["intrinsic_abs_sum"] / count
    weighted_intrinsic_abs_mean = totals["weighted_intrinsic_abs_sum"] / count
    denominator = max(external_abs_mean, 1e-8)
    return {
        "episode": float(episode),
        "environment_steps": float(environment_steps),
        "external_advantage_mean": totals["external_sum"] / count,
        "intrinsic_advantage_mean": totals["intrinsic_sum"] / count,
        "weighted_intrinsic_advantage_mean": totals[
            "weighted_intrinsic_sum"
        ] / count,
        "external_advantage_abs_mean": external_abs_mean,
        "intrinsic_advantage_abs_mean": intrinsic_abs_mean,
        "weighted_intrinsic_advantage_abs_mean": weighted_intrinsic_abs_mean,
        "intrinsic_to_external_raw_ratio": intrinsic_abs_mean / denominator,
        "intrinsic_to_external_advantage_ratio": (
            weighted_intrinsic_abs_mean / denominator
        ),
    }


def pop_completed_episode_advantage_rows(
    totals_by_episode: dict[int, dict[str, float]],
    completed_metadata: dict[int, tuple[int, int]],
) -> list[dict[str, float]]:
    """Return chart rows once every transition of a finished episode is known."""
    rows = []
    for token, (episode, environment_steps) in list(completed_metadata.items()):
        totals = totals_by_episode.pop(token, None)
        if totals is not None:
            row = episode_advantage_row(episode, environment_steps, totals)
            if row is not None:
                rows.append(row)
        del completed_metadata[token]
    return rows


def intrinsic_reward_row(agent, environment_steps: int) -> dict[str, float] | None:
    """Return the raw mean over completed, valid goal-segment rewards."""
    if not hasattr(agent, "last_worker_mean_intrinsic_reward"):
        return None
    return {
        "environment_steps": float(environment_steps),
        "intrinsic_reward_mean": float(agent.last_worker_mean_intrinsic_reward),
    }


def intrinsic_alignment_row(
    agent, environment_steps: int, n_agents: int
) -> dict[str, float] | None:
    """Return one raw update row for manager alignment and each worker reward."""
    per_agent = getattr(agent, "last_per_agent_intrinsic_reward", None)
    if (
        not hasattr(agent, "last_manager_cos_sim")
        or per_agent is None
        or len(per_agent) != n_agents
    ):
        return None
    row = {
        "environment_steps": float(environment_steps),
        "manager_cos_sim": float(agent.last_manager_cos_sim),
    }
    row.update(
        {
            f"worker_{agent_id}_raw_intrinsic_reward": float(per_agent[agent_id])
            for agent_id in range(n_agents)
        }
    )
    return row


def gc_mrl_metrics_row(
    agent, environment_steps: int, n_agents: int
) -> dict[str, float] | None:
    """Return GC-MRL learner diagnostics for one completed PPO update."""
    per_agent = getattr(agent, "last_per_agent_intrinsic_reward", None)
    if (
        not hasattr(agent, "last_manager_actor_loss")
        or per_agent is None
        or len(per_agent) != n_agents
    ):
        return None
    row = {
        "environment_steps": float(environment_steps),
        "update": float(agent.update_count),
        "manager_actor_loss": float(agent.last_manager_actor_loss),
    }
    row.update(
        {
            f"worker_{agent_id}_cosine_similarity": float(per_agent[agent_id])
            for agent_id in range(n_agents)
        }
    )
    return row


def prepare_advantage_csv(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        existing_fields = reader.fieldnames or []
    normalized_rows = [
        {field: row.get(field, "nan") for field in ADVANTAGE_FIELDS} for row in rows
    ]
    if existing_fields != ADVANTAGE_FIELDS:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ADVANTAGE_FIELDS)
            writer.writeheader()
            writer.writerows(normalized_rows)
    return [
        {key: float(value) for key, value in row.items()}
        for row in normalized_rows
    ]


def read_intrinsic_reward_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def read_intrinsic_alignment_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def read_manager_segment_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def read_gc_mrl_metric_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def plot_intrinsic_rewards(rows: list[dict[str, float]], destination: Path) -> None:
    """Plot the raw intrinsic reward means exactly as recorded, without smoothing."""
    if not rows:
        return
    environment_steps = [row["environment_steps"] for row in rows]
    rewards = [row["intrinsic_reward_mean"] for row in rows]
    fig, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.plot(environment_steps, rewards, color="tab:purple", linewidth=0.7)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Raw goal-segment intrinsic reward per rollout update (no smoothing)")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Intrinsic reward mean")
    axis.grid(alpha=0.25)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_intrinsic_alignment(
    rows: list[dict[str, float]], destination: Path, n_agents: int
) -> None:
    """Plot raw manager segment alignment and raw reward for every worker."""
    if not rows:
        return
    environment_steps = [row["environment_steps"] for row in rows]
    figure, (manager_axis, worker_axis) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    manager_axis.plot(
        environment_steps,
        [row["manager_cos_sim"] for row in rows],
        color="tab:blue",
        linewidth=0.8,
        label="manager_cos_sim",
    )
    manager_axis.axhline(0.0, color="black", linewidth=0.8)
    manager_axis.set_ylim(-1.05, 1.05)
    manager_axis.set_title("Raw manager-goal alignment and per-worker segment reward")
    manager_axis.set_ylabel("Manager cosine")
    manager_axis.grid(alpha=0.25)
    manager_axis.legend(loc="best")

    colors = plt.get_cmap("tab10")
    for agent_id in range(n_agents):
        field = f"worker_{agent_id}_raw_intrinsic_reward"
        worker_axis.plot(
            environment_steps,
            [row[field] for row in rows],
            color=colors(agent_id % 10),
            linewidth=0.7,
            label=f"worker {agent_id}",
        )
    worker_axis.axhline(0.0, color="black", linewidth=0.8)
    worker_axis.set_xlabel("Environment steps")
    worker_axis.set_ylabel("Raw goal-segment intrinsic reward")
    worker_axis.grid(alpha=0.25)
    worker_axis.legend(loc="best", ncol=4, fontsize="small")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_manager_segments(
    rows: list[dict[str, float]], destination: Path, smoothing: float
) -> None:
    """Plot the smoothed manager return and goal-actor loss for environment zero."""
    environment_rows = [row for row in rows if int(row["env_id"]) == 0]
    if not environment_rows:
        return
    figure, (return_axis, actor_axis) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    steps = [row["environment_steps"] for row in environment_rows]
    return_axis.plot(
        steps,
        smooth([row["manager_return"] for row in environment_rows], smoothing),
        color="tab:blue",
        linewidth=1.5,
        label="env 0",
    )
    actor_axis.plot(
        steps,
        smooth([row["manager_actor_loss"] for row in environment_rows], smoothing),
        color="tab:orange",
        linewidth=1.5,
        label="env 0",
    )
    for axis in (return_axis, actor_axis):
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=4, fontsize="small")
    return_axis.set_title(
        f"Manager metrics for env 0 (EMA smoothing {smoothing:.2f})"
    )
    return_axis.set_ylabel("Manager return")
    actor_axis.set_xlabel("Environment steps")
    actor_axis.set_ylabel("Manager actor loss")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_gc_mrl_metrics(
    rows: list[dict[str, float]], destination: Path, n_agents: int
) -> None:
    """Plot raw GC-MRL manager actor loss and per-agent intrinsic rewards."""
    if not rows:
        return
    steps = [row["environment_steps"] for row in rows]
    figure, (manager_axis, intrinsic_axis) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    manager_axis.plot(
        steps,
        [row["manager_actor_loss"] for row in rows],
        color="tab:red",
        linewidth=0.8,
    )
    manager_axis.axhline(0.0, color="black", linewidth=0.8)
    manager_axis.set_title("GC-MRL raw learner diagnostics per PPO update")
    manager_axis.set_ylabel("Manager actor loss")
    manager_axis.grid(alpha=0.25)

    colors = plt.get_cmap("tab10")
    for agent_id in range(n_agents):
        intrinsic_axis.plot(
            steps,
            [row[f"worker_{agent_id}_cosine_similarity"] for row in rows],
            color=colors(agent_id % 10),
            linewidth=0.8,
            label=f"worker {agent_id}",
        )
    intrinsic_axis.axhline(0.0, color="black", linewidth=0.8)
    intrinsic_axis.set_xlabel("Environment steps")
    intrinsic_axis.set_ylabel("Worker cosine similarity")
    intrinsic_axis.grid(alpha=0.25)
    intrinsic_axis.legend(loc="best", ncol=4, fontsize="small")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream) or {}
    parser = argparse.ArgumentParser(description="Train a multi-agent algorithm on a SMAC battle map.")
    parser.add_argument("--map", default=defaults.get("map", "8m"))
    parser.add_argument("--algorithm", default=defaults.get("algorithm", "feudal"), choices=sorted(ALGORITHM_MODULES))
    parser.add_argument(
        "--total-steps", type=int, default=int(defaults.get("total_steps", 1_600_000))
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 1)))
    parser.add_argument("--difficulty", default=str(defaults.get("difficulty", "7")))
    parser.add_argument("--step-mul", type=int, default=int(defaults.get("step_mul", 8)))
    parser.add_argument(
        "--n-rollout-threads",
        type=int,
        default=int(defaults.get("n_rollout_threads", 1)),
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=int(defaults.get("rollout_steps", 200))
    )
    parser.add_argument(
        "--sc2-startup-batch-size",
        type=int,
        default=int(defaults.get("sc2_startup_batch_size", 2)),
    )
    parser.add_argument("--window", type=int, default=int(defaults.get("win_rate_window", 100)))
    parser.add_argument("--plot-every", type=int, default=int(defaults.get("plot_every", 25)))
    parser.add_argument(
        "--smoothing", type=float, default=float(defaults.get("chart_smoothing", 0.99))
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=int(defaults.get("checkpoint_every", 100))
    )
    run_dir = defaults.get("run_dir")
    resume = defaults.get("resume_checkpoint")
    parser.add_argument("--run-dir", type=Path, default=Path(run_dir) if run_dir else None)
    parser.add_argument("--resume", type=Path, default=Path(resume) if resume else None)
    parser.add_argument(
        "--save-replay", action="store_true", default=bool(defaults.get("save_replay", False))
    )
    return parser.parse_args()


def smooth(values, factor: float):
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        result.append(factor * result[-1] + (1.0 - factor) * float(value))
    return result


def plot_metrics(
    rows: list[dict[str, float]],
    destination: Path,
    window: int,
    smoothing: float,
    advantage_rows: list[dict[str, float]] | None = None,
) -> None:
    if not rows:
        return
    environment_steps = np.asarray([r["environment_steps"] for r in rows])
    series = [
        ("return", "Episode return"),
        ("episode_steps", "Steps per episode"),
        ("allied_kills", "Enemy units killed"),
        ("allied_survivors", "Allied units remaining"),
        ("rolling_win_rate", f"Win rate (last {window})"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, series):
        values = [r[key] for r in rows]
        axis.plot(
            environment_steps, values, linewidth=0.8, alpha=0.18, color="tab:blue"
        )
        axis.plot(
            environment_steps,
            smooth(values, smoothing),
            linewidth=1.8,
            color="tab:blue",
        )
        axis.set_title(label)
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.25)
    axes.flat[4].set_ylim(-0.02, 1.02)
    advantage_axis = axes.flat[5]
    episode_rows = [
        row for row in (advantage_rows or [])
        if np.isfinite(row.get("episode", np.nan))
    ]
    if episode_rows:
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
                row for row in episode_rows if np.isfinite(row.get(key, np.nan))
            ]
            if not valid_rows:
                continue
            advantage_steps = [row["episode"] for row in valid_rows]
            values = [row[key] for row in valid_rows]
            advantage_axis.plot(
                advantage_steps, values, linewidth=0.7, alpha=0.18, color=color
            )
            advantage_axis.plot(
                advantage_steps,
                smooth(values, smoothing),
                linewidth=1.8,
                color=color,
                label=label,
            )
        ratio_axis = advantage_axis.twinx()
        raw_ratio_rows = [
            row
            for row in episode_rows
            if np.isfinite(row.get("intrinsic_advantage_abs_mean", np.nan))
            and np.isfinite(row.get("external_advantage_abs_mean", np.nan))
            and row["external_advantage_abs_mean"] > 0
        ]
        if raw_ratio_rows:
            raw_ratios = []
            for row in raw_ratio_rows:
                ratio = row.get("intrinsic_to_external_raw_ratio", np.nan)
                if not np.isfinite(ratio):
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
            for row in episode_rows
            if np.isfinite(row.get("intrinsic_to_external_advantage_ratio", np.nan))
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
            ratio_axis.set_ylabel("Intrinsic / external")
            ratio_axis.legend(loc="upper right", fontsize=8)
        elif raw_ratio_rows:
            ratio_axis.set_ylabel("|A_int| / |A_ext| raw")
            ratio_axis.legend(loc="upper right", fontsize=8)
        advantage_axis.legend(loc="upper left", fontsize=8)
    else:
        advantage_axis.text(
            0.5, 0.5, "No advantage data", ha="center", va="center",
            transform=advantage_axis.transAxes,
        )
    advantage_axis.axhline(0.0, color="black", linewidth=0.8)
    advantage_axis.set_title("Episode-mean raw advantages and magnitude ratios")
    advantage_axis.set_xlabel("Completed episode")
    advantage_axis.grid(alpha=0.25)
    fig.suptitle(f"SMAC training - smoothing {smoothing:.2f}")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def checkpoint(
    agent,
    episode: int,
    env_steps: int,
    path: Path,
    rolling_win_rate: float | None = None,
) -> None:
    payload = {"episode": episode, "environment_steps": env_steps}
    if rolling_win_rate is not None:
        payload["rolling_win_rate"] = rolling_win_rate
    torch.save(agent.save_model(payload), path)


def run_parallel_haa2c(args, run_dir: Path, Algorithm) -> None:
    env_count = int(args.n_rollout_threads)
    if env_count < 2:
        raise ValueError("Parallel HAA2C requires at least two rollout threads")
    env_kwargs = [
        {
            "map_name": args.map,
            "seed": args.seed + rank * 1000,
            "difficulty": args.difficulty,
            "step_mul": args.step_mul,
            "reward_only_positive": True,
            "reward_scale": True,
            "reward_scale_rate": 20,
        }
        for rank in range(env_count)
    ]
    envs = SubprocSMACVecEnv(
        env_kwargs, startup_batch_size=args.sc2_startup_batch_size
    )
    csv_file = None
    advantage_file = None
    intrinsic_reward_file = None
    intrinsic_alignment_file = None
    manager_segment_file = None
    gc_mrl_metrics_file = None
    tensorboard_writer = None
    try:
        info = envs.get_env_info()
        agent = Algorithm(
            action_num=info["n_actions"],
            n_agents=info["n_agents"],
            state_dim=info["obs_shape"],
            state_g_dim=info["state_shape"],
            episode_limit=info["episode_limit"],
            rollout_steps=args.rollout_steps,
        )
        episode = 0
        env_steps = 0
        if args.resume:
            saved = torch.load(args.resume, map_location=agent.device, weights_only=False)
            agent.load_model(saved)
            episode = int(saved.get("episode", 0))
            env_steps = int(saved.get("environment_steps", 0))

        csv_path = run_dir / "metrics.csv"
        rows: list[dict[str, float]] = []
        wins: deque[int] = deque(maxlen=args.window)
        best_win_rate = -1.0
        csv_file = csv_path.open("a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
        if csv_path.stat().st_size == 0:
            csv_writer.writeheader()
        advantage_path = run_dir / "advantages.csv"
        advantage_rows = prepare_advantage_csv(advantage_path)
        advantage_file = advantage_path.open("a", newline="", encoding="utf-8")
        advantage_writer = csv.DictWriter(advantage_file, fieldnames=ADVANTAGE_FIELDS)
        if advantage_path.stat().st_size == 0:
            advantage_writer.writeheader()
        intrinsic_reward_path = run_dir / "intrinsic_rewards.csv"
        intrinsic_reward_rows = read_intrinsic_reward_rows(intrinsic_reward_path)
        intrinsic_reward_file = intrinsic_reward_path.open(
            "a", newline="", encoding="utf-8"
        )
        intrinsic_reward_writer = csv.DictWriter(
            intrinsic_reward_file, fieldnames=INTRINSIC_REWARD_FIELDS
        )
        if intrinsic_reward_path.stat().st_size == 0:
            intrinsic_reward_writer.writeheader()
        intrinsic_alignment_path = run_dir / "intrinsic_alignment.csv"
        intrinsic_alignment_rows = read_intrinsic_alignment_rows(
            intrinsic_alignment_path
        )
        intrinsic_alignment_file = intrinsic_alignment_path.open(
            "a", newline="", encoding="utf-8"
        )
        intrinsic_alignment_fields_ = intrinsic_alignment_fields(info["n_agents"])
        intrinsic_alignment_writer = csv.DictWriter(
            intrinsic_alignment_file, fieldnames=intrinsic_alignment_fields_
        )
        if intrinsic_alignment_path.stat().st_size == 0:
            intrinsic_alignment_writer.writeheader()
        manager_segment_path = run_dir / "manager_segments.csv"
        manager_segment_rows = read_manager_segment_rows(manager_segment_path)
        manager_segment_file = manager_segment_path.open(
            "a", newline="", encoding="utf-8"
        )
        manager_segment_writer = csv.DictWriter(
            manager_segment_file, fieldnames=MANAGER_SEGMENT_FIELDS
        )
        if manager_segment_path.stat().st_size == 0:
            manager_segment_writer.writeheader()
        if args.algorithm == "gc_mrl":
            gc_mrl_metrics_path = run_dir / "gc_mrl_metrics.csv"
            gc_mrl_metric_rows = read_gc_mrl_metric_rows(gc_mrl_metrics_path)
            gc_mrl_metrics_file = gc_mrl_metrics_path.open(
                "a", newline="", encoding="utf-8"
            )
            gc_mrl_metrics_writer = csv.DictWriter(
                gc_mrl_metrics_file,
                fieldnames=gc_mrl_metric_fields(info["n_agents"]),
            )
            if gc_mrl_metrics_path.stat().st_size == 0:
                gc_mrl_metrics_writer.writeheader()
            gc_mrl_last_logged_update = int(agent.update_count)
        tensorboard_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=10)

        observations, global_states, available = envs.reset()
        alive = available[:, :, 1:].any(axis=-1)
        episode_returns = np.zeros(env_count, dtype=np.float64)
        episode_lengths = np.zeros(env_count, dtype=np.int64)
        track_episode_advantages = args.algorithm == "feudal_haa2c"
        active_episode_tokens = np.arange(env_count, dtype=np.int64)
        next_episode_token = env_count
        advantage_episode_tokens: list[np.ndarray] = []
        episode_advantage_totals: dict[int, dict[str, float]] = {}
        completed_advantage_metadata: dict[int, tuple[int, int]] = {}

        print(
            f"{args.algorithm} parallel rollout: {env_count} environments x "
            f"{args.rollout_steps} steps = {env_count * args.rollout_steps} samples/update"
        )
        while env_steps < args.total_steps:
            actions = agent.sample_actions_batch(
                observations, global_states, alive, available
            )
            (
                rewards,
                dones,
                infos,
                next_observations,
                next_global_states,
                next_available,
            ) = envs.step(actions)
            next_alive = next_available[:, :, 1:].any(axis=-1)
            agent.store_transition_batch(
                rewards,
                dones,
                next_global_states=next_global_states,
                next_local_states=next_observations,
                next_alive_masks=next_alive,
            )
            if track_episode_advantages:
                advantage_episode_tokens.append(active_episode_tokens.copy())
            env_steps += env_count
            if (
                args.algorithm == "gc_mrl"
                and agent.update_count > gc_mrl_last_logged_update
            ):
                gc_mrl_row = gc_mrl_metrics_row(
                    agent, env_steps, info["n_agents"]
                )
                if gc_mrl_row is not None:
                    gc_mrl_metric_rows.append(gc_mrl_row)
                    gc_mrl_metrics_writer.writerow(gc_mrl_row)
                    tensorboard_writer.add_scalar(
                        "gc_mrl/manager_actor_loss",
                        gc_mrl_row["manager_actor_loss"],
                        env_steps,
                    )
                    for agent_id in range(info["n_agents"]):
                        tensorboard_writer.add_scalar(
                            f"gc_mrl/cosine_similarity_worker_{agent_id}",
                            gc_mrl_row[
                                f"worker_{agent_id}_cosine_similarity"
                            ],
                            env_steps,
                        )
                    intrinsic_text = " ".join(
                        f"w{agent_id}="
                        f"{gc_mrl_row[f'worker_{agent_id}_cosine_similarity']:+.5f}"
                        for agent_id in range(info["n_agents"])
                    )
                    print(
                        f"gc_mrl_update={int(gc_mrl_row['update']):5d} "
                        f"manager_actor_loss={gc_mrl_row['manager_actor_loss']:+.5f} "
                        f"cosine_similarity {intrinsic_text}"
                    )
                gc_mrl_last_logged_update = int(agent.update_count)
            episode_returns += rewards
            episode_lengths += 1
            for segment in getattr(agent, "last_manager_segment_metrics", []):
                manager_segment = {
                    "environment_steps": float(env_steps),
                    **segment,
                }
                manager_segment_rows.append(manager_segment)
                manager_segment_writer.writerow(manager_segment)
                print(
                    "manager_segment"
                    f" env={segment['env_id']:2d}"
                    f" segment_steps={segment['segment_steps']:2d}"
                    f" end_env_steps={env_steps:9d}"
                    f" return={segment['manager_return']:+.5f}"
                    f" actor_loss={segment['manager_actor_loss']:+.5f}"
                )

            done_indices = np.flatnonzero(dones).tolist()
            for env_id in done_indices:
                episode += 1
                final_info = infos[env_id]
                won = int(bool(final_info.get("battle_won", False)))
                kills = int(final_info.get("dead_enemies", 0))
                survivors = info["n_agents"] - int(final_info.get("dead_allies", 0))
                wins.append(won)
                rolling_win_rate = float(np.mean(wins))
                row = {
                    "episode": episode,
                    "environment_steps": env_steps,
                    "episode_steps": int(episode_lengths[env_id]),
                    "return": float(episode_returns[env_id]),
                    "allied_kills": kills,
                    "allied_survivors": survivors,
                    "won": won,
                    "rolling_win_rate": rolling_win_rate,
                }
                rows.append(row)
                csv_writer.writerow(row)
                write_tensorboard_metrics(tensorboard_writer, row)
                if track_episode_advantages:
                    completed_advantage_metadata[int(active_episode_tokens[env_id])] = (
                        episode,
                        env_steps,
                    )
                print(
                    f"episode={episode:6d} env={env_id:2d} return={row['return']:9.3f} "
                    f"steps={row['episode_steps']:3d} total_steps={env_steps:9d}/{args.total_steps} "
                    f"kills={kills:2d} survivors={survivors:2d} "
                    f"win_rate({args.window})={rolling_win_rate:.3f} "
                    f"updates={getattr(agent, 'update_count', getattr(agent, 'worker_update_count', 0))}"
                )
                if episode % args.plot_every == 0:
                    plot_metrics(
                        rows, run_dir / "training_metrics.png", args.window, args.smoothing,
                        advantage_rows,
                    )
                    plot_intrinsic_rewards(
                        intrinsic_reward_rows,
                        run_dir / "intrinsic_reward_raw.png",
                    )
                    plot_intrinsic_alignment(
                        intrinsic_alignment_rows,
                        run_dir / "intrinsic_alignment_raw.png",
                        info["n_agents"],
                    )
                    plot_manager_segments(
                        manager_segment_rows,
                        run_dir / "manager_segments_env0_ema099.png",
                        args.smoothing,
                    )
                    if args.algorithm == "gc_mrl":
                        plot_gc_mrl_metrics(
                            gc_mrl_metric_rows,
                            run_dir / "gc_mrl_metrics_raw.png",
                            info["n_agents"],
                        )
                if episode % args.checkpoint_every == 0:
                    checkpoint(
                        agent,
                        episode,
                        env_steps,
                        run_dir / "checkpoints" / f"episode_{episode}.pt",
                    )
                if len(wins) == args.window and rolling_win_rate > best_win_rate:
                    best_win_rate = rolling_win_rate
                    checkpoint(
                        agent,
                        episode,
                        env_steps,
                        run_dir / "checkpoints" / "best_win_rate.pt",
                        rolling_win_rate=rolling_win_rate,
                    )
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0
                if track_episode_advantages:
                    active_episode_tokens[env_id] = next_episode_token
                    next_episode_token += 1
            if done_indices:
                reset_snapshots = envs.reset_at(done_indices)
                for env_id, snapshot in reset_snapshots.items():
                    (
                        next_observations[env_id],
                        next_global_states[env_id],
                        next_available[env_id],
                    ) = snapshot

            observations = next_observations
            global_states = next_global_states
            available = next_available
            alive = available[:, :, 1:].any(axis=-1)
            rollout_buffer = (
                agent.buffer if hasattr(agent, "buffer") else agent.worker_buffer
            )
            if len(rollout_buffer) >= args.rollout_steps:
                agent.train(next_global_state=global_states)
                update_count = getattr(
                    agent, "update_count", getattr(agent, "worker_update_count", 0)
                )
                if track_episode_advantages:
                    accumulate_episode_advantages(
                        episode_advantage_totals,
                        advantage_episode_tokens,
                        agent.last_external_advantages,
                        agent.last_intrinsic_advantages,
                        agent.last_advantage_alive_masks,
                        agent.intrinsic_coef,
                    )
                    advantage_episode_tokens.clear()
                    for episode_row in pop_completed_episode_advantage_rows(
                        episode_advantage_totals, completed_advantage_metadata
                    ):
                        advantage_rows.append(episode_row)
                        advantage_writer.writerow(episode_row)
                        step = int(episode_row["environment_steps"])
                        tensorboard_writer.add_scalar(
                            "advantage/external_mean", episode_row["external_advantage_mean"], step
                        )
                        tensorboard_writer.add_scalar(
                            "advantage/intrinsic_mean", episode_row["intrinsic_advantage_mean"], step
                        )
                        tensorboard_writer.add_scalar(
                            "advantage/weighted_intrinsic_mean",
                            episode_row["weighted_intrinsic_advantage_mean"], step,
                        )
                        tensorboard_writer.add_scalar(
                            "advantage/raw_intrinsic_to_external_ratio",
                            episode_row["intrinsic_to_external_raw_ratio"], step,
                        )
                        tensorboard_writer.add_scalar(
                            "advantage/intrinsic_to_external_ratio",
                            episode_row["intrinsic_to_external_advantage_ratio"], step,
                        )
                update_intrinsic_reward = intrinsic_reward_row(agent, env_steps)
                if update_intrinsic_reward is not None:
                    intrinsic_reward_rows.append(update_intrinsic_reward)
                    intrinsic_reward_writer.writerow(update_intrinsic_reward)
                    tensorboard_writer.add_scalar(
                        "intrinsic/reward_mean",
                        update_intrinsic_reward["intrinsic_reward_mean"],
                        env_steps,
                    )
                update_intrinsic_alignment = intrinsic_alignment_row(
                    agent, env_steps, info["n_agents"]
                )
                if update_intrinsic_alignment is not None:
                    intrinsic_alignment_rows.append(update_intrinsic_alignment)
                    intrinsic_alignment_writer.writerow(update_intrinsic_alignment)
                    tensorboard_writer.add_scalar(
                        "manager/cos_sim",
                        update_intrinsic_alignment["manager_cos_sim"],
                        env_steps,
                    )
                    for agent_id in range(info["n_agents"]):
                        tensorboard_writer.add_scalar(
                            f"intrinsic/raw_reward_worker_{agent_id}",
                            update_intrinsic_alignment[
                                f"worker_{agent_id}_raw_intrinsic_reward"
                            ],
                            env_steps,
                        )
                actor_loss = getattr(
                    agent, "last_actor_loss", getattr(agent, "last_loss_worker", 0.0)
                )
                intrinsic_metrics = ""
                if hasattr(agent, "last_intrinsic_critic_loss"):
                    intrinsic_metrics = (
                        f" intrinsic_reward={agent.last_worker_mean_intrinsic_reward:.5f}"
                        f" intrinsic_critic_loss={agent.last_intrinsic_critic_loss:.5f}"
                    )
                clip_metrics = ""
                if hasattr(agent, "last_per_agent_ratio_clip_fraction"):
                    clip_metrics = (
                        " ratio_clip_fraction="
                        f"{np.mean(agent.last_per_agent_ratio_clip_fraction):.5f}"
                    )
                print(
                    f"update={update_count:5d} batch={env_count * args.rollout_steps:5d} "
                    f"actor_loss={actor_loss:.5f} "
                    f"critic_loss={agent.last_critic_loss:.5f} "
                    f"factor={agent.last_factor_mean:.5f}"
                    f"{clip_metrics}{intrinsic_metrics}"
                )
            csv_file.flush()
            advantage_file.flush()
            intrinsic_reward_file.flush()
            intrinsic_alignment_file.flush()
            manager_segment_file.flush()
            if gc_mrl_metrics_file is not None:
                gc_mrl_metrics_file.flush()

        agent.finalize_training(
            next_global_state=global_states,
            next_local_state=observations,
            next_alive_mask=alive,
        )
        if track_episode_advantages and advantage_episode_tokens:
            accumulate_episode_advantages(
                episode_advantage_totals,
                advantage_episode_tokens,
                agent.last_external_advantages,
                agent.last_intrinsic_advantages,
                agent.last_advantage_alive_masks,
                agent.intrinsic_coef,
            )
            advantage_episode_tokens.clear()
            for episode_row in pop_completed_episode_advantage_rows(
                episode_advantage_totals, completed_advantage_metadata
            ):
                advantage_rows.append(episode_row)
                advantage_writer.writerow(episode_row)
                step = int(episode_row["environment_steps"])
                tensorboard_writer.add_scalar(
                    "advantage/external_mean", episode_row["external_advantage_mean"], step
                )
                tensorboard_writer.add_scalar(
                    "advantage/intrinsic_mean", episode_row["intrinsic_advantage_mean"], step
                )
                tensorboard_writer.add_scalar(
                    "advantage/weighted_intrinsic_mean",
                    episode_row["weighted_intrinsic_advantage_mean"], step,
                )
                tensorboard_writer.add_scalar(
                    "advantage/raw_intrinsic_to_external_ratio",
                    episode_row["intrinsic_to_external_raw_ratio"], step,
                )
                tensorboard_writer.add_scalar(
                    "advantage/intrinsic_to_external_ratio",
                    episode_row["intrinsic_to_external_advantage_ratio"], step,
                )
        if (
            args.algorithm == "gc_mrl"
            and agent.update_count > gc_mrl_last_logged_update
        ):
            gc_mrl_row = gc_mrl_metrics_row(agent, env_steps, info["n_agents"])
            if gc_mrl_row is not None:
                gc_mrl_metric_rows.append(gc_mrl_row)
                gc_mrl_metrics_writer.writerow(gc_mrl_row)
            gc_mrl_last_logged_update = int(agent.update_count)
        plot_metrics(
            rows, run_dir / "training_metrics.png", args.window, args.smoothing,
            advantage_rows,
        )
        plot_intrinsic_rewards(
            intrinsic_reward_rows, run_dir / "intrinsic_reward_raw.png"
        )
        plot_intrinsic_alignment(
            intrinsic_alignment_rows,
            run_dir / "intrinsic_alignment_raw.png",
            info["n_agents"],
        )
        plot_manager_segments(
            manager_segment_rows,
            run_dir / "manager_segments_env0_ema099.png",
            args.smoothing,
        )
        if args.algorithm == "gc_mrl":
            plot_gc_mrl_metrics(
                gc_mrl_metric_rows,
                run_dir / "gc_mrl_metrics_raw.png",
                info["n_agents"],
            )
        checkpoint(agent, episode, env_steps, run_dir / "checkpoints" / "final.pt")
        if args.save_replay:
            envs.save_replay()
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        if csv_file is not None:
            csv_file.close()
        if advantage_file is not None:
            advantage_file.close()
        if intrinsic_reward_file is not None:
            intrinsic_reward_file.close()
        if intrinsic_alignment_file is not None:
            intrinsic_alignment_file.close()
        if manager_segment_file is not None:
            manager_segment_file.close()
        if gc_mrl_metrics_file is not None:
            gc_mrl_metrics_file.close()
        envs.close()


def run_parallel_independent(args, run_dir: Path, Algorithm) -> None:
    """Run algorithms without a vectorized policy API in parallel SMAC processes.

    Each replica owns its recurrent state, replay/rollout buffer and optimizer.  This
    is deliberately different from ``run_parallel_haa2c``: sharing a single legacy
    FeUdal or QMIX instance would interleave their recurrent hidden states and make
    the resulting trajectories invalid.  Checkpoints are therefore written once per
    replica under ``checkpoints/replica_<id>_*.pt``.
    """
    env_count = int(args.n_rollout_threads)
    env_kwargs = [
        {
            "map_name": args.map,
            "seed": args.seed + rank * 1000,
            "difficulty": args.difficulty,
            "step_mul": args.step_mul,
            "reward_only_positive": True,
            "reward_scale": True,
            "reward_scale_rate": 20,
        }
        for rank in range(env_count)
    ]
    envs = SubprocSMACVecEnv(env_kwargs, startup_batch_size=args.sc2_startup_batch_size)
    csv_file = None
    tensorboard_writer = None
    try:
        info = envs.get_env_info()
        agents = [
            Algorithm(
                action_num=info["n_actions"],
                n_agents=info["n_agents"],
                state_dim=info["obs_shape"],
                state_g_dim=info["state_shape"],
                episode_limit=info["episode_limit"],
                rollout_steps=args.rollout_steps,
            )
            for _ in range(env_count)
        ]
        episode = 0
        env_steps = 0
        if args.resume:
            saved = torch.load(args.resume, map_location=agents[0].device, weights_only=False)
            for agent in agents:
                agent.load_model(saved)
            episode = int(saved.get("episode", 0))
            env_steps = int(saved.get("environment_steps", 0))

        csv_path = run_dir / "metrics.csv"
        rows: list[dict[str, float]] = []
        wins: deque[int] = deque(maxlen=args.window)
        best_win_rate = -1.0
        csv_file = csv_path.open("a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
        if csv_path.stat().st_size == 0:
            csv_writer.writeheader()
        tensorboard_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=10)

        observations, global_states, available = envs.reset()
        alive = available[:, :, 1:].any(axis=-1)
        episode_returns = np.zeros(env_count, dtype=np.float64)
        episode_lengths = np.zeros(env_count, dtype=np.int64)
        actions = []
        for env_id, agent in enumerate(agents):
            agent.episode_reset()
            action, _ = agent.sample_action(
                observations[env_id], global_states[env_id], None, None, False,
                alive[env_id], avail_actions=available[env_id],
            )
            actions.append(action)
        actions = np.asarray(actions, dtype=np.int64)

        print(
            f"{args.algorithm} parallel replicas: {env_count} independent environments "
            f"(one model/checkpoint per replica)"
        )
        while env_steps < args.total_steps:
            rewards, dones, infos, next_observations, next_global_states, next_available = envs.step(actions)
            next_alive = next_available[:, :, 1:].any(axis=-1)
            env_steps += env_count
            episode_returns += rewards
            episode_lengths += 1
            next_actions = np.zeros_like(actions)
            done_indices = np.flatnonzero(dones).tolist()

            for env_id, agent in enumerate(agents):
                action, _ = agent.sample_action(
                    next_observations[env_id], next_global_states[env_id],
                    float(rewards[env_id]), [float(rewards[env_id])] * info["n_agents"],
                    bool(dones[env_id]), next_alive[env_id], avail_actions=next_available[env_id],
                )
                next_actions[env_id] = action

            for env_id in done_indices:
                episode += 1
                final_info = infos[env_id]
                won = int(bool(final_info.get("battle_won", False)))
                kills = int(final_info.get("dead_enemies", 0))
                survivors = info["n_agents"] - int(final_info.get("dead_allies", 0))
                wins.append(won)
                rolling_win_rate = float(np.mean(wins))
                row = {
                    "episode": episode, "environment_steps": env_steps,
                    "episode_steps": int(episode_lengths[env_id]),
                    "return": float(episode_returns[env_id]), "allied_kills": kills,
                    "allied_survivors": survivors, "won": won,
                    "rolling_win_rate": rolling_win_rate,
                }
                rows.append(row)
                csv_writer.writerow(row)
                write_tensorboard_metrics(tensorboard_writer, row)
                print(
                    f"episode={episode:6d} replica={env_id:2d} return={row['return']:9.3f} "
                    f"steps={row['episode_steps']:3d} total_steps={env_steps:9d}/{args.total_steps} "
                    f"kills={kills:2d} survivors={survivors:2d} win_rate({args.window})={rolling_win_rate:.3f}"
                )
                if episode % args.plot_every == 0:
                    plot_metrics(rows, run_dir / "training_metrics.png", args.window, args.smoothing)
                if episode % args.checkpoint_every == 0:
                    for replica_id, replica in enumerate(agents):
                        checkpoint(replica, episode, env_steps, run_dir / "checkpoints" / f"replica_{replica_id}_episode_{episode}.pt")
                if len(wins) == args.window and rolling_win_rate > best_win_rate:
                    best_win_rate = rolling_win_rate
                    for replica_id, replica in enumerate(agents):
                        checkpoint(replica, episode, env_steps, run_dir / "checkpoints" / f"replica_{replica_id}_best_win_rate.pt", rolling_win_rate)
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0

            if done_indices:
                reset_snapshots = envs.reset_at(done_indices)
                for env_id, snapshot in reset_snapshots.items():
                    next_observations[env_id], next_global_states[env_id], next_available[env_id] = snapshot
                    next_alive[env_id] = next_available[env_id, :, 1:].any(axis=-1)
                    agents[env_id].episode_reset()
                    action, _ = agents[env_id].sample_action(
                        next_observations[env_id], next_global_states[env_id], None, None, False,
                        next_alive[env_id], avail_actions=next_available[env_id],
                    )
                    next_actions[env_id] = action

            observations, global_states, available, alive, actions = (
                next_observations, next_global_states, next_available, next_alive, next_actions
            )
            csv_file.flush()

        plot_metrics(rows, run_dir / "training_metrics.png", args.window, args.smoothing)
        for replica_id, replica in enumerate(agents):
            if hasattr(replica, "finalize_training"):
                replica.finalize_training()
            checkpoint(replica, episode, env_steps, run_dir / "checkpoints" / f"replica_{replica_id}_final.pt")
        if args.save_replay:
            envs.save_replay()
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        if csv_file is not None:
            csv_file.close()
        envs.close()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path("E:/SC2-runs") if Path("E:/").exists() else PROJECT_ROOT / "runs"
    run_dir = args.run_dir or (output_root / f"{datetime.now():%Y%m%d_%H%M%S}_{args.algorithm}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    Algorithm = load_algorithm_class(args.algorithm)
    if args.n_rollout_threads > 1:
        if args.algorithm in {"feudal", "gc_mrl", "haa2c", "feudal_haa2c"}:
            run_parallel_haa2c(args, run_dir, Algorithm)
        else:
            run_parallel_independent(args, run_dir, Algorithm)
        return
    env = StarCraft2Env(
        map_name=args.map,
        seed=args.seed,
        difficulty=args.difficulty,
        step_mul=args.step_mul,
        reward_only_positive=True,
        reward_scale=True,
        reward_scale_rate=20,
    )
    info = env.get_env_info()
    agent = Algorithm(
        action_num=info["n_actions"],
        n_agents=info["n_agents"],
        state_dim=info["obs_shape"],
        # Centralized components consume the full SMAC global state.
        state_g_dim=info["state_shape"],
        episode_limit=info["episode_limit"],
        rollout_steps=args.rollout_steps,
    )
    start_episode = 1
    env_steps = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=agent.device, weights_only=False)
        agent.load_model(saved)
        start_episode = int(saved.get("episode", 0)) + 1
        env_steps = int(saved.get("environment_steps", 0))

    csv_path = run_dir / "metrics.csv"
    rows: list[dict[str, float]] = []
    wins: deque[int] = deque(maxlen=args.window)
    best_win_rate = -1.0
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
    if csv_path.stat().st_size == 0:
        csv_writer.writeheader()
    tensorboard_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=10)

    try:
        episode = start_episode - 1
        while env_steps < args.total_steps:
            episode += 1
            episode_start_steps = env_steps
            env.reset()
            agent.episode_reset()
            terminated = False
            episode_return = 0.0
            final_info: dict = {}

            obs = np.asarray(env.get_obs(), dtype=np.float32)
            state = np.asarray(env.get_state(), dtype=np.float32)
            avail = np.asarray(env.get_avail_actions(), dtype=np.bool_)
            alive = avail[:, 1:].any(axis=1)
            actions, _ = agent.sample_action(
                local_state=obs,
                global_state=state,
                reward_ext_m=None,
                reward_ext_w=None,
                done=False,
                alive_mask=alive,
                avail_actions=avail,
            )

            while not terminated:
                reward, terminated, final_info = env.step(actions)
                env_steps += 1
                episode_return += float(reward)

                next_obs = np.asarray(env.get_obs(), dtype=np.float32)
                next_state = np.asarray(env.get_state(), dtype=np.float32)
                next_avail = np.asarray(env.get_avail_actions(), dtype=np.bool_)
                next_alive = next_avail[:, 1:].any(axis=1)
                actions, _ = agent.sample_action(
                    local_state=next_obs,
                    global_state=next_state,
                    reward_ext_m=float(reward),
                    reward_ext_w=[float(reward)] * info["n_agents"],
                    done=terminated,
                    alive_mask=next_alive,
                    avail_actions=next_avail,
                )

            won = int(bool(final_info.get("battle_won", False)))
            kills = int(final_info.get("dead_enemies", 0))
            survivors = info["n_agents"] - int(final_info.get("dead_allies", 0))
            wins.append(won)
            rolling_win_rate = float(np.mean(wins))
            row = {
                "episode": episode,
                "environment_steps": env_steps,
                "episode_steps": env_steps - episode_start_steps,
                "return": episode_return,
                "allied_kills": kills,
                "allied_survivors": survivors,
                "won": won,
                "rolling_win_rate": rolling_win_rate,
            }
            rows.append(row)
            csv_writer.writerow(row)
            write_tensorboard_metrics(tensorboard_writer, row)
            csv_file.flush()

            print(
                f"episode={episode:6d} return={episode_return:9.3f} "
                f"steps={row['episode_steps']:3d} "
                f"total_steps={env_steps:9d}/{args.total_steps} "
                f"kills={kills:2d} survivors={survivors:2d} "
                f"win_rate({args.window})={rolling_win_rate:.3f}"
            )
            if episode % args.plot_every == 0:
                plot_metrics(
                    rows, run_dir / "training_metrics.png", args.window, args.smoothing
                )
            if episode % args.checkpoint_every == 0:
                checkpoint(agent, episode, env_steps, run_dir / "checkpoints" / f"episode_{episode}.pt")
            if len(wins) == args.window and rolling_win_rate > best_win_rate:
                best_win_rate = rolling_win_rate
                checkpoint(
                    agent,
                    episode,
                    env_steps,
                    run_dir / "checkpoints" / "best_win_rate.pt",
                    rolling_win_rate=rolling_win_rate,
                )

        if hasattr(agent, "finalize_training"):
            agent.finalize_training()
        plot_metrics(rows, run_dir / "training_metrics.png", args.window, args.smoothing)
        checkpoint(agent, episode, env_steps, run_dir / "checkpoints" / "final.pt")
        if args.save_replay:
            env.save_replay()
    finally:
        tensorboard_writer.close()
        csv_file.close()
        env.close()


if __name__ == "__main__":
    main()
