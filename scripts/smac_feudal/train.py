from __future__ import annotations

import argparse
import csv
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("training_config.yaml")
for source_dir in (PROJECT_ROOT, PROJECT_ROOT / "vendor" / "smac", PROJECT_ROOT / "vendor" / "pysc2"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from module.Algorithm.FeUdal.algorithm import Algorithm
from smac.env import StarCraft2Env


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


def parse_args() -> argparse.Namespace:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream) or {}
    parser = argparse.ArgumentParser(description="Train the existing FeUdal agent on a SMAC battle map.")
    parser.add_argument("--map", default=defaults.get("map", "8m"))
    parser.add_argument(
        "--total-steps", type=int, default=int(defaults.get("total_steps", 1_600_000))
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 1)))
    parser.add_argument("--difficulty", default=str(defaults.get("difficulty", "7")))
    parser.add_argument("--step-mul", type=int, default=int(defaults.get("step_mul", 8)))
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
    rows: list[dict[str, float]], destination: Path, window: int, smoothing: float
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
    axes.flat[5].axis("off")
    fig.suptitle(f"FeUdal on SMAC — smoothing {smoothing:.2f}")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def checkpoint(agent: Algorithm, episode: int, env_steps: int, path: Path) -> None:
    payload = {"episode": episode, "environment_steps": env_steps}
    torch.save(agent.save_model(payload), path)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.run_dir or (
        PROJECT_ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    env = StarCraft2Env(
        map_name=args.map,
        seed=args.seed,
        difficulty=args.difficulty,
        step_mul=args.step_mul,
        reward_only_positive=False,
        reward_scale=False,
    )
    info = env.get_env_info()
    agent = Algorithm(
        action_num=info["n_actions"],
        n_agents=info["n_agents"],
        state_dim=info["obs_shape"],
        # The uploaded FeUdal manager requires concatenated local observations.
        state_g_dim=info["n_agents"] * info["obs_shape"],
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
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
    if csv_path.stat().st_size == 0:
        csv_writer.writeheader()

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
            avail = np.asarray(env.get_avail_actions(), dtype=np.bool_)
            alive = avail[:, 1:].any(axis=1)
            actions, _ = agent.sample_action(
                local_state=obs,
                global_state=obs.reshape(-1),
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
                next_avail = np.asarray(env.get_avail_actions(), dtype=np.bool_)
                next_alive = next_avail[:, 1:].any(axis=1)
                actions, _ = agent.sample_action(
                    local_state=next_obs,
                    global_state=next_obs.reshape(-1),
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

        plot_metrics(rows, run_dir / "training_metrics.png", args.window, args.smoothing)
        checkpoint(agent, episode, env_steps, run_dir / "checkpoints" / "final.pt")
        if args.save_replay:
            env.save_replay()
    finally:
        csv_file.close()
        env.close()


if __name__ == "__main__":
    main()
