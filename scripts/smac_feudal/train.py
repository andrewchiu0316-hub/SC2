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


def write_tensorboard_metrics(writer: SummaryWriter, row: dict[str, float]) -> None:
    """Write only the four user-facing training metrics to TensorBoard."""
    step = int(row["episode"])
    writer.add_scalar("win_rate", float(row["rolling_win_rate"]), step)
    writer.add_scalar("return", float(row["return"]), step)
    writer.add_scalar("enemy_kills", float(row["allied_kills"]), step)
    writer.add_scalar("allied_survivors", float(row["allied_survivors"]), step)


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
    tensorboard_writer = None
    try:
        info = envs.get_env_info()
        agent = Algorithm(
            action_num=info["n_actions"],
            n_agents=info["n_agents"],
            state_dim=info["obs_shape"],
            state_g_dim=info["state_shape"],
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
        tensorboard_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=10)

        observations, global_states, available = envs.reset()
        alive = available[:, :, 1:].any(axis=-1)
        episode_returns = np.zeros(env_count, dtype=np.float64)
        episode_lengths = np.zeros(env_count, dtype=np.int64)

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
            env_steps += env_count
            episode_returns += rewards
            episode_lengths += 1

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
                print(
                    f"episode={episode:6d} env={env_id:2d} return={row['return']:9.3f} "
                    f"steps={row['episode_steps']:3d} total_steps={env_steps:9d}/{args.total_steps} "
                    f"kills={kills:2d} survivors={survivors:2d} "
                    f"win_rate({args.window})={rolling_win_rate:.3f} "
                    f"updates={getattr(agent, 'update_count', getattr(agent, 'worker_update_count', 0))}"
                )
                if episode % args.plot_every == 0:
                    plot_metrics(
                        rows, run_dir / "training_metrics.png", args.window, args.smoothing
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
                actor_loss = getattr(
                    agent, "last_actor_loss", getattr(agent, "last_loss_worker", 0.0)
                )
                print(
                    f"update={update_count:5d} batch={env_count * args.rollout_steps:5d} "
                    f"actor_loss={actor_loss:.5f} "
                    f"critic_loss={agent.last_critic_loss:.5f} "
                    f"factor={agent.last_factor_mean:.5f}"
                )
            csv_file.flush()

        agent.finalize_training(
            next_global_state=global_states,
            next_local_state=observations,
            next_alive_mask=alive,
        )
        plot_metrics(rows, run_dir / "training_metrics.png", args.window, args.smoothing)
        checkpoint(agent, episode, env_steps, run_dir / "checkpoints" / "final.pt")
        if args.save_replay:
            envs.save_replay()
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        if csv_file is not None:
            csv_file.close()
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
        if args.algorithm in {"feudal", "haa2c", "feudal_haa2c"}:
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
