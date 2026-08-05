from __future__ import annotations

import multiprocessing as mp
import time
import traceback

import numpy as np


def _snapshot(env):
    observations = np.asarray(env.get_obs(), dtype=np.float32)
    global_state = np.asarray(env.get_state(), dtype=np.float32)
    available_actions = np.asarray(env.get_avail_actions(), dtype=np.bool_)
    return observations, global_state, available_actions


def _worker(remote, parent_remote, env_kwargs):
    parent_remote.close()
    env = None
    try:
        from smac.env import StarCraft2Env

        env = StarCraft2Env(**env_kwargs)
        while True:
            command, payload = remote.recv()
            if command == "get_env_info":
                result = env.get_env_info()
            elif command == "reset":
                last_error = None
                for attempt in range(3):
                    try:
                        env.reset()
                        last_error = None
                        break
                    except Exception:
                        last_error = traceback.format_exc()
                        try:
                            env.close()
                        except Exception:
                            pass
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            env = StarCraft2Env(**env_kwargs)
                if last_error is not None:
                    raise RuntimeError(
                        "SC2 failed to start after 3 attempts:\n" + last_error
                    )
                result = _snapshot(env)
            elif command == "step":
                reward, terminated, info = env.step(payload)
                result = (float(reward), bool(terminated), info, *_snapshot(env))
            elif command == "save_replay":
                env.save_replay()
                result = None
            elif command == "close":
                remote.send(("ok", None))
                break
            else:
                raise ValueError(f"Unknown vector-environment command: {command}")
            remote.send(("ok", result))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        remote.close()


class SubprocSMACVecEnv:
    """Persistent SMAC environments running in independent spawned processes."""

    def __init__(self, env_kwargs: list[dict], startup_batch_size: int = 2):
        if not env_kwargs:
            raise ValueError("At least one environment is required")
        if startup_batch_size < 1:
            raise ValueError("startup_batch_size must be at least one")
        context = mp.get_context("spawn")
        pipes = [context.Pipe() for _ in env_kwargs]
        self.remotes = [pair[0] for pair in pipes]
        worker_remotes = [pair[1] for pair in pipes]
        self.processes = []
        self.startup_batch_size = int(startup_batch_size)
        self.closed = False
        for remote, worker_remote, kwargs in zip(
            self.remotes, worker_remotes, env_kwargs
        ):
            process = context.Process(
                target=_worker,
                args=(worker_remote, remote, kwargs),
                daemon=True,
            )
            process.start()
            worker_remote.close()
            self.processes.append(process)

    def _receive(self, remote):
        status, payload = remote.recv()
        if status == "error":
            raise RuntimeError(f"SMAC worker failed:\n{payload}")
        return payload

    def get_env_info(self):
        self.remotes[0].send(("get_env_info", None))
        return self._receive(self.remotes[0])

    def reset(self):
        snapshots = []
        for start in range(0, len(self.remotes), self.startup_batch_size):
            batch = self.remotes[start : start + self.startup_batch_size]
            for remote in batch:
                remote.send(("reset", None))
            snapshots.extend(self._receive(remote) for remote in batch)
        return self._stack(snapshots)

    def reset_at(self, indices):
        indices = list(indices)
        for index in indices:
            self.remotes[index].send(("reset", None))
        return {
            index: self._receive(self.remotes[index]) for index in indices
        }

    def step(self, actions):
        for remote, env_actions in zip(self.remotes, actions):
            remote.send(("step", env_actions.tolist()))
        results = [self._receive(remote) for remote in self.remotes]
        rewards, dones, infos, observations, states, available = zip(*results)
        return (
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            list(infos),
            np.stack(observations),
            np.stack(states),
            np.stack(available),
        )

    @staticmethod
    def _stack(snapshots):
        observations, states, available = zip(*snapshots)
        return np.stack(observations), np.stack(states), np.stack(available)

    def save_replay(self):
        for remote in self.remotes:
            remote.send(("save_replay", None))
        for remote in self.remotes:
            self._receive(remote)

    def close(self):
        if self.closed:
            return
        for remote, process in zip(self.remotes, self.processes):
            if process.is_alive():
                remote.send(("close", None))
        for remote, process in zip(self.remotes, self.processes):
            if process.is_alive():
                try:
                    self._receive(remote)
                except (EOFError, BrokenPipeError):
                    pass
            remote.close()
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.closed = True
