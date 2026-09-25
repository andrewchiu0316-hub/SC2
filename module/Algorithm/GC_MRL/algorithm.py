"""GC-MRL: the EPyMARL ``mappo_ns_feudal`` algorithm for the SC2 runner.

The existing project runner has a small, algorithm-agnostic API instead of
EPyMARL's ``EpisodeBatch`` API.  This adapter keeps the original algorithm
unchanged at the learning boundary: it stores complete episodes, pads a batch
of four, and applies EPyMARL's non-shared worker PPO plus deterministic
cosine-manager objective.
"""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from module.Algorithm.base_algorithm import BaseAlgorithm
from module.Algorithm.GC_MRL.model import DeterministicManager, NonSharedWorkers
from Tools import utils


class RunningMeanStd:
    """Exact running-moment update used by epymarl's reward standardiser."""

    def __init__(self, shape: tuple[int, ...], device: torch.device, epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = float(epsilon)

    def update(self, values: torch.Tensor) -> None:
        values = values.reshape(-1, values.shape[-1])
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].to(self.mean.device)
        self.var = state["var"].to(self.var.device)
        self.count = float(state["count"])


class Algorithm(BaseAlgorithm):
    """Standalone SC2 adapter for the GC-MRL configuration in epymarl."""

    def __init__(self, action_num: int, **kwargs):
        super().__init__()
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        self.cfg = utils.load_config(config_path)
        self.args = SimpleNamespace(**self.cfg)

        self.n_actions = int(action_num)
        self.n_agents = int(kwargs["n_agents"])
        self.obs_dim = int(kwargs["state_dim"])
        self.state_dim = int(kwargs["state_g_dim"])
        self.goal_dim = self.obs_dim  # ``goal_dim_mode: obs`` in the EPyMARL config.
        self.batch_size = int(self.cfg["batch_size"])
        self.c_step = int(self.cfg["c_step"])
        self.gamma = float(self.cfg["gamma"])
        self.episode_limit = int(kwargs.get("episode_limit", 0))
        self.rollout_steps = int(kwargs.get("rollout_steps", 0))

        self.workers = NonSharedWorkers(
            self.n_agents,
            self.obs_dim + self.goal_dim,
            int(self.cfg["hidden_dim"]),
            self.n_actions,
            bool(self.cfg["use_rnn"]),
        ).to(self.device)
        self.manager = DeterministicManager(
            self.state_dim,
            self.n_agents,
            self.goal_dim,
            int(self.cfg["hidden_dim"]),
            bool(self.cfg["goal_tanh"]),
            float(self.cfg["goal_scale"]),
        ).to(self.device)
        self.old_workers = copy.deepcopy(self.workers).to(self.device)
        self.old_manager = copy.deepcopy(self.manager).to(self.device)
        self.params = list(self.workers.parameters()) + list(self.manager.parameters())
        self.optimiser = torch.optim.Adam(self.params, lr=float(self.cfg["lr"]))
        self.rew_ms = RunningMeanStd((1,), self.device)

        # Completed episodes are the equivalent of EPyMARL's replay buffer.
        self.completed_episodes: list[dict] = []
        self.buffer = self.completed_episodes
        self._last_step_cache: dict | None = None
        self._episodes: list[dict] = []
        self._times: torch.Tensor | None = None
        self._goals: torch.Tensor | None = None
        self._raw_goals: torch.Tensor | None = None
        self._worker_hidden: torch.Tensor | None = None

        self.update_count = 0
        self.just_updated = False
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_factor_mean = 1.0
        self.last_worker_actor_loss = 0.0
        self.last_worker_value_loss = 0.0
        self.last_manager_actor_loss = 0.0
        self.last_manager_value_loss = 0.0
        self.last_worker_intrinsic_mean = 0.0
        self.last_per_agent_intrinsic_reward = [0.0] * self.n_agents

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _new_episode(self) -> dict:
        return {
            "state": [], "obs": [], "actions": [], "avail": [],
            "goals": [], "raw_goals": [], "manager_mask": [], "reward": [],
            "final_state": None, "final_obs": None,
        }

    def _ensure_vector_state(self, env_count: int) -> None:
        if self._times is not None and self._times.numel() == env_count:
            return
        self._times = torch.zeros(env_count, dtype=torch.long, device=self.device)
        self._goals = torch.zeros(env_count, self.n_agents, self.goal_dim, device=self.device)
        self._raw_goals = torch.zeros_like(self._goals)
        self._worker_hidden = self.workers.initial_hidden(env_count, self.device)
        self._episodes = [self._new_episode() for _ in range(env_count)]
        self._last_step_cache = None

    def _reset_vector_envs(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        self._times[indices] = 0
        self._goals[indices] = 0
        self._raw_goals[indices] = 0
        self._worker_hidden[indices] = 0
        for env_id in indices.tolist():
            self._episodes[env_id] = self._new_episode()

    def _record_cached_transition(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_global_states: np.ndarray,
        next_local_states: np.ndarray,
    ) -> None:
        if self._last_step_cache is None:
            raise RuntimeError("sample_actions_batch must run before store_transition_batch")

        cache = self._last_step_cache
        for env_id in range(len(self._episodes)):
            episode = self._episodes[env_id]
            episode["state"].append(cache["state"][env_id].cpu())
            episode["obs"].append(cache["obs"][env_id].cpu())
            episode["actions"].append(cache["actions"][env_id].cpu())
            episode["avail"].append(cache["avail"][env_id].cpu())
            episode["goals"].append(cache["goals"][env_id].cpu())
            episode["raw_goals"].append(cache["raw_goals"][env_id].cpu())
            episode["manager_mask"].append(float(cache["manager_mask"][env_id].item()))
            episode["reward"].append(float(rewards[env_id]))

            if dones[env_id]:
                episode["final_state"] = torch.as_tensor(next_global_states[env_id], dtype=torch.float32)
                episode["final_obs"] = torch.as_tensor(next_local_states[env_id], dtype=torch.float32)
                self.completed_episodes.append(episode)

        self._times.add_(1)
        terminal = torch.as_tensor(dones, dtype=torch.bool, device=self.device).nonzero().flatten()
        self._reset_vector_envs(terminal)
        self._last_step_cache = None

        if len(self.completed_episodes) >= self.batch_size:
            with torch.enable_grad():
                self.train()

    def sample_actions_batch(
        self, local_states, global_states, alive_masks, available_actions, deterministic: bool = False
    ):
        del alive_masks  # EPyMARL's GC-MRL learner uses filled masks, not live-agent masks.
        obs = torch.as_tensor(np.asarray(local_states, dtype=np.float32), device=self.device)
        states = torch.as_tensor(np.asarray(global_states, dtype=np.float32), device=self.device)
        avail = torch.as_tensor(np.asarray(available_actions, dtype=np.bool_), device=self.device)
        if obs.ndim != 3 or states.ndim != 2:
            raise ValueError("Expected batched observations [env, agent, obs] and states [env, state]")
        self._ensure_vector_state(obs.shape[0])

        due = self._times.remainder(self.c_step).eq(0)
        with torch.no_grad():
            if due.any():
                manager_out = self.manager(states[due])
                self._goals[due] = manager_out["goal"]
                self._raw_goals[due] = manager_out["raw_goal"]

            worker_inputs = torch.cat([obs, self._goals], dim=-1)
            logits, _, hidden = self.workers(worker_inputs, self._worker_hidden)
            logits = logits.masked_fill(~avail, -1e10)
            pi = torch.softmax(logits, dim=-1)
            actions = pi.argmax(dim=-1) if deterministic else torch.distributions.Categorical(probs=pi).sample()

        self._worker_hidden = hidden.detach()
        self._last_step_cache = {
            "state": states.detach(), "obs": obs.detach(), "actions": actions.detach(),
            "avail": avail.detach(), "goals": self._goals.detach().clone(),
            "raw_goals": self._raw_goals.detach().clone(), "manager_mask": due.float().detach(),
        }
        return actions.cpu().numpy()

    def store_transition_batch(self, rewards, dones, **kwargs) -> None:
        next_global_states = kwargs.get("next_global_states")
        next_local_states = kwargs.get("next_local_states")
        if next_global_states is None or next_local_states is None:
            raise ValueError("GC-MRL needs next_global_states and next_local_states to build intrinsic rewards")
        reward_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
        done_array = np.asarray(dones, dtype=np.bool_).reshape(-1)
        self._record_cached_transition(
            reward_array,
            done_array,
            np.asarray(next_global_states, dtype=np.float32),
            np.asarray(next_local_states, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # EPyMARL PPOLearner objective
    # ------------------------------------------------------------------
    def _make_batch(self, episodes: list[dict]) -> dict[str, torch.Tensor]:
        # EPyMARL's EpisodeBatch is allocated to the environment's fixed episode
        # limit.  Preserve that padding so reward normalisation and masks have
        # the same values as ``mappo_ns_feudal``.
        max_t = max(self.episode_limit, max(len(episode["reward"]) for episode in episodes))
        batch_size = len(episodes)
        dev = self.device
        states = torch.zeros(batch_size, max_t + 1, self.state_dim, device=dev)
        obs = torch.zeros(batch_size, max_t + 1, self.n_agents, self.obs_dim, device=dev)
        actions = torch.zeros(batch_size, max_t, self.n_agents, dtype=torch.long, device=dev)
        avail = torch.zeros(batch_size, max_t, self.n_agents, self.n_actions, dtype=torch.bool, device=dev)
        goals = torch.zeros(batch_size, max_t, self.n_agents, self.goal_dim, device=dev)
        raw_goals = torch.zeros_like(goals)
        manager_mask = torch.zeros(batch_size, max_t, device=dev)
        rewards = torch.zeros(batch_size, max_t, 1, device=dev)
        filled = torch.zeros(batch_size, max_t, device=dev)

        for batch_id, episode in enumerate(episodes):
            length = len(episode["reward"])
            states[batch_id, :length] = torch.stack(episode["state"]).to(dev)
            states[batch_id, length] = episode["final_state"].to(dev)
            obs[batch_id, :length] = torch.stack(episode["obs"]).to(dev)
            obs[batch_id, length] = episode["final_obs"].to(dev)
            actions[batch_id, :length] = torch.stack(episode["actions"]).to(dev)
            avail[batch_id, :length] = torch.stack(episode["avail"]).to(dev)
            goals[batch_id, :length] = torch.stack(episode["goals"]).to(dev)
            raw_goals[batch_id, :length] = torch.stack(episode["raw_goals"]).to(dev)
            manager_mask[batch_id, :length] = torch.tensor(episode["manager_mask"], device=dev)
            rewards[batch_id, :length, 0] = torch.tensor(episode["reward"], device=dev)
            filled[batch_id, :length] = 1
        return {
            "state": states, "obs": obs, "actions": actions, "avail": avail,
            "goals": goals, "raw_goals": raw_goals, "manager_mask": manager_mask,
            "reward": rewards, "filled": filled,
        }

    def _worker_sequence(self, workers, obs, goals, avail):
        batch_size, time_steps = obs.shape[:2]
        hidden = workers.initial_hidden(batch_size, self.device)
        pis, values = [], []
        for time_id in range(time_steps):
            inputs = torch.cat([obs[:, time_id], goals[:, time_id]], dim=-1)
            logits, value, hidden = workers(inputs, hidden)
            logits = logits.masked_fill(~avail[:, time_id], -1e10)
            pis.append(torch.softmax(logits, dim=-1))
            values.append(value.squeeze(-1))
        return torch.stack(pis, dim=1), torch.stack(values, dim=1)

    def _build_returns(self, rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[:, 0])
        for time_id in reversed(range(rewards.shape[1])):
            running = rewards[:, time_id] + self.gamma * running * mask[:, time_id]
            returns[:, time_id] = running
        return returns

    def _masked_normalize(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum() + 1e-8
        mean = (values * mask).sum() / denom
        variance = (((values - mean) * mask).square()).sum() / denom
        return (values - mean) / (variance.sqrt() + 1e-8)

    def _manager_outputs(self, states: torch.Tensor):
        batch_size, time_steps = states.shape[:2]
        out = self.manager(states.reshape(-1, self.state_dim))
        return (
            out["goal"].view(batch_size, time_steps, self.n_agents, self.goal_dim),
            out["value"].view(batch_size, time_steps),
        )

    def _manager_targets_and_cos(self, batch, manager_rewards, values_full, goals_full):
        filled = batch["filled"]
        manager_mask = batch["manager_mask"]
        batch_size, time_steps = filled.shape
        targets = torch.zeros_like(manager_rewards)
        cosine = torch.zeros_like(manager_rewards)
        valid = manager_mask * filled
        for time_id in range(time_steps):
            if time_id % self.c_step:
                continue
            end_nominal = min(time_id + self.c_step, time_steps)
            max_length = end_nominal - time_id
            segment_filled = filled[:, time_id:end_nominal]
            valid_length = segment_filled.sum(dim=1).long().clamp(min=1)
            end_index = (time_id + valid_length).clamp(max=time_steps)
            discounts = torch.tensor(
                [self.gamma**offset for offset in range(max_length)], device=self.device
            ).view(1, max_length)
            ret = (manager_rewards[:, time_id:end_nominal] * segment_filled * discounts).sum(dim=1)
            can_bootstrap = (valid_length == max_length) & (end_index < time_steps)
            batch_index = torch.arange(batch_size, device=self.device)
            safe_index = end_index.clamp(max=time_steps - 1)
            bootstrap_mask = can_bootstrap.float() * filled[batch_index, safe_index]
            ret = ret + (self.gamma ** valid_length.float()) * values_full[batch_index, end_index].detach() * bootstrap_mask
            targets[:, time_id] = ret * valid[:, time_id]

            delta_obs = batch["obs"][batch_index, end_index] - batch["obs"][:, time_id]
            cos_per_agent = F.cosine_similarity(delta_obs, goals_full[:, time_id], dim=-1, eps=1e-8)
            cosine[:, time_id] = cos_per_agent.mean(dim=-1) * valid[:, time_id]
        return targets, cosine, valid

    def _train_batch(self, batch: dict[str, torch.Tensor]) -> float:
        rewards = batch["reward"]
        self.rew_ms.update(rewards)
        rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var.clamp(min=1e-6))
        rewards = rewards.expand(-1, -1, self.n_agents)
        agent_mask = batch["filled"].unsqueeze(-1).expand_as(rewards)
        manager_rewards = rewards.mean(dim=-1)
        intrinsic = F.cosine_similarity(batch["obs"][:, 1:] - batch["obs"][:, :-1], batch["goals"], dim=-1, eps=1e-8)
        intrinsic = intrinsic * agent_mask
        worker_rewards = rewards + float(self.cfg["worker_intrinsic_coef"]) * intrinsic

        with torch.no_grad():
            old_pi, _ = self._worker_sequence(
                self.old_workers, batch["obs"][:, :-1], batch["goals"], batch["avail"]
            )
            old_pi = torch.where(agent_mask.unsqueeze(-1).eq(0), torch.ones_like(old_pi), old_pi)
            old_taken = torch.gather(old_pi, 3, batch["actions"].unsqueeze(-1)).squeeze(-1)
            old_log_prob = torch.log(old_taken + 1e-10)

        total_loss = torch.zeros((), device=self.device)
        for _ in range(int(self.cfg["epochs"])):
            pi, worker_values = self._worker_sequence(
                self.workers, batch["obs"][:, :-1], batch["goals"], batch["avail"]
            )
            pi = torch.where(agent_mask.unsqueeze(-1).eq(0), torch.ones_like(pi), pi)
            taken = torch.gather(pi, 3, batch["actions"].unsqueeze(-1)).squeeze(-1)
            log_prob = torch.log(taken + 1e-10)
            worker_returns = self._build_returns(worker_rewards, agent_mask)
            worker_advantage = (worker_returns - worker_values).detach()
            worker_ratio = torch.exp(log_prob - old_log_prob.detach())
            worker_surrogate = torch.minimum(
                worker_ratio * worker_advantage,
                worker_ratio.clamp(1 - float(self.cfg["eps_clip"]), 1 + float(self.cfg["eps_clip"])) * worker_advantage,
            )
            worker_actor_loss = -(worker_surrogate * agent_mask).sum() / (agent_mask.sum() + 1e-8)
            worker_value_loss = ((worker_returns.detach() - worker_values).square() * agent_mask).sum() / (agent_mask.sum() + 1e-8)
            worker_entropy = -(pi * torch.log(pi + 1e-10)).sum(dim=-1)
            worker_entropy_loss = -(worker_entropy * agent_mask).sum() / (agent_mask.sum() + 1e-8)

            manager_goals, manager_values_full = self._manager_outputs(batch["state"])
            manager_values = manager_values_full[:, :-1]
            targets, manager_cos, manager_mask = self._manager_targets_and_cos(
                batch, manager_rewards, manager_values_full, manager_goals
            )
            manager_advantage = (targets - manager_values).detach()
            if self.cfg["manager_advantage_norm"]:
                manager_advantage = self._masked_normalize(manager_advantage, manager_mask)
            clip = self.cfg.get("manager_advantage_clip")
            if clip is not None:
                manager_advantage = manager_advantage.clamp(-float(clip), float(clip))
            manager_actor_loss = -(manager_advantage * manager_cos * manager_mask).sum() / (manager_mask.sum() + 1e-8)
            manager_value_loss = ((targets.detach() - manager_values).square() * manager_mask).sum() / (manager_mask.sum() + 1e-8)

            total_loss = (
                worker_actor_loss
                + float(self.cfg["worker_value_coef"]) * worker_value_loss
                + float(self.cfg["entropy_coef"]) * worker_entropy_loss
                + float(self.cfg["manager_actor_coef"]) * manager_actor_loss
                + float(self.cfg["manager_value_coef"]) * manager_value_loss
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("GC-MRL produced a non-finite PPO loss")
            self.optimiser.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.params, float(self.cfg["grad_norm_clip"]))
            self.optimiser.step()

        self.old_workers.load_state_dict(self.workers.state_dict())
        self.old_manager.load_state_dict(self.manager.state_dict())
        self.update_count += 1
        self.just_updated = True
        self.last_worker_actor_loss = float(worker_actor_loss.item())
        self.last_worker_value_loss = float(worker_value_loss.item())
        self.last_manager_actor_loss = float(manager_actor_loss.item())
        self.last_manager_value_loss = float(manager_value_loss.item())
        self.last_actor_loss = self.last_worker_actor_loss + self.last_manager_actor_loss
        self.last_critic_loss = self.last_worker_value_loss + self.last_manager_value_loss
        self.last_factor_mean = 1.0
        self.last_worker_intrinsic_mean = float((intrinsic * agent_mask).sum().item() / (agent_mask.sum().item() + 1e-8))
        per_agent_count = agent_mask.sum(dim=(0, 1)).clamp_min(1.0)
        self.last_per_agent_intrinsic_reward = (
            (intrinsic * agent_mask).sum(dim=(0, 1)) / per_agent_count
        ).detach().cpu().tolist()
        return float(total_loss.item())

    def train(self, **kwargs):
        del kwargs
        if not self.completed_episodes:
            return None
        episode_count = min(self.batch_size, len(self.completed_episodes))
        episodes = self.completed_episodes[:episode_count]
        del self.completed_episodes[:episode_count]
        return self._train_batch(self._make_batch(episodes))

    # ------------------------------------------------------------------
    # Single-environment compatibility and lifecycle/checkpoint helpers
    # ------------------------------------------------------------------
    def sample_action(
        self, local_state, global_state, reward_ext_m, reward_ext_w, done,
        alive_mask, task_idx=None, avail_actions=None,
    ):
        del reward_ext_w, task_idx
        if avail_actions is None:
            raise ValueError("GC-MRL requires SMAC available-action masks")
        if reward_ext_m is not None and self._last_step_cache is not None:
            reward = float(np.asarray(reward_ext_m, dtype=np.float32).mean())
            self.store_transition_batch(
                np.asarray([reward]), np.asarray([done]),
                next_global_states=np.asarray(global_state, dtype=np.float32)[None],
                next_local_states=np.asarray(local_state, dtype=np.float32)[None],
            )
        if done:
            available = np.asarray(avail_actions, dtype=np.bool_)
            return available.argmax(axis=-1).tolist(), np.zeros((1, 1, self.n_agents, 1), dtype=np.int32)
        actions = self.sample_actions_batch(
            np.asarray(local_state, dtype=np.float32)[None],
            np.asarray(global_state, dtype=np.float32)[None],
            np.asarray(alive_mask, dtype=np.bool_)[None],
            np.asarray(avail_actions, dtype=np.bool_)[None],
        )
        return actions[0].tolist(), np.zeros((1, 1, self.n_agents, 1), dtype=np.int32)

    def episode_reset(self):
        if self._times is not None and self._times.numel() == 1:
            self._reset_vector_envs(torch.tensor([0], device=self.device))
        self._last_step_cache = None

    def finalize_training(self, **kwargs):
        del kwargs
        result = None
        while self.completed_episodes:
            with torch.enable_grad():
                result = self.train()
        return result

    def update_target_network(self):
        return None

    def save_model(self, checkpoint: dict):
        checkpoint.update({
            "gc_mrl_workers": self.workers.state_dict(),
            "gc_mrl_manager": self.manager.state_dict(),
            "gc_mrl_optimiser": self.optimiser.state_dict(),
            "gc_mrl_reward_stats": self.rew_ms.state_dict(),
            "gc_mrl_update_count": self.update_count,
        })
        return checkpoint

    def load_model(self, checkpoint: dict):
        self.workers.load_state_dict(checkpoint["gc_mrl_workers"])
        self.manager.load_state_dict(checkpoint["gc_mrl_manager"])
        self.old_workers.load_state_dict(self.workers.state_dict())
        self.old_manager.load_state_dict(self.manager.state_dict())
        if "gc_mrl_optimiser" in checkpoint:
            self.optimiser.load_state_dict(checkpoint["gc_mrl_optimiser"])
        if "gc_mrl_reward_stats" in checkpoint:
            self.rew_ms.load_state_dict(checkpoint["gc_mrl_reward_stats"])
        self.update_count = int(checkpoint.get("gc_mrl_update_count", 0))
        return self
