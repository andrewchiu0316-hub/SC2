import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from module.Algorithm.base_algorithm import BaseAlgorithm
from module.Algorithm.HAA2C.model.model import CentralValueCritic, HAA2CActor, ValueNorm
from Tools import utils


class Algorithm(BaseAlgorithm):
    """HARL-style HAA2C with vectorized rollout collection."""

    def __init__(self, action_num: int, **kwargs):
        super().__init__()
        self.cfg = utils.load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        )
        self.args = SimpleNamespace(**self.cfg)
        self.action_num = int(action_num)
        self.n_agents = int(kwargs["n_agents"])
        self.obs_dim = int(kwargs["state_dim"])
        self.global_state_dim = int(kwargs["state_g_dim"])

        self.gamma = float(self.cfg["gamma"])
        self.gae_lambda = float(self.cfg["gae_lambda"])
        self.rollout_steps = int(kwargs.get("rollout_steps", self.cfg["rollout_steps"]))
        self.a2c_epoch = int(self.cfg["a2c_epoch"])
        self.critic_epoch = int(self.cfg["critic_epoch"])
        self.entropy_coef = float(self.cfg["entropy_coef"])
        self.value_loss_coef = float(self.cfg["value_loss_coef"])
        self.value_clip = float(self.cfg["value_clip"])
        self.huber_delta = float(self.cfg["huber_delta"])
        self.max_grad_norm = float(self.cfg["max_grad_norm"])
        self.fixed_order = bool(self.cfg.get("fixed_order", False))
        self.use_clipped_value_loss = bool(self.cfg.get("use_clipped_value_loss", True))
        self.use_value_norm = bool(self.cfg.get("use_value_norm", True))

        hidden_dim = int(self.cfg["hidden_dim"])
        self.actors = nn.ModuleList(
            [HAA2CActor(self.obs_dim, hidden_dim, self.action_num) for _ in range(self.n_agents)]
        ).to(self.device)
        self.critic = CentralValueCritic(self.global_state_dim, hidden_dim).to(self.device)
        self.value_normalizer = ValueNorm(
            beta=float(self.cfg.get("value_norm_beta", 0.99999))
        ).to(self.device)

        optimizer_eps = float(self.cfg["optimizer_eps"])
        self.actor_optimizers = [
            torch.optim.Adam(
                actor.parameters(), lr=float(self.cfg["actor_learning_rate"]), eps=optimizer_eps
            )
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.cfg["critic_learning_rate"]),
            eps=optimizer_eps,
        )

        self.buffer = []
        self._last_step_cache = None
        self.update_count = 0
        self.just_updated = False
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_factor_mean = 1.0
        self.last_sequential_order = list(range(self.n_agents))
        self.last_per_agent_policy_loss = [0.0] * self.n_agents
        self.last_per_agent_ratio = [1.0] * self.n_agents

    def _distribution(self, agent_id, observations, available_actions):
        logits = self.actors[agent_id](observations)
        logits = logits.masked_fill(~available_actions, -1e10)
        return torch.distributions.Categorical(logits=logits)

    def _raw_value(self, global_states):
        normalized = self.critic(global_states)
        if self.use_value_norm:
            return self.value_normalizer.denormalize(normalized)
        return normalized

    def sample_actions_batch(
        self, local_states, global_states, alive_masks, available_actions, deterministic=False
    ):
        observations = torch.as_tensor(
            np.asarray(local_states, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        states = torch.as_tensor(
            np.asarray(global_states, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        alive = torch.as_tensor(
            np.asarray(alive_masks, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        available = torch.as_tensor(
            np.asarray(available_actions, dtype=np.bool_), dtype=torch.bool, device=self.device
        )
        if observations.ndim != 3:
            raise ValueError("Batched observations must have shape [env, agent, feature]")

        actions = []
        log_probs = []
        with torch.no_grad():
            for agent_id in range(self.n_agents):
                distribution = self._distribution(
                    agent_id, observations[:, agent_id], available[:, agent_id]
                )
                action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
                actions.append(action)
                log_probs.append(distribution.log_prob(action))
            old_raw_values = self._raw_value(states)

        action_tensor = torch.stack(actions, dim=1)
        self._last_step_cache = {
            "obs": observations,
            "state": states,
            "actions": action_tensor,
            "old_log_probs": torch.stack(log_probs, dim=1),
            "old_raw_value": old_raw_values,
            "alive": alive,
            "available_actions": available,
        }
        return action_tensor.cpu().numpy()

    def store_transition_batch(self, rewards, dones, **kwargs):
        del kwargs
        if self._last_step_cache is None:
            raise RuntimeError("sample_actions_batch must be called before storing a transition")
        rewards_tensor = torch.as_tensor(
            np.asarray(rewards, dtype=np.float32), dtype=torch.float32, device=self.device
        ).reshape(-1)
        done_tensor = torch.as_tensor(
            np.asarray(dones, dtype=np.float32), dtype=torch.float32, device=self.device
        ).reshape(-1)
        if rewards_tensor.shape[0] != self._last_step_cache["obs"].shape[0]:
            raise ValueError("Reward batch size does not match the sampled action batch")
        self.buffer.append(
            {
                key: value.detach().clone()
                for key, value in self._last_step_cache.items()
            }
            | {"reward": rewards_tensor, "done": done_tensor}
        )
        self._last_step_cache = None

    def _compute_gae(self, rewards, old_values, done, next_global_states):
        with torch.no_grad():
            if next_global_states is None:
                bootstrap = torch.zeros_like(old_values[-1])
            else:
                bootstrap = self._raw_value(next_global_states)
                bootstrap = torch.where(done[-1] > 0.5, torch.zeros_like(bootstrap), bootstrap)
            next_values = torch.empty_like(old_values)
            next_values[:-1] = old_values[1:]
            next_values[-1] = bootstrap
            deltas = rewards + self.gamma * (1.0 - done) * next_values - old_values
            advantages = torch.zeros_like(rewards)
            running_gae = torch.zeros_like(rewards[-1])
            for step in reversed(range(len(rewards))):
                running_gae = deltas[step] + (
                    self.gamma * self.gae_lambda * (1.0 - done[step]) * running_gae
                )
                advantages[step] = running_gae
        return advantages, advantages + old_values

    def _train_actors(self, observations, actions, available, old_log_probs, advantages, alive):
        factor = torch.ones(len(observations), device=self.device)
        order = (
            list(range(self.n_agents))
            if self.fixed_order
            else torch.randperm(self.n_agents).cpu().tolist()
        )
        losses = [0.0] * self.n_agents
        ratios = [1.0] * self.n_agents

        for agent_id in order:
            active = alive[:, agent_id]
            active_count = active.sum()
            if active_count <= 0:
                continue
            actor_obs = observations[:, agent_id]
            actor_actions = actions[:, agent_id]
            actor_available = available[:, agent_id]
            with torch.no_grad():
                before_log_prob = self._distribution(
                    agent_id, actor_obs, actor_available
                ).log_prob(actor_actions)

            valid_advantages = advantages[active > 0]
            actor_advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std(unbiased=False) + 1e-5
            )
            for _ in range(self.a2c_epoch):
                distribution = self._distribution(agent_id, actor_obs, actor_available)
                log_prob = distribution.log_prob(actor_actions)
                importance_ratio = torch.exp(log_prob - old_log_probs[:, agent_id])
                policy_loss = -(
                    factor.detach() * importance_ratio * actor_advantages.detach() * active
                ).sum() / active_count
                entropy = (distribution.entropy() * active).sum() / active_count
                actor_loss = policy_loss - self.entropy_coef * entropy
                optimizer = self.actor_optimizers[agent_id]
                optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                optimizer.step()

            with torch.no_grad():
                after_log_prob = self._distribution(
                    agent_id, actor_obs, actor_available
                ).log_prob(actor_actions)
                update_ratio = torch.exp(after_log_prob - before_log_prob)
                update_ratio = torch.where(active > 0, update_ratio, torch.ones_like(update_ratio))
                factor.mul_(update_ratio)
                losses[agent_id] = float(policy_loss.item())
                ratios[agent_id] = float(update_ratio[active > 0].mean().item())

        self.last_sequential_order = order
        self.last_per_agent_policy_loss = losses
        self.last_per_agent_ratio = ratios
        self.last_factor_mean = float(factor.mean().item())
        valid_losses = [losses[i] for i in range(self.n_agents) if alive[:, i].sum() > 0]
        self.last_actor_loss = float(np.mean(valid_losses)) if valid_losses else 0.0

    def _huber_loss(self, error):
        absolute = error.abs()
        quadratic = torch.minimum(absolute, torch.tensor(self.huber_delta, device=error.device))
        return 0.5 * quadratic.square() + self.huber_delta * (absolute - quadratic)

    def _train_critic(self, global_states, old_raw_values, raw_returns):
        if self.use_value_norm:
            self.value_normalizer.update(raw_returns)
            returns = self.value_normalizer.normalize(raw_returns)
            old_values = self.value_normalizer.normalize(old_raw_values)
        else:
            returns = raw_returns
            old_values = old_raw_values

        value_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.critic_epoch):
            values = self.critic(global_states)
            original_loss = self._huber_loss(returns.detach() - values)
            if self.use_clipped_value_loss:
                clipped_values = old_values + (values - old_values).clamp(
                    -self.value_clip, self.value_clip
                )
                clipped_loss = self._huber_loss(returns.detach() - clipped_values)
                value_loss = torch.maximum(original_loss, clipped_loss).mean()
            else:
                value_loss = original_loss.mean()
            self.critic_optimizer.zero_grad()
            (self.value_loss_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        self.last_critic_loss = float(value_loss.item())

    def train(self, next_global_state=None, **kwargs):
        del kwargs
        if not self.buffer:
            return None
        observations = torch.stack([item["obs"] for item in self.buffer])
        global_states = torch.stack([item["state"] for item in self.buffer])
        actions = torch.stack([item["actions"] for item in self.buffer])
        old_log_probs = torch.stack([item["old_log_probs"] for item in self.buffer])
        old_raw_values = torch.stack([item["old_raw_value"] for item in self.buffer])
        rewards = torch.stack([item["reward"] for item in self.buffer])
        done = torch.stack([item["done"] for item in self.buffer])
        alive = torch.stack([item["alive"] for item in self.buffer])
        available = torch.stack([item["available_actions"] for item in self.buffer])
        next_states = None
        if next_global_state is not None:
            next_states = torch.as_tensor(
                np.asarray(next_global_state, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1, self.global_state_dim)

        advantages, returns = self._compute_gae(rewards, old_raw_values, done, next_states)
        time_steps, env_count = rewards.shape
        batch_size = time_steps * env_count
        self._train_actors(
            observations.reshape(batch_size, self.n_agents, self.obs_dim),
            actions.reshape(batch_size, self.n_agents),
            available.reshape(batch_size, self.n_agents, self.action_num),
            old_log_probs.reshape(batch_size, self.n_agents),
            advantages.reshape(batch_size),
            alive.reshape(batch_size, self.n_agents),
        )
        self._train_critic(
            global_states.reshape(batch_size, self.global_state_dim),
            old_raw_values.reshape(batch_size),
            returns.reshape(batch_size),
        )
        self.buffer = []
        self.update_count += 1
        self.just_updated = True
        return self.last_actor_loss + self.last_critic_loss

    def sample_action(
        self,
        local_state,
        global_state,
        reward_ext_m,
        reward_ext_w,
        done,
        alive_mask,
        task_idx=None,
        avail_actions=None,
    ):
        del reward_ext_w, task_idx
        self.just_updated = False
        if reward_ext_m is not None and self._last_step_cache is not None:
            reward = float(np.mean(reward_ext_m)) if isinstance(
                reward_ext_m, (list, tuple, np.ndarray)
            ) else float(reward_ext_m)
            self.store_transition_batch([reward], [done])
        if len(self.buffer) >= self.rollout_steps:
            with torch.enable_grad():
                self.train(next_global_state=np.asarray(global_state)[None])
        if done:
            available = np.asarray(avail_actions, dtype=np.bool_)
            return available.argmax(axis=-1).tolist(), np.zeros(
                (1, 1, self.n_agents, 1), dtype=np.int32
            )
        actions = self.sample_actions_batch(
            np.asarray(local_state)[None],
            np.asarray(global_state)[None],
            np.asarray(alive_mask)[None],
            np.asarray(avail_actions)[None],
        )
        return actions[0].tolist(), np.zeros((1, 1, self.n_agents, 1), dtype=np.int32)

    def episode_reset(self):
        self._last_step_cache = None

    def finalize_training(self, next_global_state=None, **kwargs):
        del kwargs
        if self.buffer:
            with torch.enable_grad():
                return self.train(next_global_state=next_global_state)
        return None

    def update_target_network(self):
        return

    def save_model(self, checkpoint):
        checkpoint["actors"] = self.actors.state_dict()
        checkpoint["critic"] = self.critic.state_dict()
        checkpoint["actor_optimizers"] = [
            optimizer.state_dict() for optimizer in self.actor_optimizers
        ]
        checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()
        checkpoint["value_normalizer"] = self.value_normalizer.state_dict()
        checkpoint["haa2c_update_count"] = self.update_count
        checkpoint["haa2c_rollout_steps"] = self.rollout_steps
        return checkpoint

    def load_model(self, checkpoint):
        self.actors.load_state_dict(checkpoint["actors"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_optimizers" in checkpoint:
            for optimizer, state in zip(self.actor_optimizers, checkpoint["actor_optimizers"]):
                optimizer.load_state_dict(state)
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "value_normalizer" in checkpoint:
            self.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
        self.update_count = int(checkpoint.get("haa2c_update_count", 0))
        return self
