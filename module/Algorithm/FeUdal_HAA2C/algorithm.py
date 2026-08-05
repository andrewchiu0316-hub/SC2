import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.Algorithm.base_algorithm import BaseAlgorithm
from module.Algorithm.FeUdal_HAA2C.model.model import (
    CentralValueCritic,
    FeudalManager,
    FeudalWorkerActor,
)
from Tools import utils


class Algorithm(BaseAlgorithm):
    """FeUdal hierarchy trained with HARL HAA2C-style worker updates."""

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
        self.manager_c_steps = int(self.cfg["manager_c_steps"])
        self.worker_update_steps = int(
            kwargs.get("rollout_steps", self.cfg["worker_update_steps"])
        )
        self.a2c_epoch = int(self.cfg["a2c_epoch"])
        self.critic_epoch = int(self.cfg["critic_epoch"])
        self.entropy_coef = float(self.cfg["entropy_coef"])
        self.value_loss_coef = float(self.cfg["value_loss_coef"])
        self.use_clipped_value_loss = bool(
            self.cfg.get("use_clipped_value_loss", True)
        )
        self.value_clip = float(self.cfg["value_clip"])
        self.huber_delta = float(self.cfg["huber_delta"])
        self.max_grad_norm = float(self.cfg["max_grad_norm"])
        self.fixed_order = bool(self.cfg.get("fixed_order", False))
        self.manager_advantage_clip = float(
            self.cfg["manager_advantage_clip"]
        )
        self.manager_value_loss_coef = float(
            self.cfg["manager_value_loss_coef"]
        )

        worker_hidden_dim = int(self.cfg["worker_hidden_dim"])
        manager_hidden_dim = int(self.cfg["manager_hidden_dim"])
        critic_hidden_dim = int(self.cfg["critic_hidden_dim"])
        self.goal_dim = worker_hidden_dim

        self.manager = FeudalManager(
            state_dim=self.global_state_dim,
            n_agents=self.n_agents,
            hidden_dim=manager_hidden_dim,
            goal_dim=self.goal_dim,
        ).to(self.device)
        self.workers = nn.ModuleList(
            [
                FeudalWorkerActor(
                    obs_dim=self.obs_dim,
                    goal_dim=self.goal_dim,
                    hidden_dim=worker_hidden_dim,
                    n_actions=self.action_num,
                )
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.critic = CentralValueCritic(
            self.global_state_dim, critic_hidden_dim
        ).to(self.device)

        optimizer_eps = float(self.cfg["optimizer_eps"])
        self.manager_optimizer = torch.optim.Adam(
            self.manager.parameters(),
            lr=float(self.cfg["manager_learning_rate"]),
            eps=optimizer_eps,
        )
        self.worker_optimizers = [
            torch.optim.Adam(
                worker.parameters(),
                lr=float(self.cfg["actor_learning_rate"]),
                eps=optimizer_eps,
            )
            for worker in self.workers
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.cfg["critic_learning_rate"]),
            eps=optimizer_eps,
        )

        self.manager_hidden = None
        self.worker_hidden = None
        self.manager_segment_start_hidden = None
        self.worker_rollout_start_hidden = None
        self.current_goal = None
        self.manager_buffer = []
        self.worker_buffer = []
        self._last_step_cache = None
        self.parallel_mode = False
        self.parallel_env_count = 0
        self.parallel_manager_hidden = None
        self.parallel_worker_hidden = None
        self.parallel_worker_rollout_start_hidden = None
        self.parallel_current_goal = None
        self.parallel_goal_remaining = None
        self.parallel_segments = None
        self._parallel_step_cache = None

        self.last_loss_worker = 0.0
        self.last_loss_manager = 0.0
        self.last_critic_loss = 0.0
        self.last_factor_mean = 1.0
        self.last_sequential_order = list(range(self.n_agents))
        self.last_per_agent_policy_loss = [0.0] * self.n_agents
        self.last_per_agent_ratio = [1.0] * self.n_agents
        self.last_manager_cos_sim = 0.0
        self.last_manager_advantage = 0.0
        self.last_manager_value_loss = 0.0
        self.last_manager_return = 0.0
        self.last_manager_value_pred = 0.0
        self.last_worker_mean_reward = 0.0
        self.worker_update_count = 0
        self.manager_update_count = 0
        self.just_updated = False
        self.episode_reset()

    @staticmethod
    def _detach_hidden(hidden):
        return tuple(part.detach().clone() for part in hidden)

    def _store_transition(
        self,
        cache,
        external_reward,
        done,
    ):
        transition = {
            "obs": cache["obs"].detach().clone(),
            "state": cache["state"].detach().clone(),
            "goal": cache["goal"].detach().clone(),
            "actions": cache["actions"].detach().clone(),
            "old_log_probs": cache["old_log_probs"].detach().clone(),
            "old_value": cache["old_value"].detach().clone(),
            "external_reward": torch.tensor(
                external_reward, dtype=torch.float32, device=self.device
            ),
            "done": bool(done),
            "alive": cache["alive"].detach().clone(),
            "available_actions": cache["available_actions"].detach().clone(),
        }
        self.manager_buffer.append(transition)
        self.worker_buffer.append(transition)

    def _evaluate_actor(
        self,
        agent_id,
        observations,
        goals,
        actions,
        available_actions,
        done,
    ):
        worker = self.workers[agent_id]
        hidden = self._detach_hidden(
            self.worker_rollout_start_hidden[agent_id]
        )
        chunk_starts = [0]
        chunk_starts.extend(
            index + 1
            for index in torch.nonzero(done[:-1] > 0.5).flatten().tolist()
        )
        chunk_ends = chunk_starts[1:] + [len(observations)]
        logits_chunks = []
        for chunk_id, (start, end) in enumerate(
            zip(chunk_starts, chunk_ends)
        ):
            if chunk_id > 0:
                hidden = worker.init_hidden(1, self.device)
            chunk_logits, hidden = worker(
                observations[start:end, agent_id].unsqueeze(1),
                goals[start:end, agent_id].unsqueeze(1),
                hidden,
            )
            logits_chunks.append(chunk_logits[:, 0])
        logits = torch.cat(logits_chunks, dim=0)
        logits = logits.masked_fill(
            ~available_actions[:, agent_id], -1e10
        )
        distribution = torch.distributions.Categorical(logits=logits)
        return (
            distribution.log_prob(actions[:, agent_id]),
            distribution.entropy(),
        )

    def _huber_loss(self, error):
        absolute = error.abs()
        quadratic = torch.minimum(
            absolute, torch.tensor(self.huber_delta, device=error.device)
        )
        linear = absolute - quadratic
        return 0.5 * quadratic.square() + self.huber_delta * linear

    def _compute_advantages(
        self,
        rewards,
        old_values,
        done,
        next_global_state,
    ):
        with torch.no_grad():
            if done[-1]:
                bootstrap_value = torch.tensor(0.0, device=self.device)
            else:
                next_state = torch.as_tensor(
                    next_global_state, dtype=torch.float32, device=self.device
                ).reshape(1, -1)
                bootstrap_value = self.critic(next_state)[0]

            next_values = torch.empty_like(old_values)
            next_values[:-1] = old_values[1:]
            next_values[-1] = bootstrap_value

            deltas = rewards + self.gamma * (1.0 - done) * next_values - old_values
            advantages = torch.zeros_like(deltas)
            running_gae = torch.tensor(0.0, device=self.device)
            for step in reversed(range(len(rewards))):
                running_gae = deltas[step] + (
                    self.gamma
                    * self.gae_lambda
                    * (1.0 - done[step])
                    * running_gae
                )
                advantages[step] = running_gae
            returns = advantages + old_values
        return advantages, returns

    def _train_workers(
        self,
        observations,
        goals,
        actions,
        available_actions,
        old_log_probs,
        advantages,
        alive,
        done,
    ):
        factor = torch.ones(len(observations), device=self.device)
        if self.fixed_order:
            order = list(range(self.n_agents))
        else:
            order = torch.randperm(self.n_agents).cpu().tolist()
        self.last_sequential_order = order

        policy_losses = [0.0] * self.n_agents
        ratios = [1.0] * self.n_agents

        for agent_id in order:
            active = alive[:, agent_id]
            active_count = active.sum()
            if active_count <= 0:
                continue

            with torch.no_grad():
                before_log_prob, _ = self._evaluate_actor(
                    agent_id,
                    observations,
                    goals,
                    actions,
                    available_actions,
                    done,
                )

            agent_advantage = advantages.clone()
            valid_advantage = agent_advantage[active > 0]
            agent_advantage = (
                agent_advantage - valid_advantage.mean()
            ) / (valid_advantage.std(unbiased=False) + 1e-5)

            for _ in range(self.a2c_epoch):
                log_prob, entropy = self._evaluate_actor(
                    agent_id,
                    observations,
                    goals,
                    actions,
                    available_actions,
                    done,
                )
                importance_ratio = torch.exp(
                    log_prob - old_log_probs[:, agent_id]
                )
                policy_loss = -(
                    factor.detach()
                    * importance_ratio
                    * agent_advantage.detach()
                    * active
                ).sum() / active_count
                entropy_mean = (entropy * active).sum() / active_count
                actor_loss = policy_loss - self.entropy_coef * entropy_mean

                optimizer = self.worker_optimizers[agent_id]
                optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.workers[agent_id].parameters(), self.max_grad_norm
                )
                optimizer.step()

            with torch.no_grad():
                after_log_prob, _ = self._evaluate_actor(
                    agent_id,
                    observations,
                    goals,
                    actions,
                    available_actions,
                    done,
                )
                update_ratio = torch.exp(after_log_prob - before_log_prob)
                update_ratio = torch.where(
                    active > 0, update_ratio, torch.ones_like(update_ratio)
                )
                factor = factor * update_ratio
                ratios[agent_id] = float(update_ratio[active > 0].mean().item())
                policy_losses[agent_id] = float(policy_loss.item())

        self.last_per_agent_policy_loss = policy_losses
        self.last_per_agent_ratio = ratios
        self.last_factor_mean = float(factor.mean().item())
        valid_losses = [
            policy_losses[index]
            for index in range(self.n_agents)
            if alive[:, index].sum() > 0
        ]
        self.last_loss_worker = (
            float(np.mean(valid_losses)) if valid_losses else 0.0
        )

    def _train_critic(self, global_states, old_values, returns):
        value_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.critic_epoch):
            values = self.critic(global_states)
            original_error = returns.detach() - values
            original_loss = self._huber_loss(original_error)

            if self.use_clipped_value_loss:
                clipped_values = old_values + (values - old_values).clamp(
                    -self.value_clip, self.value_clip
                )
                clipped_error = returns.detach() - clipped_values
                clipped_loss = self._huber_loss(clipped_error)
                value_loss = torch.maximum(original_loss, clipped_loss).mean()
            else:
                value_loss = original_loss.mean()

            critic_loss = self.value_loss_coef * value_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            self.critic_optimizer.step()
        self.last_critic_loss = float(value_loss.item())

    def _train_manager(
        self,
        global_states,
        observations,
        next_local_state,
        external_rewards,
        done,
        next_global_state,
        alive,
    ):
        manager_hidden = self._detach_hidden(
            self.manager_segment_start_hidden
        )
        goals, manager_values, manager_hidden_after = self.manager(
            global_states[0].view(1, 1, -1), manager_hidden
        )
        goal_vectors = goals[0, 0]
        manager_value = manager_values[0, 0]
        end_observation = torch.as_tensor(
            next_local_state, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            manager_return = torch.tensor(0.0, device=self.device)
            discount = 1.0
            terminal_hit = False
            for step in range(len(external_rewards)):
                manager_return = (
                    manager_return + discount * external_rewards[step]
                )
                discount *= self.gamma
                if done[step] > 0.5:
                    terminal_hit = True
                    break

            if not terminal_hit:
                next_state = torch.as_tensor(
                    next_global_state,
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 1, -1)
                _, next_manager_values, _ = self.manager(
                    next_state,
                    self._detach_hidden(manager_hidden_after),
                )
                manager_return = (
                    manager_return
                    + discount * next_manager_values[0, 0]
                )

        latent_deltas = []
        with torch.no_grad():
            for agent_id, worker in enumerate(self.workers):
                start_latent = worker.state_encoder(
                    observations[0, agent_id]
                )
                end_latent = worker.state_encoder(
                    end_observation[agent_id]
                )
                latent_deltas.append(end_latent - start_latent)
        latent_deltas = torch.stack(latent_deltas)

        cosine_per_agent = (
            F.normalize(latent_deltas, dim=-1, eps=1e-8)
            * F.normalize(goal_vectors, dim=-1, eps=1e-8)
        ).sum(dim=-1)
        active = alive[0]
        cosine = (cosine_per_agent * active).sum() / active.sum().clamp_min(1.0)
        raw_manager_advantage = manager_return - manager_value
        manager_advantage = raw_manager_advantage.clamp(
            -self.manager_advantage_clip,
            self.manager_advantage_clip,
        ).detach()
        goal_loss = -manager_advantage * cosine
        value_loss = F.smooth_l1_loss(
            manager_value,
            manager_return.detach(),
            beta=1.0,
        )
        manager_loss = (
            goal_loss + self.manager_value_loss_coef * value_loss
        )

        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        nn.utils.clip_grad_norm_(
            self.manager.parameters(), self.max_grad_norm
        )
        self.manager_optimizer.step()

        self.last_loss_manager = float(manager_loss.item())
        self.last_manager_cos_sim = float(cosine.item())
        self.last_manager_advantage = float(
            raw_manager_advantage.detach().item()
        )
        self.last_manager_value_loss = float(value_loss.item())
        self.last_manager_return = float(manager_return.item())
        self.last_manager_value_pred = float(manager_value.detach().item())

    def _init_parallel_state(self, env_count):
        self.parallel_mode = True
        self.parallel_env_count = int(env_count)
        self.parallel_manager_hidden = self.manager.init_hidden(env_count, self.device)
        self.parallel_worker_hidden = [
            worker.init_hidden(env_count, self.device) for worker in self.workers
        ]
        self.parallel_worker_rollout_start_hidden = [
            self._detach_hidden(hidden) for hidden in self.parallel_worker_hidden
        ]
        self.parallel_current_goal = torch.zeros(
            env_count,
            self.n_agents,
            self.goal_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.parallel_goal_remaining = torch.zeros(
            env_count, dtype=torch.long, device=self.device
        )
        self.parallel_segments = [None] * env_count
        self.worker_buffer = []

    def _evaluate_actor_parallel(
        self,
        agent_id,
        observations,
        goals,
        actions,
        available_actions,
        done,
    ):
        worker = self.workers[agent_id]
        hidden = self._detach_hidden(
            self.parallel_worker_rollout_start_hidden[agent_id]
        )
        logits = []
        for step in range(len(observations)):
            step_logits, hidden = worker(
                observations[step, :, agent_id].unsqueeze(0),
                goals[step, :, agent_id].unsqueeze(0),
                hidden,
            )
            logits.append(step_logits[0])
            keep = (1.0 - done[step]).view(1, -1, 1)
            hidden = (hidden[0] * keep, hidden[1] * keep)
        logits = torch.stack(logits).masked_fill(
            ~available_actions[:, :, agent_id], -1e10
        )
        distribution = torch.distributions.Categorical(logits=logits)
        return (
            distribution.log_prob(actions[:, :, agent_id]),
            distribution.entropy(),
        )

    def _compute_advantages_parallel(
        self, rewards, old_values, done, next_global_states
    ):
        with torch.no_grad():
            if next_global_states is None:
                bootstrap = torch.zeros_like(old_values[-1])
            else:
                next_states = torch.as_tensor(
                    np.asarray(next_global_states, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(-1, self.global_state_dim)
                bootstrap = self.critic(next_states)
                bootstrap = torch.where(
                    done[-1] > 0.5, torch.zeros_like(bootstrap), bootstrap
                )
            next_values = torch.empty_like(old_values)
            next_values[:-1] = old_values[1:]
            next_values[-1] = bootstrap
            deltas = rewards + self.gamma * (1.0 - done) * next_values - old_values
            advantages = torch.zeros_like(rewards)
            running_gae = torch.zeros_like(rewards[-1])
            for step in reversed(range(len(rewards))):
                running_gae = deltas[step] + (
                    self.gamma
                    * self.gae_lambda
                    * (1.0 - done[step])
                    * running_gae
                )
                advantages[step] = running_gae
        return advantages, advantages + old_values

    def _train_workers_parallel(
        self,
        observations,
        goals,
        actions,
        available_actions,
        old_log_probs,
        advantages,
        alive,
        done,
    ):
        factor = torch.ones_like(advantages)
        order = (
            list(range(self.n_agents))
            if self.fixed_order
            else torch.randperm(self.n_agents).cpu().tolist()
        )
        losses = [0.0] * self.n_agents
        ratios = [1.0] * self.n_agents
        for agent_id in order:
            active = alive[:, :, agent_id]
            active_count = active.sum()
            if active_count <= 0:
                continue
            with torch.no_grad():
                before_log_prob, _ = self._evaluate_actor_parallel(
                    agent_id,
                    observations,
                    goals,
                    actions,
                    available_actions,
                    done,
                )
            valid_advantages = advantages[active > 0]
            agent_advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std(unbiased=False) + 1e-5
            )
            for _ in range(self.a2c_epoch):
                log_prob, entropy = self._evaluate_actor_parallel(
                    agent_id,
                    observations,
                    goals,
                    actions,
                    available_actions,
                    done,
                )
                importance_ratio = torch.exp(
                    log_prob - old_log_probs[:, :, agent_id]
                )
                policy_loss = -(
                    factor.detach()
                    * importance_ratio
                    * agent_advantages.detach()
                    * active
                ).sum() / active_count
                entropy_mean = (entropy * active).sum() / active_count
                actor_loss = policy_loss - self.entropy_coef * entropy_mean
                optimizer = self.worker_optimizers[agent_id]
                optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.workers[agent_id].parameters(), self.max_grad_norm
                )
                optimizer.step()
            with torch.no_grad():
                after_log_prob, _ = self._evaluate_actor_parallel(
                    agent_id,
                    observations,
                    goals,
                    actions,
                    available_actions,
                    done,
                )
                update_ratio = torch.exp(after_log_prob - before_log_prob)
                update_ratio = torch.where(
                    active > 0, update_ratio, torch.ones_like(update_ratio)
                )
                factor.mul_(update_ratio)
                losses[agent_id] = float(policy_loss.item())
                ratios[agent_id] = float(update_ratio[active > 0].mean().item())
        self.last_sequential_order = order
        self.last_per_agent_policy_loss = losses
        self.last_per_agent_ratio = ratios
        self.last_factor_mean = float(factor.mean().item())
        valid_losses = [
            losses[index]
            for index in range(self.n_agents)
            if alive[:, :, index].sum() > 0
        ]
        self.last_loss_worker = float(np.mean(valid_losses)) if valid_losses else 0.0

    def _train_manager_segments(self, segments):
        if not segments:
            return None
        losses = []
        cosines = []
        advantages = []
        value_losses = []
        returns = []
        predictions = []
        for segment in segments:
            goals, manager_values, hidden_after = self.manager(
                segment["state"].view(1, 1, -1),
                self._detach_hidden(segment["hidden"]),
            )
            goal_vectors = goals[0, 0]
            manager_value = manager_values[0, 0]
            with torch.no_grad():
                manager_return = torch.tensor(0.0, device=self.device)
                discount = 1.0
                for reward in segment["rewards"]:
                    manager_return = manager_return + discount * reward
                    discount *= self.gamma
                if not segment["done"]:
                    _, next_values, _ = self.manager(
                        segment["next_state"].view(1, 1, -1),
                        self._detach_hidden(hidden_after),
                    )
                    manager_return = manager_return + discount * next_values[0, 0]
                latent_deltas = []
                for agent_id, worker in enumerate(self.workers):
                    start_latent = worker.state_encoder(segment["obs"][agent_id])
                    end_latent = worker.state_encoder(
                        segment["next_obs"][agent_id]
                    )
                    latent_deltas.append(end_latent - start_latent)
                latent_deltas = torch.stack(latent_deltas)
            cosine_per_agent = (
                F.normalize(latent_deltas, dim=-1, eps=1e-8)
                * F.normalize(goal_vectors, dim=-1, eps=1e-8)
            ).sum(dim=-1)
            active = segment["alive"]
            cosine = (cosine_per_agent * active).sum() / active.sum().clamp_min(1.0)
            raw_advantage = manager_return - manager_value
            clipped_advantage = raw_advantage.clamp(
                -self.manager_advantage_clip, self.manager_advantage_clip
            ).detach()
            value_loss = F.smooth_l1_loss(
                manager_value, manager_return.detach(), beta=1.0
            )
            losses.append(
                -clipped_advantage * cosine
                + self.manager_value_loss_coef * value_loss
            )
            cosines.append(cosine.detach())
            advantages.append(raw_advantage.detach())
            value_losses.append(value_loss.detach())
            returns.append(manager_return.detach())
            predictions.append(manager_value.detach())
        manager_loss = torch.stack(losses).mean()
        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
        self.manager_optimizer.step()
        self.last_loss_manager = float(manager_loss.item())
        self.last_manager_cos_sim = float(torch.stack(cosines).mean().item())
        self.last_manager_advantage = float(torch.stack(advantages).mean().item())
        self.last_manager_value_loss = float(torch.stack(value_losses).mean().item())
        self.last_manager_return = float(torch.stack(returns).mean().item())
        self.last_manager_value_pred = float(torch.stack(predictions).mean().item())
        self.manager_update_count += len(segments)
        self.just_updated = True
        return self.last_loss_manager

    def _train_parallel(self, next_global_state):
        if not self.worker_buffer:
            return None
        observations = torch.stack([item["obs"] for item in self.worker_buffer])
        global_states = torch.stack([item["state"] for item in self.worker_buffer])
        goals = torch.stack([item["goal"] for item in self.worker_buffer])
        actions = torch.stack([item["actions"] for item in self.worker_buffer])
        old_log_probs = torch.stack(
            [item["old_log_probs"] for item in self.worker_buffer]
        )
        old_values = torch.stack([item["old_value"] for item in self.worker_buffer])
        rewards = torch.stack(
            [item["external_reward"] for item in self.worker_buffer]
        )
        done = torch.stack([item["done"] for item in self.worker_buffer])
        alive = torch.stack([item["alive"] for item in self.worker_buffer])
        available = torch.stack(
            [item["available_actions"] for item in self.worker_buffer]
        )
        advantages, returns = self._compute_advantages_parallel(
            rewards, old_values, done, next_global_state
        )
        self._train_workers_parallel(
            observations,
            goals,
            actions,
            available,
            old_log_probs,
            advantages,
            alive,
            done,
        )
        time_steps, env_count = rewards.shape
        batch_size = time_steps * env_count
        self._train_critic(
            global_states.reshape(batch_size, self.global_state_dim),
            old_values.reshape(batch_size).detach(),
            returns.reshape(batch_size),
        )
        self.last_worker_mean_reward = float(rewards.mean().item())
        self.worker_buffer = []
        self.parallel_worker_rollout_start_hidden = [
            self._detach_hidden(hidden) for hidden in self.parallel_worker_hidden
        ]
        self.worker_update_count += 1
        self.just_updated = True
        return self.last_loss_worker + self.last_critic_loss

    def train(
        self,
        next_global_state=None,
        next_local_state=None,
        next_alive_mask=None,
    ):
        if self.parallel_mode:
            return self._train_parallel(next_global_state)
        del next_local_state, next_alive_mask
        if not self.worker_buffer:
            return None

        observations = torch.stack(
            [transition["obs"] for transition in self.worker_buffer]
        )
        global_states = torch.stack(
            [transition["state"] for transition in self.worker_buffer]
        )
        goals = torch.stack(
            [transition["goal"] for transition in self.worker_buffer]
        )
        actions = torch.stack(
            [transition["actions"] for transition in self.worker_buffer]
        )
        old_log_probs = torch.stack(
            [
                transition["old_log_probs"]
                for transition in self.worker_buffer
            ]
        )
        old_values = torch.stack(
            [transition["old_value"] for transition in self.worker_buffer]
        )
        external_rewards = torch.stack(
            [
                transition["external_reward"]
                for transition in self.worker_buffer
            ]
        )
        done = torch.tensor(
            [
                float(transition["done"])
                for transition in self.worker_buffer
            ],
            dtype=torch.float32,
            device=self.device,
        )
        alive = torch.stack(
            [transition["alive"] for transition in self.worker_buffer]
        )
        available_actions = torch.stack(
            [
                transition["available_actions"]
                for transition in self.worker_buffer
            ]
        )

        advantages, returns = self._compute_advantages(
            external_rewards,
            old_values,
            done,
            next_global_state,
        )

        self._train_workers(
            observations,
            goals,
            actions,
            available_actions,
            old_log_probs,
            advantages,
            alive,
            done,
        )
        self._train_critic(
            global_states,
            old_values.detach(),
            returns,
        )

        self.last_worker_mean_reward = float(external_rewards.mean().item())
        self.worker_buffer = []
        self.worker_rollout_start_hidden = [
            self._detach_hidden(hidden) for hidden in self.worker_hidden
        ]
        self.worker_update_count += 1
        self.just_updated = True
        return self.last_loss_worker + self.last_critic_loss

    def _update_manager(
        self,
        next_global_state,
        next_local_state,
    ):
        if not self.manager_buffer:
            return None

        global_states = torch.stack(
            [transition["state"] for transition in self.manager_buffer]
        )
        observations = torch.stack(
            [transition["obs"] for transition in self.manager_buffer]
        )
        external_rewards = torch.stack(
            [
                transition["external_reward"]
                for transition in self.manager_buffer
            ]
        )
        done = torch.tensor(
            [
                float(transition["done"])
                for transition in self.manager_buffer
            ],
            dtype=torch.float32,
            device=self.device,
        )
        alive = torch.stack(
            [transition["alive"] for transition in self.manager_buffer]
        )
        self._train_manager(
            global_states,
            observations,
            next_local_state,
            external_rewards,
            done,
            next_global_state,
            alive,
        )
        self.manager_buffer = []
        self.manager_update_count += 1
        self.just_updated = True
        return self.last_loss_manager

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
        observations = torch.as_tensor(
            np.asarray(local_state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        state = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        )
        alive = torch.as_tensor(
            alive_mask, dtype=torch.float32, device=self.device
        )
        available = torch.as_tensor(
            avail_actions, dtype=torch.bool, device=self.device
        )

        if reward_ext_m is not None and self._last_step_cache is not None:
            cache = self._last_step_cache
            if isinstance(reward_ext_m, (list, tuple, np.ndarray)):
                external_reward = float(np.mean(reward_ext_m))
            else:
                external_reward = float(reward_ext_m)

            self._store_transition(
                cache,
                external_reward,
                done,
            )

        worker_update_due = (
            len(self.worker_buffer) >= self.worker_update_steps
        )
        manager_update_due = (
            len(self.manager_buffer) >= self.manager_c_steps
            or (done and self.manager_buffer)
        )
        if worker_update_due:
            with torch.enable_grad():
                self.train(
                    next_global_state=state,
                    next_local_state=observations,
                    next_alive_mask=alive,
                )
        if manager_update_due:
            with torch.enable_grad():
                self._update_manager(
                    next_global_state=state,
                    next_local_state=observations,
                )

        if done:
            self._last_step_cache = None
            fallback = available.long().argmax(dim=-1)
            return fallback.cpu().tolist(), np.zeros(
                (1, 1, self.n_agents, 1), dtype=np.int32
            )

        if len(self.worker_buffer) == 0:
            self.worker_rollout_start_hidden = [
                self._detach_hidden(hidden) for hidden in self.worker_hidden
            ]

        new_manager_segment = len(self.manager_buffer) == 0
        if new_manager_segment:
            self.manager_segment_start_hidden = self._detach_hidden(
                self.manager_hidden
            )
            with torch.no_grad():
                goals, _, self.manager_hidden = self.manager(
                    state.view(1, 1, -1), self.manager_hidden
                )
            self.current_goal = goals[0, 0].detach()

        actions = []
        log_probs = []
        with torch.no_grad():
            for agent_id, worker in enumerate(self.workers):
                logits, new_hidden = worker(
                    observations[agent_id].view(1, 1, -1),
                    self.current_goal[agent_id].view(1, 1, -1),
                    self.worker_hidden[agent_id],
                )
                self.worker_hidden[agent_id] = self._detach_hidden(new_hidden)
                masked_logits = logits[0, 0].masked_fill(
                    ~available[agent_id], -1e10
                )
                distribution = torch.distributions.Categorical(
                    logits=masked_logits
                )
                action = distribution.sample()
                actions.append(action)
                log_probs.append(distribution.log_prob(action))

            old_value = self.critic(state.view(1, -1))[0]

        action_tensor = torch.stack(actions)
        old_log_prob_tensor = torch.stack(log_probs)
        self._last_step_cache = {
            "obs": observations,
            "state": state,
            "goal": self.current_goal,
            "actions": action_tensor,
            "old_log_probs": old_log_prob_tensor,
            "old_value": old_value,
            "alive": alive,
            "available_actions": available,
        }
        return action_tensor.cpu().tolist(), np.zeros(
            (1, 1, self.n_agents, 1), dtype=np.int32
        )

    def sample_actions_batch(
        self,
        local_states,
        global_states,
        alive_masks,
        available_actions,
        deterministic=False,
    ):
        observations = torch.as_tensor(
            np.asarray(local_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        states = torch.as_tensor(
            np.asarray(global_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        alive = torch.as_tensor(
            np.asarray(alive_masks, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        available = torch.as_tensor(
            np.asarray(available_actions, dtype=np.bool_),
            dtype=torch.bool,
            device=self.device,
        )
        env_count = observations.shape[0]
        if not self.parallel_mode:
            self._init_parallel_state(env_count)
        elif env_count != self.parallel_env_count:
            raise ValueError("The number of parallel FeUdal environments changed")

        due = torch.nonzero(
            self.parallel_goal_remaining <= 0, as_tuple=False
        ).flatten()
        if len(due) > 0:
            hidden_before = (
                self.parallel_manager_hidden[0][:, due].detach().clone(),
                self.parallel_manager_hidden[1][:, due].detach().clone(),
            )
            with torch.no_grad():
                goals, _, hidden_after = self.manager(
                    states[due].unsqueeze(0), hidden_before
                )
            manager_hidden = [part.detach().clone() for part in self.parallel_manager_hidden]
            manager_hidden[0][:, due] = hidden_after[0]
            manager_hidden[1][:, due] = hidden_after[1]
            self.parallel_manager_hidden = tuple(manager_hidden)
            self.parallel_current_goal[due] = goals[0].detach()
            self.parallel_goal_remaining[due] = self.manager_c_steps
            for offset, env_id_tensor in enumerate(due):
                env_id = int(env_id_tensor.item())
                self.parallel_segments[env_id] = {
                    "state": states[env_id].detach().clone(),
                    "obs": observations[env_id].detach().clone(),
                    "alive": alive[env_id].detach().clone(),
                    "hidden": (
                        hidden_before[0][:, offset : offset + 1].detach().clone(),
                        hidden_before[1][:, offset : offset + 1].detach().clone(),
                    ),
                    "rewards": [],
                }

        actions = []
        log_probs = []
        with torch.no_grad():
            for agent_id, worker in enumerate(self.workers):
                logits, hidden = worker(
                    observations[:, agent_id].unsqueeze(0),
                    self.parallel_current_goal[:, agent_id].unsqueeze(0),
                    self.parallel_worker_hidden[agent_id],
                )
                self.parallel_worker_hidden[agent_id] = self._detach_hidden(hidden)
                masked_logits = logits[0].masked_fill(
                    ~available[:, agent_id], -1e10
                )
                distribution = torch.distributions.Categorical(logits=masked_logits)
                action = (
                    distribution.probs.argmax(dim=-1)
                    if deterministic
                    else distribution.sample()
                )
                actions.append(action)
                log_probs.append(distribution.log_prob(action))
            old_values = self.critic(states)

        action_tensor = torch.stack(actions, dim=1)
        self._parallel_step_cache = {
            "obs": observations,
            "state": states,
            "goal": self.parallel_current_goal.detach().clone(),
            "actions": action_tensor,
            "old_log_probs": torch.stack(log_probs, dim=1),
            "old_value": old_values,
            "alive": alive,
            "available_actions": available,
        }
        return action_tensor.cpu().numpy()

    def store_transition_batch(
        self,
        rewards,
        dones,
        next_global_states=None,
        next_local_states=None,
        next_alive_masks=None,
    ):
        del next_alive_masks
        if self._parallel_step_cache is None:
            raise RuntimeError("sample_actions_batch must be called before storing")
        if next_global_states is None or next_local_states is None:
            raise ValueError("FeUdal parallel rollout requires next states")
        rewards_tensor = torch.as_tensor(
            np.asarray(rewards, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        done_tensor = torch.as_tensor(
            np.asarray(dones, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        next_states = torch.as_tensor(
            np.asarray(next_global_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.asarray(next_local_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        cache = self._parallel_step_cache
        self.worker_buffer.append(
            {
                key: value.detach().clone() for key, value in cache.items()
            }
            | {
                "external_reward": rewards_tensor.detach().clone(),
                "done": done_tensor.detach().clone(),
            }
        )
        completed_segments = []
        for env_id in range(self.parallel_env_count):
            segment = self.parallel_segments[env_id]
            segment["rewards"].append(rewards_tensor[env_id].detach().clone())
            self.parallel_goal_remaining[env_id] -= 1
            terminal = bool(done_tensor[env_id].item() > 0.5)
            if terminal or self.parallel_goal_remaining[env_id] <= 0:
                segment["next_state"] = next_states[env_id].detach().clone()
                segment["next_obs"] = next_observations[env_id].detach().clone()
                segment["done"] = terminal
                completed_segments.append(segment)
                self.parallel_segments[env_id] = None
            if terminal:
                manager_hidden = [
                    part.detach().clone() for part in self.parallel_manager_hidden
                ]
                manager_hidden[0][:, env_id] = 0.0
                manager_hidden[1][:, env_id] = 0.0
                self.parallel_manager_hidden = tuple(manager_hidden)
                for agent_id in range(self.n_agents):
                    worker_hidden = [
                        part.detach().clone()
                        for part in self.parallel_worker_hidden[agent_id]
                    ]
                    worker_hidden[0][:, env_id] = 0.0
                    worker_hidden[1][:, env_id] = 0.0
                    self.parallel_worker_hidden[agent_id] = tuple(worker_hidden)
                self.parallel_goal_remaining[env_id] = 0
                self.parallel_current_goal[env_id] = 0.0
        self._parallel_step_cache = None
        if completed_segments:
            with torch.enable_grad():
                self._train_manager_segments(completed_segments)

    def finalize_training(
        self,
        next_global_state=None,
        next_local_state=None,
        next_alive_mask=None,
    ):
        del next_alive_mask
        if not self.parallel_mode:
            if self.worker_buffer:
                return self.train(next_global_state=next_global_state)
            return None
        if next_global_state is not None and next_local_state is not None:
            next_states = torch.as_tensor(
                np.asarray(next_global_state, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            next_observations = torch.as_tensor(
                np.asarray(next_local_state, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            segments = []
            for env_id, segment in enumerate(self.parallel_segments):
                if segment is not None and segment["rewards"]:
                    segment["next_state"] = next_states[env_id].detach().clone()
                    segment["next_obs"] = next_observations[env_id].detach().clone()
                    segment["done"] = False
                    segments.append(segment)
                    self.parallel_segments[env_id] = None
            if segments:
                self._train_manager_segments(segments)
        if self.worker_buffer:
            return self._train_parallel(next_global_state)
        return None

    def episode_reset(self):
        worker_rollout_pending = bool(self.worker_buffer)
        self.manager_hidden = self.manager.init_hidden(1, self.device)
        self.worker_hidden = [
            worker.init_hidden(1, self.device) for worker in self.workers
        ]
        self.manager_segment_start_hidden = self._detach_hidden(
            self.manager_hidden
        )
        if not worker_rollout_pending:
            self.worker_rollout_start_hidden = [
                self._detach_hidden(hidden) for hidden in self.worker_hidden
            ]
        self.current_goal = None
        self.manager_buffer = []
        self._last_step_cache = None

    def update_target_network(self):
        return

    def save_model(self, checkpoint):
        checkpoint["manager"] = self.manager.state_dict()
        checkpoint["workers"] = self.workers.state_dict()
        checkpoint["critic"] = self.critic.state_dict()
        checkpoint["manager_optimizer"] = self.manager_optimizer.state_dict()
        checkpoint["worker_optimizers"] = [
            optimizer.state_dict() for optimizer in self.worker_optimizers
        ]
        checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()
        checkpoint["worker_update_count"] = self.worker_update_count
        checkpoint["manager_update_count"] = self.manager_update_count
        return checkpoint

    def load_model(self, checkpoint):
        self.manager.load_state_dict(checkpoint["manager"])
        self.workers.load_state_dict(checkpoint["workers"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "manager_optimizer" in checkpoint:
            self.manager_optimizer.load_state_dict(
                checkpoint["manager_optimizer"]
            )
        if "worker_optimizers" in checkpoint:
            for optimizer, state in zip(
                self.worker_optimizers, checkpoint["worker_optimizers"]
            ):
                optimizer.load_state_dict(state)
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer"]
            )
        self.worker_update_count = int(
            checkpoint.get("worker_update_count", 0)
        )
        self.manager_update_count = int(
            checkpoint.get("manager_update_count", 0)
        )
        return self
