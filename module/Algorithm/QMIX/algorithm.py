from collections import deque
import copy
import os

import numpy as np
import torch

from module.Algorithm.base_algorithm import BaseAlgorithm
from module.Algorithm.QMIX.model.model import QMixer, RNNAgent
from Tools import utils


class Algorithm(BaseAlgorithm):
    """PyMARL-style QMIX adapted to the existing step-based runner."""

    def __init__(self, action_num: int, **kwargs):
        super().__init__()
        self.cfg = utils.load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        )
        self.action_num = action_num
        self.n_agents = int(kwargs["n_agents"])
        self.obs_dim = int(kwargs["state_dim"])
        self.state_dim = int(kwargs["state_g_dim"])

        self.gamma = float(self.cfg["gamma"])
        self.batch_size = int(self.cfg["batch_size"])
        self.buffer_size = int(self.cfg["buffer_size"])
        self.train_updates_per_episode = int(
            self.cfg.get("train_updates_per_episode", 1)
        )
        self.target_update_interval = int(self.cfg["target_update_interval"])
        self.double_q = bool(self.cfg.get("double_q", True))
        self.grad_norm_clip = float(self.cfg["grad_norm_clip"])
        self.obs_last_action = bool(self.cfg.get("obs_last_action", True))
        self.obs_agent_id = bool(self.cfg.get("obs_agent_id", True))

        input_dim = self.obs_dim
        if self.obs_last_action:
            input_dim += self.action_num
        if self.obs_agent_id:
            input_dim += self.n_agents

        hidden_dim = int(self.cfg["rnn_hidden_dim"])
        self.agent = RNNAgent(input_dim, hidden_dim, self.action_num).to(self.device)
        self.mixer = QMixer(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            embed_dim=int(self.cfg["mixing_embed_dim"]),
            hypernet_layers=int(self.cfg["hypernet_layers"]),
            hypernet_embed=int(self.cfg["hypernet_embed"]),
        ).to(self.device)
        self.target_agent = copy.deepcopy(self.agent)
        self.target_mixer = copy.deepcopy(self.mixer)

        parameters = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(
            parameters,
            lr=float(self.cfg["learning_rate"]),
            alpha=float(self.cfg["optim_alpha"]),
            eps=float(self.cfg["optim_eps"]),
        )

        self.replay = deque(maxlen=self.buffer_size)
        self.total_env_steps = 0
        self.episodes_seen = 0
        self.last_target_update_episode = 0
        self.last_loss = 0.0
        self.last_grad_norm = 0.0
        self.just_updated = False

        self.hidden = None
        self.last_action = None
        self.current_episode = []
        self._last_step_cache = None
        self.episode_reset()

    @property
    def epsilon(self):
        start = float(self.cfg["epsilon_start"])
        finish = float(self.cfg["epsilon_finish"])
        anneal_steps = max(1, int(self.cfg["epsilon_anneal_time"]))
        fraction = min(1.0, self.total_env_steps / anneal_steps)
        return start + fraction * (finish - start)

    def _agent_inputs(self, observations, last_actions):
        parts = [observations]
        if self.obs_last_action:
            parts.append(last_actions)
        if self.obs_agent_id:
            batch_size = observations.shape[0]
            agent_ids = torch.eye(
                self.n_agents, dtype=observations.dtype, device=observations.device
            )
            parts.append(
                agent_ids.view(1, self.n_agents, self.n_agents).expand(
                    batch_size, -1, -1
                )
            )
        return torch.cat(parts, dim=-1)

    def _select_actions(self, q_values, available_actions):
        masked_q = q_values.clone()
        masked_q[~available_actions] = -1e9
        greedy_actions = masked_q.argmax(dim=-1).cpu().numpy()
        available_np = available_actions.cpu().numpy()
        actions = greedy_actions.copy()

        for agent_id in range(self.n_agents):
            choices = np.flatnonzero(available_np[agent_id])
            if choices.size == 0:
                actions[agent_id] = 0
            elif np.random.random() < self.epsilon:
                actions[agent_id] = int(np.random.choice(choices))
        return actions

    def sample_action(
        self,
        local_state,
        global_state,
        reward_ext_m,
        reward_ext_w,
        done,
        alive_mask,
        avail_actions=None,
        task_idx=None,
    ):
        del reward_ext_w, alive_mask, task_idx
        self.just_updated = False
        observations = np.asarray(local_state, dtype=np.float32)
        state = np.asarray(global_state, dtype=np.float32)
        available = np.asarray(avail_actions, dtype=np.bool_)

        if reward_ext_m is not None and self._last_step_cache is not None:
            transition = dict(self._last_step_cache)
            transition.update(
                reward=float(reward_ext_m),
                done=bool(done),
                next_obs=observations.copy(),
                next_state=state.copy(),
                next_avail=available.copy(),
            )
            self.current_episode.append(transition)
            self.total_env_steps += 1

        if done:
            if self.current_episode:
                self.replay.append(self.current_episode)
                self.episodes_seen += 1
                for _ in range(self.train_updates_per_episode):
                    self.train()
            self.current_episode = []
            self._last_step_cache = None
            fallback = available.argmax(axis=1).astype(np.int64)
            return fallback.tolist(), np.zeros(
                (1, 1, self.n_agents, 1), dtype=np.int32
            )

        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        last_action_tensor = torch.as_tensor(
            self.last_action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        inputs = self._agent_inputs(obs_tensor, last_action_tensor).reshape(
            self.n_agents, -1
        )

        with torch.no_grad():
            q_values, self.hidden = self.agent(inputs, self.hidden)
        available_tensor = torch.as_tensor(
            available, dtype=torch.bool, device=self.device
        )
        actions = self._select_actions(q_values, available_tensor)

        self._last_step_cache = {
            "obs": observations.copy(),
            "state": state.copy(),
            "avail": available.copy(),
            "actions": actions.copy(),
            "last_actions": self.last_action.copy(),
        }
        self.last_action.fill(0.0)
        self.last_action[np.arange(self.n_agents), actions] = 1.0
        return actions.tolist(), np.zeros(
            (1, 1, self.n_agents, 1), dtype=np.int32
        )

    def _build_batch(self, episodes):
        batch_size = len(episodes)
        max_steps = max(len(episode) for episode in episodes)
        obs = np.zeros(
            (batch_size, max_steps + 1, self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        last_actions = np.zeros(
            (batch_size, max_steps + 1, self.n_agents, self.action_num),
            dtype=np.float32,
        )
        states = np.zeros(
            (batch_size, max_steps, self.state_dim), dtype=np.float32
        )
        next_states = np.zeros_like(states)
        actions = np.zeros(
            (batch_size, max_steps, self.n_agents), dtype=np.int64
        )
        rewards = np.zeros((batch_size, max_steps, 1), dtype=np.float32)
        terminated = np.zeros_like(rewards)
        mask = np.zeros_like(rewards)
        next_avail = np.zeros(
            (batch_size, max_steps, self.n_agents, self.action_num),
            dtype=np.bool_,
        )
        next_avail[..., 0] = True

        for batch_id, episode in enumerate(episodes):
            for step, transition in enumerate(episode):
                obs[batch_id, step] = transition["obs"]
                obs[batch_id, step + 1] = transition["next_obs"]
                last_actions[batch_id, step] = transition["last_actions"]
                action_ids = transition["actions"]
                last_actions[
                    batch_id, step + 1, np.arange(self.n_agents), action_ids
                ] = 1.0
                states[batch_id, step] = transition["state"]
                next_states[batch_id, step] = transition["next_state"]
                actions[batch_id, step] = action_ids
                rewards[batch_id, step, 0] = transition["reward"]
                terminated[batch_id, step, 0] = float(transition["done"])
                next_avail[batch_id, step] = transition["next_avail"]
                mask[batch_id, step, 0] = 1.0

        return {
            "obs": torch.from_numpy(obs).to(self.device),
            "last_actions": torch.from_numpy(last_actions).to(self.device),
            "states": torch.from_numpy(states).to(self.device),
            "next_states": torch.from_numpy(next_states).to(self.device),
            "actions": torch.from_numpy(actions).to(self.device),
            "rewards": torch.from_numpy(rewards).to(self.device),
            "terminated": torch.from_numpy(terminated).to(self.device),
            "mask": torch.from_numpy(mask).to(self.device),
            "next_avail": torch.from_numpy(next_avail).to(self.device),
        }

    def _unroll_agent(self, network, observations, last_actions):
        batch_size, sequence_length, _, _ = observations.shape
        hidden = network.init_hidden(batch_size * self.n_agents, self.device)
        outputs = []
        for step in range(sequence_length):
            inputs = self._agent_inputs(
                observations[:, step], last_actions[:, step]
            ).reshape(batch_size * self.n_agents, -1)
            q_values, hidden = network(inputs, hidden)
            outputs.append(
                q_values.view(batch_size, self.n_agents, self.action_num)
            )
        return torch.stack(outputs, dim=1)

    def train(self):
        if len(self.replay) < self.batch_size:
            return None

        indices = np.random.choice(
            len(self.replay), size=self.batch_size, replace=False
        )
        episodes = [self.replay[int(index)] for index in indices]
        batch = self._build_batch(episodes)

        live_q = self._unroll_agent(
            self.agent, batch["obs"], batch["last_actions"]
        )
        chosen_q = torch.gather(
            live_q[:, :-1], dim=-1, index=batch["actions"].unsqueeze(-1)
        ).squeeze(-1)

        with torch.no_grad():
            target_q = self._unroll_agent(
                self.target_agent, batch["obs"], batch["last_actions"]
            )[:, 1:]
            target_q[~batch["next_avail"]] = -1e9

            if self.double_q:
                live_next_q = live_q[:, 1:].detach().clone()
                live_next_q[~batch["next_avail"]] = -1e9
                next_actions = live_next_q.argmax(dim=-1, keepdim=True)
                target_max_q = torch.gather(
                    target_q, dim=-1, index=next_actions
                ).squeeze(-1)
            else:
                target_max_q = target_q.max(dim=-1).values

            target_total_q = self.target_mixer(
                target_max_q, batch["next_states"]
            )
            targets = batch["rewards"] + self.gamma * (
                1.0 - batch["terminated"]
            ) * target_total_q

        chosen_total_q = self.mixer(chosen_q, batch["states"])
        td_error = chosen_total_q - targets
        masked_td_error = td_error * batch["mask"]
        loss = masked_td_error.square().sum() / batch["mask"].sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        parameters = list(self.agent.parameters()) + list(self.mixer.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.grad_norm_clip
        )
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.last_grad_norm = float(grad_norm)
        self.just_updated = True

        if (
            self.episodes_seen - self.last_target_update_episode
            >= self.target_update_interval
        ):
            self.update_target_network()
            self.last_target_update_episode = self.episodes_seen
        return self.last_loss

    def episode_reset(self):
        self.hidden = self.agent.init_hidden(self.n_agents, self.device)
        self.last_action = np.zeros(
            (self.n_agents, self.action_num), dtype=np.float32
        )
        self.current_episode = []
        self._last_step_cache = None

    def update_target_network(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save_model(self, checkpoint):
        checkpoint["agent"] = self.agent.state_dict()
        checkpoint["mixer"] = self.mixer.state_dict()
        checkpoint["target_agent"] = self.target_agent.state_dict()
        checkpoint["target_mixer"] = self.target_mixer.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint["total_env_steps"] = self.total_env_steps
        checkpoint["episodes_seen"] = self.episodes_seen
        checkpoint["last_target_update_episode"] = (
            self.last_target_update_episode
        )
        return checkpoint

    def load_model(self, checkpoint):
        self.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_agent.load_state_dict(
            checkpoint.get("target_agent", checkpoint["agent"])
        )
        self.target_mixer.load_state_dict(
            checkpoint.get("target_mixer", checkpoint["mixer"])
        )
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_env_steps = int(checkpoint.get("total_env_steps", 0))
        self.episodes_seen = int(checkpoint.get("episodes_seen", 0))
        self.last_target_update_episode = int(
            checkpoint.get("last_target_update_episode", self.episodes_seen)
        )
