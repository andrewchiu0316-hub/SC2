"""Network definitions used by the GC-MRL implementation.

This is the same non-shared worker and deterministic manager topology used by
``epymarl --config=mappo_ns``.  It is intentionally kept separate from the
older ``FeUdal`` networks in this repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkerAgent(nn.Module):
    """One independent EPyMARL RNNNS worker (feed-forward by default)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, use_rnn: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim) if use_rnn else nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None):
        x = F.relu(self.fc1(inputs))
        if self.use_rnn:
            if hidden is None:
                hidden = x.new_zeros((x.shape[0], self.hidden_dim))
            h = self.rnn(x, hidden.reshape(-1, self.hidden_dim))
        else:
            h = F.relu(self.rnn(x))
        return self.actor_head(h), self.value_head(h), h


class NonSharedWorkers(nn.Module):
    """The ``rnn_ns`` agent registry entry from epymarl."""

    def __init__(self, n_agents: int, input_dim: int, hidden_dim: int, n_actions: int, use_rnn: bool):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn
        self.agents = nn.ModuleList(
            [WorkerAgent(input_dim, hidden_dim, n_actions, use_rnn) for _ in range(n_agents)]
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_agents, self.hidden_dim, device=device)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None):
        """Evaluate ``[batch, agent, obs + goal]`` worker inputs."""
        if inputs.ndim != 3:
            raise ValueError("Worker inputs must have shape [batch, n_agents, features]")
        if hidden is None:
            hidden = self.initial_hidden(inputs.shape[0], inputs.device)

        logits, values, hiddens = [], [], []
        for agent_id, agent in enumerate(self.agents):
            agent_logits, agent_values, agent_hidden = agent(
                inputs[:, agent_id], hidden[:, agent_id]
            )
            logits.append(agent_logits.unsqueeze(1))
            values.append(agent_values.unsqueeze(1))
            hiddens.append(agent_hidden.unsqueeze(1))
        return torch.cat(logits, dim=1), torch.cat(values, dim=1), torch.cat(hiddens, dim=1)


class DeterministicManager(nn.Module):
    """EPyMARL's ``manager_mode=deterministic_cos`` manager."""

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        goal_dim: int,
        hidden_dim: int,
        goal_tanh: bool = True,
        goal_scale: float = 1.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.goal_dim = goal_dim
        self.goal_tanh = goal_tanh
        self.goal_scale = float(goal_scale)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.goal_head = nn.Linear(hidden_dim, n_agents * goal_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        raw_goal = self.goal_head(x).view(-1, self.n_agents, self.goal_dim)
        goal = self.goal_scale * torch.tanh(raw_goal) if self.goal_tanh else raw_goal
        value = self.value_head(x)
        return {"raw_goal": raw_goal, "goal": goal, "value": value}
