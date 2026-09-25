import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Modulate local state features with a manager-provided goal."""

    def __init__(self, goal_dim: int, feature_dim: int):
        super().__init__()
        self.affine = nn.Linear(goal_dim, 2 * feature_dim)

    def forward(self, features, goal):
        gamma, beta = torch.chunk(self.affine(goal), 2, dim=-1)
        return features * gamma + beta


class FeudalManager(nn.Module):
    """Per-step recurrent state encoder with goal and shared external value heads."""

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        hidden_dim: int,
        goal_dim: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.goal_dim = goal_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(hidden_dim, hidden_dim)
        self.goal_head = nn.Linear(hidden_dim, n_agents * goal_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device):
        hidden = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        cell = torch.zeros_like(hidden)
        return hidden, cell

    def forward(self, state_sequence, hidden):
        encoded = self.encoder(state_sequence)
        output, new_hidden = self.rnn(encoded, hidden)
        time_steps, batch_size, _ = output.shape
        goals = self.goal_head(output).view(
            time_steps, batch_size, self.n_agents, self.goal_dim
        )
        values = self.value_head(output).squeeze(-1)
        return F.normalize(goals, dim=-1), values, new_hidden


class FeudalWorkerActor(nn.Module):
    """Independent FeUdal worker actor: local state -> FiLM(goal) -> LSTM."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dim: int,
        n_actions: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.film = FiLMLayer(goal_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size: int, device: torch.device):
        hidden = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        cell = torch.zeros_like(hidden)
        return hidden, cell

    def forward(self, observation_sequence, goal_sequence, hidden):
        features = self.state_encoder(observation_sequence)
        modulated = F.relu(self.film(features, goal_sequence))
        output, new_hidden = self.rnn(modulated, hidden)
        return self.action_head(output), new_hidden


class IntrinsicValueCritic(nn.Module):
    """One worker's remaining-goal return; independent of its actor encoder."""

    def __init__(self, state_dim, obs_dim, goal_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + obs_dim + goal_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state, observation, goal, remaining_fraction):
        inputs = torch.cat(
            (global_state, observation, goal, remaining_fraction.unsqueeze(-1)),
            dim=-1,
        )
        return self.network(inputs).squeeze(-1)
