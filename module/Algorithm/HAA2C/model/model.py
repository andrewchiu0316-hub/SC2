import torch
import torch.nn as nn


def _init_layer(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class HAA2CActor(nn.Module):
    """Independent feed-forward actor using only one agent's observation."""

    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden_dim), gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            _init_layer(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            _init_layer(nn.Linear(hidden_dim, n_actions), gain=0.01),
        )

    def forward(self, observation):
        return self.network(observation)


class CentralValueCritic(nn.Module):
    """Centralized V critic conditioned on the SMAC global state."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            _init_layer(nn.Linear(state_dim, hidden_dim), gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            _init_layer(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            _init_layer(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, global_state):
        return self.network(global_state).squeeze(-1)


class ValueNorm(nn.Module):
    """Debiased exponential running statistics used by HARL value critics."""

    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_mean_sq", torch.zeros(1))
        self.register_buffer("debiasing_term", torch.zeros(1))

    @torch.no_grad()
    def update(self, values):
        values = values.detach()
        batch_mean = values.mean()
        batch_mean_sq = values.square().mean()
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def _statistics(self):
        if self.debiasing_term.item() == 0.0:
            return self.running_mean, torch.ones_like(self.running_mean)
        mean = self.running_mean / self.debiasing_term
        mean_sq = self.running_mean_sq / self.debiasing_term
        variance = (mean_sq - mean.square()).clamp_min(self.epsilon)
        return mean, variance

    def normalize(self, values):
        mean, variance = self._statistics()
        return (values - mean) / torch.sqrt(variance)

    def denormalize(self, values):
        mean, variance = self._statistics()
        return values * torch.sqrt(variance) + mean
