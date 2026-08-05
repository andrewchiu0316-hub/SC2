import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """Shared recurrent Q-network used by every agent."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_agents: int, device: torch.device):
        return torch.zeros(batch_agents, self.hidden_dim, device=device)

    def forward(self, inputs, hidden_state):
        features = F.relu(self.fc1(inputs))
        hidden = self.rnn(features, hidden_state.reshape(-1, self.hidden_dim))
        return self.fc2(hidden), hidden


class QMixer(nn.Module):
    """Monotonic state-conditioned mixer from the original PyMARL QMIX."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int,
        hypernet_layers: int,
        hypernet_embed: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(state_dim, embed_dim * n_agents)
            self.hyper_w_final = nn.Linear(state_dim, embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, embed_dim * n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, embed_dim),
            )
        else:
            raise ValueError("QMIX supports one or two hypernetwork layers")

        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.shape[0]
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        weights_1 = torch.abs(self.hyper_w_1(states))
        weights_1 = weights_1.view(-1, self.n_agents, self.embed_dim)
        bias_1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, weights_1) + bias_1)

        weights_final = torch.abs(self.hyper_w_final(states))
        weights_final = weights_final.view(-1, self.embed_dim, 1)
        state_value = self.value(states).view(-1, 1, 1)
        total_q = torch.bmm(hidden, weights_final) + state_value
        return total_q.view(batch_size, -1, 1)
