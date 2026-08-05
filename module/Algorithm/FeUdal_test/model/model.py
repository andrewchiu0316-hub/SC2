import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskHead(nn.Module):
    """Actor-critic output head for the worker."""
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.actor(x), self.critic(x)


class RLIR_ManagerAgent(nn.Module):
    """Feudal manager: global state -> per-agent normalized latent goals."""
    def __init__(self, input_shape: int, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_tasks = args.n_tasks
        self.goal_dim = args.goal_dim
        self.hidden_dim = args.manager_hidden_dim

        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_shape, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=False)
        self.goal_head = nn.Linear(self.hidden_dim, self.n_agents * self.goal_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def forward(self, state_seq, hidden):
        x = self.mlp_encoder(state_seq)
        out, (h, c) = self.rnn(x, hidden)
        T, B, _ = out.shape
        values = self.value_head(out).squeeze(-1)
        goals_flat = self.goal_head(out)
        goals = goals_flat.view(T, B, self.n_agents, self.goal_dim)
        goals = F.normalize(goals, dim=-1)
        return goals, values, (h, c)


class RLIR_WorkerAgent(nn.Module):
    """Worker variant for feudal_test: concat local state and goal, no FiLM."""
    def __init__(self, input_shape: int, args):
        super().__init__()
        self.args = args
        self.state_l_dim = input_shape
        self.hidden_dim = args.worker_hidden_dim
        self.n_agents = args.n_agents
        self.goal_dim = args.goal_dim
        self.n_actions = args.n_actions
        self.n_tasks = getattr(args, "n_tasks", 1)

        # Shared by the policy, intrinsic reward, and manager goal objective.
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_l_dim + self.goal_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=False)
        self.head = TaskHead(self.hidden_dim, self.n_actions)

    def init_hidden(self, batch_size: int = 1):
        ba = batch_size * self.n_agents
        device = next(self.parameters()).device
        h = torch.zeros(1, ba, self.hidden_dim, device=device)
        c = torch.zeros(1, ba, self.hidden_dim, device=device)
        return (h, c)

    def encode_state_goal(self, local_state, goal):
        worker_input = torch.cat([local_state, goal], dim=-1)
        return self.state_encoder(worker_input)

    def forward(self, local_state_seq, hidden, goal_seq, task_indices=None, return_all_heads=False):
        T, B, A, Dl = local_state_seq.shape
        flat_state = local_state_seq.view(T, B * A, Dl)
        flat_goal = goal_seq.view(T, B * A, self.goal_dim)

        features = self.encode_state_goal(flat_state, flat_goal)
        rnn_out, (h, c) = self.rnn(features, hidden)
        logits, values = self.head(rnn_out)

        final_logits = logits.view(T, B, A, self.n_actions)
        final_values = values.squeeze(-1).view(T, B, A)
        return final_logits, final_values, (h, c)
