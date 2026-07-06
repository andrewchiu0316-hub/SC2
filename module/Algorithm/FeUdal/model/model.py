import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 層

    [架構定位]
    作為 Worker 接收 Manager 抽象目標的橋樑，將目標訊號注入局部狀態特徵。

    [數學定義]
    將 Manager 產生的 Latent Goal g_t 映射為調變參數 gamma(g_t) 與 beta(g_t)，
    並對 Worker 的 Latent State z_t 執行逐特徵仿射轉換 (Affine Transformation)：
        z'_t = gamma(g_t) ⊙ z_t + beta(g_t)
    """
    def __init__(self, goal_dim, feature_dim):
        super().__init__()
        # 將 goal 映射為 2 倍的 feature_dim，前半段作為 gamma，後半段作為 beta
        #self.fc = nn.Linear(goal_dim, 2 * feature_dim)
        # 將 goal 與 task one-hot 串接
        self.fc = nn.Linear(goal_dim , 2 * feature_dim)

    def forward(self, x, goal):
        """
        [輸入維度]
        x    : Tensor [Batch, feature_dim] - 來自 state_encoder 的潛在狀態 z_t
        goal : Tensor [Batch, goal_dim]    - 來自 Manager 的抽象目標 g_t

        [輸出維度]
        return : Tensor [Batch, feature_dim] - 調變後的特徵 z'_t
        """
        # 由 g_t 產生仿射轉換參數 [gamma, beta]
        params = self.fc(goal)  # [Batch, 2 * feature_dim]
        gamma, beta = torch.chunk(params, 2, dim=-1)  # 各為 [Batch, feature_dim]

        # 對 z_t 做逐元素仿射調變，得到 z'_t
        return x * gamma + beta

       


class TaskHead(nn.Module):
    """獨立的 Task Head，包含 Actor 和 Critic"""
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.actor(x), self.critic(x)


class RLIR_ManagerAgent(nn.Module):
    """
    Feudal Manager 網路 (MT-RLIR 架構)

    [職責]
    觀察全域狀態 S_t^g，並透過 LSTM 建模時間序列脈絡，每 c_steps 產生一次宏觀決策以引導 Worker。

    [輸出分支]
    1. Goal Head : 輸出確定性 (Deterministic) 的 Latent Goal g_t。
    2. Task Head : 輸出各 Agent 的戰術分配 logits。
    3. Value Head: 估計全域狀態價值 V_m(S_t^g) 以供 Advantage 計算。
    """
    def __init__(self, input_shape: int, args):
        """
        input_shape: global state 維度 (state_g_dim)
        args 需要至少有:
            args.manager_hidden_dim
            args.goal_dim
            args.n_agents
        """
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_tasks  = args.n_tasks
        self.goal_dim = args.goal_dim
        self.hidden_dim = args.manager_hidden_dim
        

        # [重要] 這裡的 input_shape 會在 algorithm.py 傳入時
        # 自動計算為 n_agents * (local_state_dim + n_tasks)
        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_shape, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=False)
        self.goal_head = nn.Linear(self.hidden_dim, self.n_agents * self.goal_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    # -------------------------------------------------
    #  hidden 初始化
    # -------------------------------------------------
    def init_hidden(self, batch_size: int = 1):
        """
        Manager LSTM hidden state 初始化：
            h, c : [1, B, hidden_dim]
        """
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)

    # -------------------------------------------------
    #  前向傳遞
    # -------------------------------------------------
    def forward(self, state_seq, hidden):
        """
        Manager 的前向傳遞 (處理時間序列)

        [參數]
        state_seq : Tensor [T, B, state_g_dim] - 全域狀態序列 S_(t:t+T)^g
        hidden    : Tuple(h, c)，每個維度為 [1, B, hidden_dim] - RNN 隱藏狀態

        [回傳值]
        goals       : Tensor [T, B, n_agents, goal_dim] - L2 正規化後的 Latent Goal g_t
        task_logits : Tensor [T, B, n_agents, n_tasks]  - 各 Agent 的任務機率分佈 (未過 Softmax)
        values      : Tensor [T, B]                     - Critic 預測價值 V_m
        new_hidden  : Tuple(h', c')                     - 更新後的 RNN 隱藏狀態
        """
        # 先將全域狀態編碼為可供 LSTM 處理的隱表示，再沿時間維度滾動更新
        x = self.mlp_encoder(state_seq)      # [T,B,H]
        out, (h, c) = self.rnn(x, hidden)    # [T,B,H]
        T, B, H = out.shape

        # Value Head: 估計 V_m(S_t^g)
        values = self.value_head(out).squeeze(-1)   # [T,B]

        # Goal Head: 為每個 Agent 產生確定性的 latent goal g_t
        goals_flat = self.goal_head(out)  # [T,B,A*d]
        goals = goals_flat.view(T, B, self.n_agents, self.goal_dim)  # [T,B,A,d]
        # L2 normalize 後可讓後續 cosine similarity 類型目標更穩定
        goals = F.normalize(goals, dim=-1)

        # Task Head: 為每個 Agent 輸出對各 task 的分配 logits(被移除了)
        #task_logits_flat = self.task_head(out)  # [T,B,A*n_tasks]
        #task_logits = task_logits_flat.view(T, B, self.n_agents, self.n_tasks)  # [T,B,A,n_tasks]

        # 第二個輸出位置保留 task_logits，供 Workspace/Algorithm 進行戰術選擇
        return goals,  values, (h, c)


class RLIR_WorkerAgent(nn.Module):
    """
    Feudal Worker 網路 (經典流水線：State -> MLP -> FiLM -> RNN -> Task Head)

    [結構流程]
    1. State Encoder (MLP) : 接收原始 State (state_l_dim)，輸出潛在特徵 z_t (hidden_dim)
    2. FiLM Layer          : 使用 g_t 產生的參數對 z_t 進行縮放與偏移，提早注入 Manager 訊號
    3. RNN (LSTM)          : 接收調變後的整合特徵，沿著時間軸建模帶有目標意識的戰術脈絡
    4. Single Task Head    : 單一共用腦袋輸出動作與價值
    """
    def __init__(self, input_shape: int, args):
        super().__init__()
        self.args = args
        self.state_l_dim = input_shape
        self.hidden_dim  = args.worker_hidden_dim
        self.n_agents    = args.n_agents
        self.goal_dim    = args.goal_dim
        self.n_actions   = args.n_actions
        self.n_tasks     = getattr(args, "n_tasks", 1)
        
        # 1. State Encoder (MLP)：特徵萃取，輸出維度為 hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_l_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # 2. FiLM Layer：接收 goal_dim 的目標訊號，並對 hidden_dim 的特徵進行調變
        self.film = FiLMLayer(goal_dim=self.goal_dim, feature_dim=self.hidden_dim)
        
        # 3. RNN (LSTM)：接收 FiLM 調變後的特徵 (hidden_dim)，輸出隱藏狀態 (hidden_dim)
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=False)

        # 4. 單一 Task Head：處理所有任務的 Actor-Critic 輸出層
        self.head = TaskHead(self.hidden_dim, self.n_actions)

    def init_hidden(self, batch_size: int = 1):
        """
        Worker 的 LSTM 看到的有效 batch 大小為 B * A
        """
        BA = batch_size * self.n_agents
        device = next(self.parameters()).device
        h = torch.zeros(1, BA, self.hidden_dim, device=device)
        c = torch.zeros(1, BA, self.hidden_dim, device=device)
        return (h, c)

    def forward(self, local_state_seq, hidden, goal_seq, task_indices=None, return_all_heads=False):
        """
        Worker 的前向傳遞
        """
        T, B, A, Dl = local_state_seq.shape
        flat_state = local_state_seq.view(T, B * A, Dl)
        flat_goal = goal_seq.view(T, B * A, self.goal_dim)
        
        # 1. 原始狀態進入 MLP，編碼成有意義的特徵 (MLP)
        state_features = self.state_encoder(flat_state)
        
        # 2. 透過 FiLM 使用 Manager 的目標訊號進行空間縮放與調變 (FiLM)
        modulated_features = self.film(state_features, flat_goal)
        modulated_features = F.relu(modulated_features)
        
        # 3. 將融合目標後的特徵送入 RNN，提煉時間序列特徵 (RNN)
        rnn_out, (h, c) = self.rnn(modulated_features, hidden)
        
        # 4. 送入單一 Task Head 輸出結果 (Head)
        logits, values = self.head(rnn_out)
        
        final_logits = logits.view(T, B, A, self.n_actions)
        final_values = values.squeeze(-1).view(T, B, A)
        
        return final_logits, final_values, (h, c)