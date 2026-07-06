from module.Algorithm.FeUdal.model.model import RLIR_ManagerAgent, RLIR_WorkerAgent
from module.Algorithm.base_algorithm import BaseAlgorithm

import torch
import torch.nn.functional as F
from Tools import utils
from types import SimpleNamespace
import os
import numpy as np   
#from Tools.pcgrad import PCGrad

# 給module是fc的演算法(worker只有單個rnn)
# 給module是mlp的演算法(worker只有單個rnn)
class Algorithm(BaseAlgorithm):
    """
    Feudal RLIR 演算法：
    - 1 個 Manager：吃 global state，每 c_steps 產生 goal，拿外在獎勵做 n-step actor-critic。
    - 多個 Worker：吃 local state + 各自 goal，用 Categorical 決策 + value head，reward=內在+外在。
    - on-policy：每 c_steps 收一段 segment 再 train 一次，不用 replay buffer。
    """
    def __init__(self, action_num: int, **kwargs):
        super().__init__()

        # 讀取 FeUdal 的設定檔
        cfg_dict = utils.load_config(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config.yaml"
        ))
        # 轉成 namespace，方便用 args.xxx 存取
        self.cfg = cfg_dict
        self.args = SimpleNamespace(**cfg_dict)

        # ----------------- 來自 main.py / Worksapce 的基本資訊 -----------------
        self.action_num  = action_num
        self.n_agents    = kwargs["n_agents"]
        self.state_l_dim = kwargs["state_dim"]      # local state 維度

        # 需要 main.py 在 get_algorithm() 裡多傳 state_g_dim
        #   之後你要在 main.py / Worksapce 把 global state 維度傳進來
        self.state_g_dim = kwargs["state_g_dim"]
        # Global state 在此版本預期為「把所有 agent 的 local state 串起來」
        # 因此理論上 state_g_dim 應該等於 n_agents * state_l_dim
        # （如果你的 global_state 不是這樣定義，請同步調整 sample_action 的拆分邏輯）
        if self.state_g_dim != self.n_agents * self.state_l_dim:
            raise ValueError(
                f"state_g_dim mismatch: got {self.state_g_dim}, "
                f"expected n_agents*state_dim={self.n_agents*self.state_l_dim}"
            )

        # 塞進 args 給網路使用
        self.args.n_agents  = self.n_agents
        self.args.n_actions = self.action_num

        # ----------------- 超參：全部從 config.yaml 讀 -----------------
        self.gamma        = self.cfg["gamma"]
        self.lam          = self.cfg["lambda"]
        self.c_steps      = self.cfg["c_steps"]
        self.alpha_int    = self.cfg["alpha"]          # 內在獎勵權重
        self.eta_ext      = self.cfg["eta"]            # 外在獎勵權重
        self.beta_mgr_v   = self.cfg["beta"]           # manager value loss 權重
        self.entropy_coef = self.cfg["entropy_coef"]
        # [NEW] Task head entropy（讓 task policy 不會太早塌縮）
        self.task_entropy_coef = self.cfg.get("task_entropy_coef", 0.01)
        # goal_dim / state_dim_d 將在下方綁定為 latent space 維度

        # ================= [修改重點 1] 維度設定 =================
        # 讓 Goal 存在於 Worker 的 Hidden 空間中 (Latent Space)，而非原始 State 空間
        self.latent_dim = self.cfg.get("worker_hidden_dim", 256)

        # Manager 輸出的單一 Agent Goal 維度
        self.manager_goal_dim = self.latent_dim
        self.state_dim_d = self.latent_dim

        # Worker 接收的 Goal 維度
        self.worker_goal_dim = self.latent_dim
        # ========================================================

        # Task head：預設 3 種戰術（可由 config.yaml 的 n_tasks 覆蓋）
        self.n_tasks = int(self.cfg.get("n_tasks", 5))

        # ----------------- 建立 Manager / Worker 網路 -----------------
        # Manager: 輸入 global state，輸出「每個 agent 的 local goal」，並可輸出「每個 agent 的 task」
        manager_args = SimpleNamespace(**cfg_dict)
        manager_args.n_agents = self.n_agents
        manager_args.n_actions = self.action_num
        manager_args.goal_dim = self.manager_goal_dim
        manager_args.state_dim_d = self.state_dim_d
        manager_args.n_tasks = self.n_tasks

        self.manager = RLIR_ManagerAgent(
            input_shape=self.state_g_dim,
            args=manager_args
        ).to(self.device)

        # Worker: 輸入 local state，接收 local goal（每個 agent 一個 slice）
        worker_args = SimpleNamespace(**cfg_dict)
        worker_args.n_agents = self.n_agents
        worker_args.n_actions = self.action_num
        # Worker goal_dim = local goal dim
        worker_args.goal_dim = self.worker_goal_dim
        worker_args.n_tasks = self.n_tasks
        worker_args.state_dim_d = self.state_dim_d

        self.worker = RLIR_WorkerAgent(
            input_shape=self.state_l_dim,
            args=worker_args
        ).to(self.device)

        # Optimizers
        mgr_lr = self.cfg["manager_lr"]
        wkr_lr = self.cfg["worker_lr"]

        self.manager_opt = torch.optim.Adam(self.manager.parameters(), lr=mgr_lr)
        self.worker_opt  = torch.optim.Adam(self.worker.parameters(),  lr=wkr_lr)
        # [NEW] 用 PCGrad 包住 worker optimizer（多任務 head 用）
        #self.worker_pcgrad = PCGrad(self.worker_opt)


        # 初始化 RNN hidden
        self.manager_hidden = self.manager.init_hidden(batch_size=1)
        self.worker_hidden  = self.worker.init_hidden(batch_size=1)

        # 【新增記憶書籤，用來保存訓練起點的記憶】
        self.segment_start_manager_hidden = self.manager_hidden
        self.segment_start_worker_hidden  = self.worker_hidden

        # on-policy segment buffer（一段長度 <= c_steps）
        self.seg_buffer = []   # list[dict]，每個 dict 是一個 time step 的資料
        self.seg_step   = 0

        # 暫存上一個 step 的 (s, g, logp, v)，因為 reward 是「下一次 call sample_action 時才知道」
        self._last_step_cache = None

        # 紀錄最近一次 train 的 loss，給 log 用
        self.last_loss_worker  = 0.0
        self.last_loss_manager = 0.0
        # [新增] 全域獎勵放大倍率，確保 Actor 與 Critic 權重平衡
        self.reward_scale = 100.0
        # ================= [新增] 詳細 Loss 紀錄變數 =================
        self.last_worker_policy_loss = 0.0
        self.last_worker_value_loss  = 0.0
        self.last_manager_policy_loss = 0.0
        self.last_manager_value_loss  = 0.0

        # -------- [NEW]  Tensorboard 細部輸出 --------
        self.last_manager_goal_loss = 0.0
        self.last_manager_task_loss = 0.0
        self.last_per_agent_reward = [0.0] * self.n_agents
        self.just_updated = False
        # -----------------------------------------------------
        
        # [NEW] 新增：詳細數值監控變數
        self.last_manager_cos_sim    = 0.0  # Cosine Similarity
        self.last_manager_advantage  = 0.0  # Advantage (A_mgr)
        self.last_manager_return     = 0.0  # Manager 的 R_ext
        self.last_manager_value_pred = 0.0  # [新增] Manager 的 V_m (Prediction)
        self.last_worker_mean_reward = 0.0  # Worker 的平均混合獎勵
        
        # [新增] per-agent loss（保留 agent 維度，沿時間維度平均）
        self.last_per_agent_policy_loss = [0.0] * self.n_agents
        self.last_per_agent_value_loss  = [0.0] * self.n_agents
        self.last_per_agent_cos_sim     = [0.0] * self.n_agents
        # ===========================================================

    

    # ----------------- buffer helper -----------------
    def _store_step(self, s_l, s_g, goal, task_idx, action, logp_w, v_w, r_ext_m, r_ext_w, r_int, done, alive_mask, avail_actions=None):
        """把一個 time step 的資料丟進 segment buffer。"""
        step = {
            "s_l": s_l.detach().clone(),         # [A, Dl]
            "s_g": s_g.detach().clone(),         # [Dg]
            "goal": goal.detach().clone(),       # [A, d]
            "task_idx": torch.as_tensor(task_idx, dtype=torch.long, device=s_l.device),  # [A] 或 [Scalar]
            "a": action.detach().clone(),        # [A]
            "logp_w": logp_w.detach().clone(),   # [A]
            "v_w": v_w.detach().clone(),         # [A]
            "r_ext_m": r_ext_m.detach().clone(), # [A]
            "r_ext_w": r_ext_w.detach().clone(), # [A]
            "r_int": r_int.detach().clone(),     # [A]
            "done": bool(done),                  # 使用 sample_action 傳入的「當下 done」
            "alive": alive_mask.detach().clone(),  # [A]
            # [新增] 將 Mask 轉為 bool tensor 存入
            "avail_actions": torch.tensor(avail_actions, dtype=torch.bool, device=s_l.device) if avail_actions is not None else None,
        }
        self.seg_buffer.append(step)
        self.seg_step += 1

    # ----------------- Worker / Manager update -----------------
    #  next_alive_mask 參數 (預設給 None 確保相容性)
    def train(self, next_global_state=None, next_alive_mask=None):
        """
        用 seg_buffer 的一段資料做一次 manager + worker 更新。
        next_global_state: 用於計算 Bootstrap value (當 done=False 時)
        """
        if len(self.seg_buffer) == 0:
            return None

        device = self.device
        L = len(self.seg_buffer)        # segment 長度
        A = self.n_agents
        d = self.state_dim_d
        Dl = self.state_l_dim          # local state 維度

        # 把 list[dict] -> tensor
        s_l = torch.stack([step["s_l"] for step in self.seg_buffer], dim=0).to(device)   # [L,A,Dl]
        s_g = torch.stack([step["s_g"] for step in self.seg_buffer], dim=0).to(device)   # [L,Dg]
        goal = torch.stack([step["goal"] for step in self.seg_buffer], dim=0).to(device) # [L,A,Dl]
        task_idx_seq = torch.stack([step["task_idx"] for step in self.seg_buffer], dim=0).to(device)  # [L,A] 或 [L]
        a = torch.stack([step["a"] for step in self.seg_buffer], dim=0).to(device)       # [L,A]
        
        # [新增] 讀取存在 buffer 裡的 r_int
        r_int = torch.stack([step["r_int"] for step in self.seg_buffer], dim=0).to(device)  # [L,A]

        r_ext_m = torch.stack([step["r_ext_m"] for step in self.seg_buffer], dim=0).to(device)  # [L,A]
        r_ext_w = torch.stack([step["r_ext_w"] for step in self.seg_buffer], dim=0).to(device)  # [L,A]
        done = torch.as_tensor(
            [float(step["done"]) for step in self.seg_buffer],
            dtype=torch.float32, device=device
        )                                                          # [L]
        alive = torch.stack([step["alive"] for step in self.seg_buffer], dim=0).to(device)  # [L,A]
        # [新增] 從 Buffer 讀出這段軌跡的 Mask，形狀為 [L, A, n_actions]
        avail_actions = torch.stack([step["avail_actions"] for step in self.seg_buffer], dim=0).to(device)

        # ================== 1. Worker: 重新 forward 得到 logits / value / logp ==================
        # 把這一段的 local_state / goal 視為長度 L 的序列，重新跑一次 worker
        s_l_seq  = s_l.view(L, 1, A, Dl)     # [L,1,A,Dl]

        # task_idx_seq 原本可能是 [L]（舊格式），這裡統一成 [L,A]
        if task_idx_seq.dim() == 1:
            task_idx_seq = task_idx_seq.view(L, 1).expand(L, A)

        # 直接使用 goal，不拼接 task onehot
        goal_seq = goal.view(L, 1, A, -1)  # [L,1,A,Dl]

        # =========================================================
        # [修改] 呼叫單一 Task Head 的 Worker，直接拿取 logits 與 values
        # =========================================================
        h0_w = self.segment_start_worker_hidden
        task_idx_seq_worker = task_idx_seq.view(L, 1, A)
        
        logits_seq, v_w_seq, _ = self.worker(s_l_seq, h0_w, goal_seq, task_indices=task_idx_seq_worker)

        logits = logits_seq[:, 0]  # [L,A,n_actions]
        v_w = v_w_seq[:, 0]        # [L,A]

        # [新增] 在 train() 也要套用 Action Mask，避免算出錯誤的機率與 Entropy
        # =========================================================
        logits[~avail_actions] = -1e10

        dist_w = torch.distributions.Categorical(logits=logits)
        logp_w = dist_w.log_prob(a.long())   # [L, A]
        
        # ====== Worker Reward：使用 Worker 專屬外在獎勵 ======
        r_w = self.alpha_int * r_int + self.eta_ext * r_ext_w    # [L, A]
        
        # ----------------- Worker GAE -----------------
        with torch.no_grad():
            # 取得「下一步是否存活」的遮罩 (Dead Agent 的未來價值必須為 0)
            alive_next = torch.zeros_like(alive)
            alive_next[:-1] = alive[1:]

            # 精確處理 Segment 邊界的死亡狀態
            if next_alive_mask is not None:
                next_alive_t = torch.as_tensor(next_alive_mask, dtype=alive.dtype, device=device).view(-1)
                alive_next[-1] = next_alive_t
            else:
                alive_next[-1] = alive[-1]  # 最後一步的狀態近似為當下的 alive

            # bootstrap：把下一步 value 視為 shift 一格
            v_w_next = torch.zeros_like(v_w)
            v_w_next[:-1] = v_w[1:]
            
            # 避免截斷誤差。如果遊戲還沒結束，最後一步的 V 不能是 0
            v_w_next[-1] = torch.where(done[-1] > 0.5, torch.zeros_like(v_w[-1]), v_w[-1])
            
            # 死亡截斷。如果下一步該 agent 死了，未來的 Bootstrap 必須強制作廢 (乘上 0)
            v_w_next_masked = v_w_next * alive_next

            # 計算 TD Error (使用 v_w_next_masked)
            delta = r_w + self.gamma * (1.0 - done.view(L, 1)) * v_w_next_masked - v_w   # [L,A]
        
            adv = torch.zeros_like(delta)
            gae = torch.zeros(A, device=device)
            for t in reversed(range(L)):
                # GAE 不能從「已經死亡的未來狀態」回傳回來
                # next_alive 決定了這條船在 t+1 是否還活著，死了就截斷未來的 GAE 傳遞
                next_alive = alive[t+1] if t < L - 1 else alive[-1]
                
                # 同時受到全局 done 與 個體 next_alive 的截斷
                gae = delta[t] + self.gamma * self.lam * (1.0 - done[t]) * next_alive * gae
                adv[t] = gae
        
            R_w = adv + v_w
        
        # ----------------- Worker Loss (for logging) -----------------
        # mask 死掉的船
        valid_mask = alive  # [L,A]
        
        total_denom = valid_mask.sum().clamp(min=1.0)
        agent_denom = valid_mask.sum(dim=0).clamp(min=1.0)  # [A]

        # Advantage 正規化 (僅計算存活的有效步數)
        adv_det = adv.detach()
        adv_masked = adv_det * valid_mask
        adv_mean = adv_masked.sum() / total_denom
        adv_std = torch.sqrt((((adv_det - adv_mean) * valid_mask) ** 2).sum() / total_denom + 1e-8)
        adv_norm = ((adv_det - adv_mean) / adv_std) * valid_mask
        
        # actor：改用 adv_norm 來計算 Policy Loss
        term = - (adv_norm * logp_w).sum(dim=0) / agent_denom  # [A]
        per_agent_policy_loss = torch.nan_to_num(term, nan=0.0)
        
        # 2. 計算個別 Agent 的 Value Loss（沿時間維度平均，保留 agent 維度）
        #per_agent_value_loss  = ((R_w.detach() - v_w)**2 * valid_mask).sum(dim=0) / agent_denom  # [A]
        
        # 3. 計算原本的總 Loss (用來做 backward)
        # 為了保持跟原本數學定義一致，這裡還是用總合除以總步數
        worker_policy_loss = - (adv_norm * logp_w).sum() / total_denom
        # critic：R_w 不對 v_w 反傳，所以用 R_w.detach()
        #worker_value_loss  = ((R_w.detach() - v_w)**2 * valid_mask).sum() / total_denom

        # 新的寫法 (Smooth L1 Loss)：
        w_loss_unreduced = F.smooth_l1_loss(v_w, R_w.detach(), reduction='none', beta=1.0)
        per_agent_value_loss = (w_loss_unreduced * valid_mask).sum(dim=0) / agent_denom
        worker_value_loss    = (w_loss_unreduced * valid_mask).sum() / total_denom

        # entropy：直接用剛剛同一個 dist_w 就好，不用再 forward 一次
        entropy = dist_w.entropy()                          # [L,A]
        worker_entropy = (entropy * valid_mask).sum() / total_denom
        # =================================================================
        # [新增] 計算並保留每個 Agent 的獨立 Entropy (沿時間維度平均)
        per_agent_entropy = (entropy * valid_mask).sum(dim=0) / agent_denom  # [A]
        # =================================================================
        #worker_loss = worker_policy_loss + worker_value_loss - self.entropy_coef * worker_entropy
        worker_loss = (worker_policy_loss * self.reward_scale) + worker_value_loss - (self.entropy_coef * worker_entropy * self.reward_scale)
        

        # ================== 2. Manager: anchor-only 版本 (修正版, Global Goal + Task Head) ==================
        # 這段 segment 只有在 t=0 時 Manager 做了一次決策（產生 goal）
        # 所以只針對決策當下的 goal / task / value 計算梯度

        # 取出 segment 第一個 global state（t=0）
        s_g_at_start = s_g[0].view(1, 1, -1)   # [1, 1, Dg]

        # 【Manager 使用記憶書籤】
        h_m0 = self.segment_start_manager_hidden

        # 只 forward 一步
        goals_seq, v_m_seq, _ = self.manager(s_g_at_start, h_m0)

        # 取出 [T=0, B=0] 包含所有 Agent 的 Goal，並攤平成全域向量 [A*d]
        # - 若 Manager 輸出為 [1,1,A,d]：goals_seq[0,0] => [A,d]，reshape(-1) => [A*d]
        # - 若 Manager 輸出為 [1,1,1,Dg]：goals_seq[0,0] => [1,Dg]，reshape(-1) => [Dg]
        global_goal_with_grad = goals_seq[0, 0].reshape(-1)

        
        v_m_start = v_m_seq[0, 0]

        # ---- (1) 外在 n-step return ----
        # 【Manager 看的是全域表現，所以把 r_ext_m [L, A] 沿著 A 取平均變成 [L]】
        r_ext_manager = r_ext_m.mean(dim=-1)

        # 計算從 t=0 到 t=L-1 的累積回報（遇到 done 提早截斷）
        with torch.no_grad():
            R_ext = torch.tensor(0.0, device=device)
            discount = 1.0
            terminal_hit = False
            for t in range(L):
                R_ext = R_ext + discount * r_ext_manager[t]
                discount *= self.gamma
                if done[t].item() > 0.5:
                    terminal_hit = True
                    break

            # ---------------- [修改核心重點] ----------------
            # 如果這段 segment 結束了，但遊戲還沒結束 (terminal_hit = False)
            # 我們必須加上「未來預期價值」 (Bootstrap Value)
            if not terminal_hit:
                v_bootstrap = torch.tensor(0.0, device=device)

                # 如果有傳入 next_global_state，就用網路預測未來價值
                if next_global_state is not None:
                    # 1. 轉成 Tensor [1, 1, Dg]
                    # 注意：如果 next_global_state 是 numpy array，要先轉 tensor
                    if isinstance(next_global_state, np.ndarray):
                        s_g_next = torch.from_numpy(next_global_state).float().to(device)
                    elif isinstance(next_global_state, torch.Tensor):
                        s_g_next = next_global_state.float().to(device)
                    else:
                        # 防呆：如果是 list / tuple
                        s_g_next = torch.tensor(next_global_state, dtype=torch.float32, device=device)

                    # 使用 reshape 避免潛在的 Contiguous 記憶體錯誤
                    s_g_next = s_g_next.reshape(1, 1, -1)

                    # 2. 直接使用 Manager 當下最新的記憶來預測未來
                    h_m_next = (self.manager_hidden[0].detach(), self.manager_hidden[1].detach())

                    # 3. Forward pass (不需要梯度)
                    _, v_m_next_seq, _ = self.manager(s_g_next, h_m_next)
                    v_bootstrap = v_m_next_seq[0, 0].detach()  # 務必 detach，不要傳梯度

                # 加上折現後的未來價值
                R_ext = R_ext + discount * v_bootstrap
            # ----------------------------------------------

        # Advantage（只針對 t=0 的決策）
        A_mgr = R_ext - v_m_start
        
        # [修改重點 1]：將 Manager 的 Advantage 進行截斷，防止梯度爆炸
        # 由於此處 batch=1 無法做正規化，使用 clamp 限制影響力
        #A_mgr_clipped = torch.clamp(A_mgr, min=-15.0, max=15.0).detach()
        # 因為前面獎勵放大了 100 倍，這裡的容忍範圍也要等比例放大到 1500
        A_mgr_clipped = torch.clamp(A_mgr, min=-1500.0, max=1500.0).detach()
        # [新增] 取得 t=0 的存活遮罩，Manager 不該對死掉的 Agent 更新策略
        valid_agents_t0 = alive[0]  # [A]
        denom_t0 = valid_agents_t0.sum().clamp(min=1.0)

        # ---- (2) Goal Loss：Cosine Similarity ----
        # 使用 Worker 的 state_encoder 將起終點 State 轉換到 Latent Space
        with torch.no_grad():
            z_start = self.worker.state_encoder(s_l[0])   # [A, latent_dim]
            z_end = self.worker.state_encoder(s_l[-1])    # [A, latent_dim]

        delta_mgr_agents = z_end - z_start                     # [A, latent_dim]
        goal_agents = global_goal_with_grad.view(A, d)         # [A, latent_dim]

        delta_norm = F.normalize(delta_mgr_agents, dim=-1, eps=1e-8)
        goal_norm_grad = F.normalize(goal_agents, dim=-1, eps=1e-8)

        cos_sim_per_agent = (delta_norm.detach() * goal_norm_grad).sum(dim=-1)  # [A]
        
        # 只計算活著的 Agent 的平均 Cosine Similarity
        cos_sim = (cos_sim_per_agent * valid_agents_t0).sum() / denom_t0

        # 圖中公式：- Sum [ A_t * cos(...) ]
        loss_goal = - A_mgr_clipped * cos_sim

        

        # ---- (4) Value Loss ----
        #manager_value_loss  = (R_ext.detach() - v_m_start) ** 2
        #loss_value = self.beta_mgr_v * manager_value_loss
        # 新的寫法 (Smooth L1 Loss)：
        manager_value_loss = F.smooth_l1_loss(v_m_start, R_ext.detach(), beta=1.0)
        
        loss_value = self.beta_mgr_v * manager_value_loss

        manager_loss = loss_goal  + loss_value

        # ================== 3. 反向傳遞 ==================
        # 確保每次更新前清空 Worker 梯度
        self.worker_opt.zero_grad()
        
        # =========================================================
        # [修改] 單一 Head 直接反向傳遞 worker_loss
        # =========================================================
        if valid_mask.sum() > 0:
            worker_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=5.0)
            self.worker_opt.step()
        else:
            self.worker_opt.zero_grad()

        self.manager_opt.zero_grad()
        manager_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=5.0)
        self.manager_opt.step()

        self.last_loss_worker  = float(worker_loss.item())
        self.last_loss_manager = float(manager_loss.item())

        # ================= [新增] 儲存詳細 Loss =================
        self.last_worker_policy_loss = float(worker_policy_loss.item())
        self.last_worker_value_loss  = float(worker_value_loss.item())
        
        # [新增] 將 Tensor [A] 轉成 list[float] 存起來，給 Workspace 用
        self.last_per_agent_policy_loss = per_agent_policy_loss.detach().cpu().tolist()
        self.last_per_agent_value_loss  = per_agent_value_loss.detach().cpu().tolist()
        # =================================================================
        # [新增這行] 把算好的 per_agent_entropy 交給 Workspace 抓取
        self.last_per_agent_entropy = per_agent_entropy.detach().cpu().tolist()
        # =================================================================
        # -------- [修改] 拆分 Manager 的 Policy(Goal) 與 Task Loss --------
        self.last_manager_goal_loss = float(loss_goal.item())
        self.last_manager_task_loss = 0.0
        
        # policy loss 現在包含：goal + task
        self.last_manager_policy_loss = float(loss_goal.item())
        self.last_manager_value_loss  = float(manager_value_loss.item())
        
        # [NEW] 新增：儲存您要求的詳細分析指標
        self.last_manager_cos_sim    = float(cos_sim.item())      # 餘弦相似度 (scalar)
        self.last_manager_advantage  = float(A_mgr.item())        # Advantage (scalar)
        # 1. 這是 R_t (Target)
        self.last_manager_return     = float(R_ext.item())        # Manager Return (scalar)
        # 2. [新增] 這是 V_m (Prediction)
        self.last_manager_value_pred = float(v_m_start.item())
        self.last_worker_mean_reward = float(r_w.mean().item())   # Worker Reward 取平均

        # -------- [新增] 儲存 Worker 各個 Agent 的綜合獎勵 (沿著時間軸 T 平均) --------
        self.last_per_agent_reward = r_w.mean(dim=0).detach().cpu().tolist()

        # 儲存每個 Agent 當下的 Cosine Similarity（針對 t=0 的單步決策）
        self.last_per_agent_cos_sim = cos_sim_per_agent.detach().cpu().tolist()
        # =======================================================

        # 清掉這段 segment
        self.seg_buffer = []
        self.seg_step   = 0

        return self.last_loss_worker + self.last_loss_manager

    # ----------------- sample_action：一邊收資料、一邊決定何時 train -----------------
    def sample_action(self,
                      local_state,
                      global_state,
                      reward_ext_m,
                      reward_ext_w,
                      done,
                      alive_mask,
                      task_idx=None,
                      avail_actions=None):
        """
        Feudal 版 sample_action 介面（之後需同步修改 Worksapce）：

        local_state : list/np.ndarray, shape [n_agents, state_l_dim]
        global_state: np.ndarray, shape [state_g_dim]
        reward_ext_m: None (第一步) 或 list[float] (上一個 step 的 Manager 外在獎勵)
        reward_ext_w: None (第一步) 或 list[float] (上一個 step 的 Worker 外在獎勵)
        done        : bool, 當前這個 obs 是否 episode 結束
        alive_mask  : list[bool], len = n_agents，alive=1 才算 loss
        """
        device = self.device
        A = self.n_agents

        # -------- [新增] 每次進來先歸零標記 --------
        self.just_updated = False
        # [新增] 全域獎勵放大倍率，確保 Actor 與 Critic 權重平衡
        self.reward_scale = 100.0
        # 轉成 tensor
        # local_state 現在是 list[np.ndarray]，先堆成一個 np.ndarray 再轉 torch
        if isinstance(local_state, np.ndarray):
            local_state_arr = local_state.astype(np.float32, copy=False)
        else:
            local_state_arr = np.asarray(local_state, dtype=np.float32)  # [A, Dl]
        
        s_l = torch.from_numpy(local_state_arr).to(device=device)        # [A,Dl]
        s_g = torch.as_tensor(global_state, dtype=torch.float32, device=device)   # [Dg]
        alive = torch.as_tensor(alive_mask, dtype=torch.float32, device=device)   # [A]

        # ------------ 1. 儲存上一步（Cache -> Buffer）------------
        if reward_ext_m is not None and reward_ext_w is not None and self._last_step_cache is not None:
            cache = self._last_step_cache
            """
            # (A) 外在獎勵：Manager / Worker 分流
            if isinstance(reward_ext_m, (list, tuple)):
                r_ext_m_tensor = torch.tensor(reward_ext_m, dtype=torch.float32, device=device)
            else:
                r_ext_m_tensor = torch.tensor([float(reward_ext_m)] * A, dtype=torch.float32, device=device)

            if isinstance(reward_ext_w, (list, tuple)):
                r_ext_w_tensor = torch.tensor(reward_ext_w, dtype=torch.float32, device=device)
            else:
                r_ext_w_tensor = torch.tensor([float(reward_ext_w)] * A, dtype=torch.float32, device=device)
            """
            # (A) 外在獎勵：Manager / Worker 分流
              

            if isinstance(reward_ext_m, (list, tuple)):
                r_ext_m_tensor = torch.tensor(reward_ext_m, dtype=torch.float32, device=device) * self.reward_scale
            else:
                r_ext_m_tensor = torch.tensor([float(reward_ext_m)] * A, dtype=torch.float32, device=device) * self.reward_scale

            if isinstance(reward_ext_w, (list, tuple)):
                r_ext_w_tensor = torch.tensor(reward_ext_w, dtype=torch.float32, device=device) * self.reward_scale
            else:
                r_ext_w_tensor = torch.tensor([float(reward_ext_w)] * A, dtype=torch.float32, device=device) * self.reward_scale
            
            # (B) 內在獎勵：r_int = cos( Z_{t+1} - Z_t, goal )
            # 透過 Worker 的 state_encoder 將 state 壓縮到 latent space
            with torch.no_grad():
                z_curr = self.worker.state_encoder(cache["s_l"]) # [A, latent_dim]
                z_next = self.worker.state_encoder(s_l)          # [A, latent_dim]

            delta_z = z_next - z_curr  # [A, latent_dim]
            goal_curr = cache["goal"]  # [A, latent_dim]
            eps = 1e-8
            d_norm = F.normalize(delta_z, dim=-1, eps=eps)
            g_norm = F.normalize(goal_curr, dim=-1, eps=eps)
            #r_int_tensor = (d_norm * g_norm).sum(dim=-1)  # [A]
            # [修改] 將算出來的內在獎勵也乘上 self.reward_scale
            r_int_tensor = (d_norm * g_norm).sum(dim=-1) * self.reward_scale  # [A]

            # (C) 存入 Buffer：done 使用「當下傳入的 done」
            self._store_step(
                s_l       = cache["s_l"],
                s_g       = cache["s_g"],
                goal      = cache["goal"],
                task_idx  = cache.get("task_idx", 0),
                action    = cache["action"],
                logp_w    = cache["logp_w"],
                v_w       = cache["v_w"],
                r_ext_m   = r_ext_m_tensor,
                r_ext_w   = r_ext_w_tensor,
                r_int     = r_int_tensor,
                done      = done,
                alive_mask= cache["alive"],
                avail_actions= cache["avail_actions"],
            )

        # ------------ 2. 判斷是否訓練 ------------
        # 如果 segment 滿了，或剛存進去的那一步導致 done，就訓練
        if (self.seg_step >= self.c_steps) or (done and self.seg_step > 0):
            # Worksapce.action() 是在 with torch.no_grad() 裡呼叫 sample_action 的
            # 這裡必須暫時打開 autograd，讓 train() 裡的 forward + backward 能正常記錄計算圖
            with torch.enable_grad():
                # 傳入 global_state 作為 next_state
                # 如果 done=True，這個 next_global_state 其實不會被用到 (terminal_hit 會是 True)
                # 如果 done=False，這個就是我們需要的 Bootstrap 依據
                # 傳入 next_alive_mask 讓演算法精確判斷邊界生死
                self.train(next_global_state=global_state, next_alive_mask=alive)
                # -------- [新增] 標記這一步有成功進行訓練 --------
                self.just_updated = True

        # ------------ 決定這一個 step 要不要更新 goal（新 segment 開頭） ------------
        new_segment = (self.seg_step == 0)

        if new_segment:
            # 【每段新劇情開始，存下當前的記憶書籤 (detach 避免梯度回傳錯誤)】
            self.segment_start_manager_hidden = (self.manager_hidden[0].detach(), self.manager_hidden[1].detach())
            self.segment_start_worker_hidden  = (self.worker_hidden[0].detach(),  self.worker_hidden[1].detach())

            # Manager 用當前 global state 產生新的 goal 與 task
            sg_seq = s_g.view(1, 1, -1)     # [T=1,B=1,Dg]
            goals_seq, v_m_seq, self.manager_hidden = self.manager(sg_seq, self.manager_hidden)
            # 直接取出所有 Agent 的 Local Goals [A, d]
            # goals_seq[0, 0] 的形狀已經是 [n_agents, goal_dim]
            goal = goals_seq[0, 0]
            self.current_goal = goal

            
        else:
            goal   = self.current_goal
            # [修正] 刪除 Manager 抽籤 (Categorical sample) 的邏輯
        # 強制將外部的 Workspace 傳進來的 task_idx 套用到所有 Agent
        if task_idx is not None:
            if isinstance(task_idx, (int, float)):
                self.current_task_idx = np.full(A, int(task_idx), dtype=np.int64)
            else:
                self.current_task_idx = np.asarray(task_idx, dtype=np.int64).reshape(A)
        else:
            if not hasattr(self, "current_task_idx") or self.current_task_idx is None:
                self.current_task_idx = np.zeros(A, dtype=np.int64)
        # ------------ Worker：用 local_state + goal 決定 action ------------
        s_l_seq = s_l.view(1, 1, A, -1)      # [1,1,A,Dl]
        # 移除 task onehot 拼接，直接使用 goal
        goal_seq = goal.view(1, 1, A, -1)    # [1,1,A,Dl]

        task_idx_step = torch.as_tensor(self.current_task_idx, device=device).long().view(1, 1, A)
        # [修改] 移除 return_all_heads 參數
        logits_seq, v_w_seq, self.worker_hidden = \
            self.worker(s_l_seq, self.worker_hidden, goal_seq, task_indices=task_idx_step)
        logits = logits_seq[0, 0]            # [A,n_actions]
        v_w    = v_w_seq[0, 0]              # [A]
        # =================================================================
        # [新增] Action Mask 邏輯：強制把不合法的動作機率降到 0
        # =================================================================
        if avail_actions is not None:
            # 將傳進來的 list 轉成 PyTorch bool Tensor，形狀為 [A, n_actions]
            mask = torch.tensor(avail_actions, device=device, dtype=torch.bool)
            # ~mask 代表不合法的位置 (False 反轉為 True)
            # 將不合法動作的原始數值 (logits) 設為極小值，經過 Softmax 後機率就會變成 0
            logits[~mask] = -1e10
        # =================================================================
        dist_w = torch.distributions.Categorical(logits=logits)
        action = dist_w.sample()            # [A]
        logp_w = dist_w.log_prob(action)    # [A]

        # ------------ 把「這一步」的資料 cache 起來，等下一步拿到 reward 才能存 buffer ------------
        self._last_step_cache = {
            "s_l": s_l,
            "s_g": s_g,
            "goal": goal,  # 純 goal（給 intrinsic reward 用）
            "task_idx": self.current_task_idx,
            "action": action,
            "logp_w": logp_w,
            "v_w": v_w,
            "alive": alive,
            "avail_actions": avail_actions,
        }

        # 回傳給 Workspace 的 Task ID：one-hot [1,1,A,n_tasks]
        task_onehot_out = np.zeros((1, 1, A, self.n_tasks), dtype=np.int32)
        batch_indices = np.arange(A)
        task_idx_np = np.asarray(self.current_task_idx, dtype=np.int64).reshape(-1)
        if task_idx_np.shape[0] == A:
            task_onehot_out[0, 0, batch_indices, task_idx_np] = 1

        return action.detach().cpu().tolist(), task_onehot_out


    # ----------------- Episode 控制 -----------------
    def episode_reset(self):
        self.reset_hidden()

    def reset_hidden(self):
        """重置 hidden 與 segment"""
        self.manager_hidden = self.manager.init_hidden(batch_size=1)
        self.worker_hidden  = self.worker.init_hidden(batch_size=1)

        # 【重置時也要把書籤歸零】
        self.segment_start_manager_hidden = self.manager_hidden
        self.segment_start_worker_hidden  = self.worker_hidden

        self.seg_buffer = []
        self.seg_step   = 0
        self._last_step_cache = None
        self.current_goal = None
        self.current_task_idx = None


    def update_target_network(self):
        # Feudal RLIR 是 on-policy actor-critic，不用 target network
        return
        
    # ----------------- save / load model -----------------
    def save_model(self, ckpt: dict):
        ckpt["manager"] = self.manager.state_dict()
        ckpt["worker"]  = self.worker.state_dict()
        ckpt["manager_opt"] = self.manager_opt.state_dict()
        ckpt["worker_opt"]  = self.worker_opt.state_dict()
        return ckpt

    def load_model(self, ckpt: dict):
        if "manager" in ckpt:
            self.manager.load_state_dict(ckpt["manager"])
        if "worker" in ckpt:
            self.worker.load_state_dict(ckpt["worker"])
        if "manager_opt" in ckpt:
            self.manager_opt.load_state_dict(ckpt["manager_opt"])
        if "worker_opt" in ckpt:
            self.worker_opt.load_state_dict(ckpt["worker_opt"])
        return self
