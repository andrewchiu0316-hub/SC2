# FeUdal HAA2C

This hybrid keeps the high-level Manager and FiLM goal conditioning from the
repository's original FeUdal implementation, while adapting the worker update
scheme from HAA2C in [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL).

Core HAA2C behavior retained here:

- one independent actor and optimizer per agent, without parameter sharing;
- a centralized external V supplied by the Manager's recurrent value head;
- GAE computed from one shared team reward;
- one independent intrinsic critic per worker, with goal-conditioned GAE;
- randomized sequential actor updates;
- a multiplicative factor that carries importance ratios from workers already
  updated in the current sequence;
- PPO-clipped actor objectives with entropy regularization;
- multiple actor and critic epochs, Huber value loss, value clipping, and
  gradient clipping.

FeUdal-specific behavior retained here:

- the Manager processes the global state and advances its LSTM every step,
  producing the external V used by worker GAE;
- one normalized goal per worker is adopted every `manager_c_steps` (currently
  10); between these boundaries the workers keep their existing goals;
- each SMAC environment collects `rollout_steps` (currently 200) transitions;
  the batch has `rollout_steps * n_rollout_threads` team samples;
- worker rollouts may cross episode boundaries, with recurrent state reset
  independently for each environment by terminal masks during training;
- each worker uses local observation encoding followed by FiLM goal modulation;
- the shared Manager value head and external GAE use environment reward;
- intrinsic reward is one value per completed manager goal segment:
  `cosine(z_segment_end - z_segment_start, segment_start_goal)`, using each
  worker's pre-FiLM `state_encoder`. It is written to the segment's final valid
  transition before optimizer updates, then intrinsic GAE propagates it to the
  earlier actions. Zero displacement, already-dead workers, and transitions
  into death receive zero intrinsic reward;
- each intrinsic critic takes global state, its worker's local observation,
  its held goal, and remaining goal steps divided by `manager_c_steps`. It has
  two 128-unit layers and its own Adam optimizer, without actor parameter sharing;
- intrinsic GAE ends when the goal expires, the worker dies, or the episode ends.
  It bootstraps within unfinished goal segments, including rollout boundaries;
- worker `i` uses `normalize(A_ext + intrinsic_coef * A_int[..., i])`, with
  normalization over its alive samples. The entire mixed advantage is multiplied
  by `factor` and the PPO-clipped current-worker ratio surrogate. The HAA2C
  `factor` itself is not clipped. `worker_ratio_clip` defaults to 0.2;
  `intrinsic_coef` controls the intrinsic actor contribution;
- the Manager retains its external n-step return, advantage, latent displacement
  cosine objective, and segment-start value loss. Segment replay now processes
  every state before computing the segment-end bootstrap;
- the same value head also learns from every worker rollout step, using the
  unnormalized GAE target `return[t] = advantage[t] + old_value[t]`;
- value training replays `[time, environment, state]` from the saved Manager
  hidden state, resetting each environment's memory after terminal transitions;
- next-state bootstrap values are recorded before optimizer updates. Evaluating
  a bootstrap does not advance the live Manager hidden state;
- one Manager Adam optimizer handles both segment updates and per-step value
  updates. `critic_epoch`, `value_loss_coef`, and the existing value clipping /
  Huber settings control per-step value training; `manager_value_loss_coef`
  controls the segment-start value term. Both use `manager_learning_rate`;
- `last_critic_loss` now reports the per-step Manager value loss, preserving
  compatibility with the training logger. Intrinsic statistics are available as
  `last_worker_mean_intrinsic_reward`, `last_per_agent_intrinsic_reward`,
  `last_intrinsic_critic_loss`, and `last_per_agent_intrinsic_critic_loss`.
  `last_per_agent_ratio_clip_fraction` records the active-sample fraction whose
  current-worker PPO ratio was outside the clip interval in the final actor epoch.

There is no separate worker external critic or critic optimizer. Checkpoints
save the Manager, workers, individual intrinsic critics, and their optimizers. Older checkpoints can still
load those unchanged parameter layouts; their legacy `critic` and
`critic_optimizer` entries are ignored. Resumed runs use the new per-step
Manager recurrence, so their behavior differs from the previous segment-only
recurrence.

Checkpoints without intrinsic critics retain the newly initialized intrinsic
critics and emit a warning; they need training on intrinsic returns after loading.
The worker objective is a hybrid with individual goals; the original shared-reward
HAA2C guarantees do not automatically apply.

See [完整架構、公式與資料流](ARCHITECTURE.md) for the implementation details.
Select `feudal_haa2c` explicitly when running `scripts/smac_feudal/train.py`, or
set `algorithm: feudal_haa2c` in the training configuration. The launcher otherwise
uses whichever algorithm that configuration currently selects.

Run the synthetic regression tests without launching StarCraft II:

```powershell
./.venv-smac/Scripts/python.exe -m unittest discover -s tests -p test_feudal_haa2c.py -v
```
