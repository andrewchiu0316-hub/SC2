# HAA2C

This module adapts HAA2C from [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL)
to this repository's single-environment SMAC training loop.

- Each agent owns an independent feed-forward actor and optimizer.
- Actors consume local observations and respect SMAC available-action masks.
- One centralized V critic consumes the SMAC global state.
- GAE uses only the shared external environment reward.
- The runner defaults to 8 persistent SMAC subprocesses for reliable Windows
  startup. A rollout contains 200 steps from each process, producing a
  1,600-sample update batch. Set `n_rollout_threads` to 20 on a machine that can
  reliably sustain the official 4,000-sample batch.
- Episodes reset independently inside each process; terminal masks stop GAE at
  each boundary.
- Actors update in a randomized sequence. After each actor update, its action
  probability ratio is multiplied into the factor used by later actors.
- The critic uses ValueNorm, clipped Huber value loss, and gradient clipping.

There is no FeUdal Manager, goal, FiLM layer, intrinsic reward, or worker LSTM
in this standalone algorithm. `feudal_haa2c` remains available as the hybrid.
