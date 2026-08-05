# FeUdal HAA2C

This hybrid keeps the high-level Manager and FiLM goal conditioning from the
repository's original FeUdal implementation, while adapting the worker update
scheme from HAA2C in [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL).

Core HAA2C behavior retained here:

- one independent actor and optimizer per agent, without parameter sharing;
- a centralized V critic consuming the SMAC global state;
- GAE computed from one shared team reward;
- randomized sequential actor updates;
- a multiplicative factor that carries importance ratios from workers already
  updated in the current sequence;
- A2C actor objectives with entropy regularization;
- multiple actor and critic epochs, Huber value loss, value clipping, and
  gradient clipping.

FeUdal-specific behavior retained here:

- the Manager emits and updates one normalized goal per worker every 50 steps;
- eight SMAC environments each collect 200 steps, producing a joint
  1,600-sample worker and centralized-critic update batch;
- worker rollouts may cross episode boundaries, with recurrent state reset
  independently for each environment by terminal masks during training;
- each worker uses local observation encoding followed by FiLM goal modulation;
- worker GAE and the centralized critic use only environment reward;
- the Manager keeps its own value head, external n-step return, advantage,
  latent displacement cosine objective, and value loss;
- the centralized V critic is used only for worker GAE and HAA2C updates.
