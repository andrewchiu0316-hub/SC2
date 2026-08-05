# QMIX

This module adapts the QMIX implementation and default hyperparameters from
[oxwhirl/pymarl](https://github.com/oxwhirl/pymarl), licensed under Apache-2.0,
to this repository's `scripts/smac_feudal/train.py` interface.

The implementation keeps the core PyMARL design:

- one shared recurrent Q-network for all agents;
- agent inputs containing observation, previous action, and agent identity;
- epsilon-greedy exploration;
- episode replay with padded batches;
- Double Q-learning and recurrent target networks;
- a monotonic state-conditioned QMIX mixer;
- RMSprop optimization and periodic target updates.

The adapter collects transitions through `sample_action`, commits complete
episodes to replay at termination, and uses the SMAC global state only in the
mixer during centralized training.
