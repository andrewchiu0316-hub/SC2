"""Manager values and intrinsic worker updates; no StarCraft II is required."""

import copy
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from module.Algorithm.FeUdal_HAA2C.algorithm import Algorithm
from Tools import utils


class SharedManagerValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = utils.load_config(
            Path(__file__).resolve().parents[1]
            / "module/Algorithm/FeUdal_HAA2C/config.yaml"
        )

    def setUp(self):
        torch.manual_seed(7)
        self.rng = np.random.default_rng(19)

    def agent(self, **overrides):
        config = dict(self.config)
        config.update(
            worker_hidden_dim=8,
            manager_hidden_dim=12,
            intrinsic_hidden_dim=8,
            manager_c_steps=3,
            worker_update_steps=4,
            a2c_epoch=2,
            critic_epoch=2,
            intrinsic_critic_epoch=2,
            fixed_order=True,
        )
        config.update(overrides)
        with patch(
            "module.Algorithm.FeUdal_HAA2C.algorithm.utils.load_config",
            return_value=config,
        ), patch("torch.cuda.is_available", return_value=False):
            return Algorithm(4, n_agents=2, state_dim=5, state_g_dim=7)

    def snapshot(self, env_count=2):
        return (
            self.rng.normal(size=(env_count, 2, 5)).astype(np.float32),
            self.rng.normal(size=(env_count, 7)).astype(np.float32),
            np.ones((env_count, 2), dtype=np.float32),
            np.ones((env_count, 2, 4), dtype=np.bool_),
        )

    def single_step(self, agent, snapshot, reward=None, done=False):
        obs, state, alive, available = snapshot
        return agent.sample_action(
            obs[0], state[0], reward, None, done, alive[0],
            avail_actions=available[0],
        )

    def store_batch(self, agent, snapshot, dones=(False, False)):
        obs, state, alive, _ = snapshot
        agent.store_transition_batch(
            [1.0, 0.5], dones, next_global_states=state,
            next_local_states=obs, next_alive_masks=alive,
        )

    def assert_hidden_equal(self, actual, expected):
        for actual_part, expected_part in zip(actual, expected):
            torch.testing.assert_close(actual_part, expected_part)

    def test_parallel_values_and_goal_cadence_match_dense_replay(self):
        agent = self.agent(worker_update_steps=100)
        agent._init_parallel_state(2)
        hidden = agent.manager.init_hidden(2, agent.device)
        remaining = torch.zeros(2, dtype=torch.long)
        held_goals = torch.zeros(2, agent.n_agents, agent.goal_dim)
        snapshot = self.snapshot()
        with patch.object(agent, "_train_manager_segments"):
            for step in range(7):
                states = torch.from_numpy(snapshot[1])
                with torch.no_grad():
                    goals, values, hidden = agent.manager(states[None], hidden)
                due = remaining <= 0
                held_goals[due] = goals[0, due]
                remaining[due] = agent.manager_c_steps
                agent.sample_actions_batch(*snapshot)
                torch.testing.assert_close(
                    agent._parallel_step_cache["old_value"], values[0]
                )
                torch.testing.assert_close(agent.parallel_current_goal, held_goals)
                self.assert_hidden_equal(agent.parallel_manager_hidden, hidden)

                dones = torch.tensor([step == 1, step == 4])
                snapshot = self.snapshot()
                live_hidden = agent._detach_hidden(agent.parallel_manager_hidden)
                with torch.no_grad():
                    _, expected_next, _ = agent.manager(
                        torch.from_numpy(snapshot[1])[None], hidden
                    )
                self.store_batch(agent, snapshot, dones.numpy())
                torch.testing.assert_close(
                    agent.worker_buffer[-1]["next_value"],
                    expected_next[0].masked_fill(dones, 0.0),
                )
                keep = (~dones).float().reshape(1, 2, 1)
                hidden = tuple(part * keep for part in hidden)
                self.assert_hidden_equal(agent.parallel_manager_hidden, hidden)
                self.assert_hidden_equal(
                    agent.parallel_manager_hidden,
                    tuple(part * keep for part in live_hidden),
                )
                remaining -= 1
                remaining[dones] = 0
                held_goals[dones] = 0

        states = torch.stack([item["state"] for item in agent.worker_buffer])
        dones = torch.stack([item["done"] for item in agent.worker_buffer])
        old_values = torch.stack([item["old_value"] for item in agent.worker_buffer])
        with torch.no_grad():
            _, replay_values, replay_hidden = agent._evaluate_manager(
                states, dones, agent.parallel_manager_rollout_start_hidden
            )
        torch.testing.assert_close(replay_values, old_values)
        self.assert_hidden_equal(replay_hidden, agent.parallel_manager_hidden)

    def test_single_values_and_replay_cross_episode_boundary(self):
        agent = self.agent(worker_update_steps=100)
        with patch.object(agent, "_train_manager"):
            for episode_length in (2, 4):
                agent.episode_reset()
                hidden = agent.manager.init_hidden(1, agent.device)
                held_goal = None
                snapshot = self.snapshot(1)
                for step in range(episode_length):
                    with torch.no_grad():
                        goals, values, hidden = agent.manager(
                            torch.from_numpy(snapshot[1])[None], hidden
                        )
                    self.single_step(
                        agent, snapshot, reward=None if step == 0 else 1.0
                    )
                    if step % agent.manager_c_steps == 0:
                        held_goal = goals[0, 0]
                    torch.testing.assert_close(agent.current_goal, held_goal)
                    torch.testing.assert_close(
                        agent._last_step_cache["old_value"], values[0, 0]
                    )
                    self.assert_hidden_equal(agent.manager_hidden, hidden)
                    snapshot = self.snapshot(1)
                self.single_step(agent, snapshot, reward=1.0, done=True)
                self.assertEqual(agent.worker_buffer[-1]["next_value"].item(), 0.0)

        states = torch.stack([item["state"] for item in agent.worker_buffer])[:, None]
        dones = torch.tensor([item["done"] for item in agent.worker_buffer]).float()[:, None]
        old_values = torch.stack([item["old_value"] for item in agent.worker_buffer])[:, None]
        with torch.no_grad():
            _, replay_values, _ = agent._evaluate_manager(
                states, dones, agent.manager_rollout_start_hidden
            )
        torch.testing.assert_close(replay_values, old_values)
        # Reset must not discard the initial hidden for a pending rollout.
        agent.episode_reset()
        initial_hidden = agent._detach_hidden(agent.manager_rollout_start_hidden)
        loss = agent.finalize_training()
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(agent.worker_update_count, 1)
        self.assert_hidden_equal(initial_hidden, agent.manager.init_hidden(1, agent.device))

    def test_gae_handles_terminal_boundaries_and_nonterminal_bootstrap(self):
        agent = self.agent(gamma=0.9, gae_lambda=1.0)
        rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        values = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
        done = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # Nonzero terminal bootstrap must be ignored, and GAE cannot cross resets.
        advantages, returns = agent._compute_advantages(
            rewards, values, done, torch.tensor([2.0, 999.0])
        )
        expected = torch.tensor([[3.7, 10.46], [3.0, 9.4], [6.8, 6.0]])
        torch.testing.assert_close(returns, expected)
        torch.testing.assert_close(advantages, expected - values)
        single_adv, single_returns = agent._compute_advantages(
            rewards[:, 0], values[:, 0], done[:, 0], torch.tensor(2.0)
        )
        torch.testing.assert_close(single_adv, advantages[:, 0])
        torch.testing.assert_close(single_returns, returns[:, 0])

    def test_worker_return_trains_manager_value_at_non_goal_steps(self):
        agent = self.agent(use_clipped_value_loss=False)
        states = torch.randn(4, 2, 7, requires_grad=True)
        dones = torch.zeros(4, 2)
        hidden = agent.manager.init_hidden(2, agent.device)
        with torch.no_grad():
            _, old_values, _ = agent._evaluate_manager(states, dones, hidden)
        targets = old_values.clone()
        targets[1:3] += 1.0  # Only intermediate, non-goal timesteps have errors.
        value_before = agent.manager.value_head.weight.detach().clone()
        goal_before = agent.manager.goal_head.weight.detach().clone()
        agent._train_manager_value(states, old_values, targets, dones, hidden)
        self.assertGreater(agent.last_critic_loss, 0.0)
        self.assertFalse(torch.equal(value_before, agent.manager.value_head.weight))
        torch.testing.assert_close(goal_before, agent.manager.goal_head.weight)
        self.assertGreater(states.grad[1:3].abs().sum().item(), 0.0)
        self.assertFalse(hasattr(agent, "critic"))
        self.assertFalse(hasattr(agent, "critic_optimizer"))

    def test_parallel_segment_bootstrap_uses_all_intermediate_states(self):
        agent = self.agent()
        states = torch.randn(3, 7)
        hidden = agent.manager.init_hidden(1, agent.device)
        next_state = torch.randn(7)
        rewards = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        with torch.no_grad():
            _, _, full_hidden = agent.manager(states[:, None], hidden)
            _, next_value, _ = agent.manager(next_state[None, None], full_hidden)
        expected_return = sum(agent.gamma ** t * r for t, r in enumerate(rewards))
        expected_return += agent.gamma ** 3 * next_value[0, 0]
        segment = {
            "states": list(states), "hidden": hidden,
            "obs": torch.randn(2, 5), "next_obs": torch.randn(2, 5),
            "alive": torch.ones(2), "rewards": rewards,
            "next_state": next_state, "done": False,
        }
        agent._train_manager_segments([segment])
        self.assertAlmostEqual(agent.last_manager_return, expected_return.item(), places=5)

    def test_manager_alignment_target_does_not_backpropagate_to_worker_encoder(self):
        agent = self.agent()
        segment = {
            "states": [torch.randn(7) for _ in range(3)],
            "hidden": agent.manager.init_hidden(1, agent.device),
            "obs": torch.randn(2, 5),
            "next_obs": torch.randn(2, 5),
            "alive": torch.ones(2),
            "rewards": [torch.tensor(1.0) for _ in range(3)],
            "next_state": torch.randn(7),
            "done": False,
        }
        agent._train_manager_segments([segment])
        for worker in agent.workers:
            for parameter in worker.state_encoder.parameters():
                self.assertIsNone(parameter.grad)

    def test_single_segment_bootstrap_uses_all_intermediate_states(self):
        agent = self.agent()
        states = torch.randn(3, 7)
        next_state = torch.randn(7)
        rewards = torch.tensor([1.0, 2.0, 3.0])
        with torch.no_grad():
            _, _, hidden = agent.manager(
                states[:, None], agent.manager_segment_start_hidden
            )
            _, next_value, _ = agent.manager(next_state[None, None], hidden)
        expected_return = sum(agent.gamma ** t * r for t, r in enumerate(rewards))
        expected_return += agent.gamma ** 3 * next_value[0, 0]
        agent._train_manager(
            states, torch.randn(3, 2, 5), torch.randn(2, 5), rewards,
            torch.zeros(3), next_state, torch.ones(3, 2),
        )
        self.assertAlmostEqual(agent.last_manager_return, expected_return.item(), places=5)

    def test_parallel_training_partial_segments_and_checkpoint_roundtrip(self):
        agent = self.agent()
        snapshot = self.snapshot()
        value_before = agent.manager.value_head.weight.detach().clone()
        for step in range(9):
            if step == 2:
                snapshot[2][:, 1] = 0.0
                snapshot[3][:, 1] = False
                snapshot[3][:, 1, 0] = True
            actions = agent.sample_actions_batch(*snapshot)
            self.assertTrue(np.take_along_axis(snapshot[3], actions[:, :, None], axis=2).all())
            snapshot = self.snapshot()
            self.store_batch(agent, snapshot, (step == 2, step == 5))
            if len(agent.worker_buffer) == agent.worker_update_steps:
                self.assertTrue(math.isfinite(agent.train(next_global_state=snapshot[1])))
        agent.finalize_training(
            next_global_state=snapshot[1], next_local_state=snapshot[0]
        )
        self.assertEqual(agent.worker_update_count, 3)
        self.assertGreater(agent.manager_update_count, 0)
        self.assertEqual(agent.worker_buffer, [])
        self.assertTrue(all(segment is None for segment in agent.parallel_segments))
        self.assertFalse(torch.equal(value_before, agent.manager.value_head.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.manager.parameters()))

        checkpoint = copy.deepcopy(agent.save_model({"episode": 8}))
        self.assertNotIn("critic", checkpoint)
        self.assertNotIn("critic_optimizer", checkpoint)
        restored = self.agent()
        restored.load_model(checkpoint)
        for name, value in agent.manager.state_dict().items():
            torch.testing.assert_close(value, restored.manager.state_dict()[name])
        for name, value in agent.workers.state_dict().items():
            torch.testing.assert_close(value, restored.workers.state_dict()[name])
        for name, value in agent.intrinsic_critics.state_dict().items():
            torch.testing.assert_close(value, restored.intrinsic_critics.state_dict()[name])
        for original, loaded in zip(
            agent.intrinsic_critic_optimizers, restored.intrinsic_critic_optimizers
        ):
            original_states = original.state_dict()["state"]
            loaded_states = loaded.state_dict()["state"]
            self.assertEqual(original_states.keys(), loaded_states.keys())
            for parameter_id, state in original_states.items():
                for key, value in state.items():
                    torch.testing.assert_close(value, loaded_states[parameter_id][key])
        self.assertEqual(restored.worker_update_count, agent.worker_update_count)
        self.assertEqual(restored.manager_update_count, agent.manager_update_count)
        self.assertEqual(len(restored.manager_optimizer.state), len(agent.manager_optimizer.state))
        checkpoint.update(critic={"legacy": True}, critic_optimizer={"legacy": True})
        restored.load_model(checkpoint)
        self.assertNotIn("critic", restored.save_model(checkpoint))

    def test_single_training_across_worker_and_goal_update_boundaries(self):
        agent = self.agent()
        for episode_length in (5, 4):
            agent.episode_reset()
            self.single_step(agent, self.snapshot(1))
            for step in range(episode_length):
                self.single_step(
                    agent, self.snapshot(1), reward=1.0,
                    done=step == episode_length - 1,
                )
        agent.finalize_training()
        self.assertEqual(agent.worker_update_count, 3)
        self.assertEqual(agent.manager_update_count, 4)
        self.assertTrue(math.isfinite(agent.last_critic_loss))
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.manager.parameters()))

    def test_single_finalize_uses_recorded_bootstrap_without_advancing_hidden(self):
        agent = self.agent(worker_update_steps=100)
        self.single_step(agent, self.snapshot(1))
        self.single_step(agent, self.snapshot(1), reward=1.0)
        expected_bootstrap = agent.worker_buffer[-1]["next_value"].clone()
        expected_intrinsic = agent.worker_buffer[-1]["next_intrinsic_value"].clone()
        live_hidden = agent._detach_hidden(agent.manager_hidden)
        with patch.object(
            agent, "_compute_advantages", wraps=agent._compute_advantages
        ) as compute:
            self.assertTrue(math.isfinite(agent.finalize_training()))
        torch.testing.assert_close(compute.call_args_list[0].args[3], expected_bootstrap)
        torch.testing.assert_close(compute.call_args_list[1].args[3], expected_intrinsic)
        self.assert_hidden_equal(agent.manager_hidden, live_hidden)

    def test_segment_intrinsic_reward_uses_encoded_displacement_and_masks_death(self):
        agent = self.agent(worker_hidden_dim=5)
        # Worker 1 reverses the first observation coordinate in latent space.
        agent.workers[0].state_encoder = torch.nn.Identity()
        encoder = torch.nn.Linear(5, 5, bias=False)
        with torch.no_grad():
            encoder.weight.copy_(torch.diag(torch.tensor([-1., 1., 1., 1., 1.])))
        agent.workers[1].state_encoder = encoder
        obs = torch.zeros(4, 2, 5, requires_grad=True)
        next_obs = torch.zeros_like(obs)
        next_obs[0, :, 0] = 2.0       # Same raw movement, opposite latent movement.
        next_obs[1, 0, 1] = 2.0      # Perpendicular; other worker does not move.
        next_obs[2:, :, 0] = 1.0
        goals = torch.zeros_like(obs)
        goals[..., 0] = 1.0
        goals.requires_grad_()
        alive = torch.ones(4, 2)
        next_alive = torch.ones(4, 2)
        alive[2, 0] = 0.0
        next_alive[2, 1] = 0.0
        rewards, valid = agent._compute_intrinsic_rewards(
            obs, next_obs, goals, alive, next_alive
        )
        torch.testing.assert_close(
            rewards, torch.tensor([[1., -1.], [0., 0.], [0., 0.], [1., -1.]])
        )
        torch.testing.assert_close(
            valid, torch.tensor([[True, True], [True, False], [False, False], [True, True]])
        )
        self.assertFalse(rewards.requires_grad)
        self.assertIsNone(encoder.weight.grad)

    def test_intrinsic_boundary_masks_and_bootstrap(self):
        agent = self.agent()
        obs, states, alive, _ = self.snapshot(3)
        cache = {
            "obs": torch.from_numpy(obs),
            "goal": torch.randn(3, 2, agent.goal_dim),
            "alive": torch.from_numpy(alive),
            "goal_remaining": torch.tensor([1, 2, 3]),
        }
        next_obs, next_states, next_alive, _ = self.snapshot(3)
        next_alive[1, 0] = 0.0
        transition = agent._intrinsic_transition(
            cache, torch.from_numpy(next_states), torch.from_numpy(next_obs),
            torch.from_numpy(next_alive), torch.tensor([0., 0., 1.]),
        )
        torch.testing.assert_close(
            transition["intrinsic_done"], torch.tensor([[1., 1.], [1., 0.], [1., 1.]])
        )
        self.assertEqual(transition["intrinsic_reward"][1, 0].item(), 0.0)
        with torch.no_grad():
            expected = agent.intrinsic_critics[1](
                torch.from_numpy(next_states[1]), torch.from_numpy(next_obs[1, 1]),
                cache["goal"][1, 1], torch.tensor(1 / agent.manager_c_steps),
            )
        bootstrap = transition["next_intrinsic_value"]
        torch.testing.assert_close(bootstrap[1, 1], expected)
        self.assertEqual(bootstrap[transition["intrinsic_done"] > 0].abs().sum().item(), 0.0)

    def test_intrinsic_gae_cuts_goal_segments_without_cutting_team_gae(self):
        agent = self.agent(gamma=1.0, gae_lambda=1.0)
        intrinsic_done = torch.zeros(4, 1, 2)
        intrinsic_done[0, :, 1] = 1.0
        intrinsic_done[1] = 1.0
        intrinsic_done[3] = 1.0
        _, intrinsic_returns = agent._compute_advantages(
            torch.ones(4, 1, 2), torch.zeros(4, 1, 2), intrinsic_done,
            torch.full((1, 2), 999.0),
        )
        torch.testing.assert_close(
            intrinsic_returns[:, 0], torch.tensor([[2., 1.], [1., 1.], [2., 2.], [1., 1.]])
        )
        _, team_returns = agent._compute_advantages(
            torch.ones(4, 1), torch.zeros(4, 1), torch.tensor([[0.], [0.], [0.], [1.]]),
            torch.tensor([999.]),
        )
        torch.testing.assert_close(team_returns[:, 0], torch.tensor([4., 3., 2., 1.]))

    def test_stored_rewards_are_fixed_before_manager_or_worker_updates(self):
        agent = self.agent(manager_c_steps=1)
        snapshot = self.snapshot()
        agent.sample_actions_batch(*snapshot)
        action_goals = agent._parallel_step_cache["goal"].clone()
        next_snapshot = self.snapshot()
        expected, expected_valid = agent._compute_intrinsic_rewards(
            torch.from_numpy(snapshot[0]), torch.from_numpy(next_snapshot[0]),
            action_goals, torch.from_numpy(snapshot[2]), torch.from_numpy(next_snapshot[2]),
        )

        def change_networks(_segments):
            with torch.no_grad():
                agent.parallel_current_goal.zero_()
                for worker in agent.workers:
                    for parameter in worker.state_encoder.parameters():
                        parameter.zero_()

        with patch.object(agent, "_train_manager_segments", side_effect=change_networks):
            self.store_batch(agent, next_snapshot)
        stored = agent.worker_buffer[-1]
        torch.testing.assert_close(stored["intrinsic_reward"], expected)
        torch.testing.assert_close(
            stored["intrinsic_reward_valid"], expected_valid.float()
        )
        torch.testing.assert_close(stored["goal"], action_goals)
        self.assertFalse(stored["intrinsic_reward"].requires_grad)
        self.assertTrue((stored["intrinsic_done"] == 1).all())
        self.assertTrue((stored["next_intrinsic_value"] == 0).all())
        self.assertFalse(torch.equal(
            expected,
            agent._compute_intrinsic_rewards(
                torch.from_numpy(snapshot[0]), torch.from_numpy(next_snapshot[0]),
                action_goals, torch.from_numpy(snapshot[2]), torch.from_numpy(next_snapshot[2]),
            )[0],
        ))

    def test_segment_intrinsic_reward_is_written_once_at_goal_boundary(self):
        agent = self.agent(worker_hidden_dim=5, manager_c_steps=3, worker_update_steps=100)
        agent.workers[0].state_encoder = torch.nn.Identity()
        agent.workers[1].state_encoder = torch.nn.Identity()
        observations = np.zeros((1, 2, 5), dtype=np.float32)
        states = np.zeros((1, 7), dtype=np.float32)
        alive = np.ones((1, 2), dtype=np.float32)
        available = np.ones((1, 2, 4), dtype=np.bool_)
        goal = torch.zeros((1, 2, 5), device=agent.device)
        goal[..., 0] = 1.0

        for step in range(3):
            agent.sample_actions_batch(observations, states, alive, available)
            agent._parallel_step_cache["goal"] = goal.clone()
            agent.parallel_current_goal = goal.clone()
            if step == 0:
                agent.parallel_segments[0]["goal"] = goal[0].clone()
            next_observations = np.zeros_like(observations)
            next_observations[..., 0] = step + 1.0
            agent.store_transition_batch(
                [0.0], [False], next_global_states=states,
                next_local_states=next_observations, next_alive_masks=alive,
            )
            observations = next_observations

        rewards = torch.stack(
            [transition["intrinsic_reward"] for transition in agent.worker_buffer]
        )
        valid = torch.stack(
            [transition["intrinsic_reward_valid"] for transition in agent.worker_buffer]
        )
        torch.testing.assert_close(rewards[:2], torch.zeros_like(rewards[:2]))
        torch.testing.assert_close(rewards[2], torch.ones_like(rewards[2]))
        torch.testing.assert_close(valid[:2], torch.zeros_like(valid[:2]))
        torch.testing.assert_close(valid[2], torch.ones_like(valid[2]))

    def test_single_simultaneous_goal_and_worker_boundary_keeps_segment_reward(self):
        agent = self.agent(worker_hidden_dim=5, manager_c_steps=2, worker_update_steps=2)
        agent.workers[0].state_encoder = torch.nn.Identity()
        agent.workers[1].state_encoder = torch.nn.Identity()
        observations = np.zeros((1, 2, 5), dtype=np.float32)
        states = np.zeros((1, 7), dtype=np.float32)
        alive = np.ones((1, 2), dtype=np.float32)
        available = np.ones((1, 2, 4), dtype=np.bool_)
        goal = torch.zeros((2, 5), device=agent.device)
        goal[:, 0] = 1.0

        self.single_step(agent, (observations, states, alive, available))
        agent.current_goal = goal.clone()
        agent._last_step_cache["goal"] = goal.clone()
        observations[..., 0] = 1.0
        self.single_step(agent, (observations, states, alive, available), reward=0.0)
        observations[..., 0] = 2.0
        captured_rewards = []

        def capture_train(**_kwargs):
            captured_rewards.append(agent.worker_buffer[-1]["intrinsic_reward"].clone())
            return 0.0

        with patch.object(agent, "train", side_effect=capture_train):
            self.single_step(agent, (observations, states, alive, available), reward=0.0)
        self.assertEqual(len(captured_rewards), 1)
        torch.testing.assert_close(captured_rewards[0], torch.ones(2))

    def test_intrinsic_critics_are_independent_and_cannot_update_actors(self):
        agent = self.agent(use_clipped_value_loss=False)
        states = torch.randn(3, 2, 7, requires_grad=True)
        observations = torch.randn(3, 2, 2, 5, requires_grad=True)
        goals = torch.randn(3, 2, 2, agent.goal_dim, requires_grad=True)
        remaining = torch.full((3, 2), 2)
        old_values = torch.zeros(3, 2, 2, requires_grad=True)
        returns = torch.ones(3, 2, 2, requires_grad=True)
        alive = torch.ones(3, 2, 2)
        alive[..., 1] = 0.0
        before = [copy.deepcopy(critic.state_dict()) for critic in agent.intrinsic_critics]
        agent._train_intrinsic_critics(
            states, observations, goals, remaining, old_values, returns, alive
        )
        self.assertTrue(any(
            not torch.equal(value, agent.intrinsic_critics[0].state_dict()[name])
            for name, value in before[0].items()
        ))
        for name, value in before[1].items():
            torch.testing.assert_close(value, agent.intrinsic_critics[1].state_dict()[name])
        for tensor in (states, observations, goals, old_values, returns):
            self.assertIsNone(tensor.grad)
        self.assertTrue(all(parameter.grad is None for parameter in agent.workers.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in agent.manager.parameters()))
        self.assertEqual(agent.last_per_agent_intrinsic_critic_loss[1], 0.0)

    def test_mixed_advantage_and_factor_match_analytic_policy_gradient(self):
        # A Bernoulli toy policy has an analytic log-probability derivative.
        # Compare complete sequential updates, including the second actor epoch.
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                # Keep this existing factor test unclipped; clipping has its
                # own direct analytic test below.
                agent = self.agent(
                    intrinsic_coef=0.4,
                    worker_ratio_clip=0.999999,
                    entropy_coef=0.0,
                    max_grad_norm=1000.0,
                )
                shape = (4, 2) if parallel else (4,)
                count = math.prod(shape)
                external = torch.linspace(-1.0, 1.5, count).reshape(shape).requires_grad_()
                intrinsic = torch.stack((
                    torch.linspace(2.0, -1.0, count).reshape(shape).square(),
                    torch.linspace(-0.5, 1.0, count).reshape(shape).sin(),
                ), dim=-1).requires_grad_()
                features = [
                    torch.linspace(-1.5 + i * 0.2, 2.0 + i * 0.3, count).reshape(shape)
                    for i in range(2)
                ]
                alive = torch.ones(*shape, 2)
                alive.reshape(-1, 2)[1, 0] = 0.0
                actors = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False) for _ in range(2)])
                with torch.no_grad():
                    actors[0].weight.fill_(0.2)
                    actors[1].weight.fill_(-0.3)
                agent.workers = actors
                agent.worker_optimizers = [torch.optim.SGD(actor.parameters(), lr=0.05) for actor in actors]

                def log_prob(theta, feature):
                    return torch.nn.functional.logsigmoid(theta * feature)

                old_probs = torch.stack([
                    log_prob(actors[i].weight.detach().reshape(()), features[i])
                    for i in range(2)
                ], dim=-1)
                factor = torch.ones(shape)
                expected_parameters, expected_losses, expected_ratios = [], [], []
                with torch.no_grad():
                    for i in range(2):
                        active = alive[..., i]
                        theta = actors[i].weight.detach().reshape(()).clone()
                        before_log = log_prob(theta, features[i])
                        mixed = external + 0.4 * intrinsic[..., i]
                        valid = mixed[active > 0]
                        advantage = (mixed - valid.mean()) / (valid.std(unbiased=False) + 1e-5)
                        for _ in range(agent.a2c_epoch):
                            ratio = torch.exp(log_prob(theta, features[i]) - old_probs[..., i])
                            loss = -(factor * ratio * advantage * active).sum() / active.sum()
                            derivative = torch.sigmoid(-theta * features[i]) * features[i]
                            gradient = -(factor * ratio * advantage * derivative * active).sum() / active.sum()
                            theta -= 0.05 * gradient
                        update_ratio = torch.exp(log_prob(theta, features[i]) - before_log)
                        update_ratio = torch.where(active > 0, update_ratio, 1.0)
                        factor *= update_ratio
                        expected_parameters.append(theta)
                        expected_losses.append(loss.item())
                        expected_ratios.append(update_ratio[active > 0].mean().item())

                def evaluate(i, *_args):
                    return log_prob(actors[i].weight.reshape(()), features[i]), torch.zeros(shape)

                method_name = "_evaluate_actor_parallel" if parallel else "_evaluate_actor"
                train_workers = agent._train_workers_parallel if parallel else agent._train_workers
                with patch.object(agent, method_name, side_effect=evaluate):
                    train_workers(
                        torch.zeros(*shape, 2, 5), torch.zeros(*shape, 2, agent.goal_dim),
                        torch.zeros(*shape, 2, dtype=torch.long),
                        torch.ones(*shape, 2, 4, dtype=torch.bool),
                        old_probs, external, intrinsic, alive, torch.zeros(shape),
                    )
                for i in range(2):
                    torch.testing.assert_close(actors[i].weight.reshape(()), expected_parameters[i])
                    self.assertAlmostEqual(agent.last_per_agent_policy_loss[i], expected_losses[i], places=6)
                    self.assertAlmostEqual(agent.last_per_agent_ratio[i], expected_ratios[i], places=6)
                self.assertAlmostEqual(agent.last_factor_mean, factor.mean().item(), places=6)
                self.assertIsNone(external.grad)
                self.assertIsNone(intrinsic.grad)

    def test_worker_ppo_surrogate_clips_correct_side_for_both_advantage_signs(self):
        agent = self.agent(worker_ratio_clip=0.2)
        ratios = torch.tensor([1.3, 1.3, 0.7, 0.7], requires_grad=True)
        advantages = torch.tensor([2.0, -2.0, 2.0, -2.0])
        surrogate = agent._worker_ppo_surrogate(ratios, advantages)

        # For positive A, excessive upward ratios are clipped. For negative A,
        # excessively downward ratios are clipped, as in PPO's min objective.
        torch.testing.assert_close(
            surrogate, torch.tensor([2.4, -2.6, 1.4, -1.6])
        )
        surrogate.sum().backward()
        torch.testing.assert_close(ratios.grad, torch.tensor([0.0, -2.0, 2.0, 0.0]))

    def test_advantage_means_use_all_external_steps_and_alive_intrinsic_worker_steps(self):
        agent = self.agent(intrinsic_coef=0.05)
        external = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        intrinsic = torch.tensor(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[1.0, 1.2], [1.4, 1.6]],
            ]
        )
        alive = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
            ]
        )

        agent._record_worker_advantage_means(external, intrinsic, alive)

        self.assertAlmostEqual(agent.last_worker_mean_external_advantage, 2.5)
        expected_intrinsic = (0.2 + 0.4 + 0.6 + 1.0 + 1.2 + 1.6) / 6
        self.assertAlmostEqual(
            agent.last_worker_mean_intrinsic_advantage, expected_intrinsic
        )
        self.assertAlmostEqual(
            agent.last_worker_mean_abs_external_advantage, 14.0 / 6.0, places=6
        )
        self.assertAlmostEqual(
            agent.last_worker_mean_abs_intrinsic_advantage, expected_intrinsic
        )
        self.assertAlmostEqual(
            agent.last_worker_mean_abs_weighted_intrinsic_advantage,
            0.05 * expected_intrinsic,
            places=6,
        )
        self.assertAlmostEqual(
            agent.last_worker_raw_intrinsic_to_external_advantage_ratio,
            expected_intrinsic / (14.0 / 6.0),
            places=6,
        )
        self.assertAlmostEqual(
            agent.last_worker_intrinsic_to_external_advantage_ratio,
            (0.05 * expected_intrinsic) / (14.0 / 6.0),
            places=6,
        )

    def test_worker_ratio_clip_requires_a_valid_ppo_epsilon(self):
        for value in (0.0, -0.1, 1.0, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "worker_ratio_clip"):
                    self.agent(worker_ratio_clip=value)

    def test_zero_intrinsic_coefficient_recovers_external_advantage(self):
        agent = self.agent(intrinsic_coef=0.0)
        external = torch.tensor([1., 3., 100., 2.])
        intrinsic = torch.tensor([-1000., 30., -100., 22.])
        active = torch.tensor([1., 1., 0., 1.])
        result = agent._mixed_worker_advantage(external, intrinsic, active)
        valid = external[active > 0]
        torch.testing.assert_close(
            result, (external - valid.mean()) / (valid.std(unbiased=False) + 1e-5)
        )

    def test_legacy_checkpoint_keeps_initialized_intrinsic_critics_with_notice(self):
        checkpoint = copy.deepcopy(self.agent().save_model({}))
        checkpoint.pop("intrinsic_critics")
        checkpoint.pop("intrinsic_critic_optimizers")
        restored = self.agent()
        initialized = copy.deepcopy(restored.intrinsic_critics.state_dict())
        with self.assertWarnsRegex(RuntimeWarning, "no intrinsic critics"):
            restored.load_model(checkpoint)
        for name, value in initialized.items():
            torch.testing.assert_close(value, restored.intrinsic_critics.state_dict()[name])

    def test_single_and_parallel_intrinsic_transition_data_agree(self):
        single = self.agent(worker_update_steps=100)
        parallel = self.agent(worker_update_steps=100)
        parallel.load_model(copy.deepcopy(single.save_model({})))
        snapshot = self.snapshot(1)
        with patch.object(single, "_train_manager"), patch.object(parallel, "_train_manager_segments"):
            self.single_step(single, snapshot)
            for step in range(5):
                parallel.sample_actions_batch(*snapshot)
                snapshot = self.snapshot(1)
                terminal = step == 4
                self.single_step(single, snapshot, reward=1.0, done=terminal)
                parallel.store_transition_batch(
                    [1.0], [terminal], next_global_states=snapshot[1],
                    next_local_states=snapshot[0], next_alive_masks=snapshot[2],
                )
                for key in (
                    "intrinsic_reward", "old_intrinsic_value", "next_intrinsic_value",
                    "intrinsic_reward_valid", "intrinsic_done", "goal_remaining",
                ):
                    torch.testing.assert_close(
                        single.worker_buffer[-1][key], parallel.worker_buffer[-1][key][0]
                    )


if __name__ == "__main__":
    unittest.main()
