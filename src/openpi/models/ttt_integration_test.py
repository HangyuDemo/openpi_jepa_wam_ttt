"""Contract tests for the Pi0.5 temporal memory routes."""

import dataclasses
import unittest

<<<<<<< HEAD
=======
import flax.nnx as nnx
>>>>>>> 1152696 (little changes)
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config
from openpi.models import ttt_memory
from openpi.training.data_loader import TemporalWindowDataset


class TTTIntegrationTest(unittest.TestCase):
    def test_memory_sources_and_middle_future_token(self):
        query = jnp.ones((1, 3, 4), dtype=jnp.float32)
        full_a = jnp.concatenate((query[:, :2], jnp.zeros((1, 1, 4)), query[:, 2:]), axis=1)
        full_b = full_a.at[:, 2, :].set(3.0)
        state = jax.tree.map(lambda value: value[0], ttt_memory.empty_fast_weights(1, 1, 4, 8))

        implicit = ttt_memory.TemporalTTT(query_dim=4, memory_dim=4, hidden_dim=8, source="implicit_full")
        variables = implicit.init(jax.random.key(8), query, full_a, None, state, update=True)
        _, state_a = implicit.apply(variables, query, full_a, None, state, update=True)
        _, state_b = implicit.apply(variables, query, full_b, None, state, update=True)
        self.assertFalse(np.allclose(state_a.w1, state_b.w1))

        explicit = ttt_memory.TemporalTTT(query_dim=4, memory_dim=4, hidden_dim=8, source="explicit_jepa")
        memory = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)
        variables = explicit.init(jax.random.key(9), query, full_a, memory, state, update=True)
        _, state_a = explicit.apply(variables, query, full_a, memory, state, update=True)
        _, state_b = explicit.apply(variables, query, full_b, memory, state, update=True)
        np.testing.assert_allclose(state_a.w1, state_b.w1)

<<<<<<< HEAD
    def _config(self, source="explicit_jepa", context=1):
=======
    def _config(self, source="explicit_jepa", context=1, tbptt=8):
>>>>>>> 1152696 (little changes)
        return pi0_config.Pi0Config(
            pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
            action_horizon=2, use_vjepa_aux=True, vjepa_num_queries=4,
            vjepa_query_grid_size=2, vjepa_target_grid_size=3,
            vjepa_target_dim=16, ttt_enabled=True, ttt_architecture="wrapper",
            ttt_memory_source=source, ttt_num_register_tokens=2,
            ttt_num_future_tokens=1, ttt_context_length=context,
<<<<<<< HEAD
=======
            ttt_tbptt_step_size=tbptt,
>>>>>>> 1152696 (little changes)
        )

    def test_explicit_kv_comes_from_current_jepa_not_future_prediction(self):
        config = self._config()
        model = config.create(jax.random.key(0))
        observation, actions = config.fake_obs(1), config.fake_act(1)
        flow_a, aux_a = model.compute_loss_components(jax.random.key(1), observation, actions)
        changed_target = dataclasses.replace(observation, vjepa_target=-observation.vjepa_target)
        flow_b, aux_b = model.compute_loss_components(jax.random.key(1), changed_target, actions)
        np.testing.assert_allclose(flow_a, flow_b, rtol=1e-5, atol=1e-5)
        self.assertFalse(np.allclose(aux_a, aux_b))

        changed_current = dataclasses.replace(observation, vjepa_current=jnp.zeros_like(observation.vjepa_current))
        initial = model.empty_ttt_state(1)
        _, _, state_a = model._ttt_loss_frame(jax.random.key(5), observation, actions, initial, train=False)
        _, _, state_c = model._ttt_loss_frame(jax.random.key(5), changed_current, actions, initial, train=False)
        self.assertFalse(np.allclose(state_a.w1, state_c.w1))

    def test_temporal_context_and_inference_state(self):
        config = self._config("implicit_full", context=2)
        model = config.create(jax.random.key(2))
        observation, actions = config.fake_obs(1), config.fake_act(1)
        temporal_obs = jax.tree.map(
            lambda x: None if x is None else jnp.stack((x, x), axis=1), observation
        )
        temporal_obs = dataclasses.replace(
            temporal_obs, time_valid_mask=jnp.array([[False, True]])
        )
        temporal_actions = jnp.stack((actions, actions), axis=1)
        flow, aux = model.compute_loss_components(jax.random.key(3), temporal_obs, temporal_actions)
        self.assertEqual(flow.shape, (1, 2, 2))
        self.assertEqual(aux.shape, (1, 2))
        _, fast = model.sample_actions(
            jax.random.key(4), observation, num_steps=2, return_ttt_state=True
        )
        self.assertEqual(fast.step.shape, (1, 1))
        np.testing.assert_array_equal(np.asarray(fast.step[:, 0]), [1])

<<<<<<< HEAD
=======
    def test_temporal_context_with_partial_final_tbptt_segment(self):
        config = self._config("implicit_full", context=3, tbptt=2)
        model = config.create(jax.random.key(10))
        observation, actions = config.fake_obs(1), config.fake_act(1)
        temporal_obs = jax.tree.map(
            lambda x: None if x is None else jnp.stack((x, x, x), axis=1), observation
        )
        temporal_obs = dataclasses.replace(
            temporal_obs, time_valid_mask=jnp.array([[True, True, True]])
        )
        temporal_actions = jnp.stack((actions, actions, actions), axis=1)
        flow, aux = model.compute_loss_components(jax.random.key(11), temporal_obs, temporal_actions)
        self.assertEqual(flow.shape, (1, 3, 2))
        self.assertEqual(aux.shape, (1, 3))

    def test_segmented_tbptt_supports_outer_gradient(self):
        config = self._config("explicit_jepa", context=3, tbptt=2)
        model = config.create(jax.random.key(12))
        observation, actions = config.fake_obs(1), config.fake_act(1)
        temporal_obs = jax.tree.map(
            lambda x: None if x is None else jnp.stack((x, x, x), axis=1), observation
        )
        temporal_obs = dataclasses.replace(
            temporal_obs, time_valid_mask=jnp.array([[True, True, True]])
        )
        temporal_actions = jnp.stack((actions, actions, actions), axis=1)

        def loss_fn(model):
            flow, aux = model.compute_loss_components(
                jax.random.key(13), temporal_obs, temporal_actions, train=True
            )
            return jnp.mean(flow) + jnp.mean(aux)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        self.assertTrue(np.isfinite(np.asarray(loss)))
        self.assertTrue(all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(grads)))

>>>>>>> 1152696 (little changes)
    def test_episode_window_does_not_cross_boundary(self):
        class Frames:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {"actions": np.array([index], dtype=np.float32)}

        dataset = TemporalWindowDataset(Frames(), np.array([0, 0, 2, 2]), 3)
        sample = dataset[2]
        np.testing.assert_array_equal(sample["actions"][:, 0], [2, 2, 2])
        np.testing.assert_array_equal(sample["time_valid_mask"], [False, False, True])

    def test_episode_window_uses_physical_frame_stride(self):
        class Frames:
            def __len__(self):
                return 20

            def __getitem__(self, index):
                return {"actions": np.array([index], dtype=np.float32)}

        dataset = TemporalWindowDataset(Frames(), np.zeros(20, dtype=np.int64), 3, observation_stride=8)
        sample = dataset[17]
        np.testing.assert_array_equal(sample["actions"][:, 0], [1, 9, 17])
        np.testing.assert_array_equal(sample["time_valid_mask"], [True, True, True])


if __name__ == "__main__":
    unittest.main()
