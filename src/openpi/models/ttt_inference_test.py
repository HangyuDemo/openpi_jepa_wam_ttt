"""Inference state transitions across flow-matching steps and observations."""

# ruff: noqa: SLF001

from operator import itemgetter
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_module
from openpi.models import pi0
from openpi.models import ttt_memory


def _ttt_harness():
    module = ttt_memory.TemporalTTT(query_dim=4, memory_dim=4, hidden_dim=8, source="explicit_jepa")
    tokens = jnp.ones((2, 2, 4), dtype=jnp.float32)
    current_jepa = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    empty_slot = jax.tree.map(lambda x: x[0], ttt_memory.empty_fast_weights(2, 2, 4, 8))
    variables = module.init(jax.random.key(30), tokens, tokens, current_jepa, empty_slot, update=True)
    records = []

    class Harness:
        ttt_enabled = True
        ttt_memory_source = "explicit_jepa"
        ttt_update_first_denoise_only = True
        action_horizon = 2
        action_dim = 4
        sample_actions = pi0.Pi0.sample_actions

        def _ttt_prefix(self, observation, state):
            return None, None, None, None

        def _ttt_suffix(self, observation, x_t, time, prefix_mask, cache, current_jepa, state, *, update, update_mask):
            del observation, time, prefix_mask, cache
            outputs, next_slots = [], []
            for slot in range(2):
                slot_state = jax.tree.map(itemgetter(slot), state)
                output, next_slot = module.apply(
                    variables,
                    x_t,
                    x_t,
                    current_jepa,
                    slot_state,
                    update=update,
                    valid_mask=update_mask,
                )
                outputs.append(output)
                next_slots.append(next_slot)
            next_state = jax.tree.map(lambda *xs: jnp.stack(xs), *next_slots)

            def record(step, mask, before, after):
                records.append(
                    (
                        step.copy(),
                        mask.copy(),
                        tuple(weight.copy() for weight in before),
                        tuple(weight.copy() for weight in after),
                    )
                )

            jax.debug.callback(
                record,
                state.step,
                update_mask,
                (state.w1, state.b1, state.w2, state.b2),
                (next_state.w1, next_state.b1, next_state.w2, next_state.b2),
                ordered=True,
            )
            return jnp.mean(jnp.stack(outputs), axis=0), next_state

    observation = SimpleNamespace(state=jnp.ones((2, 4)), vjepa_current=current_jepa)
    initial_step = jnp.array([[5, 7], [10, 12]], dtype=jnp.int32)
    initial = ttt_memory.empty_fast_weights(2, 2, 4, 8)._replace(step=initial_step)
    return Harness(), observation, initial, records


def _assert_records(records, start_step, expected_flags):
    assert len(records) == len(expected_flags)
    for index, (step, mask, before, after) in enumerate(records):
        np.testing.assert_array_equal(step, start_step)
        np.testing.assert_array_equal(mask, np.full(2, expected_flags[index]))
        if index == 0:
            assert any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))
        else:
            for a, b in zip(before, after, strict=True):
                np.testing.assert_array_equal(a, b)


def test_first_denoise_keeps_position_and_commits_once_across_observations():
    model, observation, initial, records = _ttt_harness()
    with mock.patch.object(model_module, "preprocess_observation", side_effect=lambda rng, obs, train: obs):
        _, after_a = model.sample_actions(
            jax.random.key(31),
            observation,
            num_steps=4,
            ttt_state=initial,
            return_ttt_state=True,
        )
        np.testing.assert_array_equal(after_a.step, initial.step + 1)
        _assert_records(records, initial.step, [True, False, False, False])

        records.clear()
        _, after_b = model.sample_actions(
            jax.random.key(32),
            observation,
            num_steps=4,
            ttt_state=after_a,
            return_ttt_state=True,
        )
        np.testing.assert_array_equal(after_b.step, initial.step + 2)
        _assert_records(records, initial.step + 1, [True, False, False, False])

        records.clear()
        _, one_step = model.sample_actions(
            jax.random.key(33),
            observation,
            num_steps=1,
            ttt_state=initial,
            return_ttt_state=True,
        )
        np.testing.assert_array_equal(one_step.step, initial.step + 1)
        _assert_records(records, initial.step, [True])


def test_episode_reset_discards_persistent_ttt_state():
    from openpi.policies import policy as policy_module

    policy = object.__new__(policy_module.Policy)
    policy._ttt_state = ttt_memory.empty_fast_weights(2, 1, 4, 8)
    policy.reset()
    assert policy._ttt_state is None


def test_non_ttt_inference_keeps_original_euler_path():
    class PlainHarness:
        ttt_enabled = False
        use_vjepa_aux = False
        action_horizon = 2
        action_dim = 4
        sample_actions = pi0.Pi0.sample_actions

        def __init__(self):
            self.PaliGemma = SimpleNamespace(llm=self.llm)

        def llm(self, xs, **kwargs):
            return (None, None) if xs[1] is None else ((None, xs[1]), None)

        def embed_prefix(self, observation):
            batch = observation.state.shape[0]
            return jnp.ones((batch, 1, 4)), jnp.ones((batch, 1), bool), jnp.array([False])

        def embed_suffix(self, observation, x_t, time):
            return x_t, jnp.ones(x_t.shape[:2], bool), jnp.array([False, False]), None

        def action_out_proj(self, x):
            return x

    model = PlainHarness()
    observation = SimpleNamespace(state=jnp.ones((1, 4)))
    noise = jnp.ones((1, 2, 4))
    with mock.patch.object(model_module, "preprocess_observation", side_effect=lambda rng, obs, train: obs):
        one = model.sample_actions(jax.random.key(34), observation, num_steps=1, noise=noise)
        four = model.sample_actions(jax.random.key(35), observation, num_steps=4, noise=noise)
    np.testing.assert_allclose(one, jnp.zeros_like(noise))
    np.testing.assert_allclose(four, noise * (0.75**4), rtol=1e-6)
