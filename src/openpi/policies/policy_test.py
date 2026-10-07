# ruff: noqa: SLF001

import jax
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_seeded_jax_inference_rng_is_stable_and_does_not_advance_global_rng():
    initial_rng = jax.random.key(0)

    next_rng, first = _policy._select_jax_inference_rng(initial_rng, 123)
    repeated_next_rng, repeated = _policy._select_jax_inference_rng(initial_rng, 123)
    advanced_rng, unseeded = _policy._select_jax_inference_rng(initial_rng, None)

    np.testing.assert_array_equal(jax.random.key_data(next_rng), jax.random.key_data(initial_rng))
    np.testing.assert_array_equal(jax.random.key_data(repeated_next_rng), jax.random.key_data(initial_rng))
    np.testing.assert_array_equal(jax.random.key_data(first), jax.random.key_data(repeated))
    assert not np.array_equal(jax.random.key_data(first), jax.random.key_data(unseeded))
    assert not np.array_equal(jax.random.key_data(advanced_rng), jax.random.key_data(initial_rng))


def test_policy_recorder_forwards_seed(tmp_path):
    class FakePolicy:
        def __init__(self):
            self.seeds = []

        def infer(self, _observation, *, seed=None):
            self.seeds.append(seed)
            return {"actions": np.zeros((1, 1), dtype=np.float32)}

    inner = FakePolicy()
    recorder = _policy.PolicyRecorder(inner, str(tmp_path))

    recorder.infer({"state": np.zeros(1, dtype=np.float32)}, seed=123)

    assert inner.seeds == [123]
    assert (tmp_path / "step_0.npy").is_file()


def test_aloha_inputs_preserve_vjepa_target():
    example = aloha_policy.make_aloha_example()
    target = np.ones((576, 1408), dtype=np.float16)
    example["vjepa_target"] = target

    inputs = aloha_policy.AlohaInputs()(example)

    np.testing.assert_array_equal(inputs["vjepa_target"], target)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
