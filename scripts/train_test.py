import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax.numpy as jnp
import numpy as np

from openpi.training import config as _config

from . import train


def test_matched_we_context_is_sixteen_observations():
    for name in train._MATCHED_WE_TTT_ONLY_CONFIGS:
        model = _config._CONFIGS_DICT[name].model  # noqa: SLF001
        assert model.ttt_context_length == 16
        assert model.ttt_observation_stride == 8
        assert model.ttt_tbptt_step_size == 8


def test_queued_robotwin_context_override_is_limited_to_two_jobs():
    config = _config._CONFIGS_DICT["pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only"]  # noqa: SLF001
    stale = dataclasses.replace(config, model=dataclasses.replace(config.model, ttt_context_length=128))
    for job_id in train._QUEUED_ROBOTWIN_CONTEXT_16_JOBS:
        fixed = train._correct_queued_robotwin_context(stale, job_id)
        assert fixed.model.ttt_context_length == 16
        assert fixed.model.ttt_observation_stride == 8
        assert fixed.model.ttt_tbptt_step_size == 8
    assert train._correct_queued_robotwin_context(stale, "unrelated") is stale
    assert train._correct_queued_robotwin_context(stale, None) is stale


def test_matched_we_freezes_inactive_ttt_slots_and_future_tokens():
    config = _config._CONFIGS_DICT["pi05_libero_ttt_wrapper_explicit_jepa_ttt_only"]  # noqa: SLF001
    previous = nnx.State({
        "PaliGemma": {"llm": {"layers": {"ttt": {"memory_w1": nnx.Param(jnp.ones((18, 2)))}}}},
        "ttt_future_tokens": nnx.Param(jnp.ones((2,))),
        "ttt_register_tokens": nnx.Param(jnp.ones((3,))),
    })
    updated = previous.map(lambda _path, param: param.replace(value=param.value + 2))
    actual = train._preserve_frozen_wrapper_params(config, previous, updated).flat_state()
    masked = train._mask_frozen_wrapper_grads(config, updated).flat_state()

    wrapper = np.asarray(actual[("PaliGemma", "llm", "layers", "ttt", "memory_w1")].value)
    for index in range(18):
        np.testing.assert_array_equal(wrapper[index], [3, 3] if index in (3, 7, 11, 15) else [1, 1])
    np.testing.assert_array_equal(actual[("ttt_future_tokens",)].value, [1, 1])
    np.testing.assert_array_equal(actual[("ttt_register_tokens",)].value, [3, 3, 3])
    masked_wrapper = np.asarray(masked[("PaliGemma", "llm", "layers", "ttt", "memory_w1")].value)
    for index in range(18):
        np.testing.assert_array_equal(masked_wrapper[index], [3, 3] if index in (3, 7, 11, 15) else [0, 0])
    np.testing.assert_array_equal(masked[("ttt_future_tokens",)].value, [0, 0])
    np.testing.assert_array_equal(masked[("ttt_register_tokens",)].value, [3, 3, 3])
    assert train._effective_wrapper_trainable_count(
        config,
        {
            ("PaliGemma", "llm", "layers", "ttt", "memory_w1"): jnp.ones((18, 2)),
            ("ttt_future_tokens",): jnp.ones((2,)),
            ("ttt_register_tokens",): jnp.ones((3,)),
        },
    ) == 11


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
