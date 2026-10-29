"""Tests for inference step selection without loading a checkpoint."""

from unittest import mock

import pytest

from scripts import serve_policy


@mock.patch("scripts.serve_policy._policy_config.create_trained_policy")
@mock.patch("scripts.serve_policy._config.get_config")
def test_checkpoint_step_choices(get_config, create_trained_policy):
    checkpoint = serve_policy.Checkpoint(config="pi05_piper_jepa_ttt_joint", dir="/checkpoint")
    for steps in (1, 4, 10):
        serve_policy.create_policy(serve_policy.Args(policy=checkpoint, num_steps=steps))
        assert create_trained_policy.call_args.kwargs["sample_kwargs"] == {"num_steps": steps}
    assert get_config.call_count == 3


@mock.patch("scripts.serve_policy._policy_config.create_trained_policy")
@mock.patch("scripts.serve_policy._config.get_config")
def test_default_policy_and_invalid_step_count(get_config, create_trained_policy):
    serve_policy.create_policy(serve_policy.Args(env=serve_policy.EnvMode.LIBERO, num_steps=4))
    assert create_trained_policy.call_args.kwargs["sample_kwargs"] == {"num_steps": 4}
    with pytest.raises(ValueError, match="num_steps must be positive"):
        serve_policy.create_policy(serve_policy.Args(num_steps=0))
    assert get_config.call_count == 1
