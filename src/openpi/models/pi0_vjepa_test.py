import dataclasses

from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp

from openpi.models import pi0_config


def _dummy_config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        **kwargs,
    )


def test_disabled_model_has_no_vjepa_parameters():
    model = _dummy_config().create(jax.random.key(0))
    params = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    assert not any("vjepa_" in path for path in params)


def test_vjepa_aux_forward_and_inference_prefix():
    config = _dummy_config(
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=3,
        vjepa_target_dim=16,
    )
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(1), config.fake_act(1)

    flow_loss, aux_loss = model.compute_loss_components(jax.random.key(1), observation, actions)
    assert flow_loss.shape == (1, config.action_horizon)
    assert aux_loss is not None
    assert aux_loss.shape == (1,)
    assert jnp.all(jnp.isfinite(flow_loss))
    assert jnp.all(jnp.isfinite(aux_loss))

    inference_observation = dataclasses.replace(observation, vjepa_target=None)
    _, prefix_mask, prefix_ar_mask = model.embed_prefix(inference_observation)
    assert jnp.all(prefix_mask[:, -config.vjepa_num_queries :])
    assert prefix_ar_mask[-config.vjepa_num_queries :].tolist() == [True, False, False, False]


def test_masked_queries_do_not_change_flow_prediction():
    base_config = _dummy_config()
    aux_config = _dummy_config(
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=3,
        vjepa_target_dim=16,
        vjepa_action_attends_queries=False,
    )
    base_model = base_config.create(jax.random.key(0))
    aux_model = aux_config.create(jax.random.key(0))
    base_observation, actions = base_config.fake_obs(1), base_config.fake_act(1)
    aux_observation = dataclasses.replace(
        base_observation,
        vjepa_target=jnp.ones((1, 9, 16), dtype=jnp.float16),
    )

    base_flow, _ = base_model.compute_loss_components(jax.random.key(1), base_observation, actions)
    aux_flow, _ = aux_model.compute_loss_components(jax.random.key(1), aux_observation, actions)
    assert jnp.allclose(base_flow, aux_flow, atol=2e-3, rtol=2e-3)
