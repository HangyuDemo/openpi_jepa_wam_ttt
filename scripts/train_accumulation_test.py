"""Numerical checks for streamed GBS accumulation and validation reduction."""
import dataclasses
import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import utils
from scripts import train


class ToyPolicy(_model.BaseModel):
    def __init__(self):
        super().__init__(action_dim=1, action_horizon=2, max_token_len=1)
        self.kernel = nnx.Param(jnp.array([[0.7]], jnp.float32))

    def compute_loss_components(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state @ self.kernel
        flow = jnp.square(prediction[..., None, :] - actions).mean(axis=-1)
        aux = jnp.square(prediction[..., 0] - 0.3)
        return flow, aux

    def compute_loss(self, rng, observation, actions, *, train=False):
        flow, aux = self.compute_loss_components(rng, observation, actions, train=train)
        return flow + 0.1 * aux[..., None]

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return (observation.state @ self.kernel)[..., None, :]


def test_accumulation_matches_full_masked_batch_and_single_optimizer_step():
    config = dataclasses.replace(
        _config.get_config('debug'),
        model=pi0_config.Pi0Config(pi05=True, use_vjepa_aux=True, vjepa_aux_warmup_steps=0),
        batch_size=4, gradient_accumulation_steps=2, ema_decay=None,
    )
    model = ToyPolicy()
    graph, params = nnx.split(model)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
    state = utils.TrainState(
        step=jnp.array(0), model_def=graph, params=params, opt_state=tx.init(params), tx=tx, ema_decay=None,
    )
    observation = _model.Observation(
        images={}, image_masks={}, state=jnp.arange(12, dtype=jnp.float32).reshape(4, 3, 1) / 3,
        time_valid_mask=jnp.array([[False, False, True], [False, True, True], [True, True, True], [True, True, True]]),
    )
    actions = jnp.ones((4, 3, 2, 1))
    rng = jax.random.key(7)
    expected, expected_info = jax.jit(functools.partial(train.train_step, config))(rng, state, (observation, actions))
    _, metrics = train.compute_gradients(config, rng, state, (observation, actions))
    accumulator = (
        jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, metrics), jnp.array(0.0),
    )
    accumulate = jax.jit(functools.partial(train.accumulate_gradients, config))
    for start in (0, 2):
        microbatch = jax.tree.map(lambda value: value[start:start + 2], (observation, actions))
        accumulator = accumulate(rng, state, microbatch, accumulator)
        assert int(state.step) == 0
    assert float(accumulator[2]) == 9
    actual, info = jax.jit(functools.partial(train.apply_accumulated_gradients, config))(state, accumulator)
    assert int(actual.step) == 1
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
    for key in expected_info:
        np.testing.assert_allclose(info[key], expected_info[key], rtol=2e-6, atol=2e-6)
    full = train.validation_metrics(config, rng, state, (observation, actions))
    parts = [train.validation_metrics(config, rng, state, jax.tree.map(lambda x: x[s:s+2], (observation, actions))) for s in (0, 2)]
    summed = jax.tree.map(lambda a, b: a + b, *parts)
    for key in full:
        np.testing.assert_allclose(summed[key], full[key], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('batch,accumulation', [(64, 0), (63, 32)])
def test_invalid_accumulation_is_rejected(batch, accumulation):
    with pytest.raises(ValueError):
        dataclasses.replace(_config.get_config('debug'), batch_size=batch, gradient_accumulation_steps=accumulation)


def test_superseded_two_gpu_piper_job_stops_before_model_initialization(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '1051119')
    monkeypatch.setenv('SLURM_NTASKS', '2')
    old_submission = dataclasses.replace(
        _config.get_config('pi05_piper_jepa_ttt_joint'),
        fsdp_devices=2,
        gradient_accumulation_steps=32,
    )
    with pytest.raises(ValueError, match='earlier 2-GPU submission is obsolete'):
        train.main(old_submission)


def test_accumulated_main_validation_checkpoint_and_resume(tmp_path, monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class ToyConfig(pi0_config.Pi0Config):
        def create(self, rng):
            del rng
            return ToyPolicy()

    config = dataclasses.replace(
        _config.get_config('debug'), model=ToyConfig(pi05=True, use_vjepa_aux=True, vjepa_aux_warmup_steps=0),
        batch_size=2 * jax.device_count(), gradient_accumulation_steps=2, num_workers=0, ema_decay=None,
        num_train_steps=2, log_interval=1, save_interval=1, validation_interval=1,
        validation_fraction=0.05, validation_num_batches=2, keep_period=None,
        overwrite=False, exp_name='accumulated', checkpoint_base_dir=str(tmp_path),
        fsdp_devices=jax.device_count(),
    )

    class Loader:
        def __init__(self, sharding, num_batches):
            self.sharding, self.num_batches = sharding, num_batches

        def data_config(self):
            return _config.DataConfig()

        def online_vjepa_encoder(self):
            return None

        def __iter__(self):
            microbatch_size = jax.device_count()
            observation = _model.Observation(
                images={}, image_masks={}, state=jnp.ones((microbatch_size, 3, 1)),
                time_valid_mask=jnp.tile(jnp.array([[False, True, True]]), (microbatch_size, 1)),
            )
            batch = jax.device_put((observation, jnp.ones((microbatch_size, 3, 2, 1))), self.sharding)
            count = 0
            while self.num_batches is None or count < self.num_batches:
                yield batch
                count += 1

    def create_loader(loader_config, *, sharding, num_batches=None, **kwargs):
        del kwargs
        assert loader_config.batch_size == jax.device_count()
        return Loader(sharding, num_batches)

    monkeypatch.setattr(train._data_loader, 'create_data_loader', create_loader)
    train.main(config)
    assert (config.checkpoint_dir / 'latest' / '2' / 'params' / '_METADATA').exists()
    best_step, best_loss = train._checkpoints.read_best_validation(config.checkpoint_dir)
    assert best_step == 2 and np.isfinite(best_loss)
    train.main(dataclasses.replace(config, resume=True, num_train_steps=3))
    assert (config.checkpoint_dir / 'latest' / '3' / 'params' / '_METADATA').exists()
    assert train._checkpoints.read_best_validation(config.checkpoint_dir)[0] == 3
