import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    run_id_file = ckpt_dir / "wandb_id.txt"
    if resuming and run_id_file.exists():
        run_id = run_id_file.read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        run_id_file.write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def initialize_jax_distributed() -> None:
    """Initialize JAX multi-host mode when launched by the Slurm wrapper."""
    process_count = int(os.environ.get("SLURM_NTASKS", "1"))
    # Querying jax.process_count() initializes the XLA backend, which makes a
    # subsequent jax.distributed.initialize() fail. This state-only check does
    # not touch XLA and is therefore safe before distributed initialization.
    if process_count <= 1 or jax.distributed.is_initialized():
        return

    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if not coordinator_address:
        raise RuntimeError("JAX_COORDINATOR_ADDRESS is required for a multi-host Slurm launch")
    process_id = int(os.environ.get("SLURM_PROCID", "0"))
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=process_count,
        process_id=process_id,
    )
    logging.info(
        "Initialized JAX distributed runtime: process %d/%d, local devices=%d",
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
    )


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    trainable_shapes = traverse_util.flatten_dict(
        train_state_shape.params.filter(config.trainable_filter).to_pure_dict()
    )
    trainable_paths = ["/".join(map(str, path)) for path in trainable_shapes]
    if config.name.endswith(("_ttt_only", "_ttt_only_online")):
        def is_ttt_parameter(path: str) -> bool:
            parts = tuple(path.split("/"))
            return "ttt" in parts or any(part.startswith("ttt_") for part in parts)

        unexpected = [path for path in trainable_paths if not is_ttt_parameter(path)]
        if unexpected:
            raise ValueError(f"TTT-only config exposed non-TTT trainable parameters: {unexpected}")
        if not trainable_paths:
            raise ValueError("TTT-only config did not expose any trainable parameters")
    trainable_count = sum(int(np.prod(value.shape)) for value in trainable_shapes.values())
    logging.info(
        "Trainable parameters: %d values across %d leaves%s",
        trainable_count,
        len(trainable_paths),
        " (strict TTT-only validation passed)" if config.name.endswith("_ttt_only") else "",
    )

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        if observation.time_valid_mask is None:
            return jnp.mean(chunked_loss)
        mask = observation.time_valid_mask.astype(jnp.float32)
        return jnp.sum(chunked_loss * mask[..., None]) / (jnp.maximum(mask.sum(), 1) * chunked_loss.shape[-1])

    def vjepa_loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        flow_loss, vjepa_loss = model.compute_loss_components(rng, observation, actions, train=True)
        assert vjepa_loss is not None
        if config.model.vjepa_aux_warmup_steps > 0:
            warmup = jnp.minimum(
                state.step.astype(jnp.float32) / config.model.vjepa_aux_warmup_steps,
                1.0,
            )
        else:
            warmup = jnp.asarray(1.0, dtype=jnp.float32)
        vjepa_weight = warmup * config.model.vjepa_aux_weight
        if observation.time_valid_mask is None:
            flow_loss = jnp.mean(flow_loss)
            vjepa_loss = jnp.mean(vjepa_loss)
        else:
            mask = observation.time_valid_mask.astype(jnp.float32)
            flow_loss = jnp.sum(flow_loss * mask[..., None]) / (jnp.maximum(mask.sum(), 1) * flow_loss.shape[-1])
            vjepa_loss = jnp.sum(vjepa_loss * mask) / jnp.maximum(mask.sum(), 1)
        weighted_vjepa_loss = vjepa_weight * vjepa_loss
        total_loss = flow_loss + weighted_vjepa_loss
        return total_loss, {
            "flow_loss": flow_loss,
            "vjepa_loss": vjepa_loss,
            "vjepa_cosine": 1.0 - vjepa_loss,
            "vjepa_weight": vjepa_weight,
            "weighted_vjepa_loss": weighted_vjepa_loss,
        }

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    if getattr(config.model, "use_vjepa_aux", False) and getattr(config.model, "vjepa_aux_loss_enabled", True):
        (loss, loss_info), grads = nnx.value_and_grad(vjepa_loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )
    else:
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
        loss_info = {"flow_loss": loss} if getattr(config.model, "use_vjepa_aux", False) else {}

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **loss_info,
    }
    return new_state, info


def validation_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Array:
    """Evaluate action flow matching loss on fixed held-out episodes."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    flow_loss, _ = model.compute_loss_components(rng, observation, actions, train=False)
    if observation.time_valid_mask is None:
        return jnp.mean(flow_loss)
    mask = observation.time_valid_mask.astype(jnp.float32)
    return jnp.sum(flow_loss * mask[..., None]) / (jnp.maximum(mask.sum(), 1) * flow_loss.shape[-1])


def main(config: _config.TrainConfig):
    init_logging()
    initialize_jax_distributed()
    logging.info(f"Running on: {platform.node()}")

    if config.name.endswith("_ttt_only_online") and (
        jax.process_count() != 4 or jax.device_count() != 4 or jax.local_device_count() != 1
    ):
        raise RuntimeError(
            "Online LIBERO TTT requires four JAX processes with one GPU each; "
            f"got processes={jax.process_count()}, devices={jax.device_count()}, "
            f"local_devices={jax.local_device_count()}"
        )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    latest_checkpoint_dir = config.checkpoint_dir / "latest" if config.validation_fraction else config.checkpoint_dir
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        latest_checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled and jax.process_index() == 0,
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    validation_loader = None
    best_manager = None
    best_loss = float("inf")
    if config.validation_fraction:
        validation_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=config.validation_num_batches,
            split="validation",
            online_vjepa_encoder=data_loader.online_vjepa_encoder(),
            num_workers=0,
        )
        best_manager, _ = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir / "best",
            keep_period=None,
            overwrite=False,
            resume=True,
        )
        best_step, best_loss = _checkpoints.read_best_validation(config.checkpoint_dir)
        if best_step != best_manager.latest_step():
            raise ValueError(
                f"Best checkpoint/metric mismatch: checkpoint={best_manager.latest_step()}, metric={best_step}"
            )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pvalidation_step = None
    if validation_loader is not None:
        pvalidation_step = jax.jit(
            functools.partial(validation_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if config.wandb_enabled and jax.process_index() == 0:
                wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        completed_step = step + 1
        final_step = completed_step == config.num_train_steps
        if completed_step % config.save_interval == 0 or final_step:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, completed_step)
            checkpoint_manager.wait_until_finished()

        if validation_loader is not None and (completed_step % config.validation_interval == 0 or final_step):
            values = []
            for validation_index, validation_batch in enumerate(validation_loader):
                validation_rng = jax.random.fold_in(jax.random.key(config.validation_seed), validation_index)
                with sharding.set_mesh(mesh):
                    values.append(pvalidation_step(validation_rng, train_state, validation_batch))
            validation_loss = float(jax.device_get(jnp.mean(jnp.stack(values))))
            if not np.isfinite(validation_loss):
                raise ValueError(f"Non-finite validation flow loss at step {completed_step}: {validation_loss}")
            if jax.process_index() == 0:
                logging.info("Validation flow loss at step %d: %.6f (best %.6f)", completed_step, validation_loss, best_loss)
                if config.wandb_enabled:
                    wandb.log({"validation/flow_loss": validation_loss}, step=completed_step)
            if validation_loss < best_loss:
                assert best_manager is not None
                _checkpoints.save_state(best_manager, train_state, data_loader, completed_step)
                best_manager.wait_until_finished()
                if jax.process_index() == 0:
                    _checkpoints.write_best_validation_loss(config.checkpoint_dir, completed_step, validation_loss)
                best_loss = validation_loss

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if best_manager is not None:
        best_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
