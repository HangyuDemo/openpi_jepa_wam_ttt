import dataclasses
import functools
import logging
import os
import platform
import time
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


_MATCHED_WE_TTT_ONLY_CONFIGS = frozenset({
    "pi05_libero_ttt_wrapper_explicit_jepa_ttt_only",
    "pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only",
})
_QUEUED_ROBOTWIN_CONTEXT_16_JOBS = frozenset({"1034331", "1037807"})
_WRAPPER_TTT_PATH = ("PaliGemma", "llm", "layers", "ttt")
_DEFAULT_WRAPPER_TTT_INDICES = (3, 7, 11, 15)
_MASKED_WRAPPER_CONFIGS = _MATCHED_WE_TTT_ONLY_CONFIGS | {"pi05_piper_jepa_ttt_joint"}


def _selected_wrapper_indices(config: _config.TrainConfig) -> tuple[int, ...]:
    return config.model.ttt_layer_indices or _DEFAULT_WRAPPER_TTT_INDICES


def _correct_queued_robotwin_context(config: _config.TrainConfig, job_id: str | None) -> _config.TrainConfig:
    """Override the stale CLI value in the two already-submitted Clean5 jobs."""
    if (
        job_id in _QUEUED_ROBOTWIN_CONTEXT_16_JOBS
        and config.name == "pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only"
        and config.model.ttt_context_length == 128
        and config.model.ttt_observation_stride == 8
    ):
        return dataclasses.replace(config, model=dataclasses.replace(config.model, ttt_context_length=16))
    return config


def _effective_wrapper_trainable_count(config: _config.TrainConfig, trainable_shapes: dict) -> int:
    """Count parameters that can change, not checkpoint-compatible optimizer slots."""
    selected = _selected_wrapper_indices(config)
    count = 0
    for path, value in trainable_shapes.items():
        if path == ("ttt_future_tokens",):
            continue
        if path[:4] == _WRAPPER_TTT_PATH:
            if not value.shape or max(selected) >= value.shape[0]:
                raise ValueError(f"Invalid selected TTT layers {selected} for {path}: {value.shape}")
            count += len(selected) * int(np.prod(value.shape[1:]))
        elif path == ("ttt_register_tokens",):
            count += int(np.prod(value.shape))
        else:
            raise ValueError(f"Unexpected trainable parameter in matched WE route: {path}")
    return count


def _preserve_frozen_wrapper_params(
    config: _config.TrainConfig, old_params: nnx.State, updated_params: nnx.State
) -> nnx.State:
    """Freeze inactive wrapper slots/future tokens without changing old checkpoint shapes.

    Existing checkpoints store optimizer moments for all 18 scanned TTT slots.
    Keep that tree intact for resume, but discard every update (including AdamW
    weight decay) outside the four selected wrappers and the register tokens.
    """
    if config.name not in _MASKED_WRAPPER_CONFIGS:
        return updated_params

    selected = _selected_wrapper_indices(config)
    previous = old_params.flat_state()

    def preserve(path, param):
        if path == ("ttt_future_tokens",):
            return previous[path]
        if path[:4] != _WRAPPER_TTT_PATH:
            return param
        if not param.value.shape or max(selected) >= param.value.shape[0]:
            raise ValueError(f"Invalid selected TTT layers {selected} for {path}: {param.value.shape}")
        selected_mask = jnp.asarray([i in selected for i in range(param.value.shape[0])])
        selected_mask = selected_mask.reshape((-1,) + (1,) * (param.value.ndim - 1))
        return param.replace(value=jnp.where(selected_mask, param.value, previous[path].value))

    return updated_params.map(preserve)


def _mask_frozen_wrapper_grads(config: _config.TrainConfig, grads: nnx.State) -> nnx.State:
    """Exclude inactive slots and future tokens from AdamW and global clipping."""
    if config.name not in _MASKED_WRAPPER_CONFIGS:
        return grads

    selected = _selected_wrapper_indices(config)

    def mask(path, grad):
        if path == ("ttt_future_tokens",):
            return grad.replace(value=jnp.zeros_like(grad.value))
        if path[:4] != _WRAPPER_TTT_PATH:
            return grad
        if not grad.value.shape or max(selected) >= grad.value.shape[0]:
            raise ValueError(f"Invalid selected TTT layers {selected} for {path}: {grad.value.shape}")
        selected_mask = jnp.asarray([i in selected for i in range(grad.value.shape[0])])
        selected_mask = selected_mask.reshape((-1,) + (1,) * (grad.value.ndim - 1))
        return grad.replace(value=jnp.where(selected_mask, grad.value, jnp.zeros_like(grad.value)))

    return grads.map(mask)


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
    if config.name in _MATCHED_WE_TTT_ONLY_CONFIGS:
        logging.info(
            "Effective selected-wrapper trainable parameters: %d; optimizer slots retained for checkpoint resume",
            _effective_wrapper_trainable_count(config, trainable_shapes),
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
def compute_gradients(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[nnx.State, dict[str, at.Array]]:
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

    return grads, {"loss": loss, **loss_info}


def accumulation_weight(batch: tuple[_model.Observation, _model.Actions]) -> at.Array:
    """Weight microbatch means by valid observations, excluding temporal padding."""
    observation, actions = batch
    if observation.time_valid_mask is not None:
        return jnp.sum(observation.time_valid_mask.astype(jnp.float32))
    return jnp.asarray(actions.shape[0], dtype=jnp.float32)


def accumulate_gradients(config, rng, state, batch, accumulator):
    """Stream one microbatch without updating the optimizer or model weights."""
    grads, info = compute_gradients(config, rng, state, batch)
    weight = accumulation_weight(batch)
    grad_sum, info_sum, weight_sum = accumulator
    return (
        jax.tree.map(lambda total, value: total + value * weight, grad_sum, grads),
        jax.tree.map(lambda total, value: total + value * weight, info_sum, info),
        weight_sum + weight,
    )


def apply_gradients(config, state, grads, loss_info):
    model = nnx.merge(state.model_def, state.params)
    grads = _mask_frozen_wrapper_grads(config, grads)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params = _preserve_frozen_wrapper_params(config, params, new_params)

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
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **loss_info,
    }
    return new_state, info


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    grads, info = compute_gradients(config, rng, state, batch)
    if os.environ.get("OPENPI_ABORT_ON_NONFINITE") == "1":
        input_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(batch)]))
        loss_finite = jnp.all(jnp.isfinite(info["loss"]))
        grad_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))

        def apply(_):
            new_state, metrics = apply_gradients(config, state, grads, info)
            return new_state, {
                **metrics,
                "input_finite": input_finite,
                "loss_finite": loss_finite,
                "grad_finite": grad_finite,
            }

        def stop(_):
            # Keep the last finite state intact. The host raises before any
            # checkpoint can be written from this step.
            return state, {
                **info,
                "grad_norm": optax.global_norm(grads),
                "param_norm": jnp.asarray(jnp.nan, jnp.float32),
                "input_finite": input_finite,
                "loss_finite": loss_finite,
                "grad_finite": grad_finite,
            }

        return jax.lax.cond(input_finite & loss_finite & grad_finite, apply, stop, operand=None)
    return apply_gradients(config, state, grads, info)


def apply_accumulated_gradients(config, state, accumulator):
    grad_sum, info_sum, weight_sum = accumulator
    denominator = jnp.maximum(weight_sum, 1)
    grads = jax.tree.map(lambda value: value / denominator, grad_sum)
    info = jax.tree.map(lambda value: value / denominator, info_sum)
    return apply_gradients(config, state, grads, info)


def validation_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Array:
    """Evaluate action flow matching loss on fixed held-out episodes."""
    metrics = validation_metrics(config, rng, state, batch)
    return metrics["flow_loss_sum"] / jnp.maximum(metrics["weight"], 1)


def validation_metrics(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    flow_loss, aux_loss = model.compute_loss_components(rng, observation, actions, train=False)
    mask = (
        jnp.ones(flow_loss.shape[:-1], dtype=jnp.float32)
        if observation.time_valid_mask is None else observation.time_valid_mask.astype(jnp.float32)
    )
    metrics = {
        "flow_loss_sum": jnp.sum(jnp.mean(flow_loss, axis=-1) * mask),
        "weight": jnp.sum(mask),
    }
    if aux_loss is not None:
        metrics["vjepa_loss_sum"] = jnp.sum(aux_loss * mask)
    return metrics


def main(config: _config.TrainConfig):
    init_logging()
    if config.name == "pi05_piper_jepa_ttt_joint" and os.environ.get("SLURM_JOB_ID"):
        if (
            int(os.environ.get("SLURM_NTASKS", "0")) != 4
            or config.fsdp_devices != 4
            or config.batch_size != 64
            or config.gradient_accumulation_steps != 4
        ):
            raise ValueError(
                "Piper joint training requires four GPUs, global batch 64, "
                "per-GPU microbatch 4, and four accumulation steps."
            )
    # Slurm keeps a copy of each submitted script, so editing the .sbatch file
    # cannot change the save interval of these already-queued jobs.
    queued_save_100_jobs = {
        "930092": "pi05_libero_ttt_inline_explicit_jepa_ttt_only",
        "930140": "pi05_libero_ttt_wrapper_explicit_jepa_ttt_only",
        "935952": "pi05_libero_ttt_wrapper_implicit_full_ttt_only",
        "899790": "pi05_libero_ttt_inline_implicit_full_ttt_only",
        "930174": "pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only",
    }
    job_id = os.environ.get("SLURM_JOB_ID")
    corrected_config = _correct_queued_robotwin_context(config, job_id)
    if corrected_config is not config:
        logging.info("Slurm job %s: overriding submitted RoboTwin context from 128 to 16 observations", job_id)
        config = corrected_config
    if config.name == queued_save_100_jobs.get(job_id) and config.save_interval == 500:
        config = dataclasses.replace(config, save_interval=100)
        logging.info("Slurm job %s: checkpoint save interval overridden from 500 to 100 steps", job_id)
    initialize_jax_distributed()
    logging.info(f"Running on: {platform.node()}")

    if config.name.endswith("_ttt_only_online") and (
        jax.device_count() != jax.process_count() or jax.local_device_count() != 1
    ):
        raise RuntimeError(
            "Online LIBERO TTT requires one GPU per JAX process; "
            f"got processes={jax.process_count()}, devices={jax.device_count()}, "
            f"local_devices={jax.local_device_count()}"
        )

    microbatch_size = config.batch_size // config.gradient_accumulation_steps
    if microbatch_size % jax.device_count() != 0:
        raise ValueError(
            f"Global microbatch {microbatch_size} must be divisible by {jax.device_count()} devices."
        )
    logging.info(
        "Global batch=%d = microbatch=%d x accumulation=%d; steps count optimizer updates",
        config.batch_size, microbatch_size, config.gradient_accumulation_steps,
    )

    # Keep compilation artifacts out of the small PSC home quota when the job
    # provides a node-local or project cache directory.
    jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path(jax_cache_dir).expanduser()))

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

    loader_config = dataclasses.replace(config, batch_size=microbatch_size, gradient_accumulation_steps=1)
    data_loader = _data_loader.create_data_loader(
        loader_config,
        sharding=data_sharding,
        shuffle=True,
    )
    validation_loader = None
    best_manager = None
    best_loss = float("inf")
    if config.validation_fraction:
        validation_loader = _data_loader.create_data_loader(
            loader_config,
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
    accumulator_sharding = None
    if config.gradient_accumulation_steps > 1:
        grad_sharding = train_state_sharding.params.filter(config.trainable_filter)
        accumulator_sharding = (grad_sharding, replicated_sharding, replicated_sharding)
        paccumulate = jax.jit(
            functools.partial(accumulate_gradients, config),
            in_shardings=(
                replicated_sharding, train_state_sharding, data_sharding, accumulator_sharding,
            ),
            out_shardings=accumulator_sharding,
            donate_argnums=(3,),
        )
        papply = jax.jit(
            functools.partial(apply_accumulated_gradients, config),
            in_shardings=(train_state_sharding, accumulator_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(0, 1),
        )
        metric_names = {"loss"}
        if getattr(config.model, "use_vjepa_aux", False):
            metric_names.add("flow_loss")
            if getattr(config.model, "vjepa_aux_loss_enabled", True):
                metric_names.update({"vjepa_loss", "vjepa_cosine", "vjepa_weight", "weighted_vjepa_loss"})

        def empty_accumulator(params):
            return (
                jax.tree.map(jnp.zeros_like, params),
                {name: jnp.zeros((), jnp.float32) for name in sorted(metric_names)},
                jnp.zeros((), jnp.float32),
            )

        pempty_accumulator = jax.jit(
            empty_accumulator, in_shardings=(grad_sharding,), out_shardings=accumulator_sharding,
        )
    pvalidation_step = None
    if validation_loader is not None:
        pvalidation_step = jax.jit(
            functools.partial(validation_metrics, config),
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
        step_started = time.monotonic()
        with sharding.set_mesh(mesh):
            if config.gradient_accumulation_steps == 1:
                train_state, info = ptrain_step(train_rng, train_state, batch)
            else:
                accumulator = pempty_accumulator(train_state.params.filter(config.trainable_filter))
                for microstep in range(config.gradient_accumulation_steps):
                    accumulator = paccumulate(
                        jax.random.fold_in(train_rng, microstep), train_state, batch, accumulator,
                    )
                    # Keep the host from queuing and retaining 32 large image/token batches.
                    jax.block_until_ready(accumulator)
                    if step == start_step and microstep == 0:
                        first_metrics = jax.device_get(accumulator[1])
                        if not all(np.all(np.isfinite(value)) for value in first_metrics.values()):
                            raise ValueError("Non-finite loss in the first accumulated microbatch")
                        logging.info("First real-data joint gradient microbatch completed successfully")
                    if microstep + 1 < config.gradient_accumulation_steps:
                        batch = next(data_iter)
                train_state, info = papply(train_state, accumulator)
                jax.block_until_ready(info)
                info["optimizer_step_seconds"] = time.monotonic() - step_started
                if not all(np.all(np.isfinite(value)) for value in jax.device_get(info).values()):
                    raise ValueError(f"Non-finite training metric at optimizer step {step + 1}")
        if os.environ.get("OPENPI_ABORT_ON_NONFINITE") == "1" and config.gradient_accumulation_steps == 1:
            step_metrics = jax.device_get(info)
            if not all(
                bool(step_metrics[name]) for name in ("input_finite", "loss_finite", "grad_finite")
            ) or not all(np.all(np.isfinite(value)) for value in step_metrics.values()):
                raise ValueError(f"Non-finite training metric at optimizer step {step + 1}: {step_metrics}")
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step + 1}: {info_str}")
            if config.wandb_enabled and jax.process_index() == 0:
                wandb.log(reduced_info, step=step + 1, commit=False)
            infos = []
        batch = next(data_iter)

        completed_step = step + 1
        final_step = completed_step == config.num_train_steps
        first_joint_step = config.gradient_accumulation_steps > 1 and completed_step == 1
        if completed_step % config.save_interval == 0 or final_step or first_joint_step:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, completed_step)
            checkpoint_manager.wait_until_finished()

        validation_due = validation_loader is not None and (
            completed_step % config.validation_interval == 0 or final_step or first_joint_step
        )
        if validation_due:
            values = []
            for validation_index, validation_batch in enumerate(validation_loader):
                validation_rng = jax.random.fold_in(jax.random.key(config.validation_seed), validation_index)
                with sharding.set_mesh(mesh):
                    values.append(pvalidation_step(validation_rng, train_state, validation_batch))
            totals = jax.device_get(jax.tree.map(lambda *xs: jnp.sum(jnp.stack(xs)), *values))
            validation_info = {
                f"validation/{name.removesuffix('_sum')}": float(value / max(totals["weight"], 1))
                for name, value in totals.items() if name != "weight"
            }
            if not all(np.isfinite(value) for value in validation_info.values()):
                raise ValueError(f"Non-finite validation metrics at step {completed_step}: {validation_info}")
            validation_loss = validation_info["validation/flow_loss"]
            if jax.process_index() == 0:
                logging.info(
                    "Validation flow loss at step %d: %.6f (best %.6f)", completed_step, validation_loss, best_loss
                )
                if config.wandb_enabled:
                    wandb.log(validation_info, step=completed_step, commit=False)
            if validation_loss < best_loss:
                assert best_manager is not None
                _checkpoints.save_state(best_manager, train_state, data_loader, completed_step)
                best_manager.wait_until_finished()
                if jax.process_index() == 0:
                    _checkpoints.write_best_validation_loss(config.checkpoint_dir, completed_step, validation_loss)
                best_loss = validation_loss
                if config.wandb_enabled and jax.process_index() == 0:
                    wandb.run.summary.update({"best_step": completed_step, "best_validation_flow_loss": best_loss})
        # Commit once so training and validation metrics at the same optimizer
        # step share a W&B row instead of dropping the later validation entry.
        if config.wandb_enabled and jax.process_index() == 0 and (step % config.log_interval == 0 or validation_due):
            wandb.log({}, step=completed_step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if best_manager is not None:
        best_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
