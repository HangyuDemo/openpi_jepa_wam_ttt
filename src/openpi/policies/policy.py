from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _inject_online_vjepa_current(inputs: dict, encoder: Any) -> dict:
    """Encode the raw camera frame before policy image resizing and normalization."""
    raw_image = np.asarray(inputs.pop("vjepa_online_image", inputs["observation/image"]))
    if raw_image.ndim != 3 or raw_image.shape[-1] != 3 or raw_image.dtype != np.uint8:
        raise ValueError(f"Online V-JEPA inference expects a uint8 HWC image, got {raw_image.shape}/{raw_image.dtype}")
    inputs["vjepa_current"] = encoder.encode(
        raw_image[None, None, ...], np.zeros((1, 1), dtype=np.int64)
    )[0, 0]
    return inputs


def _select_jax_inference_rng(rng: at.KeyArrayLike, seed: int | None) -> tuple[at.KeyArrayLike, at.KeyArrayLike]:
    if seed is None:
        return tuple(jax.random.split(rng))
    return rng, jax.random.key(seed)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        vjepa_encoder: Any | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._ttt_enabled = not is_pytorch and bool(getattr(model, "ttt_enabled", False))
        self._ttt_state = None
        self._vjepa_encoder = vjepa_encoder

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions, static_argnames=("return_ttt_state",)
            ) if self._ttt_enabled else nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        seed: int | None = None,
    ) -> dict:  # type: ignore[misc]
        if seed is not None and noise is not None:
            raise ValueError("seed and explicit noise are mutually exclusive")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
            raise ValueError(f"seed must be a uint32, got {seed!r}")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if self._vjepa_encoder is not None:
            inputs = _inject_online_vjepa_current(inputs, self._vjepa_encoder)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = _select_jax_inference_rng(self._rng, seed)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if self._is_pytorch_model and seed is not None:
            generator = torch.Generator(device=self._pytorch_device)
            generator.manual_seed(seed)
            model_config = self._model.config
            noise = torch.randn(
                (1, model_config.action_horizon, model_config.action_dim),
                device=self._pytorch_device,
                dtype=torch.float32,
                generator=generator,
            )
        if noise is not None:
            if self._is_pytorch_model:
                if isinstance(noise, np.ndarray):
                    noise = torch.from_numpy(noise).to(self._pytorch_device)
            else:
                noise = jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if self._ttt_enabled:
            actions, self._ttt_state = self._sample_actions(
                sample_rng_or_pytorch_device, observation, ttt_state=self._ttt_state,
                return_ttt_state=True, **sample_kwargs,
            )
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def reset(self) -> None:
        """Reset temporal fast weights at the start of a new robot episode."""
        self._ttt_state = None

    def validate_ttt_rollout_stride(self, num_open_loop_steps: int) -> None:
        """Reject a rollout cadence that differs from temporal TTT training."""
        if not self._ttt_enabled:
            return
        expected = int(getattr(self._model, "ttt_observation_stride", 1))
        if num_open_loop_steps != expected:
            raise ValueError(
                "TTT rollout cadence mismatch: the model was trained with "
                f"ttt_observation_stride={expected}, but evaluation would execute "
                f"num_open_loop_steps={num_open_loop_steps} before the next observation"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    def reset(self) -> None:
        """Forward an episode reset to a stateful TTT policy."""
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @override
    def infer(self, obs: dict, *, seed: int | None = None) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs) if seed is None else self._policy.infer(obs, seed=seed)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
