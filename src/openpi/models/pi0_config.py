import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Optional future-representation objective. Disabled by default so existing Pi0/Pi0.5 models are unchanged.
    use_vjepa_aux: bool = False
    # Keep the JEPA query architecture while omitting the future-target loss.
    vjepa_aux_loss_enabled: bool = True
    vjepa_num_queries: int = 64
    vjepa_query_grid_size: int = 8
    vjepa_target_grid_size: int = 24
    vjepa_target_dim: int = 1408
    vjepa_aux_weight: float = 0.1
    vjepa_aux_warmup_steps: int = 1000
    vjepa_action_attends_queries: bool = False
    vjepa_disable_geometric_augmentation: bool = True

    # Opt-in temporal TTT on the Pi0.5 action expert. Existing checkpoints keep
    # the original architecture when ttt_enabled is false.
    ttt_enabled: bool = False
    ttt_architecture: str = "wrapper"  # wrapper: four blocks; inline: every block
    ttt_memory_source: str = "explicit_jepa"  # or implicit_full
    ttt_layer_indices: tuple[int, ...] = ()
    ttt_num_register_tokens: int = 16
    ttt_num_future_tokens: int = 32
    ttt_memory_hidden_dim: int = 0  # 0 selects 2 * memory_dim
    ttt_context_length: int = 1
    # Physical-frame distance between consecutive observations in a temporal
    # training window. Rollout should execute the same number of actions before
    # requesting the next observation.
    ttt_observation_stride: int = 1
    ttt_tbptt_step_size: int = 8
    # A physical observation is one memory-write event, regardless of how many
    # denoising evaluations are used to produce its action chunk.
    ttt_update_first_denoise_only: bool = True
    # Number of camera views concatenated into current-observation V-JEPA K/V.
    ttt_current_jepa_num_views: int = 1

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.use_vjepa_aux:
            if not self.pi05:
                raise ValueError("V-JEPA auxiliary training is only supported for Pi0.5")
            if self.vjepa_num_queries != self.vjepa_query_grid_size**2:
                raise ValueError("vjepa_num_queries must equal vjepa_query_grid_size squared")
            if self.vjepa_target_grid_size < 1 or self.vjepa_target_dim < 1:
                raise ValueError("V-JEPA target grid size and dimension must be positive")
            if self.vjepa_aux_weight < 0 or self.vjepa_aux_warmup_steps < 0:
                raise ValueError("V-JEPA auxiliary weight and warmup steps must be non-negative")
        if self.ttt_enabled:
            if not self.pi05:
                raise ValueError("TTT is supported only for Pi0.5")
            if self.ttt_architecture not in ("inline", "wrapper"):
                raise ValueError("ttt_architecture must be inline or wrapper")
            if self.ttt_memory_source not in ("explicit_jepa", "implicit_full"):
                raise ValueError("ttt_memory_source must be explicit_jepa or implicit_full")
            if self.ttt_memory_source == "explicit_jepa" and not self.use_vjepa_aux:
                raise ValueError("Explicit TTT requires the V-JEPA feature dimensions")
            if self.ttt_num_register_tokens < 1 or self.ttt_num_future_tokens < 1:
                raise ValueError("TTT requires register and future tokens")
            if self.ttt_context_length < 1 or self.ttt_observation_stride < 1 or self.ttt_tbptt_step_size < 1:
                raise ValueError("TTT context, observation stride, and segment length must be positive")
            if self.ttt_current_jepa_num_views < 1:
                raise ValueError("TTT current-JEPA view count must be positive")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                vjepa_target=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.vjepa_target_grid_size**2, self.vjepa_target_dim], jnp.float16
                    )
                    if self.use_vjepa_aux and self.vjepa_aux_loss_enabled
                    else None
                ),
                vjepa_current=(
                    jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.ttt_current_jepa_num_views * self.vjepa_target_grid_size**2,
                            self.vjepa_target_dim,
                        ],
                        jnp.float16,
                    )
                    if self.ttt_enabled and self.ttt_memory_source == "explicit_jepa"
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
