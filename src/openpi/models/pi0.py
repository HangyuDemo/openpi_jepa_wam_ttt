import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import ttt_memory
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_vjepa_aux = config.use_vjepa_aux
        self.vjepa_aux_loss_enabled = config.vjepa_aux_loss_enabled
        self.vjepa_num_queries = config.vjepa_num_queries
        self.vjepa_query_grid_size = config.vjepa_query_grid_size
        self.vjepa_target_grid_size = config.vjepa_target_grid_size
        self.vjepa_target_dim = config.vjepa_target_dim
        self.vjepa_aux_weight = config.vjepa_aux_weight
        self.vjepa_action_attends_queries = config.vjepa_action_attends_queries
        self.vjepa_disable_geometric_augmentation = config.vjepa_disable_geometric_augmentation
        self.ttt_enabled = config.ttt_enabled
        self.ttt_memory_source = config.ttt_memory_source
        self.ttt_context_length = config.ttt_context_length
        self.ttt_observation_stride = config.ttt_observation_stride
        self.ttt_tbptt_step_size = config.ttt_tbptt_step_size
        self.ttt_update_first_denoise_only = config.ttt_update_first_denoise_only
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.ttt_layers = action_expert_config.depth
        self.ttt_memory_dim = config.vjepa_target_dim if config.ttt_memory_source == "explicit_jepa" else action_expert_config.width
        self.ttt_hidden_dim = config.ttt_memory_hidden_dim or (2 * self.ttt_memory_dim)
        self.ttt_query_prefix = 1 + config.ttt_num_register_tokens
        if config.ttt_layer_indices:
            ttt_indices = config.ttt_layer_indices
        elif config.ttt_architecture == "inline":
            ttt_indices = tuple(range(action_expert_config.depth))
        else:
            ttt_indices = tuple(i for i in (3, 7, 11, 15) if i < action_expert_config.depth)
        if config.ttt_enabled and (
            not ttt_indices or len(set(ttt_indices)) != len(ttt_indices)
            or any(i < 0 or i >= action_expert_config.depth for i in ttt_indices)
        ):
            raise ValueError("TTT layer indices must select action-expert blocks")
        self.ttt_state_slots = len(ttt_indices) if config.ttt_enabled else 0
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                ttt_source=config.ttt_memory_source if config.ttt_enabled else None,
                ttt_layer_indices=ttt_indices if config.ttt_enabled else (),
                ttt_memory_dim=self.ttt_memory_dim if config.ttt_enabled else 0,
                ttt_hidden_dim=self.ttt_hidden_dim if config.ttt_enabled else 0,
                ttt_query_prefix=self.ttt_query_prefix if config.ttt_enabled else 0,
                ttt_action_count=config.action_horizon if config.ttt_enabled else 0,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            if config.ttt_enabled:
                self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                self.ttt_register_tokens = nnx.Param(
                    jax.random.normal(rngs.params(), (config.ttt_num_register_tokens, action_expert_config.width)) * 0.02
                )
                self.ttt_future_tokens = nnx.Param(
                    jax.random.normal(rngs.params(), (config.ttt_num_future_tokens, action_expert_config.width)) * 0.02
                )
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        if self.use_vjepa_aux:
            query_init = jax.random.normal(
                rngs.params(), (self.vjepa_num_queries, paligemma_config.width), dtype=jnp.float32
            )
            self.vjepa_query_tokens = nnx.Param(query_init * 0.02)
            self.vjepa_alignment_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_in = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_out = nnx.Linear(paligemma_config.width, self.vjepa_target_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        if self.use_vjepa_aux:
            query_tokens = jnp.broadcast_to(
                self.vjepa_query_tokens.value.astype(tokens[0].dtype),
                (tokens[0].shape[0], self.vjepa_num_queries, self.vjepa_query_tokens.value.shape[-1]),
            )
            tokens.append(query_tokens)
            input_mask.append(jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_))
            # Queries read image/language and one another. Earlier prefix tokens cannot read the queries.
            ar_mask += [True] + ([False] * (self.vjepa_num_queries - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05 or self.ttt_enabled:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        if self.ttt_enabled:
            batch = obs.state.shape[0]
            for value in (self.ttt_register_tokens.value, self.ttt_future_tokens.value):
                extra = jnp.broadcast_to(value.astype(tokens[0].dtype), (batch, *value.shape))
                tokens.append(extra)
                input_mask.append(jnp.ones(extra.shape[:2], dtype=jnp.bool_))
                ar_mask += [False] * extra.shape[1]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def empty_ttt_state(self, batch_size: int) -> ttt_memory.FastWeights:
        if not self.ttt_enabled:
            raise ValueError("TTT is disabled")
        return ttt_memory.empty_fast_weights(
            self.ttt_state_slots, batch_size, self.ttt_memory_dim, self.ttt_hidden_dim
        )

    def _ttt_prefix(self, observation: _model.Observation, state: ttt_memory.FastWeights):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache, _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=mask, positions=positions,
            ttt_state=state, return_ttt_state=True,
        )
        prediction = None
        if self.use_vjepa_aux and self.vjepa_aux_loss_enabled:
            prediction = self.predict_vjepa_target(prefix_out[:, -self.vjepa_num_queries:])
        return prefix_tokens, prefix_mask, kv_cache, prediction

    def _ttt_suffix(
        self,
        observation: _model.Observation,
        x_t: jax.Array,
        time: jax.Array,
        prefix_mask: jax.Array,
        kv_cache,
        current_jepa: jax.Array | None,
        state: ttt_memory.FastWeights,
        *,
        update: bool,
        update_mask: jax.Array | None = None,
    ):
        batch = x_t.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_mask_self = make_attn_mask(suffix_mask, suffix_ar_mask)
        suffix_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
            suffix_to_prefix = suffix_to_prefix.at[:, :, -self.vjepa_num_queries:].set(False)
        mask = jnp.concatenate((suffix_to_prefix, suffix_mask_self), axis=-1)
        positions = jnp.sum(prefix_mask, axis=1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1
        if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
            positions = positions - self.vjepa_num_queries
        (_, suffix_out), _, next_state = self.PaliGemma.llm(
            [None, suffix_tokens], mask=mask, positions=positions, kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond], ttt_state=state,
            ttt_memory_tokens=current_jepa if self.ttt_memory_source == "explicit_jepa" else None,
            ttt_valid_mask=update_mask if update_mask is not None else observation.time_valid_mask,
            ttt_update=update, return_ttt_state=True,
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon:]), next_state

    def _ttt_loss_frame(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions,
        state: ttt_memory.FastWeights, *, train: bool,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train,
            geometric_augmentation=not (self.use_vjepa_aux and self.vjepa_disable_geometric_augmentation),
        )
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prefix_tokens, prefix_mask, cache, prediction = self._ttt_prefix(observation, state)
        del prefix_tokens
        current_jepa = None
        if self.ttt_memory_source == "explicit_jepa":
            if observation.vjepa_current is None:
                raise ValueError(
                    "Explicit TTT requires observation.vjepa_current from the frozen V-JEPA encoder"
                )
            # Match LeVJEPA_WAM_TTT's jepa_current route: normalize each frozen
            # patch token before using all configured camera views as K/V.
            current_jepa = jax.lax.stop_gradient(observation.vjepa_current.astype(jnp.float32))
            current_jepa /= jnp.maximum(jnp.linalg.norm(current_jepa, axis=-1, keepdims=True), 1e-6)
        velocity, next_state = self._ttt_suffix(
            observation, x_t, time, prefix_mask, cache, current_jepa, state, update=True
        )
        flow_loss = jnp.mean(jnp.square(velocity - (noise - actions)), axis=-1)
        aux_loss = None
        if self.use_vjepa_aux and self.vjepa_aux_loss_enabled:
            if observation.vjepa_target is None:
                raise ValueError("V-JEPA auxiliary training requires observation.vjepa_target")
            target = jax.lax.stop_gradient(observation.vjepa_target.astype(jnp.float32))
            predicted = prediction.astype(jnp.float32)
            if predicted.shape != target.shape:
                raise ValueError(f"V-JEPA target shape mismatch: {predicted.shape} != {target.shape}")
            predicted = predicted / jnp.maximum(jnp.linalg.norm(predicted, axis=-1, keepdims=True), 1e-6)
            target = target / jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
            aux_loss = jnp.mean(1.0 - jnp.sum(predicted * target, axis=-1), axis=-1)
        return flow_loss, aux_loss, next_state

    def compute_loss_components(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b"] | None]:
        if self.ttt_enabled:
            state = self.empty_ttt_state(actions.shape[0])
            if actions.ndim == 3:
                flow, aux, _ = self._ttt_loss_frame(rng, observation, actions, state, train=train)
                return flow, aux
            if actions.ndim != 4 or actions.shape[1] != self.ttt_context_length:
                raise ValueError(f"TTT expects [B, {self.ttt_context_length}, horizon, dim] actions")
            time_steps = actions.shape[1]
            obs_time = jax.tree.map(
                lambda x: None if x is None else jnp.swapaxes(x, 0, 1), observation
            )
            actions_time = jnp.swapaxes(actions, 0, 1)
            # True TBPTT: build independent segments and stop gradients through
            # the fast-weight carry between segments. Merely applying
            # stop_gradient inside one long scan still lets reverse-mode AD
            # retain residuals for the entire context. Checkpointing each
            # segment makes those residuals local to `segment_size` timesteps.
            segment_size = min(self.ttt_tbptt_step_size, time_steps)
            num_segments = (time_steps + segment_size - 1) // segment_size
            padded_steps = num_segments * segment_size
            pad_steps = padded_steps - time_steps

            def pad_time(x):
                if x is None or pad_steps == 0:
                    return x
                padding = ((0, pad_steps),) + ((0, 0),) * (x.ndim - 1)
                return jnp.pad(x, padding, mode="edge")

            obs_time = jax.tree.map(pad_time, obs_time)
            if obs_time.time_valid_mask is not None and pad_steps:
                obs_time = dataclasses.replace(
                    obs_time,
                    time_valid_mask=obs_time.time_valid_mask.at[time_steps:].set(False),
                )
            actions_time = pad_time(actions_time)
            keys = jax.random.split(rng, padded_steps)

            def segment_time(x):
                if x is None:
                    return None
                return x.reshape((num_segments, segment_size, *x.shape[1:]))

            obs_segments = jax.tree.map(segment_time, obs_time)
            action_segments = segment_time(actions_time)
            key_segments = segment_time(keys)

            def frame_step(carry, inputs):
                frame, action, frame_rng = inputs
                flow, aux, next_state = self._ttt_loss_frame(frame_rng, frame, action, carry, train=train)
                return next_state, (flow, aux) if aux is not None else (flow,)

            # Rematerialize one temporal frame at a time during the backward
            # pass. A checkpoint around the whole segment only prevents
            # residuals from escaping the segment; differentiating its inner
            # scan can still materialize all `segment_size` frames together.
            # Per-frame rematerialization preserves the full gradient path
            # within each TBPTT segment while keeping large Pi0/TTT
            # activations local to one frame.
            checkpointed_frame_step = jax.checkpoint(
                frame_step,
                prevent_cse=False,
                policy=jax.checkpoint_policies.nothing_saveable,
            )

            def segment_step(carry, inputs):
                # Carry the fast weights forward as values, but truncate their
                # gradient history at every segment boundary.
                carry = jax.tree.map(jax.lax.stop_gradient, carry)
                return jax.lax.scan(checkpointed_frame_step, carry, inputs)

            checkpointed_segment_step = jax.checkpoint(
                segment_step,
                prevent_cse=False,
                policy=jax.checkpoint_policies.nothing_saveable,
            )
            _, values = jax.lax.scan(
                checkpointed_segment_step,
                state,
                (obs_segments, action_segments, key_segments),
            )

            flow_time = values[0].reshape((padded_steps, *values[0].shape[2:]))[:time_steps]
            flow = jnp.swapaxes(flow_time, 0, 1)
            if self.use_vjepa_aux and self.vjepa_aux_loss_enabled:
                aux_time = values[1].reshape((padded_steps, *values[1].shape[2:]))[:time_steps]
                aux = jnp.swapaxes(aux_time, 0, 1)
            else:
                aux = None
            return flow, aux
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            geometric_augmentation=not (self.use_vjepa_aux and self.vjepa_disable_geometric_augmentation),
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
            prefix_length = prefix_tokens.shape[1]
            query_start = prefix_length - self.vjepa_num_queries
            attn_mask = attn_mask.at[:, prefix_length:, query_start:prefix_length].set(False)
            positions = positions.at[:, prefix_length:].add(-self.vjepa_num_queries)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if not self.use_vjepa_aux or not self.vjepa_aux_loss_enabled:
            return flow_loss, None
        if observation.vjepa_target is None:
            raise ValueError("V-JEPA auxiliary training requires observation.vjepa_target")

        query_out = prefix_out[:, -self.vjepa_num_queries :]
        predicted_target = self.predict_vjepa_target(query_out)
        target = jax.lax.stop_gradient(observation.vjepa_target.astype(jnp.float32))
        if predicted_target.shape != target.shape:
            raise ValueError(f"V-JEPA prediction/target shape mismatch: {predicted_target.shape} != {target.shape}")
        predicted_target = predicted_target.astype(jnp.float32)
        predicted_target /= jnp.maximum(jnp.linalg.norm(predicted_target, axis=-1, keepdims=True), 1e-6)
        target /= jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
        aux_loss = jnp.mean(1.0 - jnp.sum(predicted_target * target, axis=-1), axis=-1)
        return flow_loss, aux_loss

    def predict_vjepa_target(self, query_out: at.Float[at.Array, "b q emb"]) -> at.Float[at.Array, "b p d"]:
        value = self.vjepa_alignment_norm(query_out)
        value = nnx.gelu(self.vjepa_alignment_in(value))
        value = self.vjepa_alignment_out(value)
        value = value.reshape(
            value.shape[0], self.vjepa_query_grid_size, self.vjepa_query_grid_size, self.vjepa_target_dim
        )
        value = jax.image.resize(
            value,
            (value.shape[0], self.vjepa_target_grid_size, self.vjepa_target_grid_size, self.vjepa_target_dim),
            method="linear",
        )
        return value.reshape(value.shape[0], self.vjepa_target_grid_size**2, self.vjepa_target_dim)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        flow_loss, aux_loss = self.compute_loss_components(rng, observation, actions, train=train)
        if aux_loss is None:
            return flow_loss

        return flow_loss + self.vjepa_aux_weight * aux_loss[..., None]

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        ttt_state: ttt_memory.FastWeights | None = None,
        return_ttt_state: bool = False,
        ttt_update: bool = True,
    ) -> _model.Actions:
        if self.ttt_enabled:
            observation = _model.preprocess_observation(None, observation, train=False)
            batch_size = observation.state.shape[0]
            if noise is None:
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            if ttt_state is None:
                ttt_state = self.empty_ttt_state(batch_size)
            _, prefix_mask, cache, _ = self._ttt_prefix(observation, ttt_state)
            current_jepa = None
            if self.ttt_memory_source == "explicit_jepa":
                if observation.vjepa_current is None:
                    raise ValueError(
                        "Explicit TTT inference requires current V-JEPA tokens in observation.vjepa_current"
                    )
                current_jepa = observation.vjepa_current.astype(jnp.float32)
                current_jepa /= jnp.maximum(jnp.linalg.norm(current_jepa, axis=-1, keepdims=True), 1e-6)

            observation_step = ttt_state.step

            def step(index, carry):
                x_t, state, committed_step = carry
                time = jnp.full((batch_size,), 1.0 - index / num_steps)
                update_mask = (
                    jnp.full((batch_size,), index == 0, dtype=jnp.bool_)
                    if self.ttt_update_first_denoise_only
                    else None
                )
                velocity, state = self._ttt_suffix(
                    observation,
                    x_t,
                    time,
                    prefix_mask,
                    cache,
                    current_jepa,
                    state,
                    update=ttt_update,
                    update_mask=update_mask,
                )
                if self.ttt_update_first_denoise_only:
                    # Keep the newly written weights, but query the same physical
                    # observation at its starting RoPE position on every denoise step.
                    committed_step = jnp.where(index == 0, state.step, committed_step)
                    state = state._replace(step=observation_step)
                return x_t - velocity / num_steps, state, committed_step

            actions, next_state, committed_step = jax.lax.fori_loop(
                0, num_steps, step, (noise, ttt_state, observation_step)
            )
            if self.ttt_update_first_denoise_only:
                next_state = next_state._replace(step=committed_step)
            return (actions, next_state) if return_ttt_state else actions
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
                prefix_attn_mask = prefix_attn_mask.at[:, :, -self.vjepa_num_queries :].set(False)
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
                positions = positions - self.vjepa_num_queries

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
