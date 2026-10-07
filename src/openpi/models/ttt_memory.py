"""Per-layer fast-weight memory for the Pi0.5 action expert.

The memory state is explicit: callers keep it across robot observations and
reset it between episodes. K/V are either frozen current-observation V-JEPA
tokens or the full post-attention action-expert sequence. The future JEPA
teacher and WAM prediction are never read by the explicit-current route.
"""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class FastWeights(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    step: jax.Array


def empty_fast_weights(layers: int, batch: int, dim: int, hidden: int) -> FastWeights:
    """One independent zero-delta state per transformer block and environment."""
    return FastWeights(
        jnp.zeros((layers, batch, dim, hidden), jnp.float32),
        jnp.zeros((layers, batch, hidden), jnp.float32),
        jnp.zeros((layers, batch, hidden, dim), jnp.float32),
        jnp.zeros((layers, batch, dim), jnp.float32),
        jnp.zeros((layers, batch), jnp.int32),
    )


def _rope(x: jax.Array, step: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    inv_freq = 10_000.0 ** (-jnp.arange(half, dtype=jnp.float32) / half)
    phase = step.astype(jnp.float32)[:, None, None] * inv_freq[None, None, :]
    cos, sin = jnp.cos(phase), jnp.sin(phase)
    left, right = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate([left * cos - right * sin, left * sin + right * cos], axis=-1)


def _mlp(weights: tuple[jax.Array, ...], x: jax.Array) -> jax.Array:
    w1, b1, w2, b2 = weights
    return nn.gelu(x @ w1 + b1) @ w2 + b2


class TemporalTTT(nn.Module):
    query_dim: int
    memory_dim: int
    hidden_dim: int
    source: str

    @nn.compact
    def __call__(
        self,
        query_tokens: jax.Array,
        full_tokens: jax.Array,
        explicit_jepa: jax.Array | None,
        state: FastWeights,
        *,
        update: bool,
        valid_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, FastWeights]:
        if self.memory_dim % 2:
            raise ValueError("TTT memory dimension must be even for temporal RoPE")
        if self.source not in ("explicit_jepa", "implicit_full"):
            raise ValueError(f"Unknown TTT memory source: {self.source}")
        x = query_tokens.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        qkv = nn.Dense(3 * self.query_dim, name="to_qkv")
        q_hidden, _, _ = jnp.split(qkv(x), 3, axis=-1)
        if self.source == "explicit_jepa":
            if explicit_jepa is None:
                raise ValueError("Explicit TTT requires current-observation V-JEPA features")
            q = nn.Dense(self.memory_dim, name="query_to_memory")(q_hidden)
            v = explicit_jepa.astype(jnp.float32)
            k = v * jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + 1e-6)
        else:
            if self.memory_dim != self.query_dim:
                raise ValueError("Implicit TTT memory must use the action-expert width")
            full = full_tokens.astype(jnp.float32)
            full = full * jax.lax.rsqrt(jnp.mean(full * full, axis=-1, keepdims=True) + 1e-6)
            _, k, v = jnp.split(qkv(full), 3, axis=-1)
            q = q_hidden
        q, k = _rope(q, state.step), _rope(k, state.step)

        base = (
            self.param("memory_w1", nn.initializers.lecun_normal(), (self.memory_dim, self.hidden_dim)),
            self.param("memory_b1", nn.initializers.zeros, (self.hidden_dim,)),
            self.param("memory_w2", nn.initializers.lecun_normal(), (self.hidden_dim, self.memory_dim)),
            self.param("memory_b2", nn.initializers.zeros, (self.memory_dim,)),
        )
        delta = (state.w1, state.b1, state.w2, state.b2)
        weights = tuple(jnp.asarray(p, jnp.float32)[None] + d for p, d in zip(base, delta, strict=True))

        def loss_fn(params, one_k, one_v):
            return jnp.mean(jnp.square(_mlp(params, one_k) - one_v))

        grads = jax.vmap(jax.grad(loss_fn), in_axes=(0, 0, 0))(weights, k, v)
        rate = jax.nn.softplus(self.param("learnable_lr", lambda _: jnp.array(1e-2, jnp.float32)))
        forget = nn.sigmoid(
            nn.Dense(1, name="forget_out")(
                nn.silu(nn.Dense(2 * self.query_dim, name="forget_in")(jnp.mean(x, axis=1)))
            )
        )[:, 0]
        active = jnp.broadcast_to(jnp.asarray(update, dtype=jnp.bool_), (query_tokens.shape[0],))
        if valid_mask is not None:
            active = active & valid_mask
        forget = forget * active.astype(forget.dtype)
        new_delta = tuple(d - rate * g * forget.reshape((-1,) + (1,) * (g.ndim - 1)) for d, g in zip(delta, grads, strict=True))
        state = FastWeights(*new_delta, state.step + active.astype(jnp.int32))
        weights = tuple(jnp.asarray(p, jnp.float32)[None] + d for p, d in zip(base, new_delta, strict=True))

        retrieved = jax.vmap(_mlp)(weights, q)
        output = nn.Dense(self.query_dim, name="memory_to_output")(retrieved)
        gate = self.param("memory_out_layerscale", nn.initializers.normal(1e-4), (self.query_dim,))
        return (output * jnp.tanh(gate)).astype(query_tokens.dtype), state
