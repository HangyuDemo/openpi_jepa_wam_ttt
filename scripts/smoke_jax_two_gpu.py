"""Exercise the Slurm JAX initialization path across two GPU tasks."""

import os

import jax
import jax.numpy as jnp

from train import initialize_jax_distributed


def main() -> None:
    initialize_jax_distributed()
    rank = jax.process_index()
    local_count = jax.local_device_count()
    global_count = jax.device_count()
    print(
        f"rank={rank} local_rank={os.environ.get('SLURM_LOCALID')} "
        f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"step_gpus={os.environ.get('SLURM_STEP_GPUS')} "
        f"jax_local={local_count} jax_global={global_count}",
        flush=True,
    )
    if local_count != 1 or global_count != 2:
        raise RuntimeError(f"Expected one local and two global GPUs, got {local_count} and {global_count}")

    result = jax.pmap(lambda x: jax.lax.psum(x, "devices"), axis_name="devices")(
        jnp.ones((1,), dtype=jnp.float32)
    )
    result.block_until_ready()
    if float(result[0]) != 2.0:
        raise RuntimeError(f"Cross-process sum was {result[0]}, expected 2")
    print(f"rank={rank} distributed sum=2 PASS", flush=True)


if __name__ == "__main__":
    main()
