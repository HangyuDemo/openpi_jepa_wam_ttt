"""Frozen V-JEPA current-frame encoder for storage-light LIBERO TTT training."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class OnlineVJepaCurrent:
    """Encode only distinct frames in a collated temporal batch on this rank's GPU."""

    def __init__(self, checkpoint: str, source_root: str, batch_size: int = 4):
        if batch_size < 1:
            raise ValueError("V-JEPA online batch size must be positive")
        script = Path(__file__).resolve().parents[3] / "scripts/precompute_vjepa_pair_targets.py"
        spec = importlib.util.spec_from_file_location("_openpi_vjepa_precompute", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load V-JEPA preprocessing from {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._preprocess_image = module.preprocess_image
        self._batch_size = batch_size

        local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
        visible = torch.cuda.device_count()
        if visible == 0:
            raise RuntimeError("Online V-JEPA current encoding requires CUDA")
        assigned = os.environ.get("JAX_LOCAL_DEVICE_IDS")
        if assigned is not None:
            if "," in assigned:
                raise ValueError("Online V-JEPA requires exactly one JAX GPU per process")
            gpu_index = int(assigned)
        else:
            gpu_index = local_rank if local_rank < visible else 0
        if not 0 <= gpu_index < visible:
            raise RuntimeError(f"JAX GPU {gpu_index} is not among {visible} CUDA-visible GPUs")
        args = SimpleNamespace(
            checkpoint=Path(checkpoint),
            vjepa_source_root=Path(source_root),
            device=f"cuda:{gpu_index}",
            min_free_gib=16.0,
        )
        self._model, self._device = module.load_target_encoder(args)
        logging.info("Online V-JEPA current encoder: rank=%s gpu=%s", local_rank, self._device)

    def encode(self, images: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Return [B,T,576,1408] fp16 tokens, matching precompute's current path."""
        from PIL import Image

        if images.ndim != 5 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f"Expected uint8 [B,T,H,W,3] images; got {images.shape}, {images.dtype}")
        if indices.shape != images.shape[:2]:
            raise ValueError(f"Image/index shape mismatch: {images.shape[:2]} != {indices.shape}")
        flat_images = images.reshape(-1, *images.shape[2:])
        _, positions, inverse = np.unique(indices.reshape(-1), return_index=True, return_inverse=True)
        unique_images = flat_images[positions]
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(unique_images), self._batch_size):
                batch = np.stack([
                    self._preprocess_image(Image.fromarray(image))
                    for image in unique_images[start : start + self._batch_size]
                ])
                tensor = torch.from_numpy(batch[:, :, None, :, :]).to(
                    device=self._device, dtype=torch.bfloat16
                )
                tokens = self._model(tensor)
                if isinstance(tokens, list):
                    tokens = tokens[-1]
                if tokens.shape[1:] != (576, 1408):
                    raise ValueError(f"Unexpected V-JEPA current shape: {tuple(tokens.shape)}")
                encoded.append(tokens.to(device="cpu", dtype=torch.float16).numpy())
                del tensor, tokens
        unique_tokens = np.concatenate(encoded, axis=0)
        return np.ascontiguousarray(unique_tokens[inverse].reshape(*indices.shape, 576, 1408))
