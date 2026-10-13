from collections import OrderedDict
from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

    def online_vjepa_encoder(self):
        """Return the rank-local frozen encoder, if one is attached."""
        raise NotImplementedError


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class TemporalWindowDataset(Dataset):
    """Left-pad strided observations from one LeRobot episode for TTT."""

    def __init__(self, dataset: Dataset, episode_starts: np.ndarray, context_length: int, observation_stride: int = 1):
        if context_length < 1 or observation_stride < 1 or len(episode_starts) != len(dataset):
            raise ValueError("Invalid temporal window metadata")
        self._dataset = dataset
        self._starts = episode_starts
        self._context = context_length
        self._stride = observation_stride

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        start = int(self._starts[index])
        candidates = index + np.arange(-self._context + 1, 1) * self._stride
        valid = candidates >= start
        indices = np.maximum(candidates, start)
        samples = [self._dataset[int(i)] for i in indices]
        result = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *samples)
        result["time_valid_mask"] = valid
        return result


def _episode_starts(dataset: Dataset) -> np.ndarray:
    source = dataset
    while hasattr(source, "_dataset"):
        source = source._dataset
    if not hasattr(source, "episode_data_index"):
        raise ValueError("Temporal TTT windows require LeRobot episode boundaries")
    result = np.empty(len(source), dtype=np.int64)
    bounds = source.episode_data_index
    for start, end in zip(bounds["from"], bounds["to"], strict=True):
        start, end = int(start), int(end)
        result[start:end] = start
    return result


def _split_episode_indices(
    dataset: Dataset, split: Literal["train", "validation"], fraction: float, seed: int,
    *, limit: int | None = None,
) -> np.ndarray:
    """Choose whole LeRobot episodes before taking any temporal windows."""
    source = dataset
    while hasattr(source, "_dataset"):
        source = source._dataset
    if not hasattr(source, "episode_data_index"):
        raise ValueError("Episode-level validation requires LeRobot episode boundaries")
    bounds = source.episode_data_index
    n_episodes = len(bounds["from"])
    if n_episodes < 2:
        raise ValueError("Episode-level validation requires at least two episodes")
    n_validation = max(1, min(n_episodes - 1, round(n_episodes * fraction)))
    held_out = set(np.random.default_rng(seed).permutation(n_episodes)[:n_validation].tolist())
    indices = np.concatenate([
        np.arange(int(start), int(end), dtype=np.int64)
        for episode, (start, end) in enumerate(zip(bounds["from"], bounds["to"], strict=True))
        if (episode in held_out) == (split == "validation")
    ])
    if limit is not None:
        if len(indices) < limit:
            raise ValueError(f"Validation needs {limit} held-out frames, found {len(indices)}")
        indices = np.sort(np.random.default_rng(seed + 1).choice(indices, size=limit, replace=False))
    return indices


class VJepaTargetDataset(Dataset):
    """Adds one precomputed V-JEPA target row to each LeRobot sample."""

    def __init__(
        self,
        dataset: Dataset,
        target_root: str | Path,
        *,
        expected_shape: tuple[int, int],
        expected_future_offset: int | None = None,
        expected_image_key: str | None = None,
        expected_num_frames: int | None = None,
        require_target: bool = True,
        require_current: bool = False,
        expected_current_image_keys: Sequence[str] = (),
        mmap_cache_size: int = 16,
    ):
        if mmap_cache_size < 1:
            raise ValueError("V-JEPA mmap cache size must be positive")
        self._dataset = dataset
        self._target_root = Path(target_root)
        self._expected_shape = expected_shape
        self._mmap_cache_size = mmap_cache_size
        self._require_target = require_target
        self._require_current = require_current
        self._expected_current_shape = (len(expected_current_image_keys) * expected_shape[0], expected_shape[1])
        self._cache: OrderedDict[tuple[str, int], np.memmap] = OrderedDict()

        manifest_path = self._target_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"V-JEPA target manifest not found: {manifest_path}")
        with manifest_path.open() as f:
            manifest = json.load(f)
        if tuple(manifest.get("target_shape", ())) != expected_shape:
            raise ValueError(
                f"V-JEPA target shape mismatch: expected {expected_shape}, manifest has {manifest.get('target_shape')}"
            )
        if manifest.get("target_dtype") != "float16":
            raise ValueError(f"Expected float16 V-JEPA targets, got {manifest.get('target_dtype')}")
        if expected_future_offset is not None and manifest.get("future_offset") != expected_future_offset:
            raise ValueError(
                "V-JEPA future offset mismatch: "
                f"expected {expected_future_offset}, manifest has {manifest.get('future_offset')}"
            )
        if expected_image_key is not None and manifest.get("image_key") != expected_image_key:
            raise ValueError(
                f"V-JEPA image key mismatch: expected {expected_image_key!r}, manifest has {manifest.get('image_key')!r}"
            )
        if expected_num_frames is not None and manifest.get("dataset_total_frames") != expected_num_frames:
            raise ValueError(
                "V-JEPA frame count mismatch: "
                f"expected {expected_num_frames}, manifest has {manifest.get('dataset_total_frames')}"
            )
        if require_current:
            if not expected_current_image_keys:
                raise ValueError("Explicit TTT requires at least one current V-JEPA image key")
            if tuple(manifest.get("current_shape", ())) != self._expected_current_shape:
                raise ValueError(
                    "Current V-JEPA shape mismatch: "
                    f"expected {self._expected_current_shape}, manifest has {manifest.get('current_shape')}"
                )
            if manifest.get("current_dtype") != "float16":
                raise ValueError(f"Expected float16 current V-JEPA tokens, got {manifest.get('current_dtype')}")
            if tuple(manifest.get("current_image_keys", ())) != tuple(expected_current_image_keys):
                raise ValueError(
                    "Current V-JEPA image-key mismatch: "
                    f"expected {tuple(expected_current_image_keys)!r}, "
                    f"manifest has {tuple(manifest.get('current_image_keys', ()))!r}"
                )
        self._chunks_size = int(manifest.get("chunks_size", 1000))
        if self._chunks_size < 1:
            raise ValueError("V-JEPA target manifest chunks_size must be positive")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = dict(self._dataset[index])
        episode_index = int(np.asarray(sample["episode_index"]).item())
        frame_index = int(np.asarray(sample["frame_index"]).item())
        if self._require_target:
            target = self._get_episode("targets", episode_index, self._expected_shape)
            if not 0 <= frame_index < target.shape[0]:
                raise IndexError(
                    f"Frame {frame_index} is outside V-JEPA target episode {episode_index} with {target.shape[0]} rows"
                )
            # The copy decouples the row from an mmap that may be evicted before collation.
            sample["vjepa_target"] = np.array(target[frame_index], copy=True)
        if self._require_current:
            current = self._get_episode("current", episode_index, self._expected_current_shape)
            if not 0 <= frame_index < current.shape[0]:
                raise IndexError(
                    f"Frame {frame_index} is outside current V-JEPA episode {episode_index} with {current.shape[0]} rows"
                )
            sample["vjepa_current"] = np.array(current[frame_index], copy=True)
        return sample

    def _get_episode(self, kind: str, episode_index: int, expected_shape: tuple[int, int]) -> np.memmap:
        cache_key = (kind, episode_index)
        if cache_key in self._cache:
            target = self._cache.pop(cache_key)
            self._cache[cache_key] = target
            return target

        path = (
            self._target_root
            / kind
            / f"chunk-{episode_index // self._chunks_size:03d}"
            / f"episode_{episode_index:06d}.npy"
        )
        target = np.load(path, mmap_mode="r", allow_pickle=False)
        if target.ndim != 3 or tuple(target.shape[1:]) != expected_shape or target.dtype != np.float16:
            self._close_mmap(target)
            raise ValueError(f"Invalid V-JEPA array: {path}, shape={target.shape}, dtype={target.dtype}")
        self._cache[cache_key] = target
        while len(self._cache) > self._mmap_cache_size:
            _, evicted = self._cache.popitem(last=False)
            self._close_mmap(evicted)
        return target

    @staticmethod
    def _close_mmap(value: np.ndarray) -> None:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if jnp.issubdtype(spec.dtype, jnp.floating):
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0).astype(spec.dtype)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_config.dataset_root)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=data_config.dataset_root,
        video_backend=data_config.video_backend,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    use_vjepa_aux = bool(getattr(model_config, "use_vjepa_aux", False))
    require_target = use_vjepa_aux and bool(getattr(model_config, "vjepa_aux_loss_enabled", True))
    require_current = bool(
        getattr(model_config, "ttt_enabled", False)
        and getattr(model_config, "ttt_memory_source", None) == "explicit_jepa"
    )
    if require_target and data_config.vjepa_target_root is None:
        raise ValueError("V-JEPA auxiliary training requires data.vjepa_target_root")
    if require_current and data_config.vjepa_target_root is None and not data_config.vjepa_online_current:
        raise ValueError("Explicit TTT requires precomputed current tokens or online V-JEPA encoding")
    if data_config.vjepa_online_current and not require_current:
        raise ValueError("Online V-JEPA current encoding requires explicit_jepa TTT")
    if data_config.vjepa_target_root is not None:
        if not use_vjepa_aux:
            raise ValueError("V-JEPA targets were configured for a model with use_vjepa_aux=False")
        if data_config.vjepa_future_offset is None or data_config.vjepa_image_key is None:
            raise ValueError("V-JEPA targets require an expected future offset and image key")
        dataset = VJepaTargetDataset(
            dataset,
            data_config.vjepa_target_root,
            expected_shape=(model_config.vjepa_target_grid_size**2, model_config.vjepa_target_dim),
            expected_future_offset=data_config.vjepa_future_offset,
            expected_image_key=data_config.vjepa_image_key,
            expected_num_frames=len(dataset),
            require_target=require_target,
            require_current=require_current and not data_config.vjepa_online_current,
            expected_current_image_keys=data_config.vjepa_current_image_keys,
            mmap_cache_size=data_config.vjepa_mmap_cache_size,
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    split: Literal["train", "validation"] = "train",
    online_vjepa_encoder=None,
    num_workers: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers if num_workers is None else num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        split=split,
        validation_fraction=config.validation_fraction,
        validation_seed=config.validation_seed,
        online_vjepa_encoder=online_vjepa_encoder,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    split: Literal["train", "validation"] = "train",
    validation_fraction: float = 0.0,
    validation_seed: int = 2026,
    online_vjepa_encoder=None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    starts = _episode_starts(dataset) if getattr(model_config, "ttt_enabled", False) and model_config.ttt_context_length > 1 else None
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    if starts is not None:
        dataset = TemporalWindowDataset(
            dataset,
            starts,
            model_config.ttt_context_length,
            model_config.ttt_observation_stride,
        )
    if validation_fraction:
        limit = None if split == "train" else batch_size * num_batches if num_batches is not None else None
        indices = _split_episode_indices(dataset, split, validation_fraction, validation_seed, limit=limit)
        if split == "validation" and framework == "jax":
            indices = indices[jax.process_index() :: jax.process_count()]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
        logging.info("%s split: %d temporal windows on process %d", split, len(dataset), jax.process_index())

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        online_vjepa_config=data_config if data_config.vjepa_online_current else None,
        online_vjepa_encoder=online_vjepa_encoder,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        online_vjepa_config: _config.DataConfig | None = None,
        online_vjepa_encoder=None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._online_vjepa = online_vjepa_encoder
        if online_vjepa_config is not None and self._online_vjepa is None:
            if framework != "jax":
                raise ValueError("Online V-JEPA current encoding currently requires the JAX loader")
            if not online_vjepa_config.vjepa_online_checkpoint or not online_vjepa_config.vjepa_online_source_root:
                raise ValueError("Online V-JEPA requires checkpoint and source-root paths")
            from openpi.training.online_vjepa import OnlineVJepaCurrent

            self._online_vjepa = OnlineVJepaCurrent(
                online_vjepa_config.vjepa_online_checkpoint,
                online_vjepa_config.vjepa_online_source_root,
                online_vjepa_config.vjepa_online_batch_size,
            )

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        # Each JAX process owns a disjoint local batch stream. The resulting
        # global batch is assembled by make_array_from_process_local_data.
        generator.manual_seed(seed + 1_000_003 * jax.process_index())
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                if self._online_vjepa is not None:
                    images = batch.pop("vjepa_online_image")
                    indices = batch.pop("vjepa_online_index")
                    batch["vjepa_current"] = self._online_vjepa.encode(images, indices)
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def online_vjepa_encoder(self):
        return getattr(self._data_loader, "_online_vjepa", None)

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
