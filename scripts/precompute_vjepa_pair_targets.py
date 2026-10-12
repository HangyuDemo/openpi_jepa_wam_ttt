#!/usr/bin/env python3
"""Precompute future-pair targets and current-observation V-JEPA tokens."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
TARGET_HEIGHT = 24
TARGET_WIDTH = 24
TARGET_DIM = 1408
TARGET_DTYPE = np.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vjepa-source-root",
        type=Path,
        required=True,
        help="Path to a clone of https://github.com/facebookresearch/vjepa2.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-key", default="image")
    parser.add_argument(
        "--future-offset",
        type=int,
        default=31,
        help="Pair I_t with I_min(t+offset,last). Offset 31 gives a 32-frame inclusive span.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--min-free-gib", type=float, default=16.0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_episodes(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    global_start = 0
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            record["global_start"] = global_start
            global_start += int(record["length"])
            episodes.append(record)
    return episodes


def expected_bytes(episodes: list[dict[str, Any]]) -> int:
    num_frames = sum(int(episode["length"]) for episode in episodes)
    bytes_per_view = num_frames * TARGET_HEIGHT * TARGET_WIDTH * TARGET_DIM * np.dtype(TARGET_DTYPE).itemsize
    # One future-pair target and one current-observation representation, both
    # from the same image view.
    return bytes_per_view * 2


def target_path(output_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return output_root / "targets" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.npy"


def current_path(output_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return output_root / "current" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.npy"


def metadata_path(output_root: Path, episode_index: int) -> Path:
    return target_path(output_root, episode_index).with_suffix(".json")


def input_path(dataset_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def video_path(dataset_root: Path, dataset_info: dict[str, Any], image_key: str, episode_index: int) -> Path:
    template = dataset_info.get("video_path")
    if not template:
        raise ValueError(f"Dataset does not define video_path for video feature {image_key!r}")
    return dataset_root / template.format(
        video_key=image_key,
        episode_chunk=episode_index // int(dataset_info.get("chunks_size", 1000)),
        episode_index=episode_index,
    )


def run_contract(
    args: argparse.Namespace, dataset_info: dict[str, Any], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    checkpoint_stat = args.checkpoint.stat()
    image_feature = dataset_info.get("features", {}).get(args.image_key)
    if image_feature is None:
        raise KeyError(f"Image key {args.image_key!r} is absent from dataset metadata")
    # Explicit TTT K/V must use exactly the same view as the future-pair
    # teacher target.
    current_image_keys = (args.image_key,)
    current_features = []
    for key in current_image_keys:
        feature = dataset_info.get("features", {}).get(key)
        if feature is None:
            raise KeyError(f"Current V-JEPA image key {key!r} is absent from dataset metadata")
        current_features.append(feature)
    return {
        "format_version": 2,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_total_episodes": len(episodes),
        "dataset_total_frames": sum(int(episode["length"]) for episode in episodes),
        "dataset_fps": dataset_info["fps"],
        "image_key": args.image_key,
        "image_storage": image_feature.get("dtype"),
        "current_image_keys": list(current_image_keys),
        "current_image_storage": [feature.get("dtype") for feature in current_features],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "encoder_key": "target_encoder",
        "future_offset": args.future_offset,
        "pair_span": args.future_offset + 1,
        "tail_policy": "clamp_future_to_last_frame",
        "input_shape": [3, 2, 384, 384],
        "current_input_shape": [3, 1, 384, 384],
        "target_shape": [TARGET_HEIGHT * TARGET_WIDTH, TARGET_DIM],
        "target_grid": [TARGET_HEIGHT, TARGET_WIDTH],
        "target_dtype": "float16",
        "current_shape": [len(current_image_keys) * TARGET_HEIGHT * TARGET_WIDTH, TARGET_DIM],
        "current_dtype": "float16",
        "current_view_order": "concatenate_patch_tokens",
        "current_normalized": False,
        "chunks_size": int(dataset_info.get("chunks_size", 1000)),
        "normalization_mean": IMAGENET_MEAN.tolist(),
        "normalization_std": IMAGENET_STD.tolist(),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def ensure_contract(output_root: Path, contract: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        # Multiple Slurm workers start together. The worker that won O_EXCL
        # writes the manifest in place, so briefly wait for its JSON to become
        # complete before validating the shared contract.
        existing = None
        for _ in range(200):
            try:
                existing = read_json(path)
                break
            except json.JSONDecodeError:
                time.sleep(0.05)
        if existing is None:
            raise RuntimeError(f"Timed out waiting for the shared manifest: {path}")
        if existing != contract:
            # Format 1 contained only future-pair targets. Preserve those files
            # and upgrade the manifest so interrupted/older preprocessing can
            # resume by generating just the missing current-observation arrays.
            legacy_matches = existing.get("format_version") == 1 and all(
                key == "format_version" or contract.get(key) == value for key, value in existing.items()
            )
            if not legacy_matches:
                raise RuntimeError(f"Output contract differs from existing manifest: {path}") from None
            write_json_atomic(path, contract)
        return

    with os.fdopen(fd, "w") as f:
        json.dump(contract, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def existing_target_is_valid(path: Path, episode_length: int) -> bool:
    return existing_array_is_valid(path, episode_length, (TARGET_HEIGHT * TARGET_WIDTH, TARGET_DIM))


def existing_array_is_valid(path: Path, episode_length: int, feature_shape: tuple[int, int]) -> bool:
    if not path.exists():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return value.shape == (episode_length, *feature_shape) and value.dtype == TARGET_DTYPE
    except (OSError, ValueError):
        return False


def decode_image(value: dict[str, Any], dataset_root: Path) -> Image.Image:
    image_bytes = value.get("bytes")
    if image_bytes is not None:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB").copy()
    image_path = value.get("path")
    if image_path is None:
        raise ValueError("Image record contains neither bytes nor path")
    path = Path(image_path)
    if not path.is_absolute():
        path = dataset_root / path
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def decode_video(path: Path, expected_frames: int) -> list[Image.Image]:
    import av

    images: list[Image.Image] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            images.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
            # Decode one extra frame so an incorrect episode length is not
            # silently accepted.
            if len(images) > expected_frames:
                break
    if len(images) != expected_frames:
        actual = f">{expected_frames}" if len(images) > expected_frames else str(len(images))
        raise ValueError(f"Video frame count mismatch for {path}: expected={expected_frames}, decoded={actual}")
    return images


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((384, 384), resample=Image.Resampling.BICUBIC)
    value = np.asarray(image, dtype=np.float32) / 255.0
    value = (value - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def import_vjepa_factory(source_root: Path):
    source_root = source_root.resolve()
    module_path = source_root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"V-JEPA 2.1 source was not found at {module_path}. Clone facebookresearch/vjepa2 and pass its root."
        )
    sys.path.insert(0, str(source_root))
    # In the upstream implementation, interpolated RoPE positions may promote bfloat16 q/k tensors to float32 while
    # v remains bfloat16. SDPA requires all three tensors to have the same dtype. Cast the rotated result back to the
    # input dtype; this is equivalent to the dtype guard used to generate the original JEPA-WAM targets.
    modules = importlib.import_module("app.vjepa_2_1.models.utils.modules")
    original_rotate = modules.rotate_queries_or_keys
    if not getattr(original_rotate, "openpi_dtype_safe", False):

        def dtype_safe_rotate_queries_or_keys(x, *rotate_args, **rotate_kwargs):
            return original_rotate(x, *rotate_args, **rotate_kwargs).to(dtype=x.dtype)

        dtype_safe_rotate_queries_or_keys.openpi_dtype_safe = True
        modules.rotate_queries_or_keys = dtype_safe_rotate_queries_or_keys
    module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
    return module.vit_giant_xformers


def load_target_encoder(args: argparse.Namespace):
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for ViT-G preprocessing, but device {args.device!r} is unavailable")
    device = torch.device(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"Refusing to load ViT-G on {device}: only {free_gib:.1f} GiB free of {total_gib:.1f} GiB; "
            f"minimum is {args.min_free_gib:.1f} GiB"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {device} does not support bfloat16")

    print(f"GPU preflight: device={device} free={free_gib:.1f}GiB total={total_gib:.1f}GiB", flush=True)
    factory = import_vjepa_factory(args.vjepa_source_root)
    model = factory(
        patch_size=16,
        img_size=(384, 384),
        num_frames=16,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "target_encoder" not in checkpoint:
        raise KeyError("Checkpoint does not contain target_encoder")
    state_dict = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in checkpoint["target_encoder"].items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    nontrivial_missing = [key for key in missing if "pos_embed" not in key]
    if nontrivial_missing or unexpected:
        raise RuntimeError(f"V-JEPA checkpoint mismatch: missing={nontrivial_missing}, unexpected={unexpected}")
    del checkpoint, state_dict

    model.requires_grad_(requires_grad=False)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"V-JEPA loaded: allocated={allocated:.1f}GiB reserved={reserved:.1f}GiB", flush=True)
    return model, device


def episode_metadata(
    contract: dict[str, Any], episode: dict[str, Any], path: Path, current: Path, clamped_count: int
) -> dict[str, Any]:
    return {
        "format_version": contract["format_version"],
        "episode_index": int(episode["episode_index"]),
        "episode_length": int(episode["length"]),
        "global_start": int(episode["global_start"]),
        "global_end_exclusive": int(episode["global_start"]) + int(episode["length"]),
        "future_offset": contract["future_offset"],
        "pair_span": contract["pair_span"],
        "tail_policy": contract["tail_policy"],
        "clamped_pair_count": clamped_count,
        "target_file": str(path),
        "target_shape": [int(episode["length"]), TARGET_HEIGHT * TARGET_WIDTH, TARGET_DIM],
        "target_dtype": contract["target_dtype"],
        "current_file": str(current),
        "current_shape": [int(episode["length"]), *contract["current_shape"]],
        "current_dtype": contract["current_dtype"],
    }


def process_episode(
    args: argparse.Namespace,
    contract: dict[str, Any],
    dataset_info: dict[str, Any],
    episode: dict[str, Any],
    model,
    device,
) -> None:
    import torch

    episode_index = int(episode["episode_index"])
    episode_length = int(episode["length"])
    output_path = target_path(args.output_root, episode_index)
    current_output_path = current_path(args.output_root, episode_index)
    meta_path = metadata_path(args.output_root, episode_index)
    clamped_count = min(args.future_offset, episode_length)
    meta = episode_metadata(contract, episode, output_path, current_output_path, clamped_count)

    target_valid = existing_target_is_valid(output_path, episode_length)
    current_shape = tuple(contract["current_shape"])
    current_valid = existing_array_is_valid(current_output_path, episode_length, current_shape)
    if target_valid and current_valid:
        if not meta_path.exists():
            write_json_atomic(meta_path, meta)
        print(f"skip episode={episode_index:06d} rows={episode_length}", flush=True)
        return
    if output_path.exists() and not target_valid:
        raise RuntimeError(f"Existing target is invalid; refusing to overwrite it: {output_path}")
    if current_output_path.exists() and not current_valid:
        raise RuntimeError(f"Existing current representation is invalid; refusing to overwrite it: {current_output_path}")

    parquet_path = input_path(args.dataset_root, episode_index)
    all_image_keys = tuple(dict.fromkeys((args.image_key, *contract["current_image_keys"])))
    storage = {
        key: dataset_info["features"][key].get("dtype")
        for key in all_image_keys
    }
    columns = ["frame_index", "episode_index", "index"]
    columns[:0] = [key for key in all_image_keys if storage[key] != "video"]
    table = pq.read_table(parquet_path, columns=columns)
    if table.num_rows != episode_length:
        raise ValueError(
            f"Episode {episode_index} length mismatch: metadata={episode_length}, parquet={table.num_rows}"
        )
    frame_indices = table["frame_index"].to_numpy()
    expected_frame_indices = np.arange(episode_length)
    if not np.array_equal(frame_indices, expected_frame_indices):
        raise ValueError(f"Episode {episode_index} frame_index is not contiguous")
    episode_indices = table["episode_index"].to_numpy()
    if not np.all(episode_indices == episode_index):
        raise ValueError(f"Episode {episode_index} parquet contains another episode index")
    global_indices = table["index"].to_numpy()
    expected_global = np.arange(int(episode["global_start"]), int(episode["global_start"]) + episode_length)
    if not np.array_equal(global_indices, expected_global):
        raise ValueError(f"Episode {episode_index} global index does not match metadata order")

    images_by_key = {}
    for key in all_image_keys:
        if storage[key] == "video":
            images_by_key[key] = decode_video(
                video_path(args.dataset_root, dataset_info, key, episode_index), episode_length
            )
        else:
            images_by_key[key] = [decode_image(value, args.dataset_root) for value in table[key].to_pylist()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-r{args.worker_rank}-p{os.getpid()}")
    current_temp_path = current_output_path.with_name(
        f".{current_output_path.name}.tmp-r{args.worker_rank}-p{os.getpid()}"
    )
    targets = None if target_valid else np.lib.format.open_memmap(
        temp_path, mode="w+", dtype=TARGET_DTYPE,
        shape=(episode_length, TARGET_HEIGHT * TARGET_WIDTH, TARGET_DIM),
    )
    current_targets = None if current_valid else np.lib.format.open_memmap(
        current_temp_path, mode="w+", dtype=TARGET_DTYPE,
        shape=(episode_length, *current_shape),
    )

    started = time.monotonic()
    first_batch = True
    try:
        with torch.inference_mode():
            for start in range(0, episode_length, args.batch_size):
                end = min(start + args.batch_size, episode_length)
                if first_batch:
                    torch.cuda.reset_peak_memory_stats(device)
                expected_shape = (end - start, TARGET_HEIGHT * TARGET_WIDTH, TARGET_DIM)
                if targets is not None:
                    current = np.stack(
                        [preprocess_image(images_by_key[args.image_key][index]) for index in range(start, end)]
                    )
                    future_indices = [min(index + args.future_offset, episode_length - 1) for index in range(start, end)]
                    future = np.stack(
                        [preprocess_image(images_by_key[args.image_key][index]) for index in future_indices]
                    )
                    pair_tensor = torch.from_numpy(np.stack((current, future), axis=2)).to(
                        device=device, dtype=torch.bfloat16, non_blocking=False
                    )
                    output = model(pair_tensor)
                    if isinstance(output, list):
                        output = output[-1]
                    if tuple(output.shape) != expected_shape:
                        raise ValueError(f"Unexpected V-JEPA output shape {tuple(output.shape)}, expected {expected_shape}")
                    targets[start:end] = output.to(device="cpu", dtype=torch.float16).numpy()
                    del pair_tensor, output

                if current_targets is not None:
                    current_views = np.stack(
                        [
                            np.stack(
                                [preprocess_image(images_by_key[key][index]) for index in range(start, end)]
                            )
                            for key in contract["current_image_keys"]
                        ],
                        axis=1,
                    ).reshape(-1, 3, 384, 384)
                    current_tensor = torch.from_numpy(current_views[:, :, None, :, :]).to(
                        device=device, dtype=torch.bfloat16, non_blocking=False
                    )
                    current_output = model(current_tensor)
                    if isinstance(current_output, list):
                        current_output = current_output[-1]
                    current_expected = (
                        (end - start) * len(contract["current_image_keys"]),
                        TARGET_HEIGHT * TARGET_WIDTH,
                        TARGET_DIM,
                    )
                    if tuple(current_output.shape) != current_expected:
                        raise ValueError(
                            f"Unexpected current V-JEPA output shape {tuple(current_output.shape)}, "
                            f"expected {current_expected}"
                        )
                    current_output = current_output.reshape(end - start, *current_shape)
                    current_targets[start:end] = current_output.to(device="cpu", dtype=torch.float16).numpy()
                    del current_tensor, current_output

                if first_batch:
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_allocated(device) / 1024**3
                    print(f"first-batch peak allocated={peak:.1f}GiB batch_size={args.batch_size}", flush=True)
                    first_batch = False
        if targets is not None:
            targets.flush()
            del targets
            os.replace(temp_path, output_path)
        if current_targets is not None:
            current_targets.flush()
            del current_targets
            os.replace(current_temp_path, current_output_path)
        write_json_atomic(meta_path, meta)
    except torch.OutOfMemoryError as error:
        if temp_path.exists():
            temp_path.unlink()
        if current_temp_path.exists():
            current_temp_path.unlink()
        raise RuntimeError(
            f"CUDA OOM at batch_size={args.batch_size}; the launcher intentionally does not retry with unsafe settings"
        ) from error
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        if current_temp_path.exists():
            current_temp_path.unlink()
        raise

    elapsed = time.monotonic() - started
    size_gib = (output_path.stat().st_size + current_output_path.stat().st_size) / 1024**3
    print(
        f"complete episode={episode_index:06d} rows={episode_length} clamped={clamped_count} "
        f"size={size_gib:.3f}GiB elapsed={elapsed:.1f}s",
        flush=True,
    )


def print_plan(contract: dict[str, Any], episodes: list[dict[str, Any]], output_root: Path) -> None:
    size_gib = expected_bytes(episodes) / 1024**3
    clamped = sum(min(int(contract["future_offset"]), int(episode["length"])) for episode in episodes)
    print(
        json.dumps(
            {**contract, "output_root": str(output_root), "expected_total_gib": size_gib, "clamped_pairs": clamped},
            indent=2,
        )
    )


def print_status(output_root: Path, episodes: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    complete = 0
    complete_frames = 0
    complete_bytes = 0
    for episode in episodes:
        path = target_path(output_root, int(episode["episode_index"]))
        current = current_path(output_root, int(episode["episode_index"]))
        target_valid = existing_target_is_valid(path, int(episode["length"]))
        current_valid = existing_array_is_valid(current, int(episode["length"]), tuple(contract["current_shape"]))
        if target_valid and current_valid:
            complete += 1
            complete_frames += int(episode["length"])
            complete_bytes += path.stat().st_size + current.stat().st_size
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "complete_episodes": complete,
                "total_episodes": len(episodes),
                "complete_frames": complete_frames,
                "total_frames": sum(int(episode["length"]) for episode in episodes),
                "complete_gib": complete_bytes / 1024**3,
                "expected_gib": expected_bytes(episodes) / 1024**3,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.future_offset < 0:
        raise ValueError("future-offset must be non-negative")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.world_size < 1 or not 0 <= args.worker_rank < args.world_size:
        raise ValueError("worker-rank must be in [0, world-size)")

    dataset_info = read_json(args.dataset_root / "meta" / "info.json")
    episodes = read_episodes(args.dataset_root / "meta" / "episodes.jsonl")
    contract = run_contract(args, dataset_info, episodes)
    if args.plan_only:
        print_plan(contract, episodes, args.output_root)
        return
    if args.status_only:
        print_status(args.output_root, episodes, contract)
        return

    ensure_contract(args.output_root, contract)
    assigned = [episode for episode in episodes if int(episode["episode_index"]) % args.world_size == args.worker_rank]
    if args.max_episodes is not None:
        assigned = assigned[: args.max_episodes]
    print(
        f"worker start rank={args.worker_rank}/{args.world_size} episodes={len(assigned)} "
        f"future_offset={args.future_offset} pair_span={args.future_offset + 1} "
        "current_view_matches_target=True "
        f"batch={args.batch_size}",
        flush=True,
    )
    model, device = load_target_encoder(args)
    for episode in assigned:
        process_episode(args, contract, dataset_info, episode, model, device)
    print(f"worker complete rank={args.worker_rank}/{args.world_size}", flush=True)


if __name__ == "__main__":
    main()
