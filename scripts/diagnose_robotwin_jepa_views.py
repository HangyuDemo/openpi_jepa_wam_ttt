#!/usr/bin/env python3
"""Infer which RoboTwin camera supervised a released Pi0.5 + V-JEPA checkpoint.

The diagnostic runs in two subprocesses so PyTorch/V-JEPA and JAX/OpenPI do not
hold GPU memory at the same time.  It compares the checkpoint's predicted
future representation with frozen V-JEPA targets made independently from each
RoboTwin camera.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "targets", "score"), default="all")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/lustre/fs1/home/ha865618/project/robotwin_lerobot/robotwin2_demo_clean_all_v21"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/lustre/fs1/home/ha865618/project/openpi_jepawam/pretrained/CokeAnd1ce_JEPA_WAM/"
            "checkpoints/openpi/pi05_robotwin_clean_20_vjepa_aux/"
            "pi05_robotwin_vjepa_delta50_no_geomaug_b256_fsdp4_gpu0123_seed42/14999"
        ),
    )
    parser.add_argument(
        "--vjepa-checkpoint",
        type=Path,
        default=Path(
            "/lustre/fs1/home/ha865618/project/openpi_jepawam/data/weights/vjepa2/vjepa2_1_vitg_384.pt"
        ),
    )
    parser.add_argument(
        "--vjepa-source-root",
        type=Path,
        default=Path("/lustre/fs1/home/ha865618/project/openpi_jepawam/third_party/vjepa2"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/lustre/fs1/home/ha865618/project/openpi_jepawam/artifacts/robotwin_jepa_view_test"),
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--future-offset", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def read_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def video_path(dataset_root: Path, episode_index: int, camera_key: str) -> Path:
    return dataset_root / "videos" / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4"


def read_video_frames(path: Path, frame_indices: tuple[int, ...]) -> list[np.ndarray]:
    """Decode selected AV1 frames with PyAV's software decoder.

    OpenCV on the cluster attempts an unavailable hardware AV1 path and can
    report missing sequence headers for otherwise valid LeRobot videos.
    """
    import av

    requested = set(frame_indices)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if len(decoded) == len(requested):
                break
    missing = requested.difference(decoded)
    if missing:
        raise RuntimeError(f"Could not decode frames {sorted(missing)} from {path}")
    return [decoded[index] for index in frame_indices]


def select_samples(dataset_root: Path, count: int, future_offset: int) -> list[dict]:
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(dataset_root / "meta" / "tasks.jsonl")}
    eligible = [row for row in episodes if int(row["length"]) > future_offset]
    if not eligible:
        raise ValueError(f"No episode is longer than future offset {future_offset}")
    chosen_positions = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    samples = []
    for position in chosen_positions:
        episode = eligible[int(position)]
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        frame_index = min(max(0, length // 3), length - future_offset - 1)
        parquet = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet, columns=["observation.state", "task_index"])
        state = np.asarray(table["observation.state"][frame_index].as_py(), dtype=np.float32)
        task_index = int(table["task_index"][frame_index].as_py())
        samples.append(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "future_index": frame_index + future_offset,
                "state": state,
                "prompt": tasks[task_index],
            }
        )
    return samples


def run_targets(args: argparse.Namespace) -> None:
    import torch

    import precompute_vjepa_pair_targets as precompute

    args.work_dir.mkdir(parents=True, exist_ok=True)
    samples = select_samples(args.dataset_root, args.samples, args.future_offset)
    load_args = argparse.Namespace(
        device=args.device,
        min_free_gib=16.0,
        vjepa_source_root=args.vjepa_source_root,
        checkpoint=args.vjepa_checkpoint,
    )
    model, device = precompute.load_target_encoder(load_args)
    current_images = []
    all_targets = []
    with torch.inference_mode():
        for sample_number, sample in enumerate(samples):
            sample_current = []
            sample_targets = []
            for camera_key in CAMERA_KEYS:
                path = video_path(args.dataset_root, sample["episode_index"], camera_key)
                current, future = read_video_frames(path, (sample["frame_index"], sample["future_index"]))
                current_preprocessed = precompute.preprocess_image(precompute.Image.fromarray(current))
                future_preprocessed = precompute.preprocess_image(precompute.Image.fromarray(future))
                pair = np.stack((current_preprocessed, future_preprocessed), axis=1)[None]
                pair_tensor = torch.from_numpy(pair).to(device=device, dtype=torch.bfloat16)
                output = model(pair_tensor)
                if isinstance(output, list):
                    output = output[-1]
                if tuple(output.shape) != (1, 576, 1408):
                    raise ValueError(f"Unexpected V-JEPA shape: {tuple(output.shape)}")
                sample_current.append(current)
                sample_targets.append(output[0].to(device="cpu", dtype=torch.float16).numpy())
                del pair_tensor, output
            current_images.append(sample_current)
            all_targets.append(sample_targets)
            print(
                f"V-JEPA targets {sample_number + 1}/{len(samples)}: "
                f"episode={sample['episode_index']} frame={sample['frame_index']}",
                flush=True,
            )

    np.savez_compressed(
        args.work_dir / "samples_and_targets.npz",
        current_images=np.asarray(current_images, dtype=np.uint8),
        targets=np.asarray(all_targets, dtype=np.float16),
        states=np.stack([sample["state"] for sample in samples]),
        prompts=np.asarray([sample["prompt"] for sample in samples]),
        episode_indices=np.asarray([sample["episode_index"] for sample in samples]),
        frame_indices=np.asarray([sample["frame_index"] for sample in samples]),
        future_indices=np.asarray([sample["future_index"] for sample in samples]),
    )
    print(f"Saved diagnostic targets to {args.work_dir / 'samples_and_targets.npz'}", flush=True)


def make_transformed_input(images, state, prompt, target, norm_stats, *, discrete_state_input: bool):
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.policies import robotwin_policy
    from openpi.training import config as training_config

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=1,
        discrete_state_input=discrete_state_input,
        use_vjepa_aux=True,
        vjepa_num_queries=64,
        vjepa_query_grid_size=8,
        vjepa_target_grid_size=24,
        vjepa_target_dim=1408,
        vjepa_action_attends_queries=False,
        vjepa_disable_geometric_augmentation=True,
    )
    raw = {
        "images": dict(zip(CAMERA_NAMES, images, strict=True)),
        "state": state,
        "prompt": prompt,
        "vjepa_target": target,
    }
    pipeline = transforms.compose(
        [
            robotwin_policy.RoboTwinInputs(),
            transforms.Normalize(norm_stats, use_quantiles=True),
            *training_config.ModelTransformFactory()(model_config).inputs,
        ]
    )
    return pipeline(raw), model_config


def run_score(args: argparse.Namespace) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    from openpi.models import model as openpi_model
    from openpi.shared import nnx_utils
    from openpi.shared import normalize

    packed = np.load(args.work_dir / "samples_and_targets.npz")
    checkpoint_asset = args.checkpoint_dir / "assets" / "local" / "robotwin_clean_20"
    norm_stats = normalize.load(checkpoint_asset)

    # Architecture is identical for both tokenization variants; create and restore once.
    _, model_config = make_transformed_input(
        packed["current_images"][0],
        packed["states"][0],
        str(packed["prompts"][0]),
        packed["targets"][0, 0],
        norm_stats,
        discrete_state_input=True,
    )
    print(f"JAX devices: {jax.devices()}", flush=True)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("JAX cannot see a GPU")
    print("Restoring released Pi0.5 + V-JEPA checkpoint...", flush=True)
    model = model_config.load(openpi_model.restore_params(args.checkpoint_dir / "params", dtype=jnp.bfloat16))
    score_fn = nnx_utils.module_jit(model.compute_loss_components, static_argnames=("train",))

    reports = {}
    for discrete_state_input in (True, False):
        rows = []
        for sample_index in range(len(packed["states"])):
            transformed = []
            for target_index in range(len(CAMERA_NAMES)):
                value, _ = make_transformed_input(
                    packed["current_images"][sample_index],
                    packed["states"][sample_index],
                    str(packed["prompts"][sample_index]),
                    packed["targets"][sample_index, target_index],
                    norm_stats,
                    discrete_state_input=discrete_state_input,
                )
                transformed.append(value)
            batched = jax.tree.map(lambda *values: jnp.asarray(np.stack(values)), *transformed)
            observation = openpi_model.Observation.from_dict(batched)
            dummy_actions = jnp.zeros((len(CAMERA_NAMES), 1, model_config.action_dim), dtype=jnp.float32)
            _, aux_loss = score_fn(jax.random.key(0), observation, dummy_actions, train=False)
            similarities = 1.0 - np.asarray(aux_loss)
            rows.append(similarities)
            print(
                f"state_in_prompt={discrete_state_input} sample={sample_index + 1}/{len(packed['states'])} "
                + " ".join(f"{name}={value:.5f}" for name, value in zip(CAMERA_NAMES, similarities, strict=True)),
                flush=True,
            )
        matrix = np.stack(rows)
        reports[str(discrete_state_input)] = {
            "mean_cosine_similarity": dict(zip(CAMERA_NAMES, matrix.mean(axis=0).tolist(), strict=True)),
            "std_cosine_similarity": dict(zip(CAMERA_NAMES, matrix.std(axis=0).tolist(), strict=True)),
            "per_sample": matrix.tolist(),
            "winner_counts": {
                name: int(np.sum(np.argmax(matrix, axis=1) == index)) for index, name in enumerate(CAMERA_NAMES)
            },
        }

    result = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(args.dataset_root),
        "future_offset": args.future_offset,
        "sample_count": int(len(packed["states"])),
        "camera_order": list(CAMERA_NAMES),
        "results": reports,
        "interpretation": (
            "The consistently highest camera is the most likely single-view JEPA supervision source. "
            "This is empirical evidence, not proof of an unavailable private training configuration."
        ),
    }
    output = args.work_dir / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved result to {output}", flush=True)


def run_all(args: argparse.Namespace) -> None:
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset-root",
        str(args.dataset_root),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--vjepa-checkpoint",
        str(args.vjepa_checkpoint),
        "--vjepa-source-root",
        str(args.vjepa_source_root),
        "--work-dir",
        str(args.work_dir),
        "--samples",
        str(args.samples),
        "--future-offset",
        str(args.future_offset),
        "--device",
        args.device,
    ]
    subprocess.run([*base, "--stage", "targets"], check=True)
    subprocess.run([*base, "--stage", "score"], check=True)


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.future_offset < 1:
        raise ValueError("samples and future-offset must be positive")
    if args.stage == "targets":
        run_targets(args)
    elif args.stage == "score":
        run_score(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
