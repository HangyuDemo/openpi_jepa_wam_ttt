"""Compute Pi0.5 Piper state/action statistics without decoding the videos."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pick_can_place_plate_150_trimmed_v21"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/pi05_piper/piper_pick_can_place_plate_150_v21"))
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.action_horizon < 1:
        raise ValueError("Action horizon must be positive")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("Validation fraction must be in [0, 1)")

    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("Expected the converted LeRobot v2.1 dataset")
    n_episodes = info["total_episodes"]
    n_validation = (
        max(1, min(n_episodes - 1, round(n_episodes * args.validation_fraction))) if args.validation_fraction else 0
    )
    held_out = set(np.random.default_rng(args.validation_seed).permutation(n_episodes)[:n_validation].tolist())
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    frames = 0
    for episode_index in range(n_episodes):
        if episode_index in held_out:
            continue
        path = args.dataset_root / info["data_path"].format(
            episode_chunk=episode_index // info["chunks_size"], episode_index=episode_index
        )
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states.shape[1:] != (14,) or actions.shape[1:] != (7,):
            raise ValueError(f"Unexpected state/action dimensions in {path}: {states.shape}, {actions.shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite state/action values in {path}")

        # LeRobot's delta_timestamps query clamps action chunks at the episode end.
        indices = np.minimum(
            np.arange(len(actions))[:, None] + np.arange(args.action_horizon)[None, :], len(actions) - 1
        )
        action_chunks = actions[indices].copy()
        action_chunks[..., :6] -= states[:, None, :6]
        state_stats.update(states)
        action_stats.update(action_chunks)
        frames += len(states)

    episodes = [json.loads(line) for line in (args.dataset_root / "meta/episodes.jsonl").read_text().splitlines()]
    expected_frames = sum(episode["length"] for index, episode in enumerate(episodes) if index not in held_out)
    if frames != expected_frames:
        raise ValueError(f"Expected {expected_frames} training frames, found {frames}")
    normalize.save(args.output_dir, {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})
    (args.output_dir / "norm_stats_provenance.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root.resolve()),
                "training_episodes": n_episodes - n_validation,
                "training_frames": frames,
                "validation_episodes": sorted(held_out),
                "validation_seed": args.validation_seed,
                "validation_fraction": args.validation_fraction,
                "action_horizon": args.action_horizon,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote state/action statistics for {frames} frames to {args.output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
