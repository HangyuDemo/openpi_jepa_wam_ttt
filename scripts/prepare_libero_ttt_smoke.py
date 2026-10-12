#!/usr/bin/env python3
"""Make a tiny local LeRobot LIBERO view for a bounded TTT smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()

    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if output.exists():
        shutil.rmtree(output)

    source_info = read_json(source / "meta" / "info.json")
    source_episodes = read_jsonl(source / "meta" / "episodes.jsonl")
    source_stats = read_jsonl(source / "meta" / "episodes_stats.jsonl")
    selected = source_episodes[: args.episodes]
    selected_stats = source_stats[: args.episodes]
    if len(selected) != args.episodes:
        raise RuntimeError(f"Requested {args.episodes} episodes, found only {len(selected)}")

    (output / "data" / "chunk-000").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)
    for new_index, episode in enumerate(selected):
        old_index = int(episode["episode_index"])
        old_chunk = old_index // int(source_info.get("chunks_size", 1000))
        old_path = source / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_index:06d}.parquet"
        new_path = output / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet"
        if not old_path.exists():
            raise FileNotFoundError(old_path)
        new_path.symlink_to(old_path)

    episodes = []
    stats = []
    for new_index, episode in enumerate(selected):
        row = dict(episode)
        row["episode_index"] = new_index
        episodes.append(row)
    for new_index, row in enumerate(selected_stats):
        item = dict(row)
        item["episode_index"] = new_index
        stats.append(item)

    info = dict(source_info)
    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(int(row["length"]) for row in episodes)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(episodes)}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    write_json(output / "meta" / "info.json", info)
    write_jsonl(output / "meta" / "episodes.jsonl", episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", stats)
    shutil.copy2(source / "meta" / "tasks.jsonl", output / "meta" / "tasks.jsonl")
    print(json.dumps({"output_root": str(output), "episodes": len(episodes), "frames": info["total_frames"]}))


if __name__ == "__main__":
    main()
