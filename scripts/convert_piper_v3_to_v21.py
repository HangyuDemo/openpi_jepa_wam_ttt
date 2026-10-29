"""Convert the downloaded Piper LeRobot v3 dataset to this repository's v2.1 layout.

The source videos are copied at episode keyframes without re-encoding. The
conversion is written to a temporary directory and published only after all
episode Parquet and video frame counts have been checked.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import av
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq

SOURCE_DEFAULT = Path("data/pick_can_place_plate_150_trimmed")
DEST_DEFAULT = Path("data/pick_can_place_plate_150_trimmed_v21")
VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")
STATS_KEYS = ("min", "max", "mean", "std", "count")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def count_video_packets(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(packet.dts is not None for packet in container.demux(stream))


def keyframe_indices(path: Path, fps: int) -> set[int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            round(float(packet.pts * stream.time_base) * fps)
            for packet in container.demux(stream)
            if packet.dts is not None and packet.is_keyframe
        }


def split_video(source: Path, output_pattern: Path, starts: list[int], lengths: list[int], fps: int) -> None:
    if count_video_packets(source) != sum(lengths):
        raise ValueError(f"Video frame count differs from episode metadata: {source}")
    missing = set(starts) - keyframe_indices(source, fps)
    if missing:
        raise ValueError(f"Cannot copy video at non-keyframe episode boundaries: {source}: {sorted(missing)[:8]}")

    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    segment_times = ",".join(f"{start / fps:.12f}" for start in starts[1:])
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_times",
            segment_times,
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ],
        check=True,
    )
    for episode_index, length in enumerate(lengths):
        output = Path(str(output_pattern) % episode_index)
        if not output.is_file() or count_video_packets(output) != length:
            raise ValueError(f"Wrong frame count for episode {episode_index}: {output}")


def convert(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info['codebase_version']}")
    if set(VIDEO_KEYS) != {key for key, feature in info["features"].items() if feature["dtype"] == "video"}:
        raise ValueError("Unexpected camera keys; review the conversion before proceeding")

    episode_rows = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    data = pq.read_table(source / "data/chunk-000/file-000.parquet")
    lengths = [int(row["length"]) for row in episode_rows]
    starts = [int(row["dataset_from_index"]) for row in episode_rows]
    if len(lengths) != info["total_episodes"] or sum(lengths) != info["total_frames"] or data.num_rows != sum(lengths):
        raise ValueError("Episode and frame totals do not agree")
    if starts != np.cumsum([0, *lengths[:-1]]).tolist():
        raise ValueError("Episode data offsets are not contiguous")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".tmp-", dir=destination.parent) as temp_dir:
        output = Path(temp_dir)
        episodes = []
        episode_stats = []
        for episode_index, (row, start, length) in enumerate(zip(episode_rows, starts, lengths, strict=True)):
            if int(row["episode_index"]) != episode_index or int(row["dataset_to_index"]) != start + length:
                raise ValueError(f"Episode {episode_index} has inconsistent index metadata")
            # The v3 Parquet schema embeds a Hugging Face `List` feature tag.
            # This repository's older `datasets` version cannot deserialize
            # that tag, but can infer the same columns from the Arrow schema.
            episode_data = data.slice(start, length).replace_schema_metadata(None)
            if set(episode_data.column("episode_index").to_pylist()) != {episode_index}:
                raise ValueError(f"Episode {episode_index} contains frames from another episode")
            if episode_data.column("frame_index").to_pylist() != list(range(length)):
                raise ValueError(f"Episode {episode_index} has non-sequential frame indices")

            parquet_path = output / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_data, parquet_path)
            episodes.append({"episode_index": episode_index, "tasks": row["tasks"], "length": length})

            stats = {}
            for key in ("action", "observation.state", *VIDEO_KEYS):
                stats[key] = {metric: row[f"stats/{key}/{metric}"] for metric in STATS_KEYS}
            episode_stats.append({"episode_index": episode_index, "stats": stats})

        tasks_table = pq.read_table(source / "meta/tasks.parquet").to_pylist()
        tasks = [
            {"task_index": int(row["task_index"]), "task": row.get("task", row.get("__index_level_0__"))}
            for row in tasks_table
        ]
        if len(tasks) != info["total_tasks"] or any(not row["task"] for row in tasks):
            raise ValueError("Task metadata is incomplete")
        write_jsonl(output / "meta/episodes.jsonl", episodes)
        write_jsonl(output / "meta/episodes_stats.jsonl", episode_stats)
        write_jsonl(output / "meta/tasks.jsonl", tasks)

        converted_info = copy.deepcopy(info)
        converted_info.update(
            codebase_version="v2.1",
            total_videos=len(episodes) * len(VIDEO_KEYS),
            total_chunks=(len(episodes) + 999) // 1000,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        for key in VIDEO_KEYS:
            height, width, channels = converted_info["features"][key]["shape"]
            converted_info["features"][key]["shape"] = [channels, height, width]
            converted_info["features"][key]["names"] = ["channels", "height", "width"]
            video_source = source / f"videos/{key}/chunk-000/file-000.mp4"
            pattern = output / f"videos/chunk-000/{key}/episode_%06d.mp4"
            split_video(video_source, pattern, starts, lengths, info["fps"])
            print(f"Validated {len(episodes)} {key} video segments", flush=True)

        write_json(output / "meta/info.json", converted_info)
        write_json(
            output / "source.json",
            {
                "repo_id": "jachansantiago/pick_can_place_plate_150_trimmed",
                "revision": "fcb248e888bba427f4c74593fae4c37d6c5de0d0",
            },
        )
        shutil.move(str(output), str(destination))
    print(f"Converted {len(episodes)} episodes and {sum(lengths)} frames to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--destination", type=Path, default=DEST_DEFAULT)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
