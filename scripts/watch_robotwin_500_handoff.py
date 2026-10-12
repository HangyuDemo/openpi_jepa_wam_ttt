"""Hand job 898547 off to an eight-GPU job after checkpoint 500 is finalized."""

import json
import os
from pathlib import Path
import subprocess
import time

from orbax.checkpoint import utils as orbax_utils


ROOT = Path("/lustre/fs1/home/ha865618/project/openpi_jepawam")
OLD_JOB_ID = 898547
CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints/pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only"
    / "robotwin_clean20_ttt_wrapper_explicit_gbs16"
)
NEW_SCRIPT = ROOT / "scripts/run_robotwin_ttt_8gpu_resume.sbatch"
HANDOFF_RECORD = CHECKPOINT_ROOT / "handoff_898547_step500.json"
POLL_SECONDS = 60


def checkpoint_is_complete(checkpoint: Path) -> bool:
    if not checkpoint.is_dir() or not orbax_utils.is_checkpoint_finalized(checkpoint):
        return False
    required = (
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "train_state/manifest.ocdbt",
        checkpoint / "params/manifest.ocdbt",
        checkpoint / "assets/local/robotwin_clean_20/norm_stats.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads(required[0].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(metadata.get("commit_timestamp_nsecs"))


def latest_finalized_checkpoint() -> Path | None:
    checkpoints = (
        path
        for path in CHECKPOINT_ROOT.iterdir()
        if path.name.isdigit() and int(path.name) >= 500 and checkpoint_is_complete(path)
    )
    return max(checkpoints, key=lambda path: int(path.name), default=None)


def old_job_state() -> str | None:
    result = subprocess.run(
        ["squeue", "--noheader", "--jobs", str(OLD_JOB_ID), "--format=%T"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def submit_successor() -> int:
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{OLD_JOB_ID}",
            "--time=2-00:00:00",
            str(NEW_SCRIPT),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return int(result.stdout.strip().split(";", 1)[0])


def write_handoff_record(new_job_id: int, checkpoint: Path) -> None:
    temp = HANDOFF_RECORD.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(
            {
                "old_job_id": OLD_JOB_ID,
                "new_job_id": new_job_id,
                "checkpoint": str(checkpoint),
                "new_script": str(NEW_SCRIPT),
                "global_batch_size": 16,
                "gpu_count": 8,
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(temp, HANDOFF_RECORD)


def main() -> None:
    print(f"Watching job {OLD_JOB_ID} for finalized checkpoint >=500 in {CHECKPOINT_ROOT}", flush=True)
    polls = 0
    while (checkpoint := latest_finalized_checkpoint()) is None:
        state = old_job_state()
        if state is None:
            raise RuntimeError(f"Job {OLD_JOB_ID} ended before checkpoint 500 was finalized")
        if polls % 10 == 0:
            print(f"Checkpoint not ready; old job state={state}", flush=True)
        polls += 1
        time.sleep(POLL_SECONDS)

    print(f"Finalized checkpoint is ready: {checkpoint}", flush=True)
    if HANDOFF_RECORD.exists():
        record = json.loads(HANDOFF_RECORD.read_text())
        new_job_id = int(record["new_job_id"])
        print(f"Successor already submitted as job {new_job_id}", flush=True)
    else:
        new_job_id = submit_successor()
        write_handoff_record(new_job_id, checkpoint)
        print(f"Submitted eight-GPU successor job {new_job_id}", flush=True)

    if old_job_state() is not None:
        subprocess.run(["scancel", str(OLD_JOB_ID)], check=True)
        print(f"Cancelled old job {OLD_JOB_ID}", flush=True)
    print(f"Handoff complete: {OLD_JOB_ID} -> {new_job_id}", flush=True)


if __name__ == "__main__":
    main()
