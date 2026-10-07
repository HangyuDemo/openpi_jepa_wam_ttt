# ruff: noqa: UP006, UP007, UP017, UP035, UP038

import collections
import contextlib
import dataclasses
import datetime
import fcntl
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
_EPISODE_STATUSES = frozenset({"success", "failure", "error"})
_RESULT_SCHEMA_VERSION = 3
_EVALUATION_PROTOCOL_VERSION = "openpi-libero-eval-v5"
_EPISODE_SEED_SCHEME = "sha256-uint32"
_EPISODE_SEED_SCHEME_VERSION = 1
_POLICY_SEED_SCHEME = "sha256-uint32-episode-inference"
_POLICY_SEED_SCHEME_VERSION = 1
_LIBERO_PLUS_TASK_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
_STANDARD_LIBERO_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
_LIBERO_PLUS_CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    run_id: Optional[str] = None  # Stable served-checkpoint ID; required for durable results/resume
    resize_size: int = 224
    replan_steps: int = 5
    policy_reconnect_attempts: int = 2  # Recreate the websocket and retry the current episode after policy errors
    policy_reconnect_backoff_seconds: float = 1.0
    max_consecutive_policy_errors: int = 3  # Stop the run instead of filling the journal with cascading errors
    policy_connect_timeout_seconds: float = 30.0  # Bound each websocket construction attempt
    policy_connect_retry_interval_seconds: float = 1.0
    policy_inference_timeout_seconds: float = 300.0  # Bound each inference response wait
    policy_use_proxy: bool = False  # Local robot policy servers should bypass ambient HTTP(S) proxies

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    benchmark_mode: str = "standard"  # Either standard or plus
    benchmark_revision: Optional[str] = None  # Exact benchmark source commit/tag; required for durable results
    classification_path: Optional[str] = None  # Optional LIBERO-Plus task_classification.json override
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_start: int = 0  # Inclusive global task index
    task_end: Optional[int] = None  # Exclusive global task index; defaults to the end of the suite
    num_task_shards: int = 1  # Split selected task IDs into this many stable, strided shards
    task_shard_id: int = 0  # Zero-based shard to evaluate

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_video: bool = True  # Disable for large-scale evaluation to avoid video I/O and frame buffering
    results_path: Optional[str] = None  # Optional JSONL file with one durable record per episode
    resume: bool = True  # Skip episodes already present in results_path
    retry_errors: bool = False  # Retry structured error records when resuming
    merge_results_paths: Optional[List[str]] = None  # Merge disjoint shard journals into results_path and exit
    summarize_results_paths: Optional[List[str]] = None  # Summarize one or more compatible journals and exit
    overwrite_results: bool = False  # Allow merge mode to replace its output file

    seed: int = 7  # Random Seed (for reproducibility)


@dataclasses.dataclass
class _EpisodeOutcome:
    success: bool
    replay_images: List[np.ndarray]
    num_steps: int
    error: Optional[Dict[str, str]] = None


@dataclasses.dataclass
class _TaskInfo:
    task_id: int
    task: Any
    name: str
    description: str
    category: Optional[str] = None
    difficulty_level: Optional[int] = None


@dataclasses.dataclass
class _EvaluationCounts:
    total_episodes: int = 0
    total_successes: int = 0

    @classmethod
    def from_records(
        cls,
        records: Dict[Tuple[str, int, int], Dict[str, Any]],
        expected_keys: Set[Tuple[str, int, int]],
    ) -> "_EvaluationCounts":
        selected_records = [record for key, record in records.items() if key in expected_keys]
        return cls(
            total_episodes=len(selected_records),
            total_successes=sum(record["status"] == "success" for record in selected_records),
        )

    def replace(self, old_record: Optional[Dict[str, Any]], new_record: Dict[str, Any]) -> None:
        if old_record is not None:
            self.total_episodes -= 1
            self.total_successes -= int(old_record["status"] == "success")
        self.total_episodes += 1
        self.total_successes += int(new_record["status"] == "success")


class _EpisodeJournal:
    """Owns one run/shard journal and keeps its latest episode records in memory."""

    def __init__(
        self,
        path: Optional[str],
        *,
        run_header: Dict[str, Any],
        resume: bool,
        retry_errors: bool,
    ):
        self.path = pathlib.Path(path).expanduser() if path else None
        self.run_header = _validate_run_header(run_header)
        self.retry_errors = retry_errors
        self.records: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._lock_file = None

        if self.path is None:
            return
        _require_durable_benchmark_revision(self.run_header, self.path)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._acquire_lifetime_lock()
        try:
            if resume:
                if self.path.exists() and self.path.stat().st_size:
                    existing_header, self.records = _load_journal(self.path)
                    _require_matching_run_header(self.run_header, existing_header, self.path)
                else:
                    _write_new_journal(self.path, self.run_header)
            else:
                _reset_jsonl(self.path)
                _append_jsonl_record(self.path, self.run_header)
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _acquire_lifetime_lock(self) -> None:
        assert self.path is not None
        lock_path = self.path.with_name(self.path.name + ".lock")
        self._lock_file = lock_path.open("a+b")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError(
                f"Results journal is already owned by another process: {self.path}. "
                "Use a distinct shard ID/path or wait for that process to finish."
            ) from exc

    def close(self) -> None:
        if self._lock_file is None:
            return
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_file.close()
            self._lock_file = None

    def should_skip(
        self,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        *,
        task_name: Optional[str] = None,
        task_description: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> bool:
        record = self.records.get((task_suite_name, task_id, episode_idx))
        if record is None or (self.retry_errors and record["status"] == "error"):
            return False

        return (
            record.get("task_name") == task_name
            and record.get("task_description") == task_description
            and record.get("seed") == seed
        )

    def append(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = _validate_episode_record(record, self.run_header)
        old_record = self.records.get(key)
        if self.path is not None:
            _append_jsonl_record(self.path, record)
        self.records[key] = record
        return old_record


def eval_libero(args: Args) -> None:
    if args.merge_results_paths and args.summarize_results_paths:
        raise ValueError("merge_results_paths and summarize_results_paths are mutually exclusive")
    if args.merge_results_paths:
        if not args.results_path:
            raise ValueError("results_path is required as the merge output path")
        merged_path = pathlib.Path(args.results_path).expanduser()
        _merge_episode_journals(
            [pathlib.Path(path).expanduser() for path in args.merge_results_paths],
            merged_path,
            overwrite=args.overwrite_results,
        )
        _log_aggregate_summary(_summarize_result_journals([merged_path]))
        return
    if args.summarize_results_paths:
        _log_aggregate_summary(
            _summarize_result_journals([pathlib.Path(path).expanduser() for path in args.summarize_results_paths])
        )
        return

    benchmark_mode = args.benchmark_mode.strip().lower()
    _validate_eval_args(args, benchmark_mode)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite {args.task_suite_name!r}; available suites: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    _validate_benchmark_protocol(args, benchmark_mode, num_tasks_in_suite)
    task_infos = _build_task_infos(args, benchmark_mode, task_suite, num_tasks_in_suite)
    task_ids = _select_task_ids(
        num_tasks_in_suite,
        args.task_start,
        args.task_end,
        args.num_task_shards,
        args.task_shard_id,
    )
    max_steps = _get_max_steps(args.task_suite_name)
    run_header = _make_run_header(
        args,
        benchmark_mode,
        task_infos,
        num_tasks_in_suite,
        max_steps,
    )
    results_path = _resolve_shard_results_path(args.results_path, args.num_task_shards, args.task_shard_id)

    with _EpisodeJournal(
        str(results_path) if results_path is not None else None,
        run_header=run_header,
        resume=args.resume,
        retry_errors=args.retry_errors,
    ) as journal:
        _evaluate_tasks(
            args,
            benchmark_mode,
            task_suite,
            task_infos,
            task_ids,
            num_tasks_in_suite,
            max_steps,
            results_path,
            journal,
        )


def _evaluate_tasks(
    args: Args,
    benchmark_mode: str,
    task_suite,
    task_infos: Sequence[_TaskInfo],
    task_ids: Sequence[int],
    num_tasks_in_suite: int,
    max_steps: int,
    results_path: Optional[pathlib.Path],
    journal: _EpisodeJournal,
) -> None:
    expected_keys = {
        (args.task_suite_name, task_id, episode_idx)
        for task_id in task_ids
        for episode_idx in range(args.num_trials_per_task)
    }
    counts = _EvaluationCounts.from_records(journal.records, expected_keys)

    logging.info("Task suite: %s", args.task_suite_name)
    logging.info("Benchmark mode: %s", benchmark_mode)
    logging.info("Run fingerprint: %s", journal.run_header["run_fingerprint"])
    logging.info(
        "Selected %d/%d tasks in [%d, %d) for shard %d/%d",
        len(task_ids),
        num_tasks_in_suite,
        args.task_start,
        num_tasks_in_suite if args.task_end is None else args.task_end,
        args.task_shard_id,
        args.num_task_shards,
    )
    if results_path is not None:
        logging.info("Episode results: %s", results_path)
        logging.info("Loaded %d matching episode records", counts.total_episodes)
    if not task_ids:
        logging.warning("The selected range and shard contain no tasks")

    client = None
    consecutive_policy_errors = 0
    abort_reason = None

    try:
        for task_id in tqdm.tqdm(task_ids):
            task_info = task_infos[task_id]
            pending_episode_indices = [
                episode_idx
                for episode_idx in range(args.num_trials_per_task)
                if not journal.should_skip(
                    args.task_suite_name,
                    task_id,
                    episode_idx,
                    task_name=task_info.name,
                    task_description=task_info.description,
                    seed=args.seed,
                )
            ]
            if not pending_episode_indices:
                logging.info("Task %d is already complete; skipping environment creation", task_id)
                continue

            try:
                initial_states = task_suite.get_task_init_states(task_id)
            except Exception as exc:
                logging.exception("Failed to load initial states for task %d", task_id)
                _record_task_setup_errors(
                    args,
                    journal,
                    counts,
                    task_info,
                    max_steps,
                    stage="initial_states",
                    exc=exc,
                    episode_indices=pending_episode_indices,
                )
                continue

            env = None
            try:
                env, task_description = _get_libero_env(task_info.task, LIBERO_ENV_RESOLUTION, args.seed)
            except Exception as exc:
                logging.exception("Failed to create environment for task %d", task_id)
                _record_task_setup_errors(
                    args,
                    journal,
                    counts,
                    task_info,
                    max_steps,
                    stage="environment_setup",
                    exc=exc,
                    episode_indices=pending_episode_indices,
                )
                continue

            try:
                if str(task_description) != task_info.description:
                    raise RuntimeError(
                        f"Task prompt changed after run fingerprinting: {task_info.description!r} != {task_description!r}"
                    )
                for episode_idx in tqdm.tqdm(pending_episode_indices):
                    logging.info("\nTask: %s", task_description)
                    logging.info("Starting task %d episode %d...", task_id, episode_idx)
                    started_at = time.monotonic()
                    episode_seed = _derive_episode_seed(
                        args.seed,
                        args.task_suite_name,
                        task_id,
                        episode_idx,
                    )

                    try:
                        initial_state = initial_states[episode_idx]
                    except Exception as exc:
                        logging.exception("Failed to select initial state for task %d episode %d", task_id, episode_idx)
                        outcome = _EpisodeOutcome(
                            success=False,
                            replay_images=[],
                            num_steps=0,
                            error=_exception_record("initial_state_lookup", exc),
                        )
                    else:
                        outcome, client = _run_episode_with_policy_retries(
                            args,
                            env,
                            client,
                            task_description,
                            initial_state,
                            max_steps,
                            episode_seed,
                        )

                    status = _outcome_status(outcome)
                    video_path, video_error = _save_replay_video(
                        args,
                        outcome.replay_images,
                        task_id,
                        episode_idx,
                        task_description,
                        run_fingerprint=journal.run_header["run_fingerprint"],
                        status=status,
                    )
                    record = _make_episode_record(
                        args,
                        journal.run_header["run_fingerprint"],
                        task_info,
                        episode_idx,
                        max_steps,
                        outcome,
                        episode_seed=episode_seed,
                        duration_seconds=time.monotonic() - started_at,
                        video_path=video_path,
                        video_error=video_error,
                    )
                    old_record = journal.append(record)
                    counts.replace(old_record, record)

                    if _is_policy_error(outcome):
                        consecutive_policy_errors += 1
                    else:
                        consecutive_policy_errors = 0
                    if consecutive_policy_errors >= args.max_consecutive_policy_errors:
                        abort_reason = (
                            f"policy circuit breaker opened after {consecutive_policy_errors} consecutive errors"
                        )

                    logging.info("Episode status: %s", status)
                    if outcome.error is not None:
                        logging.error("Episode error: %s", outcome.error)
                    logging.info("# episodes recorded so far: %d", counts.total_episodes)
                    success_rate = counts.total_successes / counts.total_episodes if counts.total_episodes else 0.0
                    logging.info("# successes: %d (%.1f%%)", counts.total_successes, success_rate * 100)
                    if abort_reason is not None:
                        logging.error("Stopping evaluation: %s", abort_reason)
                        break
            finally:
                try:
                    env.close()
                except Exception:
                    logging.exception("Failed to close environment for task %d", task_id)

            _log_task_summary(
                journal.records,
                args.task_suite_name,
                task_id,
                args.num_trials_per_task,
                journal.run_header["run_config"]["task_group_metadata"],
            )
            if abort_reason is not None:
                break
    finally:
        _close_policy_client(client)

    summary = _summarize_records(
        journal.records,
        args.task_suite_name,
        task_ids,
        args.num_trials_per_task,
        journal.run_header["run_config"]["task_group_metadata"],
    )
    _annotate_summary(
        summary,
        args,
        benchmark_mode,
        num_tasks_in_suite,
        abort_reason=abort_reason,
    )
    _log_final_summary(summary)


def _run_episode_with_policy_retries(
    args: Args,
    env,
    client,
    task_description: str,
    initial_state,
    max_steps: int,
    episode_seed: int,
) -> Tuple[_EpisodeOutcome, Any]:
    attempts = args.policy_reconnect_attempts + 1
    outcome = None
    for attempt in range(attempts):
        if client is None:
            try:
                client = _websocket_client_policy.WebsocketClientPolicy(
                    args.host,
                    args.port,
                    connect_timeout=args.policy_connect_timeout_seconds,
                    retry_interval=args.policy_connect_retry_interval_seconds,
                    inference_timeout=args.policy_inference_timeout_seconds,
                    use_proxy=args.policy_use_proxy,
                )
            except Exception as exc:
                logging.exception("Failed to initialize the policy client")
                outcome = _EpisodeOutcome(
                    success=False,
                    replay_images=[],
                    num_steps=0,
                    error=_exception_record("policy_client_setup", exc),
                )
            else:
                outcome = _run_episode(
                    args,
                    env,
                    client,
                    task_description,
                    initial_state,
                    max_steps,
                    episode_seed,
                )
        else:
            outcome = _run_episode(
                args,
                env,
                client,
                task_description,
                initial_state,
                max_steps,
                episode_seed,
            )

        assert outcome is not None
        if not _is_policy_error(outcome):
            return outcome, client

        _close_policy_client(client)
        client = None
        if attempt + 1 < attempts:
            logging.warning("Retrying episode after policy error (%d/%d)", attempt + 1, attempts - 1)
            if args.policy_reconnect_backoff_seconds:
                time.sleep(args.policy_reconnect_backoff_seconds)

    return outcome, client


def _is_policy_error(outcome: _EpisodeOutcome) -> bool:
    return outcome.error is not None and outcome.error.get("stage") in {
        "policy_client_setup",
        "policy_inference",
    }


def _close_policy_client(client) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if not callable(close):
        close = getattr(getattr(client, "_ws", None), "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logging.exception("Failed to close policy client")


def _validate_eval_args(args: Args, benchmark_mode: str) -> None:
    if benchmark_mode not in {"standard", "plus"}:
        raise ValueError(f"benchmark_mode must be 'standard' or 'plus', got {args.benchmark_mode!r}")
    if args.num_trials_per_task <= 0:
        raise ValueError(f"num_trials_per_task must be positive, got {args.num_trials_per_task}")
    if args.replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {args.replan_steps}")
    if args.num_steps_wait < 0:
        raise ValueError(f"num_steps_wait must be non-negative, got {args.num_steps_wait}")
    if args.policy_reconnect_attempts < 0:
        raise ValueError("policy_reconnect_attempts must be non-negative")
    if args.policy_reconnect_backoff_seconds < 0:
        raise ValueError("policy_reconnect_backoff_seconds must be non-negative")
    if args.max_consecutive_policy_errors <= 0:
        raise ValueError("max_consecutive_policy_errors must be positive")
    if args.policy_connect_timeout_seconds <= 0:
        raise ValueError("policy_connect_timeout_seconds must be positive")
    if args.policy_connect_retry_interval_seconds < 0:
        raise ValueError("policy_connect_retry_interval_seconds must be non-negative")
    if args.policy_inference_timeout_seconds <= 0:
        raise ValueError("policy_inference_timeout_seconds must be positive")
    if not isinstance(args.policy_use_proxy, bool):
        raise ValueError("policy_use_proxy must be a boolean")
    if args.results_path and not (args.run_id and args.run_id.strip()):
        raise ValueError("run_id is required when results_path is set so resume cannot mix different checkpoints")
    if args.results_path and not (args.benchmark_revision and args.benchmark_revision.strip()):
        raise ValueError(
            "benchmark_revision is required when results_path is set so durable results identify benchmark source"
        )


def _validate_benchmark_protocol(args: Args, benchmark_mode: str, num_tasks: int) -> None:
    expected_counts = _LIBERO_PLUS_TASK_COUNTS if benchmark_mode == "plus" else _STANDARD_LIBERO_TASK_COUNTS
    if args.task_suite_name not in expected_counts:
        raise ValueError(f"Suite {args.task_suite_name!r} is not part of the {benchmark_mode} LIBERO protocol")
    expected_tasks = expected_counts[args.task_suite_name]
    if num_tasks != expected_tasks:
        raise ValueError(
            f"{benchmark_mode} {args.task_suite_name} must contain {expected_tasks} tasks, "
            f"but the imported LIBERO package exposes {num_tasks}. Check PYTHONPATH and LIBERO_CONFIG_PATH."
        )
    if benchmark_mode == "plus" and args.num_trials_per_task != 1:
        raise ValueError(
            "LIBERO-Plus requires exactly one trial/initial state per task; "
            f"got num_trials_per_task={args.num_trials_per_task}"
        )
    if benchmark_mode == "standard" and args.num_trials_per_task != 50:
        logging.warning(
            "Standard LIBERO official evaluation uses 50 trials per task; current run uses %d",
            args.num_trials_per_task,
        )


def _build_task_infos(
    args: Args,
    benchmark_mode: str,
    task_suite,
    num_tasks: int,
) -> List[_TaskInfo]:
    classification = _load_plus_classification(args, num_tasks) if benchmark_mode == "plus" else None
    task_infos = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        name = getattr(task, "name", None)
        description = getattr(task, "language", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"Task {task_id} has no stable string name")
        if not isinstance(description, str) or not description:
            raise ValueError(f"Task {task_id} has no stable language prompt")

        category = None
        difficulty_level = None
        if classification is not None:
            metadata = classification[task_id]
            if metadata["name"] != name:
                raise ValueError(
                    f"LIBERO-Plus classification/task order mismatch at task {task_id}: "
                    f"{metadata['name']!r} != {name!r}"
                )
            category = metadata["category"]
            difficulty_level = metadata["difficulty_level"]
        task_infos.append(
            _TaskInfo(
                task_id=task_id,
                task=task,
                name=name,
                description=description,
                category=category,
                difficulty_level=difficulty_level,
            )
        )
    return task_infos


def _resolve_plus_classification_path(args: Args) -> pathlib.Path:
    if args.classification_path:
        classification_path = pathlib.Path(args.classification_path).expanduser()
    else:
        classification_path = pathlib.Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    if not classification_path.is_file():
        raise FileNotFoundError(
            f"LIBERO-Plus classification file not found: {classification_path}. "
            "Install the pinned LIBERO-Plus source or pass classification_path."
        )
    return classification_path.resolve()


def _load_plus_classification(args: Args, num_tasks: int) -> List[Dict[str, Any]]:
    classification_path = _resolve_plus_classification_path(args)

    with classification_path.open("r", encoding="utf-8") as classification_file:
        payload = json.load(classification_file)
    entries = payload.get(args.task_suite_name) if isinstance(payload, dict) else None
    if not isinstance(entries, list) or len(entries) != num_tasks:
        raise ValueError(
            f"Expected {num_tasks} classification entries for {args.task_suite_name}, "
            f"found {len(entries) if isinstance(entries, list) else 'invalid data'}"
        )

    validated = []
    for task_id, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Classification entry {task_id} is not an object")
        if entry.get("id") != task_id + 1:
            raise ValueError(
                f"Classification entry {task_id} must have one-based id {task_id + 1}, got {entry.get('id')!r}"
            )
        name = entry.get("name")
        category = entry.get("category")
        difficulty_level = entry.get("difficulty_level")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Classification entry {task_id} has no task name")
        if category not in _LIBERO_PLUS_CATEGORIES:
            raise ValueError(f"Classification entry {task_id} has unknown category {category!r}")
        if difficulty_level is not None and (
            isinstance(difficulty_level, bool) or not isinstance(difficulty_level, int)
        ):
            raise ValueError(f"Classification entry {task_id} has invalid difficulty {difficulty_level!r}")
        validated.append(
            {
                "id": task_id + 1,
                "name": name,
                "category": category,
                "difficulty_level": difficulty_level,
            }
        )
    return validated


def _make_run_header(
    args: Args,
    benchmark_mode: str,
    task_infos: Sequence[_TaskInfo],
    num_tasks: int,
    max_steps: int,
) -> Dict[str, Any]:
    run_id = args.run_id.strip() if args.run_id else (f"anonymous-{os.getpid()}-{time.time_ns()}")
    benchmark_revision = args.benchmark_revision.strip() if args.benchmark_revision else None
    classification_sha256 = _sha256_file(_resolve_plus_classification_path(args)) if benchmark_mode == "plus" else None
    common_config = {
        "run_id": run_id,
        "benchmark_mode": benchmark_mode,
        "benchmark_revision": benchmark_revision,
        "evaluation_protocol_version": _EVALUATION_PROTOCOL_VERSION,
        "episode_seed_scheme": _EPISODE_SEED_SCHEME,
        "episode_seed_scheme_version": _EPISODE_SEED_SCHEME_VERSION,
        "policy_seed_scheme": _POLICY_SEED_SCHEME,
        "policy_seed_scheme_version": _POLICY_SEED_SCHEME_VERSION,
        "seed": args.seed,
        "resize_size": args.resize_size,
        "replan_steps": args.replan_steps,
        "num_steps_wait": args.num_steps_wait,
        "num_trials_per_task": args.num_trials_per_task,
        "environment_resolution": LIBERO_ENV_RESOLUTION,
        "policy_reconnect_attempts": args.policy_reconnect_attempts,
        "policy_reconnect_backoff_seconds": args.policy_reconnect_backoff_seconds,
        "max_consecutive_policy_errors": args.max_consecutive_policy_errors,
        "policy_connect_timeout_seconds": args.policy_connect_timeout_seconds,
        "policy_connect_retry_interval_seconds": args.policy_connect_retry_interval_seconds,
        "policy_inference_timeout_seconds": args.policy_inference_timeout_seconds,
        "policy_use_proxy": args.policy_use_proxy,
        "classification_sha256": classification_sha256,
    }
    evaluation_fingerprint = _fingerprint(common_config)
    task_manifest = _build_task_manifest(benchmark_mode, task_infos, classification_sha256)
    resolved_end = num_tasks if args.task_end is None else args.task_end
    run_config = {
        **common_config,
        "task_suite_name": args.task_suite_name,
        "num_tasks": num_tasks,
        "task_start": args.task_start,
        "task_end": resolved_end,
        "max_steps": max_steps,
        "task_manifest_fingerprint": _fingerprint(task_manifest),
        "task_group_metadata": (
            [
                {
                    "category": task_info.category,
                    "difficulty_level": task_info.difficulty_level,
                }
                for task_info in task_infos
            ]
            if benchmark_mode == "plus"
            else None
        ),
    }
    return {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "record_type": "run",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_fingerprint": evaluation_fingerprint,
        "run_fingerprint": _fingerprint(run_config),
        "run_config": run_config,
    }


def _build_task_manifest(
    benchmark_mode: str,
    task_infos: Sequence[_TaskInfo],
    classification_sha256: Optional[str],
) -> Dict[str, Any]:
    bddl_root = pathlib.Path(get_libero_path("bddl_files")).expanduser()
    init_states_root = pathlib.Path(get_libero_path("init_states")).expanduser()
    hash_cache: Dict[pathlib.Path, str] = {}
    tasks = []
    for task_info in task_infos:
        bddl_path, bddl_relative = _resolve_task_bddl_file(task_info.task, bddl_root, benchmark_mode)
        init_states_path, init_states_relative = _resolve_task_init_states_file(
            task_info.task,
            init_states_root,
            benchmark_mode,
        )
        tasks.append(
            {
                "task_id": task_info.task_id,
                "name": task_info.name,
                "description": task_info.description,
                "bddl_file": bddl_relative,
                "bddl_sha256": _cached_sha256_file(bddl_path, hash_cache),
                "init_states_file": init_states_relative,
                "init_states_sha256": _cached_sha256_file(init_states_path, hash_cache),
                "category": task_info.category,
                "difficulty_level": task_info.difficulty_level,
            }
        )
    return {"classification_sha256": classification_sha256, "tasks": tasks}


def _resolve_task_bddl_file(
    task: Any,
    root: pathlib.Path,
    benchmark_mode: str,
) -> Tuple[pathlib.Path, str]:
    problem_folder = _required_task_path_field(task, "problem_folder")
    bddl_file = _required_task_path_field(task, "bddl_file")
    if benchmark_mode == "plus" and "_view_" in bddl_file and "_initstate_" in bddl_file:
        bddl_file = f"{bddl_file.split('_view_')[0]}.bddl"
    relative_path = pathlib.Path(problem_folder) / bddl_file
    return _require_artifact_file(root / relative_path, "BDDL"), relative_path.as_posix()


def _resolve_task_init_states_file(
    task: Any,
    root: pathlib.Path,
    benchmark_mode: str,
) -> Tuple[pathlib.Path, str]:
    problem_folder = _required_task_path_field(task, "problem_folder")
    filename = _required_task_path_field(task, "init_states_file")
    relative_path = pathlib.Path(problem_folder) / filename
    if benchmark_mode == "plus":
        extension = filename.split(".")[-1]
        if "_language_" in filename:
            relative_path = pathlib.Path(problem_folder) / f"{filename.split('_language_')[0]}.{extension}"
        elif "_view_" in filename:
            relative_path = pathlib.Path(problem_folder) / f"{filename.split('_view_')[0]}.{extension}"
        elif "_add_" in filename or "_level" in filename:
            relative_path = pathlib.Path("libero_newobj") / problem_folder / filename
        elif "_light_" in filename:
            relative_path = pathlib.Path(problem_folder) / f"{filename.split('_light_')[0]}.{extension}"
        elif "_tb_" in filename:
            relative_path = pathlib.Path(problem_folder) / re.sub(r"_tb_\d+", "", filename)
        elif "_table_" in filename:
            relative_path = pathlib.Path(problem_folder) / re.sub(r"_table_\d+", "", filename)
    return _require_artifact_file(root / relative_path, "init-state"), relative_path.as_posix()


def _required_task_path_field(task: Any, field: str) -> str:
    value = getattr(task, field, None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Task is missing stable {field}")
    return value


def _require_artifact_file(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_file():
        raise FileNotFoundError(f"Resolved {label} file not found: {path}")
    return path.resolve()


def _cached_sha256_file(path: pathlib.Path, cache: Dict[pathlib.Path, str]) -> str:
    fingerprint = cache.get(path)
    if fingerprint is None:
        fingerprint = _sha256_file(path)
        cache[path] = fingerprint
    return fingerprint


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derive_episode_seed(base_seed: int, task_suite_name: str, task_id: int, episode_idx: int) -> int:
    payload = {
        "scheme": _EPISODE_SEED_SCHEME,
        "version": _EPISODE_SEED_SCHEME_VERSION,
        "base_seed": base_seed,
        "task_suite_name": task_suite_name,
        "task_id": task_id,
        "episode_idx": episode_idx,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], byteorder="big")


def _derive_policy_seed(episode_seed: int, inference_index: int) -> int:
    if isinstance(inference_index, bool) or not isinstance(inference_index, int) or inference_index < 0:
        raise ValueError(f"inference_index must be a non-negative integer, got {inference_index!r}")
    payload = {
        "scheme": _POLICY_SEED_SCHEME,
        "version": _POLICY_SEED_SCHEME_VERSION,
        "episode_seed": episode_seed,
        "inference_index": inference_index,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], byteorder="big")


def _resolve_shard_results_path(
    results_path: Optional[str],
    num_task_shards: int,
    task_shard_id: int,
) -> Optional[pathlib.Path]:
    if not results_path:
        return None
    path = pathlib.Path(results_path).expanduser()
    if num_task_shards == 1:
        return path
    suffix = path.suffix or ".jsonl"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.shard-{task_shard_id:05d}-of-{num_task_shards:05d}{suffix}")


def _outcome_status(outcome: _EpisodeOutcome) -> str:
    if outcome.error is not None:
        return "error"
    return "success" if outcome.success else "failure"


def _run_episode(
    args: Args,
    env,
    client,
    task_description: str,
    initial_state,
    max_steps: int,
    episode_seed: int,
) -> _EpisodeOutcome:
    action_plan = collections.deque()
    replay_images: List[np.ndarray] = []
    t = 0
    done = False
    inference_index = 0
    stage = "episode_seed"

    try:
        np.random.seed(episode_seed)
        env.seed(episode_seed)
        stage = "reset"
        env.reset()
        stage = "set_init_state"
        obs = env.set_init_state(initial_state)

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                stage = "stabilization_step"
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            stage = "observation_preprocessing"
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            if args.save_video:
                replay_images.append(img)

            if not action_plan:
                stage = "policy_inference"
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                policy_seed = _derive_policy_seed(episode_seed, inference_index)
                action_chunk = client.infer(element, seed=policy_seed)["actions"]
                inference_index += 1
                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, but policy only predicts "
                    f"{len(action_chunk)} steps."
                )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            stage = "environment_step"
            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
            t += 1
    except Exception as exc:
        logging.exception("Caught exception during episode stage %s", stage)
        return _EpisodeOutcome(
            success=False,
            replay_images=replay_images,
            num_steps=t,
            error=_exception_record(stage, exc),
        )

    return _EpisodeOutcome(success=bool(done), replay_images=replay_images, num_steps=t)


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220  # longest training demo has 193 steps
    if task_suite_name == "libero_object":
        return 280  # longest training demo has 254 steps
    if task_suite_name == "libero_goal":
        return 300  # longest training demo has 270 steps
    if task_suite_name == "libero_10":
        return 520  # longest training demo has 505 steps
    if task_suite_name == "libero_90":
        return 400  # longest training demo has 373 steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _select_task_ids(
    num_tasks: int,
    task_start: int,
    task_end: Optional[int],
    num_task_shards: int,
    task_shard_id: int,
) -> List[int]:
    resolved_end = num_tasks if task_end is None else task_end
    if task_start < 0 or task_start > num_tasks:
        raise ValueError(f"task_start must be in [0, {num_tasks}], got {task_start}")
    if resolved_end < 0 or resolved_end > num_tasks:
        raise ValueError(f"task_end must be in [0, {num_tasks}], got {resolved_end}")
    if resolved_end < task_start:
        raise ValueError(f"task_end ({resolved_end}) must be >= task_start ({task_start})")
    if num_task_shards <= 0:
        raise ValueError(f"num_task_shards must be positive, got {num_task_shards}")
    if task_shard_id < 0 or task_shard_id >= num_task_shards:
        raise ValueError(f"task_shard_id must be in [0, {num_task_shards}), got {task_shard_id}")

    return [task_id for task_id in range(task_start, resolved_end) if task_id % num_task_shards == task_shard_id]


def _episode_key(record: Dict[str, Any]) -> Optional[Tuple[str, int, int]]:
    if (
        record.get("schema_version") != _RESULT_SCHEMA_VERSION
        or record.get("record_type") != "episode"
        or record.get("status") not in _EPISODE_STATUSES
    ):
        return None
    try:
        task_suite_name = record["task_suite_name"]
        task_id = record["task_id"]
        episode_idx = record["episode_idx"]
    except KeyError:
        return None
    if (
        not isinstance(task_suite_name, str)
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id < 0
        or isinstance(episode_idx, bool)
        or not isinstance(episode_idx, int)
        or episode_idx < 0
    ):
        return None
    return task_suite_name, task_id, episode_idx


def _validate_run_header(header: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(header, dict):
        raise ValueError("Run header must be a JSON object")
    if header.get("schema_version") != _RESULT_SCHEMA_VERSION or header.get("record_type") != "run":
        raise ValueError(f"Unsupported results schema; expected run header version {_RESULT_SCHEMA_VERSION}")
    run_config = header.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("Run header is missing run_config")
    required_config_fields = {
        "run_id",
        "benchmark_mode",
        "benchmark_revision",
        "evaluation_protocol_version",
        "episode_seed_scheme",
        "episode_seed_scheme_version",
        "policy_seed_scheme",
        "policy_seed_scheme_version",
        "seed",
        "resize_size",
        "replan_steps",
        "num_steps_wait",
        "num_trials_per_task",
        "environment_resolution",
        "task_suite_name",
        "num_tasks",
        "task_start",
        "task_end",
        "max_steps",
        "task_manifest_fingerprint",
        "task_group_metadata",
        "policy_reconnect_attempts",
        "policy_reconnect_backoff_seconds",
        "max_consecutive_policy_errors",
        "policy_connect_timeout_seconds",
        "policy_connect_retry_interval_seconds",
        "policy_inference_timeout_seconds",
        "policy_use_proxy",
        "classification_sha256",
    }
    missing_fields = sorted(required_config_fields - set(run_config))
    if missing_fields:
        raise ValueError(f"Run header is missing config fields: {missing_fields}")
    if not isinstance(run_config["run_id"], str) or not run_config["run_id"].strip():
        raise ValueError("Run header has an invalid run_id")
    if not isinstance(run_config["policy_use_proxy"], bool):
        raise ValueError("Run header has an invalid policy_use_proxy")
    if run_config["benchmark_mode"] not in {"standard", "plus"}:
        raise ValueError("Run header has an invalid benchmark_mode")
    benchmark_revision = run_config["benchmark_revision"]
    if benchmark_revision is not None and (not isinstance(benchmark_revision, str) or not benchmark_revision.strip()):
        raise ValueError("Run header has an invalid benchmark_revision")
    if run_config["evaluation_protocol_version"] != _EVALUATION_PROTOCOL_VERSION:
        raise ValueError("Run header has an unsupported evaluation_protocol_version")
    if run_config["episode_seed_scheme"] != _EPISODE_SEED_SCHEME:
        raise ValueError("Run header has an unsupported episode_seed_scheme")
    if run_config["episode_seed_scheme_version"] != _EPISODE_SEED_SCHEME_VERSION:
        raise ValueError("Run header has an unsupported episode_seed_scheme_version")
    if run_config["policy_seed_scheme"] != _POLICY_SEED_SCHEME:
        raise ValueError("Run header has an unsupported policy_seed_scheme")
    if run_config["policy_seed_scheme_version"] != _POLICY_SEED_SCHEME_VERSION:
        raise ValueError("Run header has an unsupported policy_seed_scheme_version")
    if not isinstance(run_config["task_suite_name"], str) or not run_config["task_suite_name"]:
        raise ValueError("Run header has an invalid task_suite_name")
    integer_fields = (
        "seed",
        "resize_size",
        "replan_steps",
        "num_steps_wait",
        "num_trials_per_task",
        "environment_resolution",
        "num_tasks",
        "task_start",
        "task_end",
        "max_steps",
        "episode_seed_scheme_version",
        "policy_seed_scheme_version",
        "policy_reconnect_attempts",
        "max_consecutive_policy_errors",
    )
    if any(isinstance(run_config[field], bool) or not isinstance(run_config[field], int) for field in integer_fields):
        raise ValueError("Run header contains a non-integer protocol field")
    if (
        run_config["resize_size"] <= 0
        or run_config["replan_steps"] <= 0
        or run_config["num_steps_wait"] < 0
        or run_config["num_trials_per_task"] <= 0
        or run_config["environment_resolution"] <= 0
        or run_config["num_tasks"] <= 0
        or run_config["task_start"] < 0
        or run_config["task_end"] < run_config["task_start"]
        or run_config["task_end"] > run_config["num_tasks"]
        or run_config["max_steps"] <= 0
        or run_config["policy_reconnect_attempts"] < 0
        or run_config["max_consecutive_policy_errors"] <= 0
    ):
        raise ValueError("Run header contains an out-of-range protocol field")
    float_fields = (
        "policy_reconnect_backoff_seconds",
        "policy_connect_timeout_seconds",
        "policy_connect_retry_interval_seconds",
        "policy_inference_timeout_seconds",
    )
    if any(
        isinstance(run_config[field], bool) or not isinstance(run_config[field], (int, float)) for field in float_fields
    ):
        raise ValueError("Run header contains a non-numeric policy field")
    if (
        run_config["policy_reconnect_backoff_seconds"] < 0
        or run_config["policy_connect_timeout_seconds"] <= 0
        or run_config["policy_connect_retry_interval_seconds"] < 0
        or run_config["policy_inference_timeout_seconds"] <= 0
    ):
        raise ValueError("Run header contains an out-of-range policy field")
    classification_sha256 = run_config["classification_sha256"]
    if run_config["benchmark_mode"] == "plus":
        if not re.fullmatch(r"[0-9a-f]{64}", str(classification_sha256)):
            raise ValueError("Run header has an invalid LIBERO-Plus classification_sha256")
        task_group_metadata = run_config["task_group_metadata"]
        if not isinstance(task_group_metadata, list) or len(task_group_metadata) != run_config["num_tasks"]:
            raise ValueError("Run header has invalid LIBERO-Plus task_group_metadata length")
        for task_id, group in enumerate(task_group_metadata):
            if not isinstance(group, dict) or set(group) != {"category", "difficulty_level"}:
                raise ValueError(f"Run header task group {task_id} is malformed")
            if group["category"] not in _LIBERO_PLUS_CATEGORIES:
                raise ValueError(f"Run header task group {task_id} has an invalid category")
            difficulty_level = group["difficulty_level"]
            if difficulty_level is not None and (
                isinstance(difficulty_level, bool) or not isinstance(difficulty_level, int)
            ):
                raise ValueError(f"Run header task group {task_id} has an invalid difficulty")
    elif classification_sha256 is not None:
        raise ValueError("Standard LIBERO run header unexpectedly has classification_sha256")
    elif run_config["task_group_metadata"] is not None:
        raise ValueError("Standard LIBERO run header unexpectedly has task_group_metadata")
    if not re.fullmatch(r"[0-9a-f]{64}", str(run_config["task_manifest_fingerprint"])):
        raise ValueError("Run header has an invalid task_manifest_fingerprint")
    run_fingerprint = header.get("run_fingerprint")
    if run_fingerprint != _fingerprint(run_config):
        raise ValueError("Run header run_fingerprint does not match run_config")
    common_config = {
        key: run_config[key]
        for key in (
            "run_id",
            "benchmark_mode",
            "benchmark_revision",
            "evaluation_protocol_version",
            "episode_seed_scheme",
            "episode_seed_scheme_version",
            "policy_seed_scheme",
            "policy_seed_scheme_version",
            "seed",
            "resize_size",
            "replan_steps",
            "num_steps_wait",
            "num_trials_per_task",
            "environment_resolution",
            "policy_reconnect_attempts",
            "policy_reconnect_backoff_seconds",
            "max_consecutive_policy_errors",
            "policy_connect_timeout_seconds",
            "policy_connect_retry_interval_seconds",
            "policy_inference_timeout_seconds",
            "policy_use_proxy",
            "classification_sha256",
        )
    }
    if header.get("evaluation_fingerprint") != _fingerprint(common_config):
        raise ValueError("Run header evaluation_fingerprint does not match run_config")
    return header


def _require_matching_run_header(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    path: pathlib.Path,
) -> None:
    if expected["run_fingerprint"] != actual["run_fingerprint"]:
        raise ValueError(
            f"Cannot resume {path}: run fingerprint differs. "
            f"expected={expected['run_fingerprint']} actual={actual['run_fingerprint']}"
        )
    if expected["run_config"] != actual["run_config"]:
        raise ValueError(f"Cannot resume {path}: run_config differs despite matching fingerprint")


def _require_durable_benchmark_revision(header: Dict[str, Any], path: pathlib.Path) -> None:
    if header["run_config"]["benchmark_revision"] is None:
        raise ValueError(f"Durable results journal has no benchmark_revision: {path}")


def _validate_episode_record(record: Dict[str, Any], run_header: Dict[str, Any]) -> Tuple[str, int, int]:
    key = _episode_key(record)
    if key is None:
        raise ValueError(f"Invalid episode record key/status: {record}")
    run_config = run_header["run_config"]
    if record.get("run_fingerprint") != run_header["run_fingerprint"]:
        raise ValueError(f"Episode {key} has a different run_fingerprint")
    if key[0] != run_config["task_suite_name"]:
        raise ValueError(f"Episode {key} has the wrong task suite")
    if not (run_config["task_start"] <= key[1] < run_config["task_end"]):
        raise ValueError(f"Episode {key} is outside the configured task range")
    if key[2] >= run_config["num_trials_per_task"]:
        raise ValueError(f"Episode {key} exceeds num_trials_per_task")
    if not isinstance(record.get("task_name"), str) or not isinstance(record.get("task_description"), str):
        raise ValueError(f"Episode {key} is missing strict task identity")
    if record.get("seed") != run_config["seed"]:
        raise ValueError(f"Episode {key} has a different seed")
    expected_episode_seed = _derive_episode_seed(run_config["seed"], key[0], key[1], key[2])
    if record.get("episode_seed") != expected_episode_seed:
        raise ValueError(
            f"Episode {key} has an invalid episode_seed; "
            f"expected={expected_episode_seed} actual={record.get('episode_seed')!r}"
        )
    if record.get("max_steps") != run_config["max_steps"]:
        raise ValueError(f"Episode {key} has a different max_steps")
    if isinstance(record.get("num_steps"), bool) or not isinstance(record.get("num_steps"), int):
        raise ValueError(f"Episode {key} has an invalid num_steps")
    if record["num_steps"] < 0:
        raise ValueError(f"Episode {key} has a negative num_steps")
    duration_seconds = record.get("duration_seconds")
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)) or duration_seconds < 0:
        raise ValueError(f"Episode {key} has an invalid duration_seconds")
    if not isinstance(record.get("timestamp_utc"), str):
        raise ValueError(f"Episode {key} has no timestamp_utc")
    if run_config["benchmark_mode"] == "plus":
        if record.get("category") not in _LIBERO_PLUS_CATEGORIES:
            raise ValueError(f"Episode {key} has invalid LIBERO-Plus category metadata")
        difficulty_level = record.get("difficulty_level")
        if difficulty_level is not None and (
            isinstance(difficulty_level, bool) or not isinstance(difficulty_level, int)
        ):
            raise ValueError(f"Episode {key} has invalid LIBERO-Plus difficulty metadata")
    elif record.get("category") is not None or record.get("difficulty_level") is not None:
        raise ValueError(f"Standard LIBERO episode {key} unexpectedly has Plus metadata")
    status = record["status"]
    if record.get("success") is not (status == "success"):
        raise ValueError(f"Episode {key} has inconsistent status/success fields")
    if status == "error" and not isinstance(record.get("error"), dict):
        raise ValueError(f"Episode {key} error status has no structured error")
    if status != "error" and record.get("error") is not None:
        raise ValueError(f"Episode {key} has an error payload but status={status!r}")
    return key


def _load_journal(
    path: pathlib.Path,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, int, int], Dict[str, Any]]]:
    records: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Results journal is missing or empty: {path}")

    with path.open("r", encoding="utf-8", errors="replace") as results_file:
        fcntl.flock(results_file.fileno(), fcntl.LOCK_SH)
        try:
            lines = results_file.readlines()
            nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
            if not nonempty_indices:
                raise ValueError(f"Results journal has no records: {path}")
            last_nonempty_index = nonempty_indices[-1]
            parsed_records = []
            for line_index in nonempty_indices:
                line = lines[line_index]
                line_number = line_index + 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if line_index == last_nonempty_index and not line.endswith("\n"):
                        logging.warning("Ignoring incomplete final JSONL record at %s:%d", path, line_number)
                        break
                    raise ValueError(f"Malformed JSONL record at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Non-object JSONL record at {path}:{line_number}")
                parsed_records.append((line_number, record))
        finally:
            fcntl.flock(results_file.fileno(), fcntl.LOCK_UN)

    if not parsed_records:
        raise ValueError(f"Results journal has no complete run header: {path}")
    header = _validate_run_header(parsed_records[0][1])
    _require_durable_benchmark_revision(header, path)
    for line_number, record in parsed_records[1:]:
        if record.get("record_type") == "run":
            raise ValueError(f"Unexpected second run header at {path}:{line_number}")
        try:
            key = _validate_episode_record(record, header)
        except ValueError as exc:
            raise ValueError(f"Invalid episode record at {path}:{line_number}: {exc}") from exc
        old_record = records.get(key)
        if old_record is not None and (
            old_record["task_name"] != record["task_name"]
            or old_record["task_description"] != record["task_description"]
            or old_record["seed"] != record["seed"]
        ):
            raise ValueError(f"Episode {key} changes identity within {path}")
        records[key] = record
    return header, records


def _load_episode_records(path: pathlib.Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    return _load_journal(path)[1]


def _write_new_journal(path: pathlib.Path, run_header: Dict[str, Any]) -> None:
    _reset_jsonl(path)
    _append_jsonl_record(path, run_header)


@contextlib.contextmanager
def _journal_read_locks(paths: Sequence[pathlib.Path]):
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Input journal paths must be unique")
    lock_files = []
    try:
        for path in paths:
            lock_path = path.with_name(path.name + ".lock")
            lock_file = lock_path.open("a+b")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lock_file.close()
                raise RuntimeError(f"Cannot read {path} while an evaluation process is still writing it") from exc
            lock_files.append(lock_file)
        yield
    finally:
        for lock_file in reversed(lock_files):
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()


def _reset_jsonl(path: pathlib.Path) -> None:
    with path.open("a+b") as results_file:
        fcntl.flock(results_file.fileno(), fcntl.LOCK_EX)
        try:
            results_file.truncate(0)
            results_file.flush()
            os.fsync(results_file.fileno())
        finally:
            fcntl.flock(results_file.fileno(), fcntl.LOCK_UN)


def _append_jsonl_record(path: pathlib.Path, record: Dict[str, Any]) -> None:
    encoded_record = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("a+b") as results_file:
        fcntl.flock(results_file.fileno(), fcntl.LOCK_EX)
        try:
            results_file.seek(0, os.SEEK_END)
            if results_file.tell() > 0:
                results_file.seek(-1, os.SEEK_END)
                if results_file.read(1) != b"\n":
                    results_file.seek(0)
                    contents = results_file.read()
                    last_newline = contents.rfind(b"\n")
                    results_file.truncate(last_newline + 1)
                    results_file.seek(0, os.SEEK_END)
            results_file.write(encoded_record)
            results_file.flush()
            os.fsync(results_file.fileno())
        finally:
            fcntl.flock(results_file.fileno(), fcntl.LOCK_UN)


def _merge_episode_journals(
    input_paths: Sequence[pathlib.Path],
    output_path: pathlib.Path,
    *,
    overwrite: bool,
) -> None:
    with _journal_read_locks(input_paths):
        _merge_episode_journals_locked(input_paths, output_path, overwrite=overwrite)


def _merge_episode_journals_locked(
    input_paths: Sequence[pathlib.Path],
    output_path: pathlib.Path,
    *,
    overwrite: bool,
) -> None:
    if len(input_paths) < 1:
        raise ValueError("At least one input journal is required")
    resolved_output = output_path.resolve()
    if any(path.resolve() == resolved_output for path in input_paths):
        raise ValueError("Merge output path must differ from every input path")

    base_header = None
    merged_records: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    key_sources: Dict[Tuple[str, int, int], pathlib.Path] = {}
    for input_path in input_paths:
        header, records = _load_journal(input_path)
        if base_header is None:
            base_header = header
        elif header["run_fingerprint"] != base_header["run_fingerprint"] or (
            header["run_config"] != base_header["run_config"]
        ):
            raise ValueError(f"Cannot merge {input_path}: run fingerprint/config differs from {input_paths[0]}")
        for key, record in records.items():
            if key in merged_records:
                raise ValueError(
                    f"Duplicate episode {key} across {key_sources[key]} and {input_path}; shards must be disjoint"
                )
            merged_records[key] = record
            key_sources[key] = input_path

    assert base_header is not None
    _write_complete_journal_atomic(
        output_path,
        base_header,
        [merged_records[key] for key in sorted(merged_records)],
        overwrite=overwrite,
    )
    logging.info("Merged %d episode records into %s", len(merged_records), output_path)


def _write_complete_journal_atomic(
    output_path: pathlib.Path,
    header: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    header = _validate_run_header(header)
    _require_durable_benchmark_revision(header, output_path)
    seen_keys = set()
    for record in records:
        key = _validate_episode_record(record, header)
        if key in seen_keys:
            raise ValueError(f"Duplicate episode {key} in atomic journal write")
        seen_keys.add(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(output_path.name + ".lock")
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Merge output is already owned by another process: {output_path}") from exc
        if output_path.exists() and output_path.stat().st_size and not overwrite:
            raise FileExistsError(f"Merge output already exists: {output_path}; pass overwrite_results")

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(output_path.parent),
                prefix=output_path.name + ".tmp-",
                delete=False,
            ) as temporary_file:
                temporary_path = pathlib.Path(temporary_file.name)
                for record in (header, *records):
                    temporary_file.write(
                        (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(str(temporary_path), str(output_path))
            temporary_path = None
            directory_fd = os.open(str(output_path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _exception_record(stage: str, exc: Exception) -> Dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def _make_episode_record(
    args: Args,
    run_fingerprint: str,
    task_info: _TaskInfo,
    episode_idx: int,
    max_steps: int,
    outcome: _EpisodeOutcome,
    *,
    episode_seed: int,
    duration_seconds: float,
    video_path: Optional[str],
    video_error: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    status = _outcome_status(outcome)
    return {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "record_type": "episode",
        "run_fingerprint": run_fingerprint,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task_suite_name": args.task_suite_name,
        "task_id": task_info.task_id,
        "task_name": task_info.name,
        # Keep the benchmark prompt verbatim in results; only video filenames are sanitized.
        "task_description": task_info.description,
        "category": task_info.category,
        "difficulty_level": task_info.difficulty_level,
        "episode_idx": episode_idx,
        "seed": args.seed,
        "episode_seed": episode_seed,
        "status": status,
        "success": status == "success",
        "num_steps": outcome.num_steps,
        "max_steps": max_steps,
        "duration_seconds": round(duration_seconds, 6),
        "error": outcome.error,
        "video_path": video_path,
        "video_error": video_error,
    }


def _record_task_setup_errors(
    args: Args,
    journal: _EpisodeJournal,
    counts: _EvaluationCounts,
    task_info: _TaskInfo,
    max_steps: int,
    *,
    stage: str,
    exc: Exception,
    episode_indices: Optional[Sequence[int]] = None,
) -> None:
    if episode_indices is None:
        episode_indices = range(args.num_trials_per_task)
    for episode_idx in episode_indices:
        if journal.should_skip(
            args.task_suite_name,
            task_info.task_id,
            episode_idx,
            task_name=task_info.name,
            task_description=task_info.description,
            seed=args.seed,
        ):
            continue
        outcome = _EpisodeOutcome(
            success=False,
            replay_images=[],
            num_steps=0,
            error=_exception_record(stage, exc),
        )
        record = _make_episode_record(
            args,
            journal.run_header["run_fingerprint"],
            task_info,
            episode_idx,
            max_steps,
            outcome,
            episode_seed=_derive_episode_seed(
                args.seed,
                args.task_suite_name,
                task_info.task_id,
                episode_idx,
            ),
            duration_seconds=0.0,
            video_path=None,
            video_error=None,
        )
        old_record = journal.append(record)
        counts.replace(old_record, record)


def _safe_filename_component(value: str, max_length: int = 72) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    slug = slug[:max_length].rstrip("._-") or "task"
    return f"{slug}-{digest}"


def _get_video_path(
    video_out_path: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    task_description: str,
    *,
    run_fingerprint: str,
    status: str,
) -> pathlib.Path:
    if status not in _EPISODE_STATUSES:
        raise ValueError(f"Unknown video status: {status!r}")
    suite_segment = _safe_filename_component(task_suite_name, max_length=32)
    task_segment = _safe_filename_component(task_description)
    filename = f"rollout_{suite_segment}_task-{task_id:05d}_episode-{episode_idx:03d}_{task_segment}_{status}.mp4"
    return pathlib.Path(video_out_path) / f"run-{run_fingerprint[:16]}" / filename


def _save_replay_video(
    args: Args,
    replay_images: Sequence[np.ndarray],
    task_id: int,
    episode_idx: int,
    task_description: str,
    *,
    run_fingerprint: str,
    status: str,
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    if not args.save_video:
        return None, None
    if not replay_images:
        exc = RuntimeError("No replay frames were produced")
        return None, _exception_record("video_write", exc)

    video_path = _get_video_path(
        args.video_out_path,
        args.task_suite_name,
        task_id,
        episode_idx,
        task_description,
        run_fingerprint=run_fingerprint,
        status=status,
    )
    try:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, [np.asarray(image) for image in replay_images], fps=10)
    except Exception as exc:
        logging.exception("Failed to save replay video to %s", video_path)
        return None, _exception_record("video_write", exc)
    return str(video_path), None


def _summarize_records(
    records: Dict[Tuple[str, int, int], Dict[str, Any]],
    task_suite_name: str,
    task_ids: Sequence[int],
    num_trials_per_task: int,
    task_group_metadata: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    task_summaries = []
    for task_id in task_ids:
        task_records = [
            records[(task_suite_name, task_id, episode_idx)]
            for episode_idx in range(num_trials_per_task)
            if (task_suite_name, task_id, episode_idx) in records
        ]
        successes = sum(record["status"] == "success" for record in task_records)
        failures = sum(record["status"] == "failure" for record in task_records)
        errors = sum(record["status"] == "error" for record in task_records)
        task_name = next(
            (record.get("task_name") for record in reversed(task_records) if record.get("task_name")), None
        )
        task_description = next(
            (record.get("task_description") for record in reversed(task_records) if record.get("task_description")),
            None,
        )
        category = next((record.get("category") for record in reversed(task_records) if record.get("category")), None)
        difficulty_level = next(
            (
                record.get("difficulty_level")
                for record in reversed(task_records)
                if record.get("difficulty_level") is not None
            ),
            None,
        )
        if task_group_metadata is not None:
            group = task_group_metadata[task_id]
            if category is None:
                category = group["category"]
            if not task_records:
                difficulty_level = group["difficulty_level"]
        completed = len(task_records)
        task_summaries.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "task_description": task_description,
                "category": category,
                "difficulty_level": difficulty_level,
                "episodes": completed,
                "successes": successes,
                "failures": failures,
                "errors": errors,
                "pending": num_trials_per_task - completed,
                "success_rate": successes / completed if completed else 0.0,
            }
        )

    total_episodes = sum(task["episodes"] for task in task_summaries)
    total_successes = sum(task["successes"] for task in task_summaries)
    total_failures = sum(task["failures"] for task in task_summaries)
    total_errors = sum(task["errors"] for task in task_summaries)
    total_pending = sum(task["pending"] for task in task_summaries)
    category_summaries = _group_task_summaries(task_summaries, "category")
    difficulty_summaries = _group_task_summaries(
        task_summaries,
        "difficulty_level",
        unknown_group="unknown",
    )
    return {
        "task_suite_name": task_suite_name,
        "tasks": task_summaries,
        "episodes": total_episodes,
        "successes": total_successes,
        "failures": total_failures,
        "errors": total_errors,
        "pending": total_pending,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "category_macro_success_rate": _macro_group_success_rate(category_summaries),
        "categories": category_summaries,
        "difficulty_levels": difficulty_summaries,
    }


def _group_task_summaries(
    task_summaries: Sequence[Dict[str, Any]],
    field: str,
    *,
    unknown_group: Optional[Any] = None,
) -> Dict[Any, Dict[str, Any]]:
    grouped: Dict[Any, Dict[str, Any]] = {}
    for task in task_summaries:
        group = task.get(field)
        if group is None:
            if unknown_group is None or task.get("category") is None:
                continue
            group = unknown_group
        summary = grouped.setdefault(
            group,
            {"episodes": 0, "successes": 0, "failures": 0, "errors": 0, "pending": 0},
        )
        for count_field in ("episodes", "successes", "failures", "errors", "pending"):
            summary[count_field] += task[count_field]
    for summary in grouped.values():
        summary["success_rate"] = summary["successes"] / summary["episodes"] if summary["episodes"] else 0.0
    return grouped


def _macro_group_success_rate(
    groups: Dict[Any, Dict[str, Any]],
    *,
    expected_groups: Optional[Sequence[Any]] = None,
) -> Optional[float]:
    selected_groups = list(groups) if expected_groups is None else list(expected_groups)
    if not selected_groups:
        return None
    if any(group not in groups or groups[group]["episodes"] == 0 for group in selected_groups):
        return None
    return sum(groups[group]["success_rate"] for group in selected_groups) / len(selected_groups)


def _annotate_summary(
    summary: Dict[str, Any],
    args: Args,
    benchmark_mode: str,
    num_tasks: int,
    *,
    abort_reason: Optional[str],
) -> None:
    resolved_end = num_tasks if args.task_end is None else args.task_end
    full_task_range = args.task_start == 0 and resolved_end == num_tasks
    if args.num_task_shards > 1:
        scope = "shard"
    elif not full_task_range:
        scope = "task_range"
    else:
        scope = "full_suite"
    official_trials = 1 if benchmark_mode == "plus" else 50
    complete = summary["pending"] == 0
    summary.update(
        {
            "benchmark_mode": benchmark_mode,
            "scope": scope,
            "expected_episodes": len(summary["tasks"]) * args.num_trials_per_task,
            "protocol_complete": complete,
            "official": (
                scope == "full_suite"
                and complete
                and summary["errors"] == 0
                and args.num_trials_per_task == official_trials
            ),
            "abort_reason": abort_reason,
        }
    )


def _log_task_summary(
    records: Dict[Tuple[str, int, int], Dict[str, Any]],
    task_suite_name: str,
    task_id: int,
    num_trials_per_task: int,
    task_group_metadata: Optional[Sequence[Dict[str, Any]]],
) -> None:
    task_summary = _summarize_records(
        records,
        task_suite_name,
        [task_id],
        num_trials_per_task,
        task_group_metadata,
    )["tasks"][0]
    logging.info(
        "Task %d summary: successes=%d episodes=%d failures=%d errors=%d pending=%d success_rate=%.4f",
        task_id,
        task_summary["successes"],
        task_summary["episodes"],
        task_summary["failures"],
        task_summary["errors"],
        task_summary["pending"],
        task_summary["success_rate"],
    )


def _log_final_summary(summary: Dict[str, Any]) -> None:
    if len(summary["tasks"]) <= 100:
        logging.info("Per-task results:")
        for task in summary["tasks"]:
            logging.info(
                "task_id=%d task_name=%r description=%r successes=%d episodes=%d failures=%d errors=%d "
                "pending=%d success_rate=%.4f",
                task["task_id"],
                task["task_name"],
                task["task_description"],
                task["successes"],
                task["episodes"],
                task["failures"],
                task["errors"],
                task["pending"],
                task["success_rate"],
            )
    else:
        logging.info("Per-task log omitted for %d tasks; details are in the JSONL journal", len(summary["tasks"]))
    _log_group_summary("LIBERO-Plus category", summary.get("categories", {}))
    _log_group_summary("LIBERO-Plus difficulty", summary.get("difficulty_levels", {}))
    logging.info("Final results:")
    logging.info("Result scope: %s", summary.get("scope", "selected"))
    logging.info("Total episodes: %d", summary["episodes"])
    logging.info("Expected episodes: %d", summary.get("expected_episodes", summary["episodes"] + summary["pending"]))
    logging.info("Total successes: %d", summary["successes"])
    logging.info("Total failures: %d", summary["failures"])
    logging.info("Total errors: %d", summary["errors"])
    logging.info("Total pending: %d", summary["pending"])
    logging.info("Selected success rate: %.4f", summary["success_rate"])
    if summary.get("category_macro_success_rate") is not None:
        logging.info("Category-macro success rate: %.4f", summary["category_macro_success_rate"])
    logging.info("Protocol complete: %s", summary.get("protocol_complete", summary["pending"] == 0))
    logging.info("Official score: %s", summary.get("official", False))
    if summary.get("abort_reason"):
        logging.error("Abort reason: %s", summary["abort_reason"])


def _log_group_summary(label: str, groups: Dict[Any, Dict[str, Any]]) -> None:
    if not groups:
        return
    logging.info("%s summary:", label)
    for group, summary in sorted(groups.items(), key=lambda item: str(item[0])):
        logging.info(
            "%s: successes=%d episodes=%d failures=%d errors=%d pending=%d success_rate=%.4f",
            group,
            summary["successes"],
            summary["episodes"],
            summary["failures"],
            summary["errors"],
            summary["pending"],
            summary["success_rate"],
        )


def _summarize_result_journals(input_paths: Sequence[pathlib.Path]) -> Dict[str, Any]:
    with _journal_read_locks(input_paths):
        return _summarize_result_journals_locked(input_paths)


def _summarize_result_journals_locked(input_paths: Sequence[pathlib.Path]) -> Dict[str, Any]:
    if not input_paths:
        raise ValueError("At least one results journal is required")
    evaluation_fingerprint = None
    benchmark_revision = None
    mode = None
    headers_by_suite: Dict[str, Dict[str, Any]] = {}
    records: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    key_sources: Dict[Tuple[str, int, int], pathlib.Path] = {}

    for path in input_paths:
        header, journal_records = _load_journal(path)
        config = header["run_config"]
        if benchmark_revision is None:
            benchmark_revision = config["benchmark_revision"]
        elif config["benchmark_revision"] != benchmark_revision:
            raise ValueError(f"Cannot summarize {path}: benchmark revision differs")
        if evaluation_fingerprint is None:
            evaluation_fingerprint = header["evaluation_fingerprint"]
            mode = config["benchmark_mode"]
        elif header["evaluation_fingerprint"] != evaluation_fingerprint:
            raise ValueError(f"Cannot summarize {path}: evaluation fingerprint differs")
        suite_name = config["task_suite_name"]
        old_header = headers_by_suite.get(suite_name)
        if old_header is not None and old_header["run_fingerprint"] != header["run_fingerprint"]:
            raise ValueError(f"Cannot summarize {path}: suite {suite_name} has multiple run fingerprints")
        headers_by_suite[suite_name] = header
        for key, record in journal_records.items():
            if key in records:
                raise ValueError(
                    f"Duplicate episode {key} across {key_sources[key]} and {path}; merge shards first or fix overlap"
                )
            records[key] = record
            key_sources[key] = path

    protocol_counts = _LIBERO_PLUS_TASK_COUNTS if mode == "plus" else _STANDARD_LIBERO_TASK_COUNTS
    official_trials = 1 if mode == "plus" else 50
    suite_summaries = {}
    for suite_name, header in headers_by_suite.items():
        config = header["run_config"]
        task_ids = list(range(config["task_start"], config["task_end"]))
        summary = _summarize_records(
            records,
            suite_name,
            task_ids,
            config["num_trials_per_task"],
            config["task_group_metadata"],
        )
        full_suite = config["task_start"] == 0 and config["task_end"] == config["num_tasks"]
        suite_official = (
            full_suite
            and summary["pending"] == 0
            and summary["errors"] == 0
            and suite_name in protocol_counts
            and config["num_tasks"] == protocol_counts[suite_name]
            and config["num_trials_per_task"] == official_trials
        )
        summary.update(
            {
                "benchmark_mode": mode,
                "scope": "full_suite" if full_suite else "task_range",
                "expected_episodes": len(task_ids) * config["num_trials_per_task"],
                "protocol_complete": summary["pending"] == 0,
                "official": suite_official,
                "abort_reason": None,
            }
        )
        suite_summaries[suite_name] = summary

    expected_suites = set(protocol_counts) - {"libero_90"}
    protocol_definition_valid = all(
        suite_name in protocol_counts
        and header["run_config"]["num_tasks"] == protocol_counts[suite_name]
        and header["run_config"]["num_trials_per_task"] == official_trials
        for suite_name, header in headers_by_suite.items()
    )
    total_episodes = sum(summary["episodes"] for summary in suite_summaries.values())
    total_successes = sum(summary["successes"] for summary in suite_summaries.values())
    total_failures = sum(summary["failures"] for summary in suite_summaries.values())
    total_errors = sum(summary["errors"] for summary in suite_summaries.values())
    total_pending = sum(summary["pending"] for summary in suite_summaries.values())
    category_summaries = _combine_group_summaries(suite_summaries.values(), "categories")
    difficulty_summaries = _combine_group_summaries(suite_summaries.values(), "difficulty_levels")
    category_macro_success_rate = (
        _macro_group_success_rate(category_summaries, expected_groups=_LIBERO_PLUS_CATEGORIES)
        if mode == "plus"
        else None
    )
    protocol_complete = (
        set(suite_summaries) == expected_suites
        and protocol_definition_valid
        and total_pending == 0
        and all(summary["scope"] == "full_suite" for summary in suite_summaries.values())
    )
    official = (
        set(suite_summaries) == expected_suites
        and protocol_definition_valid
        and protocol_complete
        and total_errors == 0
        and all(summary["scope"] == "full_suite" for summary in suite_summaries.values())
    )
    return {
        "evaluation_fingerprint": evaluation_fingerprint,
        "benchmark_revision": benchmark_revision,
        "benchmark_mode": mode,
        "suites": suite_summaries,
        "episodes": total_episodes,
        "expected_episodes": total_episodes + total_pending,
        "successes": total_successes,
        "failures": total_failures,
        "errors": total_errors,
        "pending": total_pending,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "category_macro_success_rate": category_macro_success_rate,
        "categories": category_summaries,
        "difficulty_levels": difficulty_summaries,
        "protocol_complete": protocol_complete,
        "official": official,
    }


def _combine_group_summaries(
    suite_summaries: Sequence[Dict[str, Any]],
    field: str,
) -> Dict[Any, Dict[str, Any]]:
    combined: Dict[Any, Dict[str, Any]] = {}
    for suite_summary in suite_summaries:
        for group, source in suite_summary.get(field, {}).items():
            target = combined.setdefault(
                group,
                {"episodes": 0, "successes": 0, "failures": 0, "errors": 0, "pending": 0},
            )
            for count_field in ("episodes", "successes", "failures", "errors", "pending"):
                target[count_field] += source[count_field]
    for summary in combined.values():
        summary["success_rate"] = summary["successes"] / summary["episodes"] if summary["episodes"] else 0.0
    return combined


def _log_aggregate_summary(summary: Dict[str, Any]) -> None:
    logging.info("Aggregate evaluation fingerprint: %s", summary["evaluation_fingerprint"])
    for suite_name in sorted(summary["suites"]):
        suite = summary["suites"][suite_name]
        logging.info(
            "%s: successes=%d episodes=%d errors=%d pending=%d success_rate=%.4f official=%s",
            suite_name,
            suite["successes"],
            suite["episodes"],
            suite["errors"],
            suite["pending"],
            suite["success_rate"],
            suite["official"],
        )
    _log_group_summary("LIBERO-Plus category", summary["categories"])
    _log_group_summary("LIBERO-Plus difficulty", summary["difficulty_levels"])
    logging.info(
        "Aggregate: successes=%d episodes=%d expected=%d failures=%d errors=%d pending=%d success_rate=%.4f",
        summary["successes"],
        summary["episodes"],
        summary["expected_episodes"],
        summary["failures"],
        summary["errors"],
        summary["pending"],
        summary["success_rate"],
    )
    if summary["category_macro_success_rate"] is not None:
        logging.info("Aggregate category-macro success rate: %.4f", summary["category_macro_success_rate"])
    logging.info("Protocol complete: %s", summary["protocol_complete"])
    logging.info("Official cross-suite score: %s", summary["official"])


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    try:
        env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    except Exception:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close environment after seed setup failed")
        raise
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
