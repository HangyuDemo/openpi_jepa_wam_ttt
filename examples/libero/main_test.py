# ruff: noqa: PT009, PT027, SLF001

import contextlib
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


def _load_eval_module():
    module_path = pathlib.Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location("_openpi_libero_eval_test_target", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)

    benchmark = types.SimpleNamespace(get_benchmark_dict=dict)
    libero_package = types.ModuleType("libero")
    libero_module = types.ModuleType("libero.libero")
    libero_module.benchmark = benchmark
    libero_module.get_libero_path = lambda _: "/tmp"
    libero_envs = types.ModuleType("libero.libero.envs")
    libero_envs.OffScreenRenderEnv = object

    image_tools = types.SimpleNamespace(
        convert_to_uint8=lambda image: image,
        resize_with_pad=lambda image, _height, _width: image,
    )
    websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=object)
    openpi_client = types.ModuleType("openpi_client")
    openpi_client.image_tools = image_tools
    openpi_client.websocket_client_policy = websocket_client_policy

    stubs = {
        spec.name: module,
        "libero": libero_package,
        "libero.libero": libero_module,
        "libero.libero.envs": libero_envs,
        "openpi_client": openpi_client,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return module


libero_eval = _load_eval_module()


def _run_header(
    *,
    run_id="checkpoint-30000",
    mode="standard",
    suite="libero_spatial",
    num_tasks=10,
    task_start=0,
    task_end=None,
    trials=50,
    manifest="a" * 64,
    benchmark_revision="test-benchmark-revision",
    policy_reconnect_attempts=2,
    policy_inference_timeout_seconds=300.0,
):
    classification_sha256 = "b" * 64 if mode == "plus" else None
    common_config = {
        "run_id": run_id,
        "benchmark_mode": mode,
        "benchmark_revision": benchmark_revision,
        "evaluation_protocol_version": libero_eval._EVALUATION_PROTOCOL_VERSION,
        "episode_seed_scheme": libero_eval._EPISODE_SEED_SCHEME,
        "episode_seed_scheme_version": libero_eval._EPISODE_SEED_SCHEME_VERSION,
        "policy_seed_scheme": libero_eval._POLICY_SEED_SCHEME,
        "policy_seed_scheme_version": libero_eval._POLICY_SEED_SCHEME_VERSION,
        "seed": 7,
        "resize_size": 224,
        "replan_steps": 5,
        "num_steps_wait": 10,
        "num_trials_per_task": trials,
        "environment_resolution": 256,
        "policy_reconnect_attempts": policy_reconnect_attempts,
        "policy_reconnect_backoff_seconds": 1.0,
        "max_consecutive_policy_errors": 3,
        "policy_connect_timeout_seconds": 30.0,
        "policy_connect_retry_interval_seconds": 1.0,
        "policy_inference_timeout_seconds": policy_inference_timeout_seconds,
        "policy_use_proxy": False,
        "classification_sha256": classification_sha256,
    }
    run_config = {
        **common_config,
        "task_suite_name": suite,
        "num_tasks": num_tasks,
        "task_start": task_start,
        "task_end": num_tasks if task_end is None else task_end,
        "max_steps": libero_eval._get_max_steps(suite),
        "task_manifest_fingerprint": manifest,
        "task_group_metadata": (
            [{"category": "Sensor Noise", "difficulty_level": None} for _ in range(num_tasks)]
            if mode == "plus"
            else None
        ),
    }
    return {
        "schema_version": libero_eval._RESULT_SCHEMA_VERSION,
        "record_type": "run",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "evaluation_fingerprint": libero_eval._fingerprint(common_config),
        "run_fingerprint": libero_eval._fingerprint(run_config),
        "run_config": run_config,
    }


def _record(
    status,
    task_id=0,
    episode_idx=0,
    description="raw prompt",
    seed=7,
    *,
    header=None,
    suite="libero_spatial",
    category=None,
    difficulty_level=None,
):
    header = header or _run_header(suite=suite)
    return {
        "schema_version": libero_eval._RESULT_SCHEMA_VERSION,
        "record_type": "episode",
        "run_fingerprint": header["run_fingerprint"],
        "timestamp_utc": "2026-07-13T00:00:01+00:00",
        "task_suite_name": suite,
        "task_id": task_id,
        "task_name": f"task_{task_id}",
        "task_description": description,
        "category": category,
        "difficulty_level": difficulty_level,
        "episode_idx": episode_idx,
        "seed": seed,
        "episode_seed": libero_eval._derive_episode_seed(seed, suite, task_id, episode_idx),
        "status": status,
        "success": status == "success",
        "num_steps": 0,
        "max_steps": header["run_config"]["max_steps"],
        "duration_seconds": 0.0,
        "error": {"stage": "test", "type": "RuntimeError", "message": "test"} if status == "error" else None,
        "video_path": None,
        "video_error": None,
    }


class TaskSelectionTest(unittest.TestCase):
    def test_defaults_preserve_standard_libero_protocol(self):
        args = libero_eval.Args()

        self.assertEqual(args.num_trials_per_task, 50)
        self.assertEqual(args.task_start, 0)
        self.assertIsNone(args.task_end)
        self.assertEqual(args.num_task_shards, 1)
        self.assertEqual(args.task_shard_id, 0)
        self.assertEqual(args.benchmark_mode, "standard")
        self.assertIsNone(args.benchmark_revision)
        self.assertIsNone(args.run_id)
        self.assertEqual(args.policy_connect_timeout_seconds, 30.0)
        self.assertEqual(args.policy_inference_timeout_seconds, 300.0)
        self.assertFalse(args.policy_use_proxy)
        self.assertTrue(args.save_video)
        self.assertIsNone(args.results_path)

    def test_range_and_strided_shard_are_stable(self):
        self.assertEqual(libero_eval._select_task_ids(12, 2, 11, 3, 1), [4, 7, 10])
        self.assertEqual(libero_eval._select_task_ids(4, 0, None, 1, 0), [0, 1, 2, 3])

    def test_invalid_range_or_shard_fails_early(self):
        invalid_arguments = [
            (10, -1, None, 1, 0),
            (10, 0, 11, 1, 0),
            (10, 5, 4, 1, 0),
            (10, 0, None, 0, 0),
            (10, 0, None, 2, 2),
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                libero_eval._select_task_ids(*arguments)

    def test_tyro_help_exposes_large_scale_options(self):
        script = """
import runpy
import sys
import types

libero_package = types.ModuleType("libero")
libero_module = types.ModuleType("libero.libero")
libero_module.benchmark = types.SimpleNamespace(get_benchmark_dict=dict)
libero_module.get_libero_path = lambda _: "/tmp"
libero_envs = types.ModuleType("libero.libero.envs")
libero_envs.OffScreenRenderEnv = object
openpi_client = types.ModuleType("openpi_client")
openpi_client.image_tools = types.SimpleNamespace()
openpi_client.websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=object)
sys.modules.update({
    "libero": libero_package,
    "libero.libero": libero_module,
    "libero.libero.envs": libero_envs,
    "openpi_client": openpi_client,
})
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, str(pathlib.Path(__file__).with_name("main.py"))],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        help_text = completed.stdout + completed.stderr
        for option in (
            "--args.task-start",
            "--args.task-end",
            "--args.num-task-shards",
            "--args.task-shard-id",
            "--args.num-trials-per-task",
            "--args.results-path",
            "--args.no-save-video",
            "--args.retry-errors",
            "--args.run-id",
            "--args.benchmark-mode",
            "--args.benchmark-revision",
            "--args.policy-connect-timeout-seconds",
            "--args.policy-connect-retry-interval-seconds",
            "--args.merge-results-paths",
            "--args.summarize-results-paths",
        ):
            self.assertIn(option, help_text)


class ProtocolValidationTest(unittest.TestCase):
    def test_plus_requires_exact_suite_size_and_one_trial(self):
        args = libero_eval.Args(
            benchmark_mode="plus",
            task_suite_name="libero_spatial",
            num_trials_per_task=1,
        )
        libero_eval._validate_benchmark_protocol(args, "plus", 2402)

        with self.assertRaisesRegex(ValueError, "must contain 2402 tasks"):
            libero_eval._validate_benchmark_protocol(args, "plus", 10)
        args.num_trials_per_task = 50
        with self.assertRaisesRegex(ValueError, "exactly one trial"):
            libero_eval._validate_benchmark_protocol(args, "plus", 2402)

    def test_plus_classification_matches_task_order_and_preserves_prompt(self):
        class Task:
            def __init__(self, task_id):
                self.name = f"plus_task_{task_id}"
                self.language = f"original prompt / perturbation_{task_id}"

        class Suite:
            def get_task(self, task_id):
                return Task(task_id)

        entries = [
            {
                "id": task_id + 1,
                "name": f"plus_task_{task_id}",
                "category": "Sensor Noise",
                "difficulty_level": task_id + 1 if task_id == 0 else None,
            }
            for task_id in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "task_classification.json"
            path.write_text(json.dumps({"libero_spatial": entries}), encoding="utf-8")
            args = libero_eval.Args(
                benchmark_mode="plus",
                task_suite_name="libero_spatial",
                num_trials_per_task=1,
                classification_path=str(path),
            )
            task_infos = libero_eval._build_task_infos(args, "plus", Suite(), 2)

        self.assertEqual(task_infos[1].description, "original prompt / perturbation_1")
        self.assertEqual(task_infos[1].category, "Sensor Noise")
        self.assertIsNone(task_infos[1].difficulty_level)

    def test_durable_results_require_benchmark_revision(self):
        args = libero_eval.Args(run_id="checkpoint", results_path="results.jsonl")
        with self.assertRaisesRegex(ValueError, "benchmark_revision is required"):
            libero_eval._validate_eval_args(args, "standard")

        args.benchmark_revision = "libero-commit-abc123"
        libero_eval._validate_eval_args(args, "standard")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "results.jsonl"
            with self.assertRaisesRegex(ValueError, "no benchmark_revision"):
                libero_eval._EpisodeJournal(
                    str(path),
                    run_header=_run_header(benchmark_revision=None),
                    resume=True,
                    retry_errors=False,
                )


class ProvenanceAndSeedTest(unittest.TestCase):
    def test_manifest_hashes_resolved_artifact_contents_and_classification(self):
        class Task:
            name = "task_view_0_0_100_0_0_initstate_1"
            language = "verbatim prompt"
            problem_folder = "libero_spatial"
            bddl_file = f"{name}.bddl"
            init_states_file = f"{name}.pruned_init"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            bddl_root = root / "bddl_files"
            init_root = root / "init_states"
            benchmark_root = root / "benchmark_root"
            bddl_path = bddl_root / "libero_spatial" / "task.bddl"
            init_path = init_root / "libero_spatial" / "task.pruned_init"
            classification_path = benchmark_root / "benchmark" / "task_classification.json"
            for path in (bddl_path, init_path, classification_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            bddl_path.write_text("bddl-v1", encoding="utf-8")
            init_path.write_bytes(b"init-v1")
            classification_path.write_text('{"libero_spatial": []}', encoding="utf-8")

            paths = {
                "bddl_files": str(bddl_root),
                "init_states": str(init_root),
                "benchmark_root": str(benchmark_root),
            }
            args = libero_eval.Args(
                run_id="checkpoint",
                benchmark_mode="plus",
                benchmark_revision="plus-commit",
                task_suite_name="libero_spatial",
                num_trials_per_task=1,
            )
            task_infos = [
                libero_eval._TaskInfo(
                    task_id=0,
                    task=Task(),
                    name=Task.name,
                    description=Task.language,
                    category="Sensor Noise",
                    difficulty_level=None,
                )
            ]
            with mock.patch.object(libero_eval, "get_libero_path", side_effect=paths.__getitem__):
                first = libero_eval._make_run_header(args, "plus", task_infos, 1, 300)
                bddl_path.write_text("bddl-v2", encoding="utf-8")
                second = libero_eval._make_run_header(args, "plus", task_infos, 1, 300)
                classification_path.write_text('{"libero_spatial": []}\n', encoding="utf-8")
                third = libero_eval._make_run_header(args, "plus", task_infos, 1, 300)

        self.assertNotEqual(
            first["run_config"]["task_manifest_fingerprint"],
            second["run_config"]["task_manifest_fingerprint"],
        )
        self.assertNotEqual(first["run_fingerprint"], second["run_fingerprint"])
        self.assertNotEqual(second["evaluation_fingerprint"], third["evaluation_fingerprint"])
        self.assertEqual(
            first["run_config"]["classification_sha256"],
            hashlib.sha256(b'{"libero_spatial": []}').hexdigest(),
        )

    def test_policy_retry_settings_and_benchmark_revision_change_fingerprints(self):
        first = _run_header(policy_reconnect_attempts=1)
        attempts_changed = _run_header(policy_reconnect_attempts=2)
        revision_changed = _run_header(benchmark_revision="different-revision", policy_reconnect_attempts=1)
        inference_timeout_changed = _run_header(
            policy_reconnect_attempts=1,
            policy_inference_timeout_seconds=600.0,
        )

        self.assertNotEqual(first["evaluation_fingerprint"], attempts_changed["evaluation_fingerprint"])
        self.assertNotEqual(first["run_fingerprint"], attempts_changed["run_fingerprint"])
        self.assertNotEqual(first["evaluation_fingerprint"], revision_changed["evaluation_fingerprint"])
        self.assertNotEqual(first["evaluation_fingerprint"], inference_timeout_changed["evaluation_fingerprint"])

    def test_episode_seed_is_stable_and_strictly_validated(self):
        first = libero_eval._derive_episode_seed(7, "libero_spatial", 3, 4)
        self.assertEqual(first, libero_eval._derive_episode_seed(7, "libero_spatial", 3, 4))
        self.assertNotEqual(first, libero_eval._derive_episode_seed(7, "libero_spatial", 3, 5))
        self.assertNotEqual(first, libero_eval._derive_episode_seed(7, "libero_goal", 3, 4))

        header = _run_header()
        record = _record("success", header=header)
        record["episode_seed"] += 1
        with self.assertRaisesRegex(ValueError, "invalid episode_seed"):
            libero_eval._validate_episode_record(record, header)

        policy_seed = libero_eval._derive_policy_seed(first, 0)
        self.assertEqual(policy_seed, libero_eval._derive_policy_seed(first, 0))
        self.assertNotEqual(policy_seed, libero_eval._derive_policy_seed(first, 1))


class EpisodeJournalTest(unittest.TestCase):
    def test_jsonl_resume_uses_latest_record_and_can_retry_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "nested" / "episodes.jsonl"
            header = _run_header()
            with libero_eval._EpisodeJournal(str(path), run_header=header, resume=True, retry_errors=False) as journal:
                journal.append(_record("success", episode_idx=0, description="prompt / unchanged", header=header))
                journal.append(_record("error", episode_idx=1, header=header))

            with path.open("ab") as results_file:
                results_file.write(b'{"incomplete":"\xe4')

            with libero_eval._EpisodeJournal(str(path), run_header=header, resume=True, retry_errors=False) as resumed:
                self.assertTrue(
                    resumed.should_skip(
                        "libero_spatial", 0, 0, task_name="task_0", task_description="prompt / unchanged", seed=7
                    )
                )
                self.assertTrue(
                    resumed.should_skip(
                        "libero_spatial", 0, 1, task_name="task_0", task_description="raw prompt", seed=7
                    )
                )
                self.assertEqual(resumed.records[("libero_spatial", 0, 0)]["task_description"], "prompt / unchanged")

            with libero_eval._EpisodeJournal(str(path), run_header=header, resume=True, retry_errors=True) as retrying:
                self.assertTrue(
                    retrying.should_skip(
                        "libero_spatial", 0, 0, task_name="task_0", task_description="prompt / unchanged", seed=7
                    )
                )
                self.assertFalse(
                    retrying.should_skip(
                        "libero_spatial", 0, 1, task_name="task_0", task_description="raw prompt", seed=7
                    )
                )
                retrying.append(_record("success", episode_idx=1, header=header))

            latest = libero_eval._load_episode_records(path)
            self.assertEqual(latest[("libero_spatial", 0, 1)]["status"], "success")

    def test_resume_rejects_stale_task_identity(self):
        header = _run_header()
        journal = libero_eval._EpisodeJournal(None, run_header=header, resume=True, retry_errors=False)
        journal.append(_record("success", description="old prompt", header=header))

        self.assertFalse(
            journal.should_skip(
                "libero_spatial",
                0,
                0,
                task_name="task_0",
                task_description="new prompt",
            )
        )
        self.assertFalse(journal.should_skip("libero_spatial", 0, 0, seed=8))

    def test_no_resume_starts_a_fresh_journal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "episodes.jsonl"
            header = _run_header()
            libero_eval._write_complete_journal_atomic(
                path, header, [_record("success", header=header)], overwrite=True
            )

            with libero_eval._EpisodeJournal(str(path), run_header=header, resume=False, retry_errors=False) as journal:
                self.assertEqual(journal.records, {})

            loaded_header, records = libero_eval._load_journal(path)
            self.assertEqual(loaded_header["run_fingerprint"], header["run_fingerprint"])
            self.assertEqual(records, {})

    def test_resume_rejects_old_schema_or_different_run_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "episodes.jsonl"
            path.write_text(json.dumps({"schema_version": 1, "record_type": "episode"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run header version"):
                libero_eval._EpisodeJournal(str(path), run_header=_run_header(), resume=True, retry_errors=False)

            header = _run_header()
            libero_eval._write_complete_journal_atomic(path, header, [], overwrite=True)
            with self.assertRaisesRegex(ValueError, "run fingerprint differs"):
                libero_eval._EpisodeJournal(
                    str(path),
                    run_header=_run_header(run_id="different-checkpoint"),
                    resume=True,
                    retry_errors=False,
                )

            reconnect_header = _run_header(policy_reconnect_attempts=1)
            libero_eval._write_complete_journal_atomic(path, reconnect_header, [], overwrite=True)
            with self.assertRaisesRegex(ValueError, "run fingerprint differs"):
                libero_eval._EpisodeJournal(
                    str(path),
                    run_header=_run_header(policy_reconnect_attempts=2),
                    resume=True,
                    retry_errors=False,
                )

    def test_same_journal_has_single_process_owner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "episodes.jsonl"
            header = _run_header()
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    libero_eval._EpisodeJournal(str(path), run_header=header, resume=True, retry_errors=False)
                )
                stack.enter_context(self.assertRaisesRegex(RuntimeError, "already owned"))
                libero_eval._EpisodeJournal(str(path), run_header=header, resume=True, retry_errors=False)

    def test_multi_shard_paths_are_independent(self):
        paths = [libero_eval._resolve_shard_results_path("/tmp/results.jsonl", 3, shard_id) for shard_id in range(3)]
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(paths[1].name, "results.shard-00001-of-00003.jsonl")

    def test_strict_merger_accepts_disjoint_shards_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            header = _run_header(trials=1)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            merged = root / "merged.jsonl"
            libero_eval._write_complete_journal_atomic(
                first, header, [_record("success", task_id=0, header=header)], overwrite=True
            )
            libero_eval._write_complete_journal_atomic(
                second, header, [_record("failure", task_id=1, header=header)], overwrite=True
            )

            libero_eval._merge_episode_journals([first, second], merged, overwrite=False)
            self.assertEqual(len(libero_eval._load_episode_records(merged)), 2)

            duplicate = root / "duplicate.jsonl"
            libero_eval._write_complete_journal_atomic(
                duplicate, header, [_record("failure", task_id=0, header=header)], overwrite=True
            )
            with self.assertRaisesRegex(ValueError, "Duplicate episode"):
                libero_eval._merge_episode_journals([first, duplicate], root / "bad.jsonl", overwrite=False)

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    libero_eval._EpisodeJournal(str(first), run_header=header, resume=True, retry_errors=False)
                )
                stack.enter_context(self.assertRaisesRegex(RuntimeError, "still writing"))
                libero_eval._merge_episode_journals([first, second], root / "live.jsonl", overwrite=False)


class ReportingTest(unittest.TestCase):
    def test_category_macro_and_micro_rates_remain_distinct(self):
        groups = {
            "large": {"episodes": 9, "successes": 9, "success_rate": 1.0},
            "small": {"episodes": 1, "successes": 0, "success_rate": 0.0},
        }

        macro_rate = libero_eval._macro_group_success_rate(groups, expected_groups=("large", "small"))
        micro_rate = sum(group["successes"] for group in groups.values()) / sum(
            group["episodes"] for group in groups.values()
        )

        self.assertEqual(macro_rate, 0.5)
        self.assertEqual(micro_rate, 0.9)
        self.assertIsNone(libero_eval._macro_group_success_rate(groups, expected_groups=("large", "missing")))

    def test_summary_counts_failures_errors_and_pending_per_task(self):
        records = {
            ("libero_spatial", 0, 0): _record(
                "success", task_id=0, episode_idx=0, category="Camera Viewpoints", difficulty_level=1
            ),
            ("libero_spatial", 0, 1): _record(
                "failure", task_id=0, episode_idx=1, category="Camera Viewpoints", difficulty_level=1
            ),
            ("libero_spatial", 1, 0): _record(
                "error", task_id=1, episode_idx=0, category="Sensor Noise", difficulty_level=2
            ),
        }

        summary = libero_eval._summarize_records(records, "libero_spatial", [0, 1], 2)

        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["pending"], 1)
        self.assertAlmostEqual(summary["success_rate"], 1 / 3)
        self.assertEqual(summary["tasks"][0]["success_rate"], 0.5)
        self.assertEqual(summary["tasks"][1]["pending"], 1)
        self.assertEqual(summary["categories"]["Camera Viewpoints"]["episodes"], 2)
        self.assertEqual(summary["categories"]["Sensor Noise"]["errors"], 1)
        self.assertEqual(summary["difficulty_levels"][1]["successes"], 1)

    def test_null_plus_difficulty_is_reported_as_unknown(self):
        records = {
            ("libero_goal", 0, 0): _record(
                "success",
                task_id=0,
                episode_idx=0,
                header=_run_header(mode="plus", suite="libero_goal", num_tasks=1, trials=1),
                suite="libero_goal",
                category="Sensor Noise",
                difficulty_level=None,
            )
        }

        summary = libero_eval._summarize_records(records, "libero_goal", [0], 1)

        self.assertEqual(summary["difficulty_levels"]["unknown"]["episodes"], 1)
        self.assertEqual(summary["difficulty_levels"]["unknown"]["successes"], 1)

    def test_pending_plus_task_uses_header_group_metadata(self):
        task_groups = [{"category": "Light Conditions", "difficulty_level": None}]

        summary = libero_eval._summarize_records(
            {},
            "libero_goal",
            [0],
            1,
            task_groups,
        )

        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["categories"]["Light Conditions"]["pending"], 1)
        self.assertEqual(summary["difficulty_levels"]["unknown"]["pending"], 1)

    def test_video_names_are_safe_and_unique(self):
        description = "../pick the mug / perturbation suffix? beta"
        run_fingerprint = "a" * 64
        first = libero_eval._get_video_path(
            "/tmp/videos",
            "libero/spatial",
            7,
            0,
            description,
            run_fingerprint=run_fingerprint,
            status="success",
        )
        second = libero_eval._get_video_path(
            "/tmp/videos",
            "libero/spatial",
            7,
            1,
            description,
            run_fingerprint=run_fingerprint,
            status="success",
        )

        self.assertEqual(first.parent, pathlib.Path("/tmp/videos/run-aaaaaaaaaaaaaaaa"))
        self.assertNotEqual(first, second)
        self.assertNotIn("/", first.name)
        self.assertNotIn("..", first.name)
        self.assertIn("task-00007", first.name)
        self.assertIn("episode-000", first.name)

    def test_cross_suite_plus_summary_is_weighted_micro_average(self):
        suite_names = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        statuses = ("success", "failure", "success", "success")
        with contextlib.ExitStack() as stack:
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(
                mock.patch.dict(
                    libero_eval._LIBERO_PLUS_TASK_COUNTS,
                    dict.fromkeys(suite_names, 1),
                    clear=True,
                )
            )
            paths = []
            for index, suite_name in enumerate(suite_names):
                status = statuses[index]
                header = _run_header(
                    mode="plus",
                    suite=suite_name,
                    num_tasks=1,
                    trials=1,
                )
                record = _record(
                    status,
                    header=header,
                    suite=suite_name,
                    category="Camera Viewpoints",
                    difficulty_level=1,
                )
                path = pathlib.Path(tmpdir) / f"{suite_name}.jsonl"
                libero_eval._write_complete_journal_atomic(path, header, [record], overwrite=True)
                paths.append(path)

            summary = libero_eval._summarize_result_journals(paths)

        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["successes"], 3)
        self.assertEqual(summary["success_rate"], 0.75)
        self.assertEqual(summary["categories"]["Camera Viewpoints"]["episodes"], 4)
        self.assertTrue(summary["protocol_complete"])
        self.assertTrue(summary["official"])

    def test_aggregate_is_incomplete_when_suites_are_missing(self):
        with contextlib.ExitStack() as stack:
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(
                mock.patch.dict(
                    libero_eval._LIBERO_PLUS_TASK_COUNTS,
                    {
                        "libero_spatial": 1,
                        "libero_object": 1,
                        "libero_goal": 1,
                        "libero_10": 1,
                    },
                    clear=True,
                )
            )
            header = _run_header(mode="plus", num_tasks=1, trials=1)
            path = pathlib.Path(tmpdir) / "spatial.jsonl"
            libero_eval._write_complete_journal_atomic(
                path,
                header,
                [_record("success", header=header, category="Sensor Noise", difficulty_level=None)],
                overwrite=True,
            )
            summary = libero_eval._summarize_result_journals([path])

        self.assertFalse(summary["protocol_complete"])
        self.assertFalse(summary["official"])

    def test_cross_suite_summary_rejects_different_benchmark_revisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_header = _run_header(mode="plus", num_tasks=1, trials=1, benchmark_revision="revision-a")
            second_header = _run_header(
                mode="plus",
                suite="libero_object",
                num_tasks=1,
                trials=1,
                benchmark_revision="revision-b",
            )
            first_path = pathlib.Path(tmpdir) / "first.jsonl"
            second_path = pathlib.Path(tmpdir) / "second.jsonl"
            libero_eval._write_complete_journal_atomic(
                first_path,
                first_header,
                [_record("success", header=first_header, category="Sensor Noise", difficulty_level=1)],
                overwrite=True,
            )
            libero_eval._write_complete_journal_atomic(
                second_path,
                second_header,
                [
                    _record(
                        "success",
                        header=second_header,
                        suite="libero_object",
                        category="Sensor Noise",
                        difficulty_level=1,
                    )
                ],
                overwrite=True,
            )

            with self.assertRaisesRegex(ValueError, "benchmark revision differs"):
                libero_eval._summarize_result_journals([first_path, second_path])


class RolloutBehaviorTest(unittest.TestCase):
    @staticmethod
    def _observation():
        return {
            "agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "robot0_eye_in_hand_image": np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.4, 0.5]),
        }

    def test_prompt_is_passed_verbatim_and_action_is_unchanged(self):
        observation = self._observation()

        class Env:
            def __init__(self):
                self.action = None
                self.seed_value = None

            def seed(self, seed):
                self.seed_value = seed

            def reset(self):
                return None

            def set_init_state(self, _initial_state):
                return observation

            def step(self, action):
                self.action = action
                return observation, 0.0, True, {}

        class Client:
            def __init__(self):
                self.element = None

            def infer(self, element, *, seed):
                self.element = element
                self.seed = seed
                return {"actions": np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]])}

        args = libero_eval.Args(num_steps_wait=0, replan_steps=1, resize_size=2, save_video=False)
        env = Env()
        client = Client()
        prompt = "pick mug / language_perturbation__suffix.json"

        outcome = libero_eval._run_episode(
            args,
            env,
            client,
            prompt,
            initial_state=object(),
            max_steps=1,
            episode_seed=123,
        )

        self.assertTrue(outcome.success)
        self.assertIsNone(outcome.error)
        self.assertEqual(client.element["prompt"], prompt)
        self.assertEqual(client.seed, libero_eval._derive_policy_seed(123, 0))
        np.testing.assert_array_equal(client.element["observation/image"], observation["agentview_image"][::-1, ::-1])
        self.assertEqual(env.action, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0])
        self.assertEqual(env.seed_value, 123)
        self.assertEqual(outcome.replay_images, [])

    def test_episode_exception_is_structured(self):
        observation = self._observation()

        class Env:
            def seed(self, _seed):
                pass

            def reset(self):
                return None

            def set_init_state(self, _initial_state):
                return observation

        class Client:
            def infer(self, _element, *, seed):
                del seed
                raise ConnectionError("server disconnected")

        args = libero_eval.Args(num_steps_wait=0, replan_steps=1, save_video=False)
        outcome = libero_eval._run_episode(
            args,
            Env(),
            Client(),
            "raw prompt",
            object(),
            max_steps=1,
            episode_seed=456,
        )

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.error["stage"], "policy_inference")
        self.assertEqual(outcome.error["type"], "ConnectionError")
        self.assertEqual(outcome.error["message"], "server disconnected")

    def test_policy_disconnect_invalidates_client_and_retries_episode(self):
        observation = self._observation()

        class Env:
            def __init__(self):
                self.seeds = []
                self.random_values = []

            def seed(self, seed):
                self.seeds.append(seed)

            def reset(self):
                self.random_values.append(int(np.random.randint(0, 2**31)))

            def set_init_state(self, _initial_state):
                return observation

            def step(self, _action):
                return observation, 0.0, True, {}

        class FailingClient:
            def __init__(self, _host, _port, **_kwargs):
                self.closed = False

            def infer(self, _element, *, seed):
                self.seed = seed
                raise ConnectionError("disconnected")

            def close(self):
                self.closed = True

        class SuccessfulClient:
            def __init__(self, _host, _port, **_kwargs):
                pass

            def infer(self, _element, *, seed):
                self.seed = seed
                return {"actions": np.array([[0.0] * 7])}

        failing_client = FailingClient("host", 1)
        successful_client = SuccessfulClient("host", 1)
        clients = iter([failing_client, successful_client])
        args = libero_eval.Args(
            num_steps_wait=0,
            replan_steps=1,
            save_video=False,
            policy_reconnect_attempts=1,
            policy_reconnect_backoff_seconds=0,
        )

        env = Env()
        episode_seed = libero_eval._derive_episode_seed(7, "libero_spatial", 0, 0)
        with mock.patch.object(
            libero_eval._websocket_client_policy,
            "WebsocketClientPolicy",
            side_effect=lambda _host, _port, **_kwargs: next(clients),
        ):
            outcome, client = libero_eval._run_episode_with_policy_retries(
                args,
                env,
                None,
                "raw prompt",
                object(),
                max_steps=1,
                episode_seed=episode_seed,
            )

        self.assertTrue(outcome.success)
        self.assertIs(client, successful_client)
        self.assertTrue(failing_client.closed)
        expected_policy_seed = libero_eval._derive_policy_seed(episode_seed, 0)
        self.assertEqual(failing_client.seed, expected_policy_seed)
        self.assertEqual(successful_client.seed, expected_policy_seed)
        self.assertEqual(env.seeds, [episode_seed, episode_seed])
        self.assertEqual(env.random_values[0], env.random_values[1])

    def test_libero_env_receives_string_bddl_path(self):
        class Task:
            language = "prompt"
            problem_folder = "libero_goal"
            bddl_file = "task.bddl"

        class Env:
            def seed(self, _seed):
                pass

        env = Env()
        with contextlib.ExitStack() as stack:
            constructor = stack.enter_context(mock.patch.object(libero_eval, "OffScreenRenderEnv", return_value=env))
            stack.enter_context(mock.patch.object(libero_eval, "get_libero_path", return_value="/tmp/bddl"))
            returned_env, description = libero_eval._get_libero_env(Task(), 256, 7)

        self.assertIs(returned_env, env)
        self.assertEqual(description, "prompt")
        self.assertIsInstance(constructor.call_args.kwargs["bddl_file_name"], str)
        self.assertEqual(constructor.call_args.kwargs["bddl_file_name"], "/tmp/bddl/libero_goal/task.bddl")

    def test_eval_records_client_setup_error_and_closes_environment(self):
        observation = self._observation()

        class Task:
            name = "task_0"
            language = "raw plus prompt / suffix"
            problem_folder = "libero_spatial"
            bddl_file = "task.bddl"
            init_states_file = "task.pruned_init"

        class Suite:
            n_tasks = 10

            def get_task(self, _task_id):
                return Task()

            def get_task_init_states(self, _task_id):
                return [object()]

        class Env:
            def __init__(self):
                self.closed = False

            def reset(self):
                return None

            def seed(self, _seed):
                pass

            def set_init_state(self, _initial_state):
                return observation

            def step(self, _action):
                return observation, 0.0, True, {}

            def close(self):
                self.closed = True

        class Client:
            def __init__(self, _host, _port, **_kwargs):
                raise ConnectionError("client setup failed")

        env = Env()
        suite = Suite()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = pathlib.Path(tmpdir) / "episodes.jsonl"
            bddl_root = pathlib.Path(tmpdir) / "bddl_files"
            init_root = pathlib.Path(tmpdir) / "init_states"
            bddl_file = bddl_root / "libero_spatial" / Task.bddl_file
            init_file = init_root / "libero_spatial" / Task.init_states_file
            bddl_file.parent.mkdir(parents=True)
            init_file.parent.mkdir(parents=True)
            bddl_file.write_text("fixture", encoding="utf-8")
            init_file.write_bytes(b"fixture")
            args = libero_eval.Args(
                run_id="checkpoint-30000",
                benchmark_revision="standard-fixture-revision",
                num_steps_wait=0,
                num_trials_per_task=1,
                task_end=2,
                replan_steps=1,
                save_video=False,
                results_path=str(results_path),
                policy_reconnect_attempts=0,
                max_consecutive_policy_errors=1,
            )
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        libero_eval.benchmark,
                        "get_benchmark_dict",
                        return_value={"libero_spatial": lambda: suite},
                    )
                )
                stack.enter_context(
                    mock.patch.object(libero_eval, "_get_libero_env", return_value=(env, Task.language))
                )
                stack.enter_context(
                    mock.patch.object(
                        libero_eval,
                        "get_libero_path",
                        side_effect={
                            "bddl_files": str(bddl_root),
                            "init_states": str(init_root),
                        }.__getitem__,
                    )
                )
                stack.enter_context(
                    mock.patch.object(libero_eval._websocket_client_policy, "WebsocketClientPolicy", Client)
                )
                stack.enter_context(mock.patch.object(libero_eval.tqdm, "tqdm", side_effect=lambda iterable: iterable))
                libero_eval.eval_libero(args)

            records = libero_eval._load_episode_records(results_path)
            record = records[("libero_spatial", 0, 0)]

        self.assertTrue(env.closed)
        self.assertEqual(len(records), 1)
        self.assertNotIn(("libero_spatial", 1, 0), records)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"]["stage"], "policy_client_setup")


if __name__ == "__main__":
    unittest.main()
