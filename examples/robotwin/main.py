"""Run the official RoboTwin 2.0 XPolicyLab evaluator against an OpenPI server."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

from adapter import OpenPIModelClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--task-name")
    tasks.add_argument("--tasks-file", type=Path, help="One RoboTwin task name per line")
    parser.add_argument("--task-config", default="demo_clean", choices=("demo_clean", "demo_randomized"))
    parser.add_argument("--results-root", type=Path, default=Path("data/robotwin_eval"))
    parser.add_argument("--checkpoint-id", required=True, help="Label used in result paths; use a unique checkpoint identifier")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"))
    parser.add_argument("--no-expert-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    root = args.robotwin_root.expanduser().resolve()
    evaluator_path = root / "scripts" / "eval_policy_xpolicylab.py"
    if not evaluator_path.is_file():
        parser.error(f"RoboTwin 2.0 XPolicyLab evaluator not found: {evaluator_path}")
    if args.test_num <= 0:
        parser.error("--test-num must be positive")
    if args.tasks_file is None:
        task_names = [args.task_name]
    else:
        task_names = [
            line.strip() for line in args.tasks_file.expanduser().read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not task_names or len(task_names) != len(set(task_names)):
            parser.error("--tasks-file must contain at least one task and no duplicates")
    results_root = args.results_root.expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    # The simulator loads embodiment assets relative to its repository root.
    os.chdir(root)
    for path in (root, root / "scripts", root / "description" / "utils"):
        sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("robotwin_official_eval", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {evaluator_path}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.build_policy_client = lambda _: OpenPIModelClient(
        args.host, args.port, args.replan_steps, timeout=args.timeout
    )
    official_save_dir = evaluator.build_eval_save_dir
    evaluator.build_eval_save_dir = lambda *values: results_root / official_save_dir(*values)

    eval_args = {
        "task_config": args.task_config,
        "policy_name": "OpenPI",
        "ckpt_setting": args.checkpoint_id,
        "action_type": "joint",
        "seed": args.seed,
        "test_num": args.test_num,
        "expert_check": not args.no_expert_check,
        "eval_batch": False,
    }
    if args.instruction_type is not None:
        eval_args["instruction_type"] = args.instruction_type
    for task_name in task_names:
        print(f"Evaluating RoboTwin task: {task_name}", flush=True)
        evaluator.main({**eval_args, "task_name": task_name})


if __name__ == "__main__":
    main()
