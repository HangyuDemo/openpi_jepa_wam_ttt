# LIBERO and LIBERO-Plus Rollout Evaluation

This repository includes a reproducible rollout evaluator for both the original LIBERO benchmark and LIBERO-Plus.
It is compatible with standard OpenPI checkpoints and with `pi05_libero_vjepa_aux` checkpoints trained with the
JEPA-WAM objective.

The evaluator is split into two processes:

```text
OpenPI environment                         Python 3.8 simulation environment
trained checkpoint                         LIBERO or LIBERO-Plus
        |                                           |
        v                 WebSocket                 v
policy server  <-------------------------->  rollout evaluator
                                                        |
                                                        v
                                           JSONL journal + optional videos
```

The V-JEPA teacher and precomputed targets are not used during rollout evaluation.

## Evaluation protocols

| Benchmark | Suites | Tasks | Trials per task | Total rollouts |
| --- | --- | ---: | ---: | ---: |
| LIBERO | Spatial, Object, Goal, Long | 10 per suite | 50 | 2,000 |
| LIBERO-Plus | Spatial, Object, Goal, Long | 2,402 / 2,518 / 2,591 / 2,519 | 1 | 10,030 |

The LIBERO-Plus protocol follows the official benchmark: every perturbed task has one associated initial state, and
`task_classification.json` maps each task to one of seven perturbation categories and an optional difficulty level.

The evaluator reports both of these cross-suite metrics:

- **Task-micro success rate**: total successes divided by all 10,030 task instances. This weights categories by their
  number of tasks and matches the `Total` aggregation used by the official LIBERO-Plus leaderboard.
- **Category-macro success rate**: the unweighted mean of the seven category success rates. This is the `Avg` used in
  the JEPA-WAM LIBERO-Plus table.

Both metrics, all seven category rates, and the available difficulty-level rates are printed so results cannot be
silently compared under different aggregation rules.

## 1. Install the simulation environments

Initialize the OpenPI submodules first:

```bash
git submodule update --init --recursive
```

For the original LIBERO environment, follow the unchanged upstream instructions in
[`examples/libero/README.md`](examples/libero/README.md).

LIBERO-Plus requires its replacement `libero` package, extra rendering dependencies, and a 6.4 GB asset archive.
On Ubuntu, install the system packages once:

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl libarchive-tools \
  libexpat1 libfontconfig1-dev libpython3-dev libmagickwand-dev
```

Then run the pinned installer:

```bash
bash scripts/setup_libero_plus.sh install
```

The installer:

- checks out `sylvestf/LIBERO-plus` at `4976dc30028e805ff8094b55501d532c48fec182`;
- downloads the official assets at Hugging Face revision `dd2bd61b7d9a6fef1abc52d606e983b41886a149`;
- verifies the archive SHA-256 before extraction;
- creates an isolated Python 3.8 environment at `examples/libero/.venv-plus`;
- installs LIBERO-Plus and this repository's `openpi-client` package;
- writes a local LIBERO path configuration under the ignored `data/` directory;
- verifies the exact four-suite task counts.

Source, downloads, assets, and generated configuration are stored under ignored paths and are not committed to Git.
Override `LIBERO_PLUS_ROOT`, `EVAL_ENV`, or `LIBERO_PLUS_DOWNLOAD_ROOT` when the data should live on another disk.

Recheck an existing installation without downloading or modifying it:

```bash
bash scripts/setup_libero_plus.sh check
```

## 2. Start the checkpoint server

Point `CHECKPOINT_DIR` at one concrete checkpoint step directory:

```bash
CHECKPOINT_DIR=/path/to/checkpoints/pi05_libero_vjepa_aux/experiment/30000 \
GPU_IDS=0 \
bash scripts/run_libero_policy_server.sh check
```

Start the server after checking the printed command:

```bash
CHECKPOINT_DIR=/path/to/checkpoints/pi05_libero_vjepa_aux/experiment/30000 \
GPU_IDS=0 \
PORT=8000 \
bash scripts/run_libero_policy_server.sh start
```

The launcher defaults to `CONFIG_NAME=pi05_libero_vjepa_aux`. Set `CONFIG_NAME=pi05_libero` to evaluate a baseline
π0.5 checkpoint.

## 3. Run a LIBERO-Plus smoke evaluation

Use a stable, human-readable `RUN_ID` that uniquely identifies the served checkpoint. The run ID, checkpoint-facing
protocol, benchmark commit, task manifest, seeds, and evaluator settings are included in the journal fingerprint.

Evaluate one perturbed task first:

```bash
RUN_ID=pi05-jepa-step30000 \
TASK_SUITE=libero_spatial \
TASK_START=0 \
TASK_END=1 \
EVAL_GPU=1 \
SAVE_VIDEO=1 \
bash scripts/run_libero_evaluation.sh plus
```

Add `DRY_RUN=1` to print and validate the resolved rollout command without creating an environment.

The default result is written to:

```text
data/libero-eval/pi05-jepa-step30000/plus-libero_spatial.jsonl
```

Re-running the same command resumes the journal and skips matching completed episodes. Structured error records are
also skipped by default; set `RETRY_ERRORS=1` to retry only those episodes.

## 4. Run the complete LIBERO-Plus benchmark

With the policy server still running, evaluate all four suites:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  RUN_ID=pi05-jepa-step30000 \
  TASK_SUITE="$suite" \
  EVAL_GPU=1 \
  SAVE_VIDEO=0 \
  bash scripts/run_libero_evaluation.sh plus
done
```

This performs all 10,030 official rollout instances. Video recording is disabled above because buffering and encoding
10,030 videos is expensive; every success, failure, or structured error is still written durably to JSONL.

## 5. Shard large suites

Task sharding is deterministic and strided. For example, to split one suite across eight simulation workers, launch
one process for each `TASK_SHARD_ID` from 0 through 7:

```bash
RUN_ID=pi05-jepa-step30000 \
TASK_SUITE=libero_spatial \
NUM_TASK_SHARDS=8 \
TASK_SHARD_ID=0 \
EVAL_GPU=1 \
bash scripts/run_libero_evaluation.sh plus
```

The base journal name is expanded automatically:

```text
plus-libero_spatial.shard-00000-of-00008.jsonl
...
plus-libero_spatial.shard-00007-of-00008.jsonl
```

Each journal has a lifetime file lock, so accidentally assigning two processes to the same shard fails immediately.

Merge the eight completed shards into one suite journal:

```bash
bash scripts/run_libero_evaluation.sh merge \
  data/libero-eval/pi05-jepa-step30000/plus-libero_spatial.jsonl \
  data/libero-eval/pi05-jepa-step30000/plus-libero_spatial.shard-*.jsonl
```

Merging rejects overlapping episodes, inconsistent run fingerprints, mixed benchmark revisions, and an existing
output unless `OVERWRITE_RESULTS=1` is explicitly set.

## 6. Produce the cross-suite summary

After all four suite journals are complete:

```bash
bash scripts/run_libero_evaluation.sh summary \
  data/libero-eval/pi05-jepa-step30000/plus-libero_spatial.jsonl \
  data/libero-eval/pi05-jepa-step30000/plus-libero_object.jsonl \
  data/libero-eval/pi05-jepa-step30000/plus-libero_goal.jsonl \
  data/libero-eval/pi05-jepa-step30000/plus-libero_10.jsonl
```

The summary reports:

- per-suite success rates;
- success rate for all seven perturbation categories;
- available difficulty-level success rates;
- task-micro and category-macro aggregate rates;
- failures, structured errors, and pending episodes;
- whether every suite is complete and satisfies the pinned protocol.

An official/complete result requires the exact four suite sizes, one trial per task, no pending tasks, no structured
errors, matching benchmark revisions, and full task ranges.

## 7. Run the original LIBERO benchmark

The same evaluator preserves the standard protocol defaults: 10 tasks and 50 initial states for each of Spatial,
Object, Goal, and Long.

```bash
RUN_ID=pi05-jepa-step30000 \
TASK_SUITE=libero_10 \
EVAL_GPU=1 \
SAVE_VIDEO=0 \
bash scripts/run_libero_evaluation.sh standard
```

Use the same four-suite loop and `summary` command as above, replacing the journal prefix `plus-` with `standard-`.

## Reproducibility and failure handling

The evaluator adds several protections beyond the minimal upstream rollout loop:

- deterministic episode seeds derived from run seed, suite, task, and episode;
- deterministic policy inference seeds, preserved across WebSocket reconnects;
- versioned seeded-inference capability negotiation between client and server;
- bounded server connection and inference timeouts;
- policy reconnect and episode retry without changing random samples;
- a circuit breaker after repeated policy failures;
- immutable run and evaluation fingerprints;
- benchmark, BDDL, init-state, and classification hashes in the run manifest;
- append-and-fsync JSONL episode records;
- atomic shard merge and read/write file locks;
- safe, unique video paths.

Use `python examples/libero/main.py --help` inside the evaluation environment for all lower-level controls.

## Tests

The evaluator and inference protocol are covered by:

```bash
uv run pytest \
  examples/libero/main_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
  packages/openpi-client/src/openpi_client/websocket_inference_protocol_test.py \
  src/openpi/policies/policy_test.py \
  src/openpi/serving/websocket_policy_server_test.py
```

The full environment smoke test additionally requires a working LIBERO-Plus installation, MuJoCo EGL, and an
available GPU.

## Attribution

LIBERO is maintained by the Lifelong Robot Learning project. LIBERO-Plus is maintained by its respective authors and
is distributed separately. This repository pins but does not redistribute either LIBERO-Plus source assets or robot
benchmark data; follow the upstream licenses and dataset terms.
