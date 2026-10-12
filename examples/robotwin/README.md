# RoboTwin 2.0 simulation evaluation

This adapter runs the installed RoboTwin 2.0 XPolicyLab evaluator and sends
policy requests to this repository's OpenPI websocket server. The evaluator
chooses task seeds and instructions, checks successes, and writes its standard
result file. The adapter maps its three RGB views and Aloha-Agilex 14D joint
state to the local Pi0.5 training inputs, then executes absolute joint targets
with RoboTwin's `joint`/`qpos` control.

Use the current RoboTwin checkout at `/home/ha865618/project/RoboTwin` (or pass
another root with `scripts/eval_policy_xpolicylab.py`). The older
`script/eval_policy.py` API used by Motus is different and is not the evaluator
used here. RoboTwin and OpenPI may run in separate Python environments on the
same host.

## Start the trained policy server

From this repository's root, in the OpenPI environment:

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_robotwin2_clean20_vjepa_aux \
  --policy.dir /absolute/path/to/checkpoints/pi05_robotwin2_clean20_vjepa_aux/experiment/step
```

Use the exact `--policy.config` used to train the checkpoint. Plain Pi0.5 and
JEPA auxiliary checkpoints both work if their checkpoint assets include the
matching `norm_stats.json`. The server reports its action horizon and, for TTT
models, its rollout stride. The client rejects a mismatched stride. Explicit
JEPA TTT checkpoints require online V-JEPA features and are rejected by this
adapter until an online encoder is supplied. Implicit TTT checkpoints are
supported, with per-episode policy reset over the websocket.

### What "online update" means for TTT

For a TTT checkpoint, inference does perform an inner gradient step: current
observation features supply K/V for a self-supervised MSE loss, and that step
changes the episode's fast-weight state before actions are read. This is
test-time training. It does not update the checkpoint's slow parameters, run
the training optimizer, or use simulator success labels. The default model
setting writes memory once per physical observation, on the first denoising
evaluation. The adapter resets fast weights at every episode boundary and
requires the trained observation stride. Plain Pi0.5 and Pi0.5 + JEPA
auxiliary checkpoints have no TTT memory update, so use those when measuring
a frozen-policy baseline. Run one evaluator per policy server: the server owns
one policy instance and its fast state is shared by its client connections.

## Run RoboTwin

In the RoboTwin simulation environment, install this repo's lightweight client
once (`pip install -e /absolute/path/to/openpi_jepawam/packages/openpi-client`),
then run from this repository's root:

```bash
python examples/robotwin/main.py \
  --robotwin-root /home/ha865618/project/RoboTwin \
  --task-name beat_block_hammer \
  --task-config demo_clean \
  --checkpoint-id clean20_vjepa_step15000 \
  --host 127.0.0.1 --port 8000 \
  --replan-steps 8 --test-num 100
```

For the single-task `pi05_robotwin2_beat_block_hammer` pilot, use its checkpoint
and change `--checkpoint-id` accordingly. Use `--task-config demo_randomized`
for the randomized split. Choose `--replan-steps` based on the intended policy
rollout cadence; for the current TTT configs it must be 8. The underlying
model predicts 50 actions and the adapter executes only the requested prefix,
then replans from fresh RGB and joint observations. The RoboTwin evaluator
still obtains observations between executed actions for its own bookkeeping.

To evaluate the same 20 Clean tasks selected by this repository's training
script, replace `--task-name beat_block_hammer` with
`--tasks-file examples/robotwin/clean20_tasks.txt`.

Results go under `data/robotwin_eval/<task>/OpenPI/<task_config>/<checkpoint_id>/`.
Pass `--results-root` to change this location. `--test-num` defaults to 100 and
the official expert seed filter is enabled; use `--no-expert-check` only when
intentionally changing that protocol. The evaluator uses the task config's
`eval_instruction` setting unless `--instruction-type` is supplied.

This adapter only supports the Aloha-Agilex 6+1+6+1 joint embodiment used by
the local RoboTwin datasets. It validates the state and policy action shape,
and fails if the model returns NaN or infinity.
