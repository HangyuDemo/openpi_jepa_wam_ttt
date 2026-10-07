# Temporal TTT in Pi0.5 + JEPA

This is the JAX/Flax adaptation of the four TTT routes in
`LeVJEPA_WAM_TTT`. It does not load the old PyTorch DiT TTT weights: Pi0.5
uses an 18-block Gemma action expert rather than the 16-block Flow DiT.
The insertion point is the same logical location, between the action
expert's attention residual and feed-forward network.

| Route | Action-expert blocks (zero based) | Fast-memory K/V |
| --- | --- | --- |
| `inline_explicit_jepa` | All 18 | Frozen V-JEPA patch features of the current observation |
| `inline_implicit_full` | All 18 | This block's complete post-attention action-expert token sequence |
| `wrapper_explicit_jepa` | 3, 7, 11, 15 | Frozen V-JEPA patch features of the current observation |
| `wrapper_implicit_full` | 3, 7, 11, 15 | This block's complete post-attention action-expert token sequence |

In all routes the action-expert suffix is `[state, 16 registers, learned
future tokens, action tokens]`. The TTT query and direct gated residual use
`[state, registers, actions]`. The learned future tokens are included in
implicit K/V but do not receive a direct TTT residual. The Pi0.5 image,
language, and V-JEPA query prefix remains in the VLM's own cache and is not
treated as implicit action-expert K/V. Explicit K/V is `vjepa_current`: the
feature-normalized output of the frozen V-JEPA encoder for every current
camera view used by the future-pair target (RoboTwin defaults to
`cam_high`). The precompute pipeline enforces this same-view contract. It is
deliberately independent of `predict_vjepa_target`.
The precomputed future-pair target supplies only the cosine auxiliary loss.

The memory is a fast two-layer GELU MLP. Per observation, K/V give an inner
MSE update, then Q reads the updated memory. Its weights are explicit state,
one copy per selected action-expert block and environment. Each training
window starts from zero fast deltas, preserves the state between observations,
and detaches its gradient every `ttt_tbptt_step_size` memory updates. Episode
starts are left-padded and masked; windows never cross a LeRobot episode.
`ttt_observation_stride` is the distance in physical frames between updates.
The provided TTT configs use stride 8, so a 128-observation window covers up
to 1017 physical frames and TBPTT detaches every 64 physical frames.
Gemma stores block parameters in a scanned stack, so wrapper mode still
allocates TTT **slow** parameters for all 18 blocks; only blocks 3, 7, 11, and
15 apply TTT and retain fast state. This costs more parameter/optimizer memory
than the four-module PyTorch wrapper.
Inference keeps the fast state in `Policy`. Call `policy.reset()` at each new
episode. Only the first denoising evaluation writes memory, so one policy
observation produces exactly one update. Execute eight open-loop environment
steps before the next policy observation to match the training stride. Since
the generic policy cannot observe how many actions the simulator executes,
the evaluation adapter should call `policy.validate_ttt_rollout_stride(8)`.
OpenPI's visual backbone is SigLIP rather than V-JEPA, so explicit deployment
must run the frozen V-JEPA encoder outside the JAX policy and add the resulting
`[576, 1408]` RoboTwin token array as `vjepa_current`.

## Training configs

The config names follow:

```
pi05_{libero|robotwin2_beat_block_hammer|robotwin2_clean20}_ttt_{inline|wrapper}_{explicit_jepa|implicit_full}_{ttt_only|action_expert}
```

`ttt_only` trains TTT slow parameters, the new register/future tokens, and the
new Pi0.5 state projection;
`action_expert` additionally trains the Pi0.5 action expert projections,
state projection, and Gemma expert weights. The V-JEPA predictor is frozen in
both, so initialize from a **trained Pi0.5 + V-JEPA checkpoint**, not the
vanilla Pi0.5 base. Each config's default `weight_loader.params_path` is a
placeholder under `checkpoints/<base-vjepa-config>/pretrained/params`; point
it to the actual checkpoint when launching training. The loader initializes
new TTT weights separately.

For example, with RoboTwin targets and normalization stats already prepared:

```bash
python scripts/train.py pi05_robotwin2_beat_block_hammer_ttt_wrapper_explicit_jepa_ttt_only \
  --exp-name robotwin_ttt_wrapper_explicit \
  --weight-loader.params-path /absolute/path/to/trained-vjepa/params \
  --model.ttt-context-length 8 \
  --model.ttt-observation-stride 8 \
  --model.ttt-tbptt-step-size 8 \
  --batch-size 1
```

The default context is 128 and the default stride is 8 to match the updated
round-4 sampling cadence; the example
overrides it to 8 for an initial smoke run. The full context increases
compute and activation memory substantially. The RoboTwin config still needs
the `vjepa_target_root` (containing both `targets/` and `current/`) and the normalization stats described in
`README_ROBOTWIN2.md`. This repository does not yet include an end-to-end
RoboTwin simulator evaluation adapter; the `Policy` itself carries state and
exposes `reset()` for one environment.
