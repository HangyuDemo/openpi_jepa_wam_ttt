# Joint Piper training checks — 2026-10-25

- Numerical accumulation: matches a complete batch with unequal temporal padding; optimizer advances once. Passed.
- Existing TTT frozen-slot regression tests: passed.
- Four simulated CPU devices: main training loop, validation, latest/best saving and resume passed.
- Small Pi0.5 JEPA+TTT model: segmented temporal outer gradients and inference online-update toggle passed (2 tests).
- Official pi05_base metadata: all existing parameter shapes match; 27 new leaves are explicitly initialized.
- Real data: the four-sequence batch passed with actions `(4,16,16,32)` and current/future JEPA tokens `(4,16,576,1408)`; values were finite. The four-device simulated training loop passed.
- Held-out set: fixed 64 windows covers all 8 episodes, disjoint from 18,264 training windows.
- Cache: all 150 episodes and 19,246 frames complete.
- W&B project created and authentication verified: https://wandb.ai/hang-yu-team/openpi-piper-real-robot-jepa-ttt
- Slurm `sbatch --test-only`: accepted normal / two nodes, 2 H100 and 4 CPUs per node / 128G per node / 72h.
- Shell syntax, Python compile and changed tracked-file whitespace checks passed.
- Storage: filesystem has free capacity; user quota reports about 2.05 TiB used against about 1 TiB limit. A 64 MiB write/fsync probe in checkpoints succeeded and was removed. Large checkpoint writing is still subject to actual quota enforcement.
- Job 1052593 was cancelled before it started. Replacement 1052846 failed because four ranks shared one Hugging Face datasets cache and one rank lost an in-progress Arrow file. The script now gives each rank a separate cache on node-local temporary storage. Job 1058390 passed data loading, but was cancelled before its first optimizer update to change from per-GPU microbatch 1 and accumulation 16 to per-GPU microbatch 4 and accumulation 4. Job 1058476 then exited at an obsolete 16-step accumulation guard. The guard, default config, and validation batch count were corrected. Job 1058487 was running on evc33 and evc36 at the last queue check; global batch remains 64.
- Full-model GPU memory/compilation is not established by these CPU checks. The revised job validates its first real GPU microbatch and saves/validates after optimizer update 1.
