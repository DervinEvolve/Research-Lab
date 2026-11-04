# Epoch 25: Raising max_lr from 1.2e-3 to 1.6e-3 on the epoch-14 champion architecture will reduce val

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7938 (Δ vs best-at-time: 0.0871)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 323.9 s

## Hypothesis
Raising max_lr from 1.2e-3 to 1.6e-3 on the epoch-14 champion architecture will reduce val_bpb below 0.7067 because a higher learning rate accelerates convergence under the tight 300s wall-clock budget.

## Change
Changed max_lr from 1.2e-3 to 1.6e-3. Modified the torch.compile guard to run a dummy forward/backward pass (with a CUDA synchronize) inside the try-except block before the timed loop, zeroing grads after success. On any lazy compilation/backend error it falls back to eager mode so the LR experiment can actually run.

## Review
Revert: the run completed cleanly, but val_bpb was 0.7938, far worse than the current best 0.7067, so it fails the keep criterion. The result is still useful because it confirms the compile/lazy-backend crash issue can be bypassed enough to get a measurement, but the LR=1.6e-3 change did not improve the model as run. Challenger's schedule confound is credible: only 2509 steps were reached against a 4100-step cosine horizon, so the bad result should not be overinterpreted as a pure peak-LR verdict.

## Challenger concerns
- The run produced a CLEAN measurement at last — 0.7938 val_bpb — but it is WORSE than the 0.7067 champion. Critically, this run only reached 2509 steps vs the champion's ~4100-step horizon, yet total_steps_estimate is still hardcoded to 4100. That means the cosine schedule never finished decaying: the LR was cut off mid-schedule at a high value, so the model ended far from its min_lr annealing. The bad number is CONFOUNDED by a schedule mismatch, not a clean verdict on max_lr=1.6e-3.
- Train loss (~0.55) is LOWER than typical, but val_bpb is HIGH — this gap is the signature of an incomplete LR decay (undertrained in the fine phase), reinforcing that the schedule, not the peak LR, drove the regression. This is a mechanism question the result does not resolve.
- This is a COMPOUND change, violating the one-hypothesis rule: it altered max_lr AND rewrote the compile guard. The guard rewrite is justified (prior crashes), but it muddies attribution — we can't cleanly separate 'LR 1.6e-3 is bad' from 'the step count / horizon shifted'.

*Ledger: results.tsv, epoch 25.*
