# Epoch 80: Driving the LR schedule by wall-clock progress (progress = elapsed / time_budget for the c

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7120 (Δ vs best-at-time: 0.0054)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 336.0 s

## Hypothesis
Driving the LR schedule by wall-clock progress (progress = elapsed / time_budget for the cosine phase, keeping the 100-step linear warmup, max_lr 1.2e-3 and min_lr_ratio 0.1) instead of the fixed total_steps_estimate=4100 will make the cosine finish exactly at the budget in either run mode. I predict val_bpb <= 0.7070 if the run is compiled (~4300 steps), and below 0.7675 if it lands in eager (~2500 steps).

## Change
Only the LR schedule changed. lr_at_step still does a 100-step linear warmup to max_lr=1.2e-3; after that, cosine progress is elapsed/time_budget (clamped to 1) with min_lr_ratio=0.1, replacing total_steps_estimate=4100. The loop passes the same elapsed and time_budget used for the wall-clock break, so the cosine reaches min LR as the budget ends in either compile or eager mode. Architecture, optimizer, compile path, weight decay, and batch shape are untouched.

## Review
The run completed 4173 steps but measured val_bpb 0.7120, worse than the current best 0.7067 and worse than e79's 0.7078 on the same compiled RMSNorm/SwiGLU path. The wall-clock cosine schedule removed the unfinished decay tail, and the Challenger's confound concern is credible: 4173 vs 4346 steps means throughput variance is entangled with the LR change. No improvement, so the change is reverted.

## Challenger concerns
- Wall-clock LR coupling breaks determinism and is confounded by throughput variance: this run completed 4173 steps (455.8k tok/s) vs e79's 4346 steps (469k tok/s), a 173-step (~5.7M token) deficit that confounds the +0.0042 val_bpb regression.
- Warmup-to-cosine discontinuity: using elapsed/time_budget without subtracting the time spent in the 100-step warmup causes cosine progress to jump from 0 to ~0.024 at step 100.
- Hypothesis falsified and repeats e77's negative result: predicted <=0.7070 in compiled mode, measured 0.7120. Removing the min_lr tail hurts val_bpb.

*Ledger: results.tsv, epoch 80.*
