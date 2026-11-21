# Epoch 100: Replacing the step-based cosine LR schedule (4100-step horizon, min_lr_ratio=0.1) with a t

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6984 (Δ vs best-at-time: 0.0042)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 330.8 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e100-15ef16b7 (43699663 bytes, SHA-256 `10a80b9a48350760356afac70e4f16462c0e8471748bd758d4f83fe30ad7ae12`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing the step-based cosine LR schedule (4100-step horizon, min_lr_ratio=0.1) with a time-based Warmup-Stable-Decay (WSD) schedule—100-step linear warmup, constant peak LR until 70% of time_budget, then linear decay to 0 at 100% of time_budget—will lower val_bpb below 0.679 and clear the 0.6805 confirmation bar.

## Change
Replaced the step-based cosine LR schedule (lr_at_step) with a time-based Warmup-Stable-Decay schedule (lr_at_time). The new function keeps the 100-step linear warmup, holds max_lr until 70% of time_budget, then decays linearly to 0 at 100% of time_budget. The training loop now passes elapsed and time_budget instead of total_steps_estimate; the unused total_steps_estimate variable was removed.

## Review
The run completed at 0.69839 val_bpb, worse than the current best of 0.6942 by 0.00419 and above the 0.6805 confirmation bar. It respected the 300-second budget and achieved 3942 steps with baseline-like memory use, but did not demonstrate improvement. The difference is within measured noise; this rejects the tested WSD recipe for adoption, not time-based scheduling generally, since schedule shape and endpoint also changed.

## Challenger concerns
- The edit bundles three schedule changes: step-based to time-based horizon, cosine to hold-then-linear shape, and LR floor 0.1x to 0. The result cannot say which of these mattered, so 'time-based scheduling is dead' is not supported.
- The result is inside the noise band. 0.6984 vs the recorded best of 0.6942 is +0.0042, below the measured noise of 0.0068. It is, however, about 0.015-0.018 worse than the recent 0.680-0.685 single runs of near-baseline variants (e92-97), which suggests the cosine baseline's true mean sits near 0.68 and WSD probably is worse than it. That comparison rests on single runs.
- Throughput was not a confound this time: 3942 steps at ~428k tok/s matches the fast-machine baseline (3947-3989 steps). The old cosine at step 3942 of 4100 would have ended at roughly 0.1x peak LR, so both schedules finish at low LR. The real difference is the time spent at peak: 70% hold versus continuous decay.
- No sign of metric gaming or budget violation. Evaluation settings are unchanged (200 batches, batch size 32, seq_len 512), train_seconds is 300.0, and VRAM and parameter count match the baseline. The LR reaches exactly 0 only at the budget boundary, so the last few steps are near-no-ops but harmless.

*Ledger: results.tsv, epoch 100.*
