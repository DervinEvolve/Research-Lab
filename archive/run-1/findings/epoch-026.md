# Epoch 26: Replacing the fixed total_steps_estimate=4100 with a dynamic estimate based on measured th

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7749 (Δ vs best-at-time: 0.0682)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 317.5 s

## Hypothesis
Replacing the fixed total_steps_estimate=4100 with a dynamic estimate based on measured throughput after warmup will reduce val_bpb below the current 0.7067 champion by allowing the cosine schedule to fully decay the learning rate to its minimum by the end of training, avoiding the 'hot LR at eval' problem identified in epoch 25.

## Change
Removed the fixed total_steps_estimate=4100 and the torch.compile path (eager-only, per e24 root-cause note). Added a 20-step timed calibration window right after warmup that measures sec/step, then sets a dynamic total_steps_estimate = calib_steps + projected_remaining_steps (with a 0.97 safety margin) so the cosine schedule decays fully to min_lr by the time-budget cutoff. The main loop now uses this dynamic horizon; all architecture, max_lr=1.2e-3, wd=0.1, warmup=100, min_lr_ratio=0.1 are unchanged.

## Review
val_bpb 0.7749 >> 0.7067 champion; run regressed. Multiple confounds (eager-only + dynamic horizon) and methodology issues (calibration steps, early min_lr, noisy 20-step estimate, 300.1s overrun) prevent clean attribution. No credible path to keep.

## Challenger concerns
- Attribution is confounded: the patch changed both the LR horizon logic and the execution path by removing torch.compile. Since e25's fallback likely ran eager after the dummy guard, this may be small, but it is still not cleanly isolated.
- The 20-step calibration window may have added real optimizer updates outside the measured train_seconds/time budget. If those steps were excluded from t_start but included in training, this violates the wall-clock methodology even though the result was worse.
- The implemented schedule did not merely 'reach min_lr by the end'; it reached min_lr around step 2350 and then spent roughly the last 160 steps at the floor. This tests early over-decay, not just avoiding a hot LR at eval.
- Throughput estimation from only 20 early steps can be biased by transient startup effects, allocator state, or dataloader/cache behavior. The resulting horizon is therefore noisy and can move the LR schedule substantially.
- train_seconds is 300.105s, and if calibration was outside that timer then actual training wall clock exceeded the nominal budget further. This is not metric gaming here, but it weakens compliance with the fixed-budget rule.

*Ledger: results.tsv, epoch 26.*
