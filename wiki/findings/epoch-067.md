# Epoch 67: Setting total_steps_estimate exactly to 2520 (measured eager throughput at batch=64) while

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7751 (Δ vs best-at-time: 0.0684)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.5 s

## Hypothesis
Setting total_steps_estimate exactly to 2520 (measured eager throughput at batch=64) while keeping max_lr=1.2e-3 and the 100-step warmup will let the cosine schedule anneal through the entire 300 s budget, dropping val_bpb below 0.75 (ideally ~0.72) versus the 0.77+ seen in mismatched-horizon eager runs.

## Change
Two surgical changes: (1) set total_steps_estimate from 4100 to 2520 to match the measured eager throughput so the cosine schedule anneals fully through the 300s budget instead of truncating at ~61% progress; (2) removed the now-pointless torch.compile try/except wrapper (compile/JIT/TF32 all confirmed dead on this A10G per e61-e66) and run eager directly. No architecture, optimizer, batch, or eval changes.

## Review
The measured val_bpb of 0.7751 is worse than the current best of 0.7067 and does not drop below the hypothesized 0.75, despite matching the cosine horizon to eager throughput (2522 steps). This essentially replicates epoch 61's eager + ~2500-step schedule (0.7838), confirming that proper annealing alone does not rescue eager mode to compile-era performance. The Challenger's note about bundling edits is minor but the negative result is clear.

## Challenger concerns
- This is not really a new methodology: epoch 61 already tested eager/no-compile with a ~2500-step cosine horizon and got 0.7838. Changing 2500 to 2520 is too small to constitute a new mechanism.
- The stated change bundles two edits: matching total_steps_estimate to eager throughput and removing the torch.compile wrapper. That slightly confounds attribution, even though compile appears dead.
- The result does not support the hypothesis: val_bpb=0.7751 is still in the bad eager regime and not below 0.75. The train loss tail improved, but validation did not.
- The run slightly exceeds the nominal 300s budget at 300.014s, though this looks like normal loop-boundary overshoot rather than cheating.
- No sign of metric gaming or validation leakage is visible from the result: eval batch size/count and evaluate_bpb path appear unchanged.

*Ledger: results.tsv, epoch 67.*
