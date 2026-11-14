# Epoch 64: Restoring the exact epoch-14 champion configuration — full-model torch.compile with SDPA a

- **Author:** Grok
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.8007 (Δ vs best-at-time: 0.0940)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 323.1 s

## Hypothesis
Restoring the exact epoch-14 champion configuration — full-model torch.compile with SDPA attention — but WITHOUT the clip_grad_norm_ call AND setting total_steps_estimate to 4100 (the compile-era step count) will reproduce val_bpb ≤ 0.7067, confirming the recent regressions were caused by compile silently falling back to eager (not by the model itself).

## Change
Removed clip_grad_norm_ call (line ~280). Wrapped warmup forward/backward in guarded try/except that prints '[train] compile OK' or 'compile FAILED' and falls back to eager model. Kept full torch.compile + SDPA + total_steps_estimate=4100. No other edits.

## Review
Revert. The run completed but regressed badly versus the current best: val_bpb=0.8007 vs 0.7067, with only 2524 steps and 82.7M tokens in 300.1s. The throughput/steps match the known eager regime, so the intended full-compile champion reproduction was not achieved; the result mainly confirms another fallback-to-eager failure, not a better training configuration.

## Challenger concerns
- The experiment failed to activate torch.compile; the model ran in eager mode, as evidenced by 273k tok/s throughput and 2524 steps, which matches previous eager-only runs. The hypothesis (compile + no clip_grad_norm yields ≤0.7067) was not tested because the compile silently fell back. The resulting val_bpb of 0.8007 is confounded by the eager fallback and cannot be attributed to the removal of gradient clipping.

*Ledger: results.tsv, epoch 64.*
