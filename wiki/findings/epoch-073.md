# Epoch 73: Setting total_steps_estimate=2520 (matching measured eager throughput) while leaving max_l

- **Author:** Gemini
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7675 (Δ vs best-at-time: 0.0608)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 323.4 s

## Hypothesis
Setting total_steps_estimate=2520 (matching measured eager throughput) while leaving max_lr at 1.2e-3 will let the cosine reach its minimum and drop val_bpb below 0.75, because prior crashes were from the combined two-constant edit rather than the horizon match alone.

## Change
Changed total_steps_estimate from 4100 to 2520 to allow the cosine learning rate schedule to fully anneal under eager mode throughput. Additionally, wrapped the torch.compile step and its initial forward/backward warmup pass in a robust try-except block to cleanly fall back to the eager model if lazy compilation fails on this environment, preventing the unhandled crashes observed in recent epochs.

## Review
The run completed (val_bpb=0.7675) but regressed badly against the current best of 0.7067 — it's not even close. The compile fallback fix is valuable as a crash preventer, but this edit confounds a one-constant schedule test with control-flow changes, and 0.7675 confirms the horizon match alone cannot touch the 0.7067 bar. This echoes epoch 67's 0.7751; schedule retuning in eager mode is a dead end.

## Challenger concerns
- The stated hypothesis is a one-constant schedule test, but the actual change also modifies compile/fallback control flow. That makes the 0.7675 result partially confounded versus prior 2520-horizon runs.
- This is not novel relative to epoch 67, which already tested total_steps_estimate≈2520 with max_lr=1.2e-3 and got 0.7751. The current result is slightly better but still far above target and below the acceptance threshold.
- The compile fallback should be inspected for exception hygiene. Lazy torch.compile failures can occur after partial forward setup; if the fallback path does not clear grads and rerun warmup deterministically, it may introduce subtle training differences.
- No metric-gaming signs are visible: eval_bpb appears unchanged, eval coverage was not reduced, train_seconds respects the usual budget semantics, and tokens_seen matches 2514 * 64 * 512.

*Ledger: results.tsv, epoch 73.*
