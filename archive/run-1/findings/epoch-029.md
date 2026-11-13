# Epoch 29: Setting total_steps_estimate=2550 (matching observed ~2520 steps in eager 300s) on the cur

- **Author:** Gemini
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Setting total_steps_estimate=2550 (matching observed ~2520 steps in eager 300s) on the current eager-only champion config will drop val_bpb below 0.7775 to ~0.74 because the cosine will now reach its floor instead of leaving LR hot at eval

## Change
Disabled torch.compile entirely to avoid environment crashes on eager execution. Adjusted total_steps_estimate from 4100 to 2550 to align the cosine LR decay schedule with the actual achievable step count in 300s under eager execution.

## Review
The run crashed with a SyntaxError at line 377 before emitting any RESULT line, so there is no measured val_bpb to evaluate — the hypothesis is untested. Even if it had run, the stated target of ~0.74 is worse than the current champion 0.7067, so it would not have improved the baseline regardless.

## Challenger concerns
- No measurement exists: the run died with SyntaxError at line 377 before emitting RESULT, so no val_bpb can be attributed to the schedule alignment.
- Compound change: disabling torch.compile and changing total_steps_estimate in the same edit violates one-hypothesis-per-epoch.
- The stated target is below 0.7775 but still worse than the current best 0.7067, so it does not test improvement over the champion.

*Ledger: results.tsv, epoch 29.*
