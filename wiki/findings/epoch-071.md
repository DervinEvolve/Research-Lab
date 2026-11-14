# Epoch 71: Setting total_steps_estimate=2500 (matching the ~2520-step eager throughput) AND lowering 

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Setting total_steps_estimate=2500 (matching the ~2520-step eager throughput) AND lowering max_lr to 8e-4 together will cut val_bpb below 0.74, because the current schedule is built for a 4100-step compile-era horizon that never completes in eager mode — so the cosine never anneals and LR stays destructively high through the whole run.

## Change
Set max_lr from 1.2e-3 to 8e-4 and total_steps_estimate from 4100 to 2500 to align the cosine schedule with observed eager throughput (~2520 steps/300s). Removed torch.compile entirely (no try/except, run_model=model) to eliminate compilation crashes documented in prior epochs.

## Review
The experiment crashed with a SyntaxError due to an invalid non-printable backspace character (U+0008) in the MLP docstring. No training occurred, and the hypothesis remains completely untested.

## Challenger concerns
- SyntaxError from U+0008 (backspace) in docstring prevents any execution; hypothesis untested

*Ledger: results.tsv, epoch 71.*
