# Epoch 61: Forcing eager mode (no compile) and setting the cosine horizon to exactly 2500 steps (the 

- **Author:** GPT-5
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7838 (Δ vs best-at-time: 0.0771)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.6 s

## Hypothesis
Forcing eager mode (no compile) and setting the cosine horizon to exactly 2500 steps (the measured eager throughput at batch=64) will drop val_bpb from the ~0.77 range of mismatched-horizon eager runs to ≈0.715, because proper annealing through the budget avoids early LR collapse seen in e57/e58.

## Change
Disabled torch.compile by replacing the compile attempt with run_model = model, forcing eager execution and avoiding the lazy Triton crash path. Changed total_steps_estimate from 4100 to 2500 to match the measured batch=64 eager throughput over the 300s budget. No architecture, optimizer, batch size, clipping, or evaluation settings were changed.

## Review
The run completed but regressed badly: 0.7837 val_bpb vs current best 0.7067, and even worse than e57's mismatched 3300-horizon eager run (0.7701). The hypothesis that matching the horizon to measured throughput would help is falsified — shortening to 2500 steps hurt, not helped. Eager mode at batch=64 is simply too slow to be competitive with the compiled 0.7067 champion.

## Challenger concerns
- Tuning the cosine schedule horizon down to 2500 steps to match eager throughput caused severe performance degradation (0.7837 val_bpb) compared to the mismatched 3300-step horizon in e57 (0.7701 val_bpb). Proper annealing to 2500 steps is a false optimization here.

*Ledger: results.tsv, epoch 61.*
