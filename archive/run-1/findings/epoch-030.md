# Epoch 30: On the eager-only (compile-off) config with total_steps_estimate=2500 to match the ~2520 s

- **Author:** Grok
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7810 (Δ vs best-at-time: 0.0744)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 319.8 s

## Hypothesis
On the eager-only (compile-off) config with total_steps_estimate=2500 to match the ~2520 steps actually achieved in 300s, val_bpb will drop from 0.7775 to below 0.74 because the cosine will reach its 0.12e-3 floor at eval instead of leaving LR hot at step 2520 (where the 4100-horizon schedule still sits at ~0.9e-3).

## Change
Hard-disabled torch.compile (run_model=model) and set total_steps_estimate=2500 to match measured eager throughput. All other code identical to current file. Single-integer + one-line change per hypothesis.

## Review
The run completed (2389 steps, 300.1s) but measured val_bpb=0.7810, worse than the current best 0.7067 and worse than epoch-28's eager-only 0.7775. Lowering total_steps_estimate to 2500 to force full cosine decay did not help; it regressed, supporting the Challenger's point that the higher late-training LR is beneficial in this step-limited regime.

## Challenger concerns
- Underestimation of the utility of 'hot' learning rates in severely step-limited regimes.

*Ledger: results.tsv, epoch 30.*
