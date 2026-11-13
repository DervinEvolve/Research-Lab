# Epoch 28: Disable torch.compile entirely (set run_model = model; no compile attempt) while keeping t

- **Author:** Grok
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7775 (Δ vs best-at-time: 0.0709)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.8 s

## Hypothesis
Disable torch.compile entirely (set run_model = model; no compile attempt) while keeping the epoch-14 champion architecture and hyperparameters unchanged. This will complete reliably and recover the champion-quality regime, potentially matching or slightly improving 0.7067 val_bpb by eliminating lazy compile crashes without changing model math.

## Change
Removed the entire torch.compile try/except block after model creation (lines ~140-145). Replaced with the single line `run_model = model`. No other code, hyperparameters, schedule, or architecture touched.

## Review
The run completed (2520 steps, 300.0s) but val_bpb regressed to 0.7775, substantially worse than the current best of 0.7067. This is a +0.0708 loss, not a match or improvement, so the hypothesis fails measured reality. The Challenger's throughput/schedule concern is plausible, and there is no result to justify keeping eager-only as the new baseline.

## Challenger concerns
- Severe throughput degradation causing learning rate schedule truncation and undertraining

*Ledger: results.tsv, epoch 28.*
