# Epoch 47: Stripping torch.compile entirely and setting the cosine schedule's total_steps_estimate to

- **Author:** Grok
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7722 (Δ vs best-at-time: 0.0655)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.5 s

## Hypothesis
Stripping torch.compile entirely and setting the cosine schedule's total_steps_estimate to 2500 (matching the actual eager step budget) will lower val_bpb below 0.75, because the model will complete a full annealing to a low final learning rate within the available steps, improving final convergence compared to the 4100-step horizon that leaves the LR too high at step 2500.

## Change
Stripped the entire torch.compile try/except block and set run_model = model directly. Changed total_steps_estimate from 4100 to 2500. No other edits.

## Review
The measured val_bpb of 0.7722 is worse than our current best of 0.7067. While matching the cosine decay horizon to the actual eager step budget of 2500 improved on previous eager runs (0.7838 in epoch 45), eager mode remains uncompetitive with the compiled baseline.

## Challenger concerns
- Premature learning rate annealing in step-starved regime
- Confounded comparison with prior eager runs

*Ledger: results.tsv, epoch 47.*
