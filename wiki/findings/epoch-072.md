# Epoch 72: Lowering max_lr to 8e-4 and total_steps_estimate to 2500 (matching the observed ~2520 step

- **Author:** Grok
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Lowering max_lr to 8e-4 and total_steps_estimate to 2500 (matching the observed ~2520 steps/300s eager throughput) will reduce val_bpb below 0.74, because the current 4100‑step horizon never completes, leaving the cosine schedule far from its minimum LR and causing noisy, unconverged updates.

## Change
Changed exactly two constants in train(): max_lr from 1.2e-3 to 8e-4 and total_steps_estimate from 4100 to 2500. No other edits. Preserves every contract, smoke path, and evaluate_bpb call.

## Review
Revert. The run crashed during lazy torch.compile/Triton compilation before producing val_bpb, train_steps, train_seconds, or any RESULT line, so there is no measured improvement over the current best 0.7067 bpb. The Challenger's objection is decisive: the try/except wraps only torch.compile construction, while the actual backend failure occurs on the first forward pass outside that guard.

## Challenger concerns
- The try-except block only wraps the lazy torch.compile call, which does not trigger actual compilation. The actual compilation happens during the first forward pass, which is outside the try-except block, causing an unhandled BackendCompilerFailed exception.

*Ledger: results.tsv, epoch 72.*
