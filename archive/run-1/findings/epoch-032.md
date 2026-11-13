# Epoch 32: Reducing weight decay from 0.1 to 0.01 in the AdamW optimizer will lower val_bpb below the

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Reducing weight decay from 0.1 to 0.01 in the AdamW optimizer will lower val_bpb below the 0.7067 champion because the model is trained in a strictly single-pass underfitting regime (seeing <10% of the data), where weight decay acts as an unnecessary constraint that slows down convergence.

## Change
Changed the AdamW weight_decay argument from 0.1 to 0.01 in the single call to model.configure_optimizer inside train(). No other hyperparameters, architecture, schedule, or training-loop logic were modified. This is the minimal surgical edit required to test the Planner's hypothesis.

## Review
The run crashed inside torch.compile/Triton (triton_backend hash failure) before emitting any RESULT line, so val_bpb was never measured — there is zero evidence about weight_decay=0.01. This is the 6th crash in a streak, and the failure is in compile infrastructure, entirely confounded with the optimizer edit. No improvement over 0.7067 can be claimed from a non-result.

## Challenger concerns
- The run crashed inside torch.compile/Triton before producing RESULT, so the weight_decay hypothesis was not tested at all.
- The infrastructure failure is confounded with prior known compile instability, not with the optimizer edit.
- The try/except around torch.compile is insufficient because compilation can fail lazily on first model invocation, outside the exception block.
- Changing compile behavior to rescue the run would change throughput and effective tokens seen, so it cannot be silently bundled with the weight_decay test.

*Ledger: results.tsv, epoch 32.*
