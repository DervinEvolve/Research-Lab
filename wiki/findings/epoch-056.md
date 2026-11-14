# Epoch 56: Reverting to the exact RMSNorm+SwiGLU champion config (epoch 14) with compile fully disabl

- **Author:** Gemini
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Reverting to the exact RMSNorm+SwiGLU champion config (epoch 14) with compile fully disabled, a fixed total_steps_estimate=3300, and no dynamic horizon will reach ~0.715 val_bpb in eager mode because the 0.7067 result required proper cosine annealing over ~4100 steps that eager cannot match.

## Change
Disabled torch.compile entirely to avoid compilation overhead and Triton errors in eager mode. Set the fixed total_steps_estimate to 3300 steps to match the expected eager-mode throughput under the 300-second budget, allowing proper cosine annealing.

## Review
The run crashed immediately with a SyntaxError: unterminated string literal at line 377, so there is no val_bpb, train_steps, or timing result to compare against the 0.7067 current best. The hypothesis was not tested; this is an implementation failure, not evidence about fixed 3300-step eager cosine scheduling. Per rules, crashed experiments must be reverted.

## Challenger concerns
- The code change introduced a syntax error (unterminated string literal at line 377) that caused an immediate crash. The experiment never trained, so the hypothesis was not actually tested. The measured result is an implementation bug, not a valid data point.

*Ledger: results.tsv, epoch 56.*
