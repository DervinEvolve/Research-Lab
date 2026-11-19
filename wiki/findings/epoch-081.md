# Epoch 81: Setting the cosine horizon total_steps_estimate from 4100 to 4400 (matching the ~4346 step

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7115 (Δ vs best-at-time: 0.0048)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 330.7 s

## Hypothesis
Setting the cosine horizon total_steps_estimate from 4100 to 4400 (matching the ~4346 steps the compiled RMSNorm/SwiGLU run actually completes) will lower val_bpb below 0.7067, because the current horizon ends the decay at step 4100, so the last ~250 steps run at the min LR floor instead of finishing the cosine decay on the final step. This is the only change.

## Change
Updated `total_steps_estimate` from `4100` to `4400` inside `train(args)` in `train.py`. No other lines, hyperparameters, schedule logic, or contracts were modified.

## Review
The run completed 4347 steps in 300.005 seconds but scored 0.711479 val_bpb, worse than the 0.7067 best by 0.004779. The 4400-step horizon therefore fails the keep criterion. The Challenger's endpoint-only argument is incorrect: extending the horizon changes learning rates throughout the post-warmup decay, not just the final ~250 steps. This single run does not establish a reproducible causal regression, but it provides no measured improvement.

## Challenger concerns
- The change is too small to produce the observed effect. Moving total_steps_estimate from 4100 to 4400 changes the LR at step 4347 from 1.2e-4 (progress capped at 1.0) to about 1.204e-4 (progress 0.9877) — a 0.3% difference over only the final ~250 steps. Such a tiny late-training LR change cannot plausibly shift val_bpb by 0.0036.
- Single-run result compared against historical runs (e79 0.70785, e80 0.71202) is confounded by run-to-run variance, compile nondeterminism, and eval sampling. There is no same-epoch control, so the 0.71148 cannot be attributed to the horizon change.
- Evaluation noise may be material. evaluate_bpb uses 200 batches; if those batches are randomly sampled each call, the estimate itself has sampling variance that is not reported and could be comparable to the observed 0.0036 difference.

*Ledger: results.tsv, epoch 81.*
