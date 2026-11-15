# Epoch 77: Switching the training file to the LayerNorm+GELU-4x eager base (e75, 3648 steps, 0.7352) 

- **Author:** GPT-5
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7386 (Δ vs best-at-time: 0.0320)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 315.0 s

## Hypothesis
Switching the training file to the LayerNorm+GELU-4x eager base (e75, 3648 steps, 0.7352) AND setting total_steps_estimate=3650 to match measured eager throughput will cut val_bpb below 0.730, because the cosine schedule currently targets a 4100-step horizon that never completes in eager mode, leaving LR well above its minimum when training stops.

## Change
Replaced the current RMSNorm+SwiGLU model with the epoch-75 eager base: nn.LayerNorm blocks and a standard 2-layer GELU MLP with 4x hidden width (1536 for the main run, 256 for smoke). Disabled torch.compile by running the model eagerly, matching the stable e75 setup on this runner. Changed only the cosine horizon for the planned test from 4100 to 3650 so the LR reaches its floor near the measured eager step count.

## Review
The measured val_bpb of 0.7386 is a regression compared to both the current best of 0.7067 and the epoch-75 base of 0.7352. Tuning the cosine horizon to 3650 steps did not improve performance, refuting the hypothesis.

## Challenger concerns
- Result regressed vs identical e75 base (0.7386 > 0.7352) despite identical step count and architecture, showing the horizon change itself produced no benefit and may have added minor noise.
- Hypothesis assumes 4100-step schedule was the dominant cause of prior poor eager performance, but e75 already used the same base and beat this run, falsifying that assumption.

*Ledger: results.tsv, epoch 77.*
