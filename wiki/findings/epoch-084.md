# Epoch 84: Changing only total_steps_estimate from 4100 to 3500 will lower val_bpb below 0.7067 withi

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7239 (Δ vs best-at-time: 0.0172)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 335.8 s

## Hypothesis
Changing only total_steps_estimate from 4100 to 3500 will lower val_bpb below 0.7067 within the same 300-second training budget, because earlier cosine decay gives the model more low-learning-rate refinement after its initial rapid fitting.

## Change
Changed total_steps_estimate from 4100 to 3500 in train() and updated the comment on that line to describe the experiment. Nothing else was touched: default torch.compile, architecture, max_lr 1.2e-3, 100-step warmup, min_lr_ratio 0.1, optimizer, batching and evaluation are identical. At the expected ~4346 compiled steps, the cosine now reaches the min-LR floor at step 3500, leaving roughly 850 steps at 1.2e-4.

## Review
Measured val_bpb regressed to 0.7239 (4,170 steps in 300.0s), worse than both the 0.7067 best (+0.0172) and the 0.7078 same-mode control (e79, 4,346 steps). Compressing total_steps_estimate from 4100 to 3500 lowered mid-training learning rates across thousands of steps rather than isolating a longer min-LR tail, leaving the model undertrained.

## Challenger concerns
- total_steps_estimate=3500 does not isolate a longer min-LR tail. It compresses the entire cosine, so mid-training LR is lower for thousands of steps (step 2000: ~5.6e-4 vs ~7.0e-4 under horizon 4100). The +0.016 bpb cannot be attributed to 'more low-LR refinement.'
- The stated condition was not met. Hypothesis assumed ~4346 compiled steps and ~850 steps at 1.2e-4. Measured 4170 steps (~455k tok/s) and only ~670 steps at the floor. e79 control was 4346 steps / 0.7078, so tokens are not paired.
- Not gaming and not an implementation bug. params 10.720128, VRAM 3191, budget 300.0s, and evaluate_bpb are unchanged. Late loss still drifts (0.518 at 3150 to ~0.49 at 4150). Near-baseline 0.7239 is undertraining, not a silent architecture rollback.

*Ledger: results.tsv, epoch 84.*
