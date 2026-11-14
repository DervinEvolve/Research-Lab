# Epoch 68: Lowering max_lr from 1.2e-3 to 6e-4 (with total_steps_estimate=2520 and 100-step warmup) w

- **Author:** GPT-5
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Lowering max_lr from 1.2e-3 to 6e-4 (with total_steps_estimate=2520 and 100-step warmup) will cut val_bpb below 0.74 because the 1.2e-3 peak was tuned for the compile-era 4100-step horizon; at only 2520 steps the LR stays too high through mid-training, adding gradient noise that the shorter schedule cannot anneal away.

## Change
Changed the training LR magnitude from `max_lr = 1.2e-3` to `max_lr = 6e-4` in `train(args)`. Changed the cosine schedule horizon from `total_steps_estimate = 4100` to `total_steps_estimate = 2520`, matching the measured eager-mode budget. No architecture, optimizer type, data, evaluation, or loop mechanics were otherwise changed.

## Review
The run crashed during the warmup step with a BackendCompilerFailed error because the lazy Triton compilation was triggered outside the try-except block. Since no training occurred, the hypothesis could not be tested and we must revert.

## Challenger concerns
- The experiment crashed with a torch.compile BackendCompilerFailed error (missing C compiler) during the warmup forward pass, before any training. The try-except around torch.compile(model) does not catch lazy compilation errors that occur when the compiled model is first invoked, so the model never trained and the hypothesis was not tested.

*Ledger: results.tsv, epoch 68.*
