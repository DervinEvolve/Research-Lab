# Epoch 79: On the current RMSNorm/SwiGLU train.py, wrapping the compile warmup in a try/except that f

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7078 (Δ vs best-at-time: 0.0012)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 332.1 s

## Hypothesis
On the current RMSNorm/SwiGLU train.py, wrapping the compile warmup in a try/except that falls back to eager (the only change, no fused AdamW, horizon, LR, or wd edits) will make runs crash-free and reproducibly complete, landing val_bpb within 0.7675 ±0.01 over ~2500 steps. If the fallback then runs a few hundred more steps than e73's 2514, val_bpb should drop below 0.76.

## Change
Wrapped the warmup forward/backward pass in `train()` with a `try/except Exception` block. If lazy `torch.compile` execution fails during warmup, it logs the error, clears any partial gradients, sets `run_model = model`, and executes an eager warmup forward/backward before starting the timed training loop. All architecture, optimizer, learning rate, and schedule settings remain unchanged.

## Review
The run completed cleanly, but 0.7078 does not improve on the current best of 0.7067, so the keep standard is not met. The hypothesis was also not actually tested: 4346 steps and 3191 MB peak VRAM are the signature of compiled mode, not the predicted ~2500-step eager fallback, so the try/except branch almost certainly never fired. What we did learn is useful: with compile working, the unchanged RMSNorm/SwiGLU file reproduces about 0.708, so 0.7067 is reachable and the 0.7675 eager runs were an environment/compile-availability effect rather than a code regression.

## Challenger concerns
- The fallback branch was almost certainly never executed. The measured throughput (~469k tok/s), 4346 completed steps, and 3191 MB peak VRAM match torch.compile mode, not the eager fallback regime predicted by the hypothesis (which expected ~2500 steps and ~0.7675 val_bpb). The try/except therefore did not test the stated change; it merely allowed a normal compile run to complete.
- The measured val_bpb of 0.7078 is within run-to-run noise of the existing compile-era best of 0.7067. Even if the fallback had triggered, this result would not demonstrate an improvement or a new mechanism; it is indistinguishable from the known compile baseline.

*Ledger: results.tsv, epoch 79.*
