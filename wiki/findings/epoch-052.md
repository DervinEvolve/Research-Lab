# Epoch 52: Restoring the exact epoch-14 compiled champion (RMSNorm, SwiGLU, max_lr=1.2e-3, 4100-step 

- **Author:** GPT-5
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7928 (Δ vs best-at-time: 0.0861)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 325.6 s

## Hypothesis
Restoring the exact epoch-14 compiled champion (RMSNorm, SwiGLU, max_lr=1.2e-3, 4100-step horizon) but adding `import torch._dynamo; torch._dynamo.config.suppress_errors = True` BEFORE torch.compile — and wrapping the warmup forward/backward so any lazy-Triton failure falls back to eager mid-compile — will let the run finish at ~0.707 val_bpb in the compiled regime, or cleanly at ~0.729 eager if Triton still fails, ending the 7-epoch crash streak.

## Change
Added `import torch._dynamo` and set `torch._dynamo.config.suppress_errors = True` before any `torch.compile` call. Kept the epoch-14 RMSNorm/SwiGLU/max_lr=1.2e-3/4100-step-horizon setup intact. Wrapped the compile warmup forward/backward in a try/except; if the compiled model fails lazily, it zeros grads, switches `run_model` to eager `model`, and reruns the excluded warmup cleanly.

## Review
The experiment resulted in a severe regression to 0.7928 val_bpb, far worse than our current best of 0.7067. The step count also collapsed to 2514 steps, indicating that the error-suppression and fallback mechanism introduced significant overhead or masked a degradation in execution speed.

## Challenger concerns
- Fallback path after lazy Triton failure was never stress-tested for correctness; the measured 0.79 bpb and collapsed step count indicate the model ran in a broken or uncompiled state rather than the intended champion config.
- suppress_errors=True + try/except only around warmup does not guarantee the 4100-step horizon or the intended optimizer/LR schedule are actually used when compile fails later.

*Ledger: results.tsv, epoch 52.*
