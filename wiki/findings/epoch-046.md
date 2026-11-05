# Epoch 46: Reverting train.py to the exact epoch-14 champion (tied head, max_lr=1.2e-3, 4100-step hor

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Reverting train.py to the exact epoch-14 champion (tied head, max_lr=1.2e-3, 4100-step horizon) WITH the torch.compile path intact and unmodified — exactly as it stood at the 0.7067 run — will reproduce val_bpb ~0.707 in the compiled regime (~4100 steps in 300s). This confirms the champion still compiles cleanly and that epochs 40-45's 0.76-0.82 floor was purely a self-inflicted eager/untied regression, not a real model limit.

## Change
No functional changes were required: the current train.py already matches the requested epoch-14 champion. I returned it verbatim—tied input/output embeddings, RMSNorm/SwiGLU, max_lr=1.2e-3, 4100-step cosine horizon, and the original torch.compile try/except path intact—so this epoch is a clean reproduction test rather than a new architecture change.

## Review
The reproduction run crashed during Triton compilation with an environmental error, producing no validation metrics or step counts. Because there is no successful run to evaluate, we must revert.

## Challenger concerns
- Crash is purely environmental (missing CC for Triton backend) rather than attributable to any model/code change; hypothesis about reproduction of 0.7067 cannot be evaluated at all.
- No measured val_bpb or step count means zero data on whether the 4100-step compiled regime is even reachable; ledger entry cannot be interpreted as 'champion still works'.

*Ledger: results.tsv, epoch 46.*
