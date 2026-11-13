# Epoch 48: Wrapping the warmup forward+backward inside the existing try/except around torch.compile (

- **Author:** Gemini
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7846 (Δ vs best-at-time: 0.0779)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 325.9 s

## Hypothesis
Wrapping the warmup forward+backward inside the existing try/except around torch.compile (so any lazy Triton failure during first use also triggers eager fallback) will eliminate the persistent compile crashes seen in epochs 44-46, allowing either a stable compiled run (~4100 steps) or clean eager fallback, and thereby reproducing val_bpb <=0.71 instead of the 0.77-0.82 eager floor.

## Change
Moved the instantiation of the optimizer before the torch.compile step. Wrapped both the torch.compile call and the warmup forward/backward pass within a single try-except block, falling back to eager mode and executing an eager warmup step upon any lazy compilation failure.

## Review
The run successfully avoided a hard crash via the fallback mechanism but reverted to eager execution, completing only 2389 steps. This resulted in a val_bpb of 0.7845, which is a severe regression from our 0.7067 compiled champion.

## Challenger concerns
- Silent fallback to eager mode masks compilation failure and introduces a learning rate schedule mismatch confounder.

*Ledger: results.tsv, epoch 48.*
