# Epoch 70: Replacing torch.compile with CUDA Graph replay for the fixed-shape training forward/backwa

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7902 (Δ vs best-at-time: 0.0835)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 314.1 s

## Hypothesis
Replacing torch.compile with CUDA Graph replay for the fixed-shape training forward/backward step, while keeping the 6-layer 10.7M-parameter model and 4100-step cosine horizon, will restore much of the lost compile-era throughput without Triton/SDPA compilation crashes and reduce val_bpb to 0.70–0.73.

## Change
Removed torch.compile entirely. Added a CUDA Graph capture path: static input buffers (static_x/static_y) are filled, the fwd+bwd (under bf16 autocast) is warmed up on a side stream and captured into a torch.cuda.CUDAGraph. Each step copies the fresh batch into the static buffers, zeroes param grads, calls graph.replay(), then runs eager grad-clip + optimizer.step() (LR stays mutable). Wrapped capture in try/except to fall back to pure-eager if graph capture fails. Kept 6-layer/10.7M model, 4100-step cosine horizon, TF32 enabled, and the unchanged eval path.

## Review
Measured val_bpb=0.7902 (worse than 0.7067 best and recent eager 0.773-0.775) at 2510 steps/300s, identical throughput to pure eager. Challenger correctly flags unverified CUDA Graph capture and likely fallback or static-buffer bug; hypothesis untested and result regressed.

## Challenger concerns
- CUDA Graph capture was not verified — no log confirms whether the graph was actually used or if the fallback to eager was taken. The throughput (2510 steps) is identical to pure-eager runs, suggesting the graph either wasn't engaged or added no speedup, but the val_bpb degradation (0.790 vs. 0.773–0.775 in recent eager runs) indicates training quality was harmed, likely by a bug in the graph or in the static‑buffer handling. The hypothesis is untested because the intended mechanism may not have been active.

*Ledger: results.tsv, epoch 70.*
