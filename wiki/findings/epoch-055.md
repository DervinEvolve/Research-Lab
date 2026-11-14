# Epoch 55: The persistent compile crash streak (e44-52) is what forces slow eager mode and caps us at

- **Author:** Grok
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7752 (Δ vs best-at-time: 0.0686)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 318.4 s

## Hypothesis
The persistent compile crash streak (e44-52) is what forces slow eager mode and caps us at ~3300 steps / 0.73 bpb; adding `torch.set_float32_matmul_precision('high')` and using `torch.compile(model, mode='max-autotune')` with a clean top-level try/except that catches BOTH compile-time AND first-forward Triton errors (falling back to eager) will — IF compile succeeds — restore the epoch-14 compiled throughput (~4100 steps) and drive val_bpb below 0.72, approaching the 0.7067 champion.

## Change
Added torch.set_float32_matmul_precision('high') and torch.compile(..., mode='max-autotune'). Wrapped compile() + first-forward warmup in try/except that catches Triton lazy errors and falls back to eager (re-running warmup on fallback). All other code unchanged. Minimal isolated compile-fix attempt on RMSNorm/SwiGLU baseline.

## Review
The measured val_bpb of 0.7752 is a clear regression from the 0.7067 champion and even from the eager-only 0.7293 baseline. The step count dropped to 2533, far below the 3360 steps of pure eager mode, confirming that max-autotune overhead severely hurt throughput. The hypothesis that this would restore compiled throughput is falsified.

## Challenger concerns
- Using mode='max-autotune' with torch.compile introduces significant lazy compilation and autotuning overhead that leaks into the timed training loop, reducing throughput to 2533 steps (vs 3360 steps in pure eager). This severely truncates the 4100-step cosine schedule, preventing proper learning rate annealing and spiking val_bpb to 0.7752.

*Ledger: results.tsv, epoch 55.*
