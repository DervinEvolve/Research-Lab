# Epoch 49: Replacing SwiGLU with a parameter-matched standard GELU MLP, replacing the custom RMSNorm 

- **Author:** GPT-5
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7352 (Δ vs best-at-time: 0.0286)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 317.0 s

## Hypothesis
Replacing SwiGLU with a parameter-matched standard GELU MLP, replacing the custom RMSNorm with PyTorch's native fused LayerNorm, and enabling fused=True in AdamW will dramatically reduce CUDA kernel launch overhead in eager mode, boosting step throughput to over 3100 steps and lowering val_bpb to <0.74.

## Change
Replaced the custom RMSNorm modules with native nn.LayerNorm (bias disabled to preserve the bias-free setup). Replaced the SwiGLU MLP with a parameter-matched standard GELU MLP: hidden 1536 for the main run and 192 for smoke. Enabled fused AdamW on CUDA and forced eager execution by removing the torch.compile path, since this experiment targets reliable eager throughput.

## Review
The run's val_bpb 0.7352 is a clear regression from the current best 0.7067, so it cannot be kept regardless of methodology. Even if it had improved, the Challenger's compound-change critique is valid: RMSNorm→LayerNorm, SwiGLU→GELU, fused AdamW, and forced eager were changed simultaneously, making it impossible to attribute the throughput gain (3678 steps vs ~2500 eager baseline) to any single mechanism. The result is scientifically uninterpretable.

## Challenger concerns
- This is a COMPOUND change: (a) RMSNorm→LayerNorm, (b) SwiGLU→GELU MLP, (c) fused AdamW, (d) forced eager. Four independent levers in one run. The 0.7352 improvement over the eager floor (0.772) cannot be attributed to any single mechanism. Throughput jumped to ~398k tok/s / 3678 steps, but we can't tell if that came from fused AdamW, the simpler MLP, or native LayerNorm.
- Architectural confound with the historical champion. The 0.7067 champion used RMSNorm+SwiGLU under COMPILE. This run changes architecture AND execution mode. Comparing 0.7352 to 0.7067 tells us nothing clean — the gap is still dominated by the throughput/compile difference, not proof that GELU+LayerNorm is worse or better than SwiGLU+RMSNorm per-step.
- Schedule confound. total_steps_estimate is still 4100 in the code, but the run only did 3678 steps — so LR never fully annealed (cosine stopped at ~progress 0.90, min_lr not reached). Part of the remaining gap to 0.7067 is schedule mismatch, not architecture. e47 showed under-annealing matters.

*Ledger: results.tsv, epoch 49.*
