# Epoch 57: Disabling torch.compile entirely to eliminate the persistent compilation/Triton crashes, w

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7701 (Δ vs best-at-time: 0.0635)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.6 s

## Hypothesis
Disabling torch.compile entirely to eliminate the persistent compilation/Triton crashes, while training the RMSNorm+SwiGLU architecture in pure eager mode with a fixed 3300-step cosine horizon and native fused AdamW (fused=True), will reliably complete the 300s budget and lower val_bpb to <0.715.

## Change
1) Removed the torch.compile try/except block; run_model is now the raw model (eager mode). 2) Set total_steps_estimate = 3300 to align cosine decay with the ~3300-step eager budget. 3) Added fused=True parameter to configure_optimizer and called it with fused=True to enable native fused AdamW for throughput. No other architecture or data-flow changes.

## Review
Measured 0.7701 val_bpb is far worse than the 0.7067 champion and even worse than e51's clean eager LayerNorm (0.7293). The run also only reached 2509 steps — contradicting the ~3300-step premise and leaving the fixed cosine horizon misaligned. And it's a compound change (disabled compile + 3300 horizon + fused=True AdamW), so the regression is unattributable. Revert.

## Challenger concerns
- The experiment is not attributable: it simultaneously disabled compile, changed the cosine horizon, and added fused=True AdamW. Since fused AdamW was itself a prior failed/crashing idea, the bad result cannot cleanly condemn eager RMSNorm+SwiGLU or the 3300-step horizon.
- The measured throughput contradicts the stated premise. The hypothesis assumes eager mode should reach ~3300 steps, but the run reached only 2509 steps with ~4207 MB VRAM, similar to prior slow/failed compile-rescue runs rather than the faster eager LayerNorm runs.
- The 3300-step cosine horizon was no longer aligned once the run achieved only 2509 steps. This means the LR schedule ended too early in training relative to the real budget, confounding optimizer/throughput effects with schedule mismatch.
- There is no sign of metric gaming, validation leakage, or eval tampering in the reported result, but train_seconds=300.099 slightly exceeds the nominal budget.

*Ledger: results.tsv, epoch 57.*
