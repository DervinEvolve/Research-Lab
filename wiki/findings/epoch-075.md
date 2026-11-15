# Epoch 75: Reverting the architecture to standard nn.LayerNorm and 2-layer GELU MLP (with 4x hidden d

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7352 (Δ vs best-at-time: 0.0286)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 322.2 s

## Hypothesis
Reverting the architecture to standard nn.LayerNorm and 2-layer GELU MLP (with 4x hidden dimension), combined with a robust try-except warmup pass that catches lazy compilation errors, will safely restore throughput to over 3500 steps and reduce val_bpb below 0.72 by using fused native PyTorch kernels instead of custom multi-op SwiGLU/RMSNorm.

## Change
Replaced custom RMSNorm and SwiGLU MLP with standard nn.LayerNorm and a 2-layer GELU MLP using 4x hidden dimension (mlp_hidden = 4 * n_embd). Removed the RMSNorm class and its initialization logic; adjusted Block and GPT to use nn.LayerNorm. Added a robust warmup with try-except around the first forward/backward to catch lazy torch.compile failures and fall back to eager mode safely.

## Review
The run completed and reached 0.7352 val_bpb in 3648 steps, which is a substantial improvement over recent eager ~0.767 runs but still worse than the current best 0.7067. Under the stated rule, we only keep changes that improve the current best; this does not. The Challenger's confound concern is also credible because LayerNorm, GELU-4x, and compile fallback changed together, so attribution is unclear.

## Challenger concerns
- Confounded change: the edit bundles TWO architectural swaps (RMSNorm→LayerNorm AND SwiGLU→GELU-4x) plus a warmup fallback. We cannot attribute the 0.7067→0.7352 delta to any single mechanism. This violates the one-hypothesis rule.
- The headline comparison is apples-to-oranges. The 0.7067 best was a COMPILE-ERA run (~4100 steps). This run is eager (3648 steps, 395k tok/s). The real control is the eager SwiGLU/RMSNorm run (~0.767 at ~2500 steps). So this is actually an IMPROVEMENT over the eager baseline, not a regression.
- Schedule mismatch persists and is now the live suspect. total_steps_estimate=4100 but run did 3648 steps — cosine never reaches min_lr. e73 showed matching horizon alone at 1.2e-3 didn't help, but that was a SLOWER model (2500 steps); at 3648 steps the horizon gap is smaller and retuning may now pay off.

*Ledger: results.tsv, epoch 75.*
