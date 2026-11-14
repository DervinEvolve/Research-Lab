# Epoch 63: Replacing the PyTorch F.scaled_dot_product_attention (SDPA) with a classic manual causal s

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.8926 (Δ vs best-at-time: 0.1859)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 329.7 s

## Hypothesis
Replacing the PyTorch F.scaled_dot_product_attention (SDPA) with a classic manual causal self-attention implementation, while wrapping the compiled warmup step in a try-except fallback block, will bypass the SDPA-related Triton compilation crashes on the A10G, restoring full-model torch.compile throughput (>4000 steps) and reducing val_bpb below 0.71.

## Change
Replaced F.scaled_dot_product_attention in CausalSelfAttention with manual matmul + causal-mask + softmax (registered a non-persistent tril mask buffer). Moved torch.compile into a GUARDED try/except that runs a real warmup forward+backward to catch lazy Triton/Inductor failures, logs 'compile OK'/'compile FAILED', and rebuilds an eager model on failure. The cosine horizon (total_steps_estimate) is now set by verified compile status: 4100 if compile_ok else 2500.

## Review
Manual causal attention replacement dropped throughput to 1286 steps (vs. 2500 eager baseline) and val_bpb to 0.8926 – a severe regression from the 0.7067 champion. The run logged no 'compile OK' line and performance was worse than pure eager, so the compile-guard hypothesis cannot be evaluated cleanly; the attention implementation itself is a fatal confound and throughput killer.

## Challenger concerns
- Manual SDPA replacement is a known throughput killer on A10G without flash-attn or memory-efficient kernels; measured 1286 steps (vs prior 2500 eager) directly explains the 0.89 val_bpb via under-training, not a test of the compile-guard hypothesis.
- Compile status now controls the cosine horizon, but the run logged no 'compile OK' line and delivered eager-like (actually worse) throughput; the attribution of 0.89 to 'compile failure handling' is confounded by the attention change itself.
- No verification that the manual causal mask produces identical logits to SDPA (within fp16 noise); a subtle off-by-one or dtype error in the tril buffer could silently degrade learning without crashing.

*Ledger: results.tsv, epoch 63.*
