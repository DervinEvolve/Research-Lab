# Epoch 78: Enabling CUDA-fused AdamW, with all architecture, batching, learning-rate, schedule, and d

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Enabling CUDA-fused AdamW, with all architecture, batching, learning-rate, schedule, and decay settings unchanged, will increase completed training steps by at least 10% over the recent approximately 2514-step RMSNorm/SwiGLU eager runs and achieve val_bpb below 0.7675 within 300 seconds by reducing optimizer kernel-launch and memory-traffic overhead.

## Change
Only the AdamW constructor changed. configure_optimizer now takes fused and passes it to torch.optim.AdamW. train() sets fused=True when device is cuda and fused=False on CPU/--smoke, after the model is already on device. Architecture, compile path, batch, LR, horizon, weight decay, and the loop are untouched.

## Review
The run crashed before any training step completed; status is 'crash' with a Triton/Inductor backend-hash failure during lazy torch.compile in warmup. No val_bpb was measured, so the fused-AdamW hypothesis was never actually tested. Revert to preserve the last working baseline.

## Challenger concerns
- Hypothesis was never tested: the run crashed in lazy torch.compile (Inductor/Triton missing C compiler) during the warmup forward pass before optimizer.step() ever ran.
- Broken compile fallback in train.py: try/except wraps torch.compile(model), which only creates a wrapper and defers Inductor/Triton compilation to the first run_model(x, y) call.
- Optimistic speedup attribution: in a 10.7M-param eager model at B=64, T=512, AdamW kernel overhead may be smaller than the hypothesized >=10% step-count gain.

*Ledger: results.tsv, epoch 78.*
