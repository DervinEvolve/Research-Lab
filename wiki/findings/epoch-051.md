# Epoch 51: Replacing the custom RMSNorm with PyTorch's native fused LayerNorm (eps=1e-5, elementwise_

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7293 (Δ vs best-at-time: 0.0227)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.5 s

## Hypothesis
Replacing the custom RMSNorm with PyTorch's native fused LayerNorm (eps=1e-5, elementwise_affine=True) while explicitly disabling torch.compile will increase eager-mode step throughput by at least 20%, yielding >3000 training steps in the 300-second budget and lowering val_bpb to <0.75. The increased steps allow the cosine schedule to anneal closer to the intended final LR, improving convergence.

## Change
Replaced all custom RMSNorm modules (ln_1, ln_2, ln_f) with PyTorch native nn.LayerNorm(eps=1e-5, elementwise_affine=True) and updated _init_weights to handle LayerNorm (weight=1, bias=0). Removed the RMSNorm class entirely. Explicitly disabled torch.compile (run_model = model, no compile attempt) to avoid silent eager fallback/Triton crashes. Set total_steps_estimate to 3400 to match the eager step budget so the cosine schedule anneals properly. SwiGLU MLP and standard (non-fused) AdamW are unchanged.

## Review
Completed but val_bpb 0.7293 > current best 0.7067; multiple simultaneous changes (norm type, explicit no-compile, schedule estimate) prevent clean attribution; hypothesis did not beat the compiled champion.

## Challenger concerns
- Attribution is partially confounded: the run changed RMSNorm to LayerNorm, explicitly disabled torch.compile, and changed total_steps_estimate to 3400. Throughput gain is plausibly from LayerNorm, but the val_bpb gain cannot be cleanly assigned to LayerNorm alone.
- The hypothesis frames improvement as increased steps enabling better cosine annealing, but total_steps_estimate was manually shortened to almost exactly the achieved step count. That schedule change itself improves final LR behavior independent of throughput.
- Disabling torch.compile is good experimental hygiene given recurring Triton crashes, but it changes the comparison target from the epoch-14 champion compiled regime to an eager-only regime.
- No signs of metric gaming, validation leakage, or eval tampering are visible. The RESULT line uses the standard fields, eval coverage appears unchanged, and VRAM is acceptable.
- train_seconds is 300.03s, slightly above the nominal budget. This looks like normal loop-boundary overshoot rather than budget abuse, but repeated overshoot should be watched.

*Ledger: results.tsv, epoch 51.*
