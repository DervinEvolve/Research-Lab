# Epoch 53: In the hard-disabled eager LayerNorm+SwiGLU configuration from epoch 51, increasing train 

- **Author:** Grok
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7521 (Δ vs best-at-time: 0.0454)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 317.6 s

## Hypothesis
In the hard-disabled eager LayerNorm+SwiGLU configuration from epoch 51, increasing train batch_size from 64 to 96 while retuning the cosine horizon to the actual ~2700-step budget will lower val_bpb versus 0.7293, because the A10G is under-filled at batch 64 and the run should see materially more tokens without exceeding VRAM.

## Change
Minimal edits: (1) replaced all RMSNorm with nn.LayerNorm(eps=1e-5) in Block/GPT and removed RMSNorm class; (2) hard-disabled compile (run_model=model, removed try/except); (3) batch_size=96 and total_steps_estimate=2700 in non-smoke path. All other code (SwiGLU, schedule, etc.) untouched.

## Review
val_bpb 0.7521 is worse than the e51 eager LN+SwiGLU baseline (0.7293) and far from the current best (0.7067). The larger batch reduced steps to 2188 (vs 3360) and did not increase token throughput enough to compensate, resulting in poor convergence.

## Challenger concerns
- Mismatched learning rate annealing due to overestimating step budget
- Hypothesis disproven: Larger batch size decreased token throughput

*Ledger: results.tsv, epoch 53.*
