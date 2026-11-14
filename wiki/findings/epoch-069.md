# Epoch 69: Replacing the 6-layer model with a 4-layer model at the same width/head/seq/batch, run ful

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7733 (Δ vs best-at-time: 0.0667)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 313.1 s

## Hypothesis
Replacing the 6-layer model with a 4-layer model at the same width/head/seq/batch, run fully eager with total_steps_estimate≈3500, will raise throughput enough that extra updates/tokens outweigh lost depth and reduce val_bpb below 0.73, possibly challenging the 0.7067 best.

## Change
Changed non-smoke n_layer from 6 to 4, removed torch.compile entirely (run_model = model), and set total_steps_estimate to 3500 to match expected throughput increase. All other hyperparameters (width, heads, seq_len, batch, optimizer, LR, warmup) unchanged.

## Review
Run completed (3677 steps) but val_bpb=0.7733 is worse than 0.7067 best. Depth cut from 6→4 raised throughput yet produced clear regression; Challenger warnings on underparameterization and schedule mismatch are confirmed by the numbers.

## Challenger concerns
- Severe underparameterization due to depth reduction without width compensation
- LR schedule truncation due to underestimated step count

*Ledger: results.tsv, epoch 69.*
