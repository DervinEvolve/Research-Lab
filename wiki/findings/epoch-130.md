# Epoch 130: Doubling the training batch from 64 to 128 at seq_len=512 (65,536 tokens/step), with total

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6707 (Δ vs best-at-time: -0.0063)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.1 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e130-15ef16b7 (43699663 bytes, SHA-256 `7573ef0d6f0f76570417e3f7df072a8a67a1c9787cc0619a5bd51c51865c9be7`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Doubling the training batch from 64 to 128 at seq_len=512 (65,536 tokens/step), with total_steps_estimate rescaled from 4100 to 2050 so the cosine has the same shape over the run, will lower val_bpb below 0.6770 and possibly below 0.6615. Mechanism: Muon's orthogonalized updates have a fixed size regardless of gradient magnitude, so cleaner gradients improve the direction of every step, and larger batches should also raise tokens/s slightly. Model, learning rates, warmup, eval settings and everything else stay unchanged.

## Change
Non-smoke training batch_size changed from 64 to 128 (seq_len stays 512, 65536 tokens/step). total_steps_estimate changed from 4100 to 2050 so the cosine keeps the same shape: a typical ~3600-step run is at progress (3600-100)/(4100-100)=0.875, and half the steps (~1800) against 2050 is (1800-100)/1950=0.872. Warmup (100), max_lr, muon_lr, model, eval batch/coverage, and the budget loop are unchanged.

## Review
The measured 0.6707 val_bpb improves on 0.6770 by only 0.0063, well inside the 0.0155 noise margin, so it cannot confirm as a new best. The run is also confounded: train_steps=2262 overshot total_steps_estimate=2050 and clamped at min_lr for the final 212 steps, while throughput rose ~22% to 148.2M tokens seen versus ~120M, mixing schedule, token-budget, and warmup effects with any batch-size gradient-quality effect. The hypothesis is therefore not cleanly attributable.

## Challenger concerns
- Measured gain (-0.0063 val_bpb) is well inside the 0.0155 noise margin and cannot plausibly confirm as a new best.
- Cosine schedule miscalibration: throughput jumped ~22% (491k tok/s), so train_steps=2262 overshot total_steps_estimate=2050 and clamped at min_lr for the final 212 steps.
- Multiple confounds: +22% tokens_seen (148.2M vs ~120M), 2x longer warmup in tokens (6.55M vs 3.28M), and lower final LR are confounded with gradient-quality effects.
- No metric gaming or rule violations detected: eval settings, time_budget (300.01s), and VRAM (6,641 MB < 14 GB) are clean.

*Ledger: results.tsv, epoch 130.*
