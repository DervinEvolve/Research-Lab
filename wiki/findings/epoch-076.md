# Epoch 76: Reducing weight_decay from 0.1 to 0.01 will lower val_bpb by at least 0.02 (to ~0.745 or b

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7895 (Δ vs best-at-time: 0.0828)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 323.2 s

## Hypothesis
Reducing weight_decay from 0.1 to 0.01 will lower val_bpb by at least 0.02 (to ~0.745 or below) because the current strong regularization is preventing the small model from fitting the training distribution within the limited number of eager-mode steps.

## Change
Single-line optimizer change: weight_decay passed to configure_optimizer lowered from 0.1 to 0.01 (line now reads weight_decay=0.01). Also wrapped the compile warmup forward/backward in a try-except that falls back to eager, per the e73/e74/e75 notebook guidance, since the box's inductor crashes lazily — this prevents a crash but does not alter the hypothesis test. Everything else (architecture, schedule, batch shape, RESULT contract, --smoke) is unchanged.

## Review
Run completed cleanly (2513 steps) but val_bpb=0.7895 regressed vs baseline 0.7236 and current best 0.7067; hypothesis target of ~0.745 missed by 0.044. Challenger correctly flags non-paired test (RMSNorm/SwiGLU + schedule mismatch) and lack of control, so result is tainted and does not support keep.

## Challenger concerns
- The run is not a clean paired A/B for weight_decay because the compile warmup fallback was also changed and there is no same-code control with weight_decay=0.1, total_steps_estimate=4100, RMSNorm/SwiGLU, and identical fallback behavior.
- The hypothesis target appears benchmarked against the recent LayerNorm+GELU eager result, but this experiment ran the reverted RMSNorm+SwiGLU model at ~2513 steps, not the 3648-step LayerNorm+GELU configuration.
- The bad result may be partly schedule-mediated: with only 2513 steps and total_steps_estimate=4100, LR remains relatively high late in training. Lower regularization under a high-LR schedule may worsen validation without proving wd=0.01 is intrinsically bad.
- No metric-gaming indicators are visible: eval uses prepare.evaluate_bpb, eval coverage is unchanged, RESULT is valid JSON, and the loop stops at the time budget.

*Ledger: results.tsv, epoch 76.*
