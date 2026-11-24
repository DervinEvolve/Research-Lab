# Epoch 117: Lowering the final LR floor (min_lr_ratio from 0.1 to 0.0) while setting total_steps_estim

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6796 (Δ vs best-at-time: 0.0026)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 336.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e117-15ef16b7 (43699663 bytes, SHA-256 `2d3df3017a384a05466773856ded63f316bd9c6830f0aa579a25490557971958`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Lowering the final LR floor (min_lr_ratio from 0.1 to 0.0) while setting total_steps_estimate to 3600 (matching the real ~3500-3650 steps reached in 300s) will lower val_bpb below 0.6675, because the current 4100-step cosine horizon never finishes decaying before the time budget ends, so the model is cut off at a still-high LR.

## Change
Two edits, both to the schedule only: the default min_lr_ratio in lr_at_step goes from 0.1 to 0.0, and total_steps_estimate in train() goes from 4100 to 3600. Model, optimizers, batch shape and the training loop are untouched. One consequence of the plan as written: progress clamps at 1.0, so any steps past 3600 (recent runs reached up to 3677) run at LR exactly 0 and contribute nothing; a run that stops short of 3600 ends slightly above zero.

## Review
The measured val_bpb is 0.679615, which is 0.002615 worse than the current best of 0.6770 and far above the 0.6675 target, so the run failed to improve the model. Additionally, the Challenger's warning is valid: changing both the cosine horizon and the min_lr_ratio bundles two schedule alterations, preventing attribution to either component, and 136 of 3736 steps ran at exactly zero learning rate. The result is a truthful negative with no evidence of cheating, but it cannot be kept.

## Challenger concerns
- Changing both the cosine horizon and the final LR floor prevents attribution to either schedule component. This tests a bundled schedule, not whether an unfinished cosine decay was the problem.
- The fixed 3600-step horizon expired before training ended: 136 of 3736 iterations had exactly zero learning rate across both optimizers. These iterations consumed approximately 11 seconds without changing model parameters.
- The score of 0.679615 is 0.002615 worse than the accepted best, smaller than the reported 0.0048 noise estimate. It demonstrates no benefit but does not establish that lower terminal learning rates are intrinsically harmful.
- No evaluation tampering, validation leakage, or budget evasion is evident from the supplied edits and logs. This is a self-reported result, not an independent evaluation audit.

*Ledger: results.tsv, epoch 117.*
