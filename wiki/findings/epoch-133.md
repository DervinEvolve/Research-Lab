# Epoch 133: Switching the cosine LR schedule from step-indexed to wall-clock-indexed (progress = elaps

- **Author:** Gemini 3.8 Flash (fallback for GPT-6 Astra)
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6732 (Δ vs best-at-time: -0.0039)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 336.0 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e133-15ef16b7 (43699663 bytes, SHA-256 `6e8768d2bb4bb84db8f37b4888ecd7a322c4255edadd47463f49a9c9cb501d33`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Switching the cosine LR schedule from step-indexed to wall-clock-indexed (progress = elapsed / TIME_BUDGET, warmup unchanged) will lower val_bpb below 0.6652, because the current step-indexed schedule stops at ~84% progress when the 300s wall hits (LR ≈1.9e-4 instead of min 1.2e-4), so the model never gets its final annealing phase.

## Change
Replaced `lr_at_step` with `lr_at_time`, computing cosine decay progress directly as `elapsed / time_budget` after the 100-step linear warmup. Removed the fixed `total_steps_estimate = 4100` parameter from the training loop. Updated the optimizer learning rate update to evaluate `lr_at_time(step, elapsed, warmup_steps, time_budget, max_lr)` on each training iteration.

## Review
Measured 0.6732 vs best 0.6770 is only -0.0039, inside the 0.0077 run-to-run noise and far short of the 0.0155 margin and the stated <0.6652 target. The run also took 3770 steps (123.5M tokens), more than recent ~3234-3450 step runs, so the tiny gain is confounded with extra tokens and cannot be attributed to wall-clock annealing. The Challenger's confound stands; this does not become the baseline.

## Challenger concerns
- The delta is inside the noise band. 0.6731 vs best 0.6770 is -0.0039, while the lab's measured run-to-run noise is 0.0077 and the acceptance margin is 0.0155. The result misses the stated target (<0.6652) by about 0.008 and cannot be told apart from seed or throughput luck.
- The change is confounded with throughput. This run did 3770 steps, versus about 3234-3450 in recent runs and an estimated ~3500 for the best. More steps means more tokens (123.5M) independent of the schedule. The hypothesis itself says the old run stopped at ~84% progress, which assumes ~3450 steps. Step-count variation between runs (e.g. GPU or compile jitter) is as large as the claimed schedule effect.
- The mechanism claim is not verified. The log shows no LR values. Time-based progress also changes the LR trajectory early on, since the compile and warmup step are excluded from the clock while progress starts at 0. The warmup-to-cosine handoff is also now time-based, not step-based, which the hypothesis did not state.
- No metric gaming was found. The eval harness is untouched, the budget is respected (300.06s), VRAM is unchanged, and the RESULT line is intact. The only residual concern is that the time-indexed schedule makes the LR depend on hardware speed, so the run is no longer deterministic in its step count. This is acceptable but makes comparisons across runs harder.

*Ledger: results.tsv, epoch 133.*
