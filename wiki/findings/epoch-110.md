# Epoch 110: Making the LR schedule wall-clock based (progress = elapsed/TIME_BUDGET_S, cosine to min_l

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6754 (Δ vs best-at-time: -0.0016)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated)
- **Lab repeat:** within noise: mean 0.6754 of two runs (0.6754 and 0.6755) beats 0.6770 by 0.0016, not more than the 0.0137 margin; not kept (margin: twice the measured noise of 0.0068, from 13 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not. Both scores were printed by the generated train.py and are unverified: a repeat checks luck, not honesty
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.7 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e110-15ef16b7 (43699663 bytes, SHA-256 `04cea8983768f406c8669c7e00dc15cf8b7367c4a25c1b6f2098a1f9b8a53075`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Making the LR schedule wall-clock based (progress = elapsed/TIME_BUDGET_S, cosine to min_lr 0.1x, warmup unchanged) instead of the fixed 4100-step estimate will lower val_bpb below 0.6770 (target ~0.665), because the run finishes at ~3450-3750 steps so cosine currently ends ~85-90% complete with terminal LR above the floor; a time-based schedule fully anneals within the 300s budget.

## Change
Replaced the step-based cosine LR function lr_at_step with lr_at_time, which keeps the first warmup_steps linear in steps and then computes cosine progress as elapsed/time_budget instead of (step-warmup)/(4100-warmup). Updated the training loop to call lr_at_time(step, elapsed, time_budget, warmup_steps, max_lr), and removed the now-unused total_steps_estimate=4100 assignment. No other model, optimizer, data, or budget behavior was changed.

## Review
The run completed cleanly in 300.06s (3,724 steps, 122.0M tokens, 3,488.5 MB VRAM) and lowered val_bpb from 0.6770 to 0.6754 (-0.00164). The Challenger raised no cheating or methodological confound concerns, noting only that the 0.00164 bpb delta is within measured run-to-run noise (0.0068) and below the 0.0137 confirmation margin. Keeping the run triggers the lab's repeat check while cleanly adopting a wall-clock cosine schedule. [LAB: keep not taken. within noise: mean 0.6754 of two runs (0.6754 and 0.6755) beats 0.6770 by 0.0016, not more than the 0.0137 margin; not kept (margin: twice the measured noise of 0.0068, from 13 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not.]

## Challenger concerns
- The improvement is only 0.00164 bpb, well below measured run-to-run noise (0.0068) and the required confirmation margin (0.0137). This is a valid candidate result, not evidence of a reliable improvement.
- Using elapsed/time_budget after step-based warmup changes the entire post-warmup cosine trajectory, not merely its endpoint. It also starts the cosine slightly past its peak when warmup ends.

*Ledger: results.tsv, epoch 110.*
