# Epoch 103: Changing only total_steps_estimate from 4100 to 3450 will lower val_bpb below 0.6770 by le

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6769 (Δ vs best-at-time: -0.0001)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 335.3 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e103-15ef16b7 (43699663 bytes, SHA-256 `53da58fa5aee09e67190f89ba3ad03b86b428e502b6c34714d86be99dbbfeb9c`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Changing only total_steps_estimate from 4100 to 3450 will lower val_bpb below 0.6770 by letting the existing cosine schedule reach its 0.1 learning-rate floor within the 300-second QK-normalized training run. Keep the 100-step warmup, peak learning rates, QK-norm, batch size, architecture, and evaluation unchanged.

## Change
Changed only total_steps_estimate from 4100 to 3450 in train(), on the line immediately after warmup_steps = 100. The existing cosine (100-step warmup, min_lr_ratio=0.1) now reaches its 0.1 floor at step 3450 and stays clamped there if the run goes slightly past, instead of stopping near 0.155 of peak. QK-norm, peak learning rates, batch size, architecture, optimizer, and evaluation are unchanged.

## Review
The measured 0.676917 val_bpb beats the current best 0.6770 by only 0.00008, which is ~83x smaller than the measured run-to-run noise of 0.0068 and ~165x smaller than the 0.0137 confirmation margin. The Challenger's confound concern is also credible: 3709 train steps vs 3464 in epoch 102 means the schedule change is entangled with +245 extra steps and ~8M extra tokens, so any tiny gain cannot be attributed to the horizon change alone.

## Challenger concerns
- Effect size (0.0001 val_bpb) is 68x smaller than measured run-to-run noise (0.0068) and 137x smaller than the confirmation margin (0.0137).
- Throughput jumped +7.1% (3709 steps vs 3464 in e102), confounding the schedule change with +245 extra steps (~8.0M tokens) and 259 steps clamped at the 0.1 LR floor.

*Ledger: results.tsv, epoch 103.*
