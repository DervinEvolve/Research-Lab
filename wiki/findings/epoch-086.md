# Epoch 86: Increasing the training batch_size from 64 to 256 (only this change, leaving LR, schedule,

- **Author:** GPT-6 Astra
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7889 (Δ vs best-at-time: 0.0822)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 338.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e86-15ef16b7 (43699663 bytes, SHA-256 `61d92f8c46b62559aa4cbabe895b1bb6ae52049e510d11ea0848e369aaa8d431`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Increasing the training batch_size from 64 to 256 (only this change, leaving LR, schedule, and model untouched) will lower val_bpb below 0.7067 because the current run uses only ~3.2GB of the 24GB A10G and sees ~140M tokens; a 4x larger batch improves GPU utilization and total tokens seen within the same 300s, giving the underfit model more data.

## Change
Changed only batch_size = 64 to batch_size = 256 in the non-smoke branch of train(). Model, optimizer, learning-rate schedule, time-budget handling, evaluation coverage, smoke configuration, and RESULT output are unchanged.

## Review
The run completed but measured 0.7889 val_bpb at 1,161 steps, far worse than the 0.7067 best and even the 0.7236 baseline. Batch 256 bought only +6.8% tokens (152.2M vs 142.4M) while cutting optimizer updates 3.74x (1,161 vs 4,346), and with total_steps_estimate left at 4100 the cosine only reached ~26% progress, ending near 1.02e-3 instead of the 1.2e-4 floor. Peak VRAM also rose from 3,191 MB to 11,755 MB for negligible throughput gain.

## Challenger concerns
- Truncated cosine LR schedule confounds the 0.7889 result: total_steps_estimate remained 4100 while actual train_steps fell to 1161, so LR only reached ~26.5% of the cosine phase (~1.02e-3 at end instead of 1.2e-4).
- Batch_size=64 already nearly saturates A10G compute (~474k vs 506k tok/s): 4x batch size gained only +6.8% tokens (152.2M vs 142.4M) while cutting optimizer updates by 3.74x (1161 vs 4346) at an unscaled LR.
- Peak VRAM jumped 3.68x (3,191 MB to 11,755 MB) for negligible throughput gain (+6.8% tok/s), leaving little headroom below 14 GB.

*Ledger: results.tsv, epoch 86.*
