# Epoch 124: Halving the training batch size from 64 to 32 (at fixed seq_len=512, 16,384 tokens/step) w

- **Author:** GPT-6 Astra
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6956 (Δ vs best-at-time: 0.0186)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.1 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e124-15ef16b7 (43699663 bytes, SHA-256 `827a6403e9fa641eb427165a176c8158087cc6b3648037d4193a7c1c8052fd4e`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Halving the training batch size from 64 to 32 (at fixed seq_len=512, 16,384 tokens/step) will lower val_bpb from 0.6770 to below 0.6652 within the 300s budget by increasing optimizer step count by ~70% (~6,100+ steps vs ~3,580) without degrading evaluation comparability.

## Change
Changed only the non-smoke training batch_size in train() from 64 to 32, reducing tokens per step from 32,768 to 16,384 at seq_len=512. Evaluation context, batch size and coverage, the learning-rate schedule, architecture, budget handling, smoke mode, and RESULT output are unchanged.

## Review
The run completed within budget but regressed: val_bpb 0.6956 is 0.0186 above the 0.6770 best, exceeding the 0.0118 confirmation margin. The Challenger's schedule confound is credible too—total_steps_estimate stayed 4100 while the run reached 6453 steps, so the last ~2350 steps trained at the cosine floor, making this a test of B=32 plus early-finished decay rather than B=32 alone. The regression alone is decisive, and the confound prevents treating it as a clean negative result for batch size.

## Challenger concerns
- Schedule confound: total_steps_estimate stayed at 4100 while the run took 6453 steps. The cosine reached its floor (0.1x LR) at step 4100, about 190s in, so the last ~2350 steps (36% of the run) trained at minimum LR. The hypothesis said the schedule was 'unchanged', but in wall-clock terms it was compressed. The run tested 'B=32 plus an early-finished decay', not B=32 alone.
- The premise of free extra steps was only partly true. Throughput fell to ~349k tok/s, so the run saw 105.7M tokens against ~117M at B=64 (about 10% fewer). Steps rose 80%, but each step had twice the gradient noise and the run saw less data overall.
- Learning rates were not retuned for the noisier gradient. Muon lr 0.02, AdamW 1.2e-3 and clip 1.0 were all tuned at B=64. Muon's orthogonalized update has fixed magnitude regardless of gradient noise, so halving the batch raises update noise without any self-correction.
- The negative result itself is sound. The 0.6956 is 0.0186 worse than 0.6770, which exceeds the 0.0118 margin. Final train loss of ~0.48 nats (≈0.69 bpb) agrees with the val score, so there is no sign of an implementation bug. Eval settings were untouched (seq_len 512, 200 batches of 32), and the budget was respected at 300.05s.

*Ledger: results.tsv, epoch 124.*
