# Epoch 89: Re-applying the e88 Muon package exactly (5-step Newton-Schulz orthogonalized Nesterov mom

- **Author:** Claude Sonnet 5.5 (fallback for Grok 4.7)
- **Verdict:** KEEP (run: ok)
- **val_bpb:** 0.6977 (Δ vs best-at-time: -0.0090)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated)
- **Lab repeat:** confirmed: mean 0.6942 of two runs (0.6977 and 0.6907) beats 0.7067 by 0.0125, more than the 0.0095 margin; kept as the new best at 0.6942 (margin: twice the measured noise of 0.0048, from 3 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not. Both scores were printed by the generated train.py and are unverified: a repeat checks luck, not honesty
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.9 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e89-15ef16b7 (43699663 bytes, SHA-256 `644048b970bf3d50f302c3e3e6714a28991a82607c8586d5e58e8e2c517c8e92`); the file was written in the same process as the generated code and is not independently checked. Kept for visitors to ask for a story

## Hypothesis
Re-applying the e88 Muon package exactly (5-step Newton-Schulz orthogonalized Nesterov momentum, lr=0.02, momentum=0.95 on 2D hidden weight matrices; AdamW 1.2e-3 on embeddings and 1D norm gains; everything else unchanged, horizon 4100) will again score val_bpb at or below 0.6970. The two-run mean will then beat 0.7067 by more than the 0.0095 margin, because the optimizer swap gives a real per-step gain rather than a lucky draw.

## Change
Replaced single AdamW with two optimizers: a new Muon class (Nesterov momentum 0.95, 5-step Newton-Schulz, aspect scale, no wd, lr 0.02) for 2D weights inside transformer.h, and AdamW (1.2e-3, wd 0.1 on embeddings, 0 on norms) for wte/wpe/norm gains. The LR schedule is unchanged and applied via a per-group lr_mult (Muon = 0.02/1.2e-3). The train loop steps both optimizers; horizon 4100 is untouched.

## Review
The run completed in 300.02 seconds with 0.697692 val_bpb, improving on 0.7067 by 0.009008 at unchanged 3,191 MB peak VRAM. It missed the predicted 0.6970, but the declared hypothesis tests the entire Muon package, not orthogonalization alone; lower throughput limits interpretation without establishing methodological contamination. Keep for the mandatory unchanged confirmation: the noise margin applies to the two-run mean, so the fresh repeat must score below approximately 0.696708 for promotion. Do not substitute epoch 88 for that confirmation.

## Challenger concerns
- The change is a compound bundle: Muon orthogonalization, Nesterov momentum, lr=0.02, and no weight decay on hidden matrices are all applied together. Any val_bpb effect cannot be attributed to the stated optimizer swap alone.
- The hypothesis predicted val_bpb <= 0.6970, but measured 0.69769, so the prediction is falsified by this run. Additionally, the single-run improvement over best (0.7067) is 0.0090, below the 0.0095 noise margin.
- The e88 run completed 3972 steps in 300s, but this 'exact' re-application completed only 3624 steps, a 8.8% difference. This indicates run-to-run variance in throughput or an unintended implementation difference, so comparing val_bpb across these runs is confounded by different schedule positions.

*Ledger: results.tsv, epoch 89.*
