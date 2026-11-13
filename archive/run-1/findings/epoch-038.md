# Epoch 38: Setting total_steps_estimate to 2350 to align the cosine decay schedule with the actual 30

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7865 (Δ vs best-at-time: 0.0798)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.7 s

## Hypothesis
Setting total_steps_estimate to 2350 to align the cosine decay schedule with the actual 300-second eager-mode step capacity (while disabling torch.compile to avoid compilation latency and crashes) will lower val_bpb below 0.74 because the learning rate will successfully decay to its 10% floor (1.2e-4) by the end of training instead of ending 4.5x higher.

## Change
Two minimal edits: (1) replaced the torch.compile try/except block with a direct `run_model = model` assignment (eager mode) since compile is confirmed dead on this env (e31-37); (2) changed `total_steps_estimate` from 4100 to 2350 to align the cosine decay horizon with actual eager-mode step capacity (~2400 steps observed in e33/e34). Everything else — seq_len=512, batch_size=64, eval seq_len pinned to train seq_len=512, RESULT contract, --smoke, prepare imports — is untouched.

## Review
Measured val_bpb 0.7865 is far above the champion 0.7067 and indistinguishable from prior eager runs (~0.789–0.792). The schedule alignment did not produce the predicted <0.74 improvement; the gap to the compiled baseline remains large. No credible improvement over current best, so revert.

## Challenger concerns
- Hypothesis predicted val_bpb <0.74 from schedule alignment alone, but measured 0.786 is statistically indistinguishable from prior eager runs (0.789-0.792). Effect size is zero; the schedule tweak produced no detectable improvement.
- No verification that cosine actually reached the claimed floor; total_steps_estimate is only a heuristic and the loop uses elapsed time, not step count, to stop.
- Single-run result; training stochasticity (data order, init) can easily produce ±0.005-0.01 bpb swings. Claiming 'no improvement' is safe, but any future 'improvement' from similar tweaks would require multiple seeds.

*Ledger: results.tsv, epoch 38.*
