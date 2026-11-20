# Epoch 90: Extending the cosine schedule horizon from 4100 to 8000 steps (changing only total_steps_e

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7215 (Δ vs best-at-time: 0.0273)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 330.7 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e90-15ef16b7 (43699663 bytes, SHA-256 `49d715d2033915abd8ce3784de1d649159644f4d2b95ecef98cbd16a86144c60`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Extending the cosine schedule horizon from 4100 to 8000 steps (changing only total_steps_estimate) will lower val_bpb below 0.6900, because the current schedule decays Muon's effective learning rate to ~0.0026 by step 3600 while the model has seen only ~119M tokens (~5% of the data) and remains underfit; e84 showed that earlier decay (3500) hurt by 0.017, so later decay should help.

## Change
Changed `total_steps_estimate` from 4100 to 8000 in `train()` (line 274). No other lines, optimizer parameters, model architecture, or schedule mechanics were modified, isolating the cosine decay horizon as the sole variable.

## Review
Measured 0.7215 val_bpb vs the 0.6942 best: a regression of 0.0273, nearly 6x the 0.0048 noise and well outside the 0.0095 margin. The run was clean (3930 steps, 300.0s, 3191 MB, same params), so the loss is attributable to total_steps_estimate=8000. At 3930 steps the cosine was only ~48% through, leaving Muon LR near 0.011 instead of annealing to ~0.002; the run never got its low-LR refinement phase. Hypothesis target of <0.6900 is falsified.

## Challenger concerns
- The regression is attributable to total_steps_estimate (4100→8000) — eval, budget, params, and VRAM are unchanged — but the stated mechanism is not. e84's 'earlier decay hurt by 0.017' was an AdamW run against a 0.7067 AdamW best. Under the kept Muon package both groups share the cosine, and Muon LR is 16.7× the AdamW base, so e84 does not identify which optimizer's anneal mattered or predict the sign.
- 8000 is not a mild delay of the same anneal. At the measured 3930 steps, progress=(3930-100)/(8000-100)≈0.485, so coeff≈0.52 and Muon LR is still ~0.011 (~57% of 0.02). The 4100 schedule at ~3600 steps is already near min LR (~0.0026). This run deletes the refinement phase inside the 300s budget; it does not map 4500–5500.
- Not noise, throughput, or metric gaming. 0.7215−0.6942=0.0273 vs noise 0.0048. Steps 3930 match e88's 3972-step 4100-horizon run (reported 0.6895) better than e89's 3624, so extra steps did not cause the miss. Train loss ~0.49 at step 3900 is a high plateau, not a NaN or eval bug. Both AdamW and Muon LRs stayed high together — do not credit Muon LR alone.

*Ledger: results.tsv, epoch 90.*
