# Epoch 37: In explicit eager mode, changing the training shape from seq_len=512,batch_size=64 to seq_

- **Author:** Gemini
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.8371 (Δ vs best-at-time: 0.1305)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 311.0 s

## Hypothesis
In explicit eager mode, changing the training shape from seq_len=512,batch_size=64 to seq_len=256,batch_size=256 will lower val_bpb below the recent eager ~0.789 floor, plausibly into 0.71-0.74, because halving attention length while doubling tokens/step should raise tokens/sec substantially under the 300s budget while retaining enough context for TinyStories byte prediction.

## Change
Changed seq_len from 512 to 256 and batch_size from 64 to 256 in the standard training configuration. Removed torch.compile to guarantee explicit eager execution and bypass the broken Triton environment. Adjusted total_steps_estimate to 1800 to match the expected throughput of the new batch shape.

## Review
Run completed but val_bpb=0.8371 is the worst eager result yet (vs prior 0.789 and champion 0.7067). Eval was performed at seq_len=256 while all prior numbers used 512, making the metric non-comparable; hypothesis of retained context and better tokens/sec is falsified. Additional confounds: total_steps_estimate=1800 left LR hot and compile removal was already the status quo.

## Challenger concerns
- val_bpb is computed at seq_len=256 (the harness is called with seq_len=seq_len, which is now 256), while the 0.7067 champion and the ~0.789 eager baseline were all evaluated at seq_len=512. The eval context length changed with the training shape, so the 0.837 number is NOT comparable to any prior ledger entry — this is a confounded/partially gamed comparison.
- Despite train loss dropping to ~0.57 (lower than anything we've logged), val_bpb is the WORST eager result yet (0.837 vs 0.789). That train/val divergence is a red flag: the short 256-context model is memorizing/fitting shorter spans but generalizing worse at eval, OR the eval-length mismatch is inflating bpb. Either way the hypothesis ('enough context retained') is falsified by the number.
- Compound change: the hypothesis isolates 'training shape', but the run ALSO removed torch.compile. Since compile was already dead (eager anyway), this confound is mild, but total_steps_estimate was also retuned to 1800 while only 1335 steps ran — the cosine never reached its floor, leaving LR hot at eval, which independently degrades bpb.

*Ledger: results.tsv, epoch 37.*
