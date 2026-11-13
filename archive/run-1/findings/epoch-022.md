# Epoch 22: Raising max_lr from 1.2e-3 to 2.0e-3, with the same 4100-step cosine horizon, warmup=100, 

- **Author:** Grok
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Raising max_lr from 1.2e-3 to 2.0e-3, with the same 4100-step cosine horizon, warmup=100, and all other settings unchanged (including the compile-fix guard), will reduce val_bpb below the current champion 0.7067 because the model is fit-limited under the 300s budget and a higher peak LR allows larger parameter updates per step, enabling faster convergence within the fixed step count.

## Change
Single-line surgical edit: max_lr changed from 1.2e-3 to 2.0e-3 (line ~280). All else (schedule, batch, compile guard, optimizer, etc.) untouched to isolate the LR hypothesis exactly.

## Review
The experiment crashed before producing a RESULT line or any measured val_bpb, so it cannot be kept under the lab rules. The current best remains 0.7067 val_bpb. The Challenger's concern is decisive: torch.compile failed lazily during the unprotected warmup pass, so this run did not actually test max_lr=2.0e-3.

## Challenger concerns
- Lazy compilation bypasses the torch.compile try-except block, causing a crash during the unprotected warmup pass.

*Ledger: results.tsv, epoch 22.*
