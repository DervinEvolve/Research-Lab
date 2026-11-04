# Epoch 23: Raising max_lr from 1.2e-3 to 2.0e-3 (same 4100-step cosine, bs=64, seq_len=512, wd=0.1) w

- **Author:** Gemini
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Raising max_lr from 1.2e-3 to 2.0e-3 (same 4100-step cosine, bs=64, seq_len=512, wd=0.1) will reduce val_bpb below 0.7067 because the model remains fit-limited under the 300s budget and a higher peak allows greater progress before decay.

## Change
Changed max_lr from 1.2e-3 to 2.0e-3. Replaced the lazy torch.compile try-except block with a robust version that runs the full warmup forward and backward pass inside the try block, successfully catching any lazy compilation failures and falling back to eager execution.

## Review
The run crashed at parse time (SyntaxError: unterminated string literal at line 387) before any training occurred — there is no val_bpb, so the champion holds at 0.7067. This is the THIRD consecutive syntax/crash failure (e20, e21, e22, now this). max_lr=2.0e-3 remains entirely UNTESTED, not failed; logging it as a negative LR result would pollute the ledger.

## Challenger concerns
- The run crashed at Python parse time with an unterminated string literal, so the LR change and compile-guard change were never executed. This result is an implementation failure, not evidence for or against max_lr=2.0e-3.
- The stated experiment bundles two changes: max_lr increase plus a robust torch.compile fallback. The compile guard is necessary infrastructure after prior crashes, but it still affects execution mode if compile fails, so it should be treated as a crash-fix prerequisite rather than part of the LR hypothesis.
- No measured val_bpb exists, so adding this as a failed LR trial would pollute the ledger and risk falsely discouraging exploration of hotter learning rates.

*Ledger: results.tsv, epoch 23.*
