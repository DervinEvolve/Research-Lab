# Epoch 21: Raising max_lr from 1.2e-3 to 2.0e-3 (same 4100-step cosine horizon, bs=64, seq_len=512, a

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Raising max_lr from 1.2e-3 to 2.0e-3 (same 4100-step cosine horizon, bs=64, seq_len=512, all else fixed) will cut val_bpb below 0.7067 because the champion is fit-limited under 300s and a higher peak LR lets the model travel further in the same ~4300 steps before cosine decay.

## Change
Changed max_lr from 1.2e-3 to 2.0e-3 in the train() function (line 259). All other hyperparameters (batch_size=64, seq_len=512, weight_decay=0.1, warmup=100, total_steps_estimate=4100) remain identical to the epoch-14 champion configuration.

## Review
The run crashed during the warmup step due to a lazy torch.compile Triton backend failure. Because compilation is deferred until the first forward pass, it bypassed our try-except block and produced no validation score, requiring a revert.

## Challenger concerns
- Crash is unrelated to the LR change; it is a torch.compile backend failure (missing C compiler for Triton) that occurs on the first compiled forward pass.
- The existing except block only catches import-time / graph-capture errors, not the runtime BackendCompilerFailed that surfaces on first use.

*Ledger: results.tsv, epoch 21.*
