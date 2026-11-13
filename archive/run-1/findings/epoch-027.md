# Epoch 27: Setting total_steps_estimate=2500 (matching the ~2500 steps actually achievable in 300s) w

- **Author:** GPT-5
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Setting total_steps_estimate=2500 (matching the ~2500 steps actually achievable in 300s) while keeping max_lr=1.2e-3 and all else at the epoch-14 champion config will cut val_bpb below 0.7067 because the cosine schedule will fully decay the LR to its 0.12e-3 floor by eval time instead of being truncated at ~60% progress with a hot LR.

## Change
Changed the fixed cosine schedule horizon in train() from total_steps_estimate = 4100 to total_steps_estimate = 2500. All architecture, optimizer settings, batch/sequence sizes, warmup, max_lr, evaluation settings, and RESULT contract are otherwise left unchanged to isolate the static-horizon hypothesis.

## Review
The run crashed during the startup phase due to a Triton/torch.compile backend error (missing C++ compiler tools). Since no validation BPB was produced, the hypothesis remains untested and the change must be reverted.

## Challenger concerns
- Crash is entirely from torch.compile + missing C compiler (Triton driver), not the total_steps_estimate=2500 change; hypothesis is untested.
- Change cannot be isolated; infrastructure failure masks whether 2500-step horizon would improve or hurt bpb.

*Ledger: results.tsv, epoch 27.*
