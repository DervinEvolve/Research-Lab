# Epoch 44: Adding a learned bias term to the lm_head while keeping input/output weights tied will low

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Adding a learned bias term to the lm_head while keeping input/output weights tied will lower val_bpb, because byte prediction has a highly skewed unigram distribution and the current bias-free/RMSNorm model has no direct global logit prior.

## Change
Changed the lm_head linear layer from bias=False to bias=True in GPT.__init__ to add a learned bias term for byte-frequency priors. The input/output embedding weight tying is preserved (wte.weight = lm_head.weight). The bias is initialized to zero via the existing _init_weights logic. No other architectural or training changes were made.

## Review
Run crashed during deferred torch.compile (Triton backend hash failure) before any training or eval occurred, yielding no val_bpb. Hypothesis therefore untested; current best remains 0.7067. Try/except around compile is ineffective as noted by Challenger.

## Challenger concerns
- The try-except block around torch.compile is ineffective because compilation is deferred until the first forward pass, which occurs outside the try-except block.

*Ledger: results.tsv, epoch 44.*
