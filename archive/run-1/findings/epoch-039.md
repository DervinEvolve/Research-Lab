# Epoch 39: Switching the AdamW LR schedule to a higher peak (max_lr=2.4e-3) with a cosine horizon ali

- **Author:** Grok
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Switching the AdamW LR schedule to a higher peak (max_lr=2.4e-3) with a cosine horizon aligned to the real eager step budget (total_steps_estimate=2500) will cut val_bpb below the 0.78 eager floor by >0.02, because the current 1.2e-3 peak was tuned for the ~4100-step compiled regime and in eager mode the model only takes ~2400 steps — it is undertrained, so a larger LR lets each of the fewer updates make more progress.

## Change
Minimal edit: max_lr=2.4e-3 (was 1.2e-3) and total_steps_estimate=2500 (was 4100) in train(). Exactly the two levers in the hypothesis; nothing else touched.

## Review
The run crashed before producing a RESULT line, so there is no measured val_bpb, train_steps, or tokens_seen to compare against the 0.7067 current best. The traceback shows the still-enabled torch.compile/Triton path failed, so the LR hypothesis was not actually tested. Under the rules, crashes and confounded runs must be reverted.

## Challenger concerns
- The experiment does not isolate the intended LR effect because the training environment still attempts torch.compile, which crashes with a missing C compiler before any meaningful training occurs. The observed crash is a confounding factor that prevents the hypothesis from being tested at all.

*Ledger: results.tsv, epoch 39.*
