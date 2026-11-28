# Epoch 141: Adding a zero-initialized, strictly causal, case-folded word-prefix hash embedding (16,384

- **Author:** Claude Sonnet 5.5 (fallback for Grok 4.7)
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6575 (Δ vs best-at-time: -0.0196)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e141-15ef16b7 (68866124 bytes, SHA-256 `fe7552391bda757ca92326d8cc4bec508dfe6bf5d6bd2857e726709b807b5e8c`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding a zero-initialized, strictly causal, case-folded word-prefix hash embedding (16,384 buckets x 384 dim, hashing contiguous [A-Za-z] runs up to 12 bytes with A-Z folded to a-z and non-letters masked to zero) to the input embedding sum will lower val_bpb from 0.6770 to <=0.6580 within 300s.

## Change
Added a zero-initialized wword embedding (16384 x n_embd) in GPT. word_prefix_ids() builds a causal rolling hash (multiplier 31) over the contiguous [A-Za-z] run ending at position t, up to 12 bytes, with A-Z folded to a-z. Non-letter positions are masked to zero. The embedding is added to tok+pos in forward, and sits in the AdamW emb group by default.

## Review
The run completed in 300.05s and scored 0.65746, beating 0.6770 by 0.01954 and meeting the 0.6580 prediction. However, the Challenger's capacity confound remains unresolved: parameters increased from 10.72M to 17.01M, while the improvement over e139's equally sized, non-folded table is only 0.0021, below the measured 0.0077 noise. Under the stated no-credible-confound keep rule, this promising result cannot be retained; it does not establish a case-folding benefit or demonstrate cheating.

## Challenger concerns
- The case-folding part of the hypothesis cannot be attributed. This run scored 0.6575 against e139's 0.6596 for the same table without folding, a 0.0021 difference that is well inside the 0.0077 run-to-run noise.
- Capacity is confounded with the mechanism. Parameters rise from 10.72M to 17.01M, and e138's bigram/trigram tables at the identical size got 0.6639. The gain could come from any large sparse input table rather than from word-prefix structure.
- Causality was described but I have not seen the code. A hash that includes any byte after t, or an off-by-one that folds in the target byte, would be val leakage through the model's own forward pass.
- The margin is cleared by a single run only. The delta is 0.0195 against a margin of 0.0155, so the repeat must score about 0.6655 or better for the mean to confirm.
- Several implementation details are unverified: weight decay of 0.1 applies to a sparse zero-init table and shrinks rarely hit buckets, the multiplier-31 hash into 16,384 buckets will collide, and the behaviour for words longer than 12 letters is unspecified.

*Ledger: results.tsv, epoch 141.*
