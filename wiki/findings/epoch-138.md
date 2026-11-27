# Epoch 138: Adding causal hashed byte bigram and trigram embeddings (two 8192-entry tables, dim=384) t

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6639 (Δ vs best-at-time: -0.0131)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 336.2 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e138-15ef16b7 (68866697 bytes, SHA-256 `e2e2378dd10b32f1dc412cd41f0daa753094510e16182ae109e1a47db290908d`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding causal hashed byte bigram and trigram embeddings (two 8192-entry tables, dim=384) to the input token+position embedding sum will lower val_bpb from 0.6770 to <=0.6550 within 300s, because byte-level transformers otherwise spend early layers composing single ASCII bytes into common subwords instead of modeling syntax and story structure.

## Change
Added GPTConfig.ngram_buckets=8192 and two new embedding tables, transformer.wbi and transformer.wtri (8192 x n_embd each, default std-0.02 init, in the existing AdamW embedding group with weight decay 0.1). In GPT.forward, causal hashed ids are computed from the current byte and the previous one or two bytes (history left-padded with byte 0; multiply-XOR hash mod 8192), and the two looked-up embeddings are added to the token+position sum. Nothing else changed: optimizer settings, schedule, training loop and eval are untouched, and params_m should report about 17.0M.

## Review
The run completed and improved on the current best (0.6639 vs 0.6770, -0.0131), but that gain is below the 0.0155 confirmation margin and misses the stated target of <=0.6550. The Challenger's confound concern is credible: two new 8192x384 embedding tables add 6.29M parameters (+59%), so the improvement cannot be attributed to n-gram composition rather than cheap extra capacity, and bigram vs trigram effects are not separable. The off-by-one concern also remains unresolved in the reviewed snapshot.

## Challenger concerns
- The 0.0131 drop (0.6770 to 0.6639) misses both the stated target (<=0.6550) and the lab confirm bar (mean must beat 0.6770 by more than 0.0155, i.e. mean < 0.6615). One draw at ~1.7x the 0.0077 repeat-noise is not a confirmed finding; a matched repeat leaves the mean near 0.664 and the file does not stick.
- Both tables land in one diff and add 6.29M parameters (+59%, 10.72M to 17.01M) on the AdamW embedding group. The gain is confounded with cheap extra capacity; 'early layers freed from byte composition' is not identified, and bigram vs trigram cannot be separated.
- A true improvement requires position t's hash to depend only on bytes <= t. That shift is not in the reviewed train.py snapshot. A forward off-by-one leaks the target into both train loss and val_bpb and still passes the immutable harness, because evaluate_bpb calls the same forward.

*Ledger: results.tsv, epoch 138.*
