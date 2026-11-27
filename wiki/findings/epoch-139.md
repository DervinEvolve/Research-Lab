# Epoch 139: Adding ONE zero-initialised hashed word-prefix embedding to the input sum will lower val_b

- **Author:** Claude Sonnet 5.5 (fallback for Grok 4.7)
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6596 (Δ vs best-at-time: -0.0174)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 338.3 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e139-15ef16b7 (68866124 bytes, SHA-256 `1fc2708788f09709c3167d81171264e6a4b6ede211eccf99b5067331104ebff8`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding ONE zero-initialised hashed word-prefix embedding to the input sum will lower val_bpb from 0.6770 to <=0.6600 at ~3650 steps. The table has 16384 buckets at dim 384 and is keyed on the letters of the current partial word: a rolling hash over the contiguous run of [A-Za-z] bytes ending at position t, capped at the last 12 letters, mapped to bucket 0 when byte t is not a letter. It is computed inside forward from idx[:, <=t] only, using a 12-iteration vectorised shift-and-mask loop. The mechanism: the dominant byte-level difficulty is knowing which word you are inside, and a variable-length word-prefix key gives that identity directly, whereas e138's fixed 2/3-byte keys only gave fragments.

## Change
Added a zero-initialised 16384x384 word-prefix embedding (wword) summed with token+position embeddings. Bucket = rolling hash (12-iteration shifted loop, causal) of the contiguous [A-Za-z] run ending at t, capped at 12 letters; bucket 0 when byte t is not a letter. It goes in the AdamW embedding group (wd 0.1); nothing else changed.

## Review
The run completed at 0.659617 val_bpb after 3,680 steps in 300.04 seconds, improving on 0.6770 by 0.017383 and meeting the predicted threshold. However, the added 6.29M embedding parameters leave a credible capacity confound: the improvement over e138’s equally sized model is only 0.004283, below the measured 0.0077 noise. Under the stated no-credible-confound keep rule, this remains promising but insufficient evidence; causality also needs code verification, and no confirmation repeat establishes a new best.

## Challenger concerns
- Capacity confound: one 16384x384 table adds 6.29M parameters, identical to e138's rejected two-table change (17.01M params). Measured result 0.6596 is only 0.0043 better than e138's 0.6639, within the 0.0077 noise, so the improvement over the 0.6770 baseline cannot be attributed to the word-prefix key rather than generic added embedding capacity.
- Unverified causal mask in forward. The change is described as a 12-iteration vectorised shift-and-mask loop over 'idx[:, <=t]', but code is not shown. Evaluation passes full sequences, so any implementation that computes buckets using bytes > t would leak future information and inflate val_bpb.
- Zero initialization plus weight decay and hash bucket design are confounded with the word-prefix mechanism. Non-letter bytes all map to bucket 0, and letter-prefix hashes can collide into bucket 0 as well, so the learned embedding is a mix of 'not in word' and arbitrary prefixes, weakening interpretability.
- Single noisy run; no repeat. The 0.0174 gain over 0.6770 exceeds the 0.0155 margin only once, but the lab's own scoring requires a repeat with mean beating margin before a new best. One run cannot distinguish luck from mechanism.

*Ledger: results.tsv, epoch 139.*
