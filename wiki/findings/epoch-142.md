# Epoch 142: Adding one zero-initialized embedding looked up by a strictly causal case-folded word-pref

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6564 (Δ vs best-at-time: -0.0206)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.1 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e142-15ef16b7 (68866124 bytes, SHA-256 `653a6032eb53941895b43ee775e989c24fab84e2317983cff68c70dee1b96077`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding one zero-initialized embedding looked up by a strictly causal case-folded word-prefix hash (16384 buckets × 384; at position t the key is a rolling hash of the contiguous [A-Za-z] run ending at t, length capped at 12, A-Z folded to a-z, vector multiplied by 0 on non-letters) will lower val_bpb from 0.6770 to <=0.660 at roughly 3500-3700 steps. The lookup at t is a function only of idx[:, :t+1]. A score >=0.670 falsifies the claim that this causal lexical feature, rather than leakage or noise, produced the e139/e141 gains.

## Change
Added a zero-initialized `transformer.wword` embedding (16384 x n_embd, AdamW embedding group with wd 0.1, +6.29M params) and a `GPT.prefix_hash` method; `forward` now adds the looked-up vector, multiplied by 0 on non-letter bytes, to token + position embeddings. The key at position t is built by walking backwards over idx[t], idx[t-1], ... idx[t-11] (left-padded shifts, so only current and earlier bytes are read), folding A-Z to a-z, stopping at the first non-letter or sequence start, with a polynomial hash (x31) reduced mod the prime 16381. Causality holds by construction, but I did not run a future-byte perturbation test here. Nothing else changed: optimizer settings, schedule, batch shape and eval call are as before.

## Review
Measured 0.6564 beats the 0.6770 best by 0.0206, clearing the 0.0155 margin on this single run, but this is the fourth capacity-matched variant of the same word-prefix hash family (e138 0.6639, e139 0.6596, e141 0.6575), and all were reverted for capacity confounding and missing unchanged-code repeat. The modulus change to 16381 and absence of a future-byte perturbation test leave the causal lexical mechanism unverified; the gain cannot be separated from the +6.29M parameters.

## Challenger concerns
- The improvement clears the margin on this run, but the required unchanged-code confirmation is still missing.
- Causality is plausible from the description, but the actual modified code and a perturbation test were not supplied.
- The result supports the complete embedding intervention, not a unique lexical or case-folding mechanism. The proposed 0.670 falsification threshold cannot distinguish noise from leakage or implementation errors.
- This is not an exact replication of e141: the reported modulus changed from 16384 to 16381, and backward polynomial accumulation needs an explicitly documented convention.

*Ledger: results.tsv, epoch 142.*
