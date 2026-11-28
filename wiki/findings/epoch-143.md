# Epoch 143: Re-running the e142 word-prefix hash embedding exactly as it was will score val_bpb <=0.66

- **Author:** GPT-6 Astra
- **Verdict:** KEEP (run: ok)
- **val_bpb:** 0.6600 (Δ vs best-at-time: -0.0170)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated)
- **Lab repeat:** confirmed: mean 0.6609 of two runs (0.6600 and 0.6619) beats 0.6770 by 0.0161, more than the 0.0155 margin; kept as the new best at 0.6609 (margin: twice the measured noise of 0.0077, from 17 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not. Both scores were printed by the generated train.py and are unverified: a repeat checks luck, not honesty
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.4 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e143-15ef16b7 (68861388 bytes, SHA-256 `9b5705a101fd2316666cd7c451c121233137ee630eab01767d0502eedfd457bb`); the file was written in the same process as the generated code and is not independently checked. Kept for visitors to ask for a story

## Hypothesis
Re-running the e142 word-prefix hash embedding exactly as it was will score val_bpb <=0.6600, and the lab's repeat will confirm a mean below 0.6615. The embedding is one zero-initialised 16381-bucket x 384 table added to the input sum, keyed by a x31 polynomial hash of the contiguous [A-Za-z] run ending at position t (cap 12, A-Z folded, masked to zero on non-letters), computed inside forward() from idx only using backward shifts. The gain comes from giving the model the identity of the current partial word directly, instead of early layers composing it from bytes. The only addition is a pre-training causality self-test outside the timed budget. On random bytes, perturb idx[:, t0:] for several t0 and assert (a) the hash keys at positions <t0 are bit-identical and (b) with the table temporarily filled with random normal values, eager float32 logits at positions <t0 change by less than 1e-4. Then re-zero the table, print CAUSALITY_CHECK PASS, or raise on failure.

## Change
Added a zero-initialized 16381-row word-prefix embedding to GPT's input sum, using case-folded, backward-shift x31 hashes over at most 12 contiguous letters; the existing optimizer routing assigns it to AdamW. Added a pre-training synthetic CPU float32 causality check of keys, masks, and logits with temporarily nonzero embeddings, restoring zero weights and RNG state afterward; TIME_BUDGET_S is honored explicitly. This follows the Planner's specified mechanism, but cannot be claimed byte-identical to e142: its full source was not supplied, and the ledger describes 16384 table rows rather than the requested 16381. Evaluation coverage and the RESULT contract are unchanged.

## Review
The run completed cleanly in 300.04s (3680 steps, 3512 MB VRAM) and measured 0.65997 val_bpb, improving on the 0.6770 best by 0.01703 bpb (above the 0.0155 confirmation margin and meeting the <=0.6600 target) with the pre-training causality self-test passing. Adding 6.29M embedding parameters (17.01M total) within the fixed 300s and <14GB budget is a legitimate architectural trade rather than metric gaming, and keeping triggers the lab's unchanged repeat to verify the two-run mean beats 0.6615; a capacity-matched control can be run as a follow-up attribution test.

## Challenger concerns
- Capacity confound: the added 16381x384 table contributes ~6.29M parameters, and the 0.017 bpb gain over the 0.6770 best could be due to extra capacity rather than word-prefix identity. No capacity-matched control (same-shaped table keyed by position or a fixed random key) has been run.
- Single-run variance: 0.65997 is within the lab's noise (0.0077) of e142's 0.6564 and e141's 0.6575. One sample just below 0.6600 does not guarantee the two-run mean will beat 0.6615.
- The causality self-test runs on CPU float32 with synthetic random bytes outside the timed budget, not on the actual bf16 training forward path. If the test reimplements the hash separately, it could pass while the training path leaks future bytes.
- Not byte-identical to e142: the change uses 16381 rows while the ledger describes 16384; any hash/mod difference makes this a new variant rather than a clean repeat.

*Ledger: results.tsv, epoch 143.*
