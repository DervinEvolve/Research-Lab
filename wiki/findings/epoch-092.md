# Epoch 92: Splitting fused c_attn (one 1152x384 matrix, Muon-orthogonalized jointly at aspect scale s

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6799 (Δ vs best-at-time: -0.0143)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated)
- **Lab repeat:** within noise: mean 0.6872 of two runs (0.6799 and 0.6945) beats 0.6942 by 0.0070, not more than the 0.0095 margin; not kept (margin: twice the measured noise of 0.0048, from 5 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not. Both scores were printed by the generated train.py and are unverified: a repeat checks luck, not honesty
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 331.1 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e92-15ef16b7 (43706219 bytes, SHA-256 `4076927e78f952ad47ff6a71e80c85d23493442d17f1ed36f016fc0aeffdf56c`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Splitting fused c_attn (one 1152x384 matrix, Muon-orthogonalized jointly at aspect scale sqrt(3)) into three Linear layers Q, K, and V (each 384x384, each independently Newton-Schulz orthogonalized at scale 1), with Muon lr 0.02, AdamW, horizon 4100, wpe, SwiGLU, and init std unchanged, will lower val_bpb below 0.6847 because Muon's spectral update should whiten each projection separately rather than coupling Q/K/V directions in one fat matrix.

## Change
In CausalSelfAttention, replaced the fused c_attn Linear (n_embd -> 3*n_embd) with three separate bias-free Linears c_q, c_k, c_v (each 384x384), and changed forward to call them instead of splitting one output. Nothing else was edited: the existing Muon grouping (2D weights under transformer.h.) picks the three matrices up automatically, each orthogonalized on its own at aspect scale 1 instead of sqrt(3). Parameter count, init std 0.02, LRs, horizon 4100, and wpe are unchanged. One unavoidable side effect: the init RNG draw order differs, so the Q/K/V starting values are not identical to the fused layout.

## Review
The run completed 3814 steps in 300.05s and scored 0.67986 val_bpb, beating the current best 0.6942 by 0.01434 and clearing the 0.6847 target. No cheating or evaluation irregularity appears. The Challenger's warnings concern mechanistic attribution (split-QKV package vs independent orthogonalization, aspect multiplier, init RNG), not the validity of the single intended architectural change; the lab's two-run mean check will guard against a lucky single run. [LAB: keep not taken. within noise: mean 0.6872 of two runs (0.6799 and 0.6945) beats 0.6942 by 0.0070, not more than the 0.0095 margin; not kept (margin: twice the measured noise of 0.0048, from 5 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not.]

## Challenger concerns
- The result supports the split-QKV package, not independent orthogonalization as the isolated cause. Splitting also changes Muon's explicit aspect multiplier from sqrt(3) to 1 and changes initialization RNG ordering.
- The 0.01434 improvement over the recorded best clears the single-run target, but the required two-run mean has not yet been established.
- Fixed-time performance is the right endpoint, but differences in completed steps and terminal learning rate complicate mechanistic attribution. The candidate reaches 3814 steps versus 3624 in the displayed e89 primary run.

*Ledger: results.tsv, epoch 92.*
