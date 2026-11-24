# Epoch 114: Adding RoPE (base=10000, applied to Q and K after the rms_norm, cos/sin buffers cast to q.

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6727 (Δ vs best-at-time: -0.0043)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 339.8 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e114-15ef16b7 (45276331 bytes, SHA-256 `e4743f60874dab8c99584735c39eaf0bd93fc59b52103a2597707be7e709df45`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding RoPE (base=10000, applied to Q and K after the rms_norm, cos/sin buffers cast to q.dtype) while keeping the learned absolute wpe will lower val_bpb from 0.6770 to below 0.6675 in the 300s budget. Relative-offset information in attention is the active ingredient of e113's gain; deleting wpe was incidental. Nothing else changes.

## Change
Added RoPE (base=10000) only inside CausalSelfAttention. After the existing Q/K rms_norm, Q and K are rotated with precomputed rope_cos/rope_sin buffers (half-split convention), cast to q.dtype so bf16 autocast stays valid. Learned absolute wpe, the optimizer, the schedule, and the training loop are unchanged.

## Review
RoPE with learned absolute positions retained scored 0.67272 after 3,668 steps in 300.07s, improving on 0.6770 by only 0.00428—below the 0.0095 acceptance margin. The run appears methodologically clean, but the gain is not established beyond noise. Its 0.00489 gap from epoch 113 also does not establish that removing wpe matters; neither causal interpretation is supported by these single-run comparisons.

## Challenger concerns
- The measured val_bpb 0.6727 is only 0.0043 below the current best 0.6770, well inside the 0.0095 repeat margin. This is not evidence for RoPE as the active ingredient; it is within run-to-run noise and could be a lucky draw. The hypothesis claims relative-offset information is the active ingredient and deleting wpe was incidental, but the e113 run (RoPE with wpe removed) scored 0.6678, while this run (RoPE with wpe retained) scores 0.6727. The 0.0049 degradation when wpe is retained directly contradicts 'deleting wpe was incidental' and suggests wpe removal was a confound or active contributor in e113.

*Ledger: results.tsv, epoch 114.*
