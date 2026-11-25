# Epoch 123: Replacing learned absolute position embeddings (wpe) with rotary position embeddings (RoPE

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6734 (Δ vs best-at-time: -0.0036)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 344.9 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e123-15ef16b7 (43044364 bytes, SHA-256 `6778de86017871429fe285a9ebb74daaefcf714fea81641a355e25d31aaf6d4c`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing learned absolute position embeddings (wpe) with rotary position embeddings (RoPE) applied to the RMS-normalized q and k in every attention layer will lower val_bpb from 0.6770 to below 0.6652 in 300s, because relative position encoding generalizes across window offsets and removes the cold-start dependence on absolute wpe slots that each get few updates.

## Change
Removed the learned `wpe` embedding from `GPT` (and its subtraction in `num_params()` and addition in `GPT.forward`), registered precomputed `rope_cos` and `rope_sin` buffers (base 10000, `head_dim = n_embd // n_head`) on `GPT`, and passed them through `Block.forward` to `CausalSelfAttention.forward`. Added `apply_rotary_emb` and applied it to `q` and `k` immediately after `F.rms_norm`, casting `cos`/`sin` to `x.dtype` to preserve bfloat16 autocast.

## Review
RoPE measured 0.6734 val_bpb at 3,348 steps, only 0.0036 below the 0.6770 best. That is inside the 0.0059 run-to-run noise, far short of the 0.0118 confirmation margin, and misses the hypothesis's own target of below 0.6652. It also cost about 7% of steps (3,348 vs roughly 3,580-3,700) and added about 120 MB VRAM. A gain this size cannot be told apart from noise, so the baseline stays, consistent with the e120 ruling on 0.6736.

## Challenger concerns
- The improvement is only 0.00360 bpb, below both the measured 0.0059 noise and the 0.0118 confirmation margin. This run does not establish the claimed improvement below 0.6652.
- The proposed mechanism overstates absolute-position undertraining: every position appears in every full training window. RoPE changes the positional inductive bias but cannot recover context missing before a window boundary.
- The fixed step-based schedule couples runtime overhead to optimization progress. At 3348 steps, the 4100-step cosine schedule remains less decayed than in faster runs. This is a valid fixed-budget comparison, but it does not isolate RoPE's representational benefit.
- The supplied change description is not sufficient to audit rotary indexing or evaluation comparability. No gaming is evident, but these checks matter after the earlier evaluation-context artifact.

*Ledger: results.tsv, epoch 123.*
