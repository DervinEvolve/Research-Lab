# Epoch 115: Replacing learned absolute wpe with RoPE using base=1000 (instead of e113's base=10000) wi

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6693 (Δ vs best-at-time: -0.0077)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.0 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e115-15ef16b7 (44489326 bytes, SHA-256 `e85b317382f4cef38092b6b2543b52d952059b8e2621bc218109ef28780922fa`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing learned absolute wpe with RoPE using base=1000 (instead of e113's base=10000) will lower val_bpb from 0.6770 to below 0.6675 under the 300s budget, because T=512 byte sequences need faster rotary frequencies to encode short-range relative positions.

## Change
Removed learned absolute position embeddings (`wpe`) from `GPT.__init__`, `GPT.num_params`, and `GPT.forward`. Added `apply_rotary_emb` and precomputed RoPE `cos_cached`/`sin_cached` buffers with `base=1000.0` in `CausalSelfAttention.__init__`, applying them to `q` and `k` (cast to `q.dtype`) right after `F.rms_norm` in `CausalSelfAttention.forward`.

## Review
The run completed at 0.66932 val_bpb (3677 steps), a gain of 0.00768 over 0.6770, which misses both the 0.0095 margin and the hypothesis's own target of below 0.6675. The edit also bundles wpe deletion with RoPE, so against the kept best it is the e113 package again and not a test of base. The only base contrast is e115 against e113 (0.66932 vs 0.66783), a difference of +0.0015 in the wrong direction and about 0.3x the 0.0048 noise, so base=1000 shows no measurable benefit.

## Challenger concerns
- 0.66932 cannot be attributed to base=1000. The edit also deletes learned wpe, so vs the kept best (absolute wpe, no RoPE) this is the same package as e113, not a base test. The only base contrast is e115 0.66932 at 3677 steps vs e113 0.66783 at 3564 steps: +0.00149, wrong direction, ~0.3x the 0.0048 noise, despite ~113 extra steps that should have helped if base were neutral.
- Misses the keep bar and is a single unreplicated draw. Gain vs 0.6770 is 0.00768, short of the 0.0095 margin (needs <=0.6675). It is also worse than e113, so it neither confirms that near-miss nor justifies a confirmation repeat.
- The mechanism claim is underpowered and slightly mis-specified. Lowering base from 10000 to 1000 does not speed the fastest rotary channel (theta_0 = base^0 = 1); it only increases rotation on higher-index pairs (at T=512 the slowest pair goes from ~0.075 rad to ~0.63 rad). Clearing 0.6675 required beating e113 by ~0.0004, far below resolvable noise.

*Ledger: results.tsv, epoch 115.*
