# Epoch 112: Applying Muon independently to the Q, K, and V row blocks of each packed c_attn.weight, in

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6721 (Δ vs best-at-time: -0.0049)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e112-15ef16b7 (43699663 bytes, SHA-256 `eefe8d9fa8f7419e62e24dc2cd2f2db884a6edca4427c5cda3332c705d76902f`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Applying Muon independently to the Q, K, and V row blocks of each packed c_attn.weight, instead of orthogonalizing their concatenation, will lower val_bpb below 0.6675 under the same 300-second budget. Independent updates should remove unnecessary coupling between query, key, and value learning.

## Change
In GPT.configure_optimizers, routed parameters ending with '.c_attn.weight' into a separate Muon parameter group with qkv_split=True while keeping all other transformer 2D weights in the default group (qkv_split=False). In Muon.__init__ and Muon.step, added support for qkv_split: after updating the single momentum buffer and computing the Nesterov-adjusted gradient, c_attn.weight is split into 3 equal row blocks (Q, K, V) along dim=0, each block is independently passed through zeropower_ns (5 steps) and scaled by its own aspect ratio max(1.0, c.size(0)/c.size(1))**0.5, and the blocks are concatenated before applying the parameter update. Model architecture, initialization, forward pass, momentum buffer, and all hyperparameters remain unchanged.

## Review
0.6721 is only 0.0049 below the 0.6770 best — equal to the 0.0048 run-to-run noise, half the 0.0095 confirmation margin, and short of the stated 0.6675 target. A keep would fail the lab repeat test even if the second score matched. The Challenger's confounds are real: 3639 steps shifts the fixed 4100-step cosine, and splitting c_attn drops aspect scaling from sqrt(3)≈1.732 to 1.0, a ~42% cut in the update that is not the hypothesized decoupling.

## Challenger concerns
- Effect size is inside the noise band. 0.6721 is 0.0049 below the 0.6770 best; measured run-to-run noise is 0.0048 and the confirmation margin is 0.0095. A single run cannot distinguish this from luck, and it misses the hypothesis's own stated target (below 0.6675).
- Step-count / schedule confound. The run got 3639 steps, versus roughly 3417-3724 across recent runs of this architecture. Because the cosine is fixed at a 4100-step estimate, more steps means both more data and a lower terminal LR. e110 showed that fuller annealing alone gives about -0.0016.
- The change bundles two mechanisms: decoupled orthogonalization of Q, K and V, and a drop in the c_attn update scale. Aspect scaling goes from sqrt(3)=1.732 for the packed 1152x384 matrix to 1.0 per 384x384 block, so the effective c_attn step size falls by about 42%. Any gain could be a plain LR reduction on c_attn rather than 'removing coupling'.
- No integrity concerns found. The eval call, the time-budget check and the data path are untouched; the momentum buffer is shared and only the NS step is split. VRAM and params are identical to the baseline. Added complexity is modest, but it is only justified by a confirmed gain (rule 6).

*Ledger: results.tsv, epoch 112.*
