# Epoch 96: Untying the input byte embedding from the output head (delete the line `self.transformer.w

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6875 (Δ vs best-at-time: -0.0067)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 330.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e96-15ef16b7 (44093196 bytes, SHA-256 `edfa7ffc6773798a7e55fa21071406ef5cd03214a92ab712207d34164e76b6fc`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Untying the input byte embedding from the output head (delete the line `self.transformer.wte.weight = self.lm_head.weight`, leaving two independent 256x384 matrices, both init std 0.02, both in the existing AdamW weight-decay group at lr 1.2e-3) will lower val_bpb below 0.6795. The mechanism: with 256 byte tokens, the tied matrix must serve both as the input code for a byte's identity and as the output classifier over next-byte contexts, and it receives summed gradients from both roles. Untying costs only 98K parameters and no measurable throughput. Everything else stays unchanged: fused c_attn with the default sqrt(3) aspect, Muon lr 0.02, horizon 4100, wpe, SwiGLU.

## Change
Removed the single line `self.transformer.wte.weight = self.lm_head.weight` from `GPT.__init__`. This unties the input byte embedding from the output head, leaving two independent 256x384 matrices. Both are initialized with std 0.02 by the existing `_init_weights` and both remain in the AdamW weight-decay group. No other code, hyperparameters, or schedules were changed.

## Review
Untying wte and lm_head scored 0.6875 val_bpb (3989 steps, 130.7M tokens, 10.818M params), improving on the 0.6942 best by only 0.0067. That gain is smaller than the 0.0073 run-to-run noise and 0.0080 short of the 0.6795 confirmation bar, meaning a repeat would need ~0.6715 to confirm. Adding 98K parameters for a sub-noise difference that also confounds representation separation with gradient scaling does not justify keeping.

## Challenger concerns
- 0.6875 beats the 0.6942 best by only 0.0067, less than the 0.0073 noise and 0.008 short of the stated 0.6795 bar. A confirmation repeat cannot realistically clear the margin: the repeat would need to score near 0.6715 for the mean to fall under 0.6795.
- Untying confounds separate byte codes with update scale. The tied matrix took g_embed+g_head in one AdamW step; each new matrix gets only its own gradient at the same 1.2e-3. The null does not show separate codes cannot help, and a win would not have isolated the stated mechanism either.
- The edit is not a silent no-op and shows no metric gaming. params_m rose by 98304 (256*384), VRAM stayed ~3192, 3989 steps / 130.7M tokens match recent throughput, and evaluate_bpb coverage was unchanged.

*Ledger: results.tsv, epoch 96.*
