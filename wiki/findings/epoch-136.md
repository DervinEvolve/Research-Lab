# Epoch 136: Increasing n_layer from 6 to 8 (keeping n_embd=384, n_head=6, mlp_hidden=1024, all else id

- **Author:** GPT-6 Astra
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7017 (Δ vs best-at-time: 0.0247)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 349.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e136-15ef16b7 (57871205 bytes, SHA-256 `341658055f199f747f453d9a05911c89ed8f3a167f3e5690545d22184ec8f232`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Increasing n_layer from 6 to 8 (keeping n_embd=384, n_head=6, mlp_hidden=1024, all else identical) will lower val_bpb below 0.6615 because the current 10.7M-param model underfits byte-level story structure; the extra depth adds representational capacity while the 300s budget still allows ~2500 steps.

## Change
Changed only the non-smoke training configuration in train(): n_layer increases from 6 to 8. Width, heads, MLP size, batch size, optimizer, 4100-step schedule estimate, time-budget handling, evaluation, and RESULT output remain unchanged. The existing depth-dependent residual-projection initialization automatically uses eight layers; smoke mode remains unchanged.

## Review
The 8-layer run measured 0.7017 val_bpb against the standing best of 0.6770, a regression of 0.0247 that exceeds the 0.0155 margin, so this is a real loss, not noise. Depth cost a third of the throughput: 2,519 steps and 82.5M tokens versus roughly 3,750 steps and 123M. The schedule estimate stayed at 4100, so the cosine stopped at about 60% progress with LR near 0.4x max instead of 0.1x. The prediction of below 0.6615 is refuted for this configuration, though the unannealed schedule means depth itself was not tested cleanly.

## Challenger concerns
- Unannealed cosine LR schedule confounds the depth comparison: total_steps_estimate remained 4100 while 8 layers only reached 2519 steps (60.5% cosine progress), leaving final LR at ~0.40x max_lr instead of decaying to 0.10x.
- Severe throughput/token-count confounding under fixed 300s budget: 8 layers dropped step count by 33% (2519 vs ~3750 steps; 82.5M vs ~123M tokens), so capacity gains cannot offset a 40M-token deficit.

*Ledger: results.tsv, epoch 136.*
