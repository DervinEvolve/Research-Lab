# Epoch 94: Reducing only the fused QKV Muon update multiplier from sqrt(3) to 1 will lower mean val_b

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6848 (Δ vs best-at-time: -0.0093)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated)
- **Lab repeat:** within noise: mean 0.6812 of two runs (0.6848 and 0.6775) beats 0.6942 by 0.0130, not more than the 0.0147 margin; not kept (margin: twice the measured noise of 0.0073, from 7 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not. Both scores were printed by the generated train.py and are unverified: a repeat checks luck, not honesty
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 336.9 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e94-15ef16b7 (43699727 bytes, SHA-256 `db4c5d276f0cd9ef410351f9ee7b93758a71ce73dd5eeac32b82367d640a3296`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Reducing only the fused QKV Muon update multiplier from sqrt(3) to 1 will lower mean val_bpb below 0.684, because jointly orthogonalized attention projections currently receive excessively large updates. Keep fused c_attn, Muon peak LR 0.02, and all other settings unchanged.

## Change
In GPT.configure_optimizers, when a 2D parameter inside transformer.h ends with '.attn.c_attn.weight', I set a per-parameter attribute p.muon_aspect_mult = 1.0 before appending it to muon_params. In Muon.step, the aspect multiplier now reads this override via getattr(p, 'muon_aspect_mult', None); when absent it falls back to the original max(1.0, rows/cols)**0.5. This reduces only the fused QKV projection update from sqrt(3) to 1, leaving all other Muon/AdamW updates, initialization, the 4100-step horizon, and evaluation untouched.

## Review
The run finished cleanly at 0.68485 val_bpb after 3599 steps and 117.9M tokens, 0.00935 below the 0.6942 best, with params (10.72M) and VRAM (3191 MB) unchanged. That meets the keep bar: completed, improved, and the Challenger's noise warning is a confirmation-power concern, not cheating or a confound. The stated target of a mean below 0.684 was not met by this single score, and 0.00935 sits inside the 0.0147 margin, so the repeat must land near 0.674 or this will be recorded within noise. [LAB: keep not taken. within noise: mean 0.6812 of two runs (0.6848 and 0.6775) beats 0.6942 by 0.0130, not more than the 0.0147 margin; not kept (margin: twice the measured noise of 0.0073, from 7 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not.]

## Challenger concerns
- Result (0.68485) is within the lab's noise band and unlikely to clear the confirmation bar.
- Mutating nn.Parameter with an ad-hoc attribute (p.muon_aspect_mult) instead of using optimizer param_groups.

*Ledger: results.tsv, epoch 94.*
