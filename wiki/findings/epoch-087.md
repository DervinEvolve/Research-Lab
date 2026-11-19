# Epoch 87: Enabling fused AdamW on CUDA, with no other changes, will reduce optimizer overhead enough

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7155 (Δ vs best-at-time: 0.0089)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 330.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e87-15ef16b7 (43699663 bytes, SHA-256 `8924bb37f2ea310af6a25c1a0e7a3344a74a67a1a7c75fdc9eb2aac687a5c4b7`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Enabling fused AdamW on CUDA, with no other changes, will reduce optimizer overhead enough to improve 300-second validation performance below 0.7047 val_bpb. Change only the AdamW constructor to pass fused=next(self.parameters()).is_cuda, preserving the CPU path, parameter groups, learning rate, betas, and schedule.

## Change
Updated GPT.configure_optimizer in train.py to pass fused=next(self.parameters()).is_cuda to torch.optim.AdamW. No other lines were modified, preserving the CPU smoke path, parameter groups, learning rate, betas, and schedule.

## Review
Fused AdamW scored 0.7155 val_bpb at 4344 steps in 300.06s, worse than both the 0.7067 best and the 0.7078 compiled control at ~4346 steps. Throughput was unchanged (~471k tok/s, 142.3M tokens, 3191 MB), so the hypothesized optimizer-overhead win did not occur. No cheating; a clean negative, and this file must not become the baseline.

## Challenger concerns
- The stated mechanism did not occur. Fused AdamW was supposed to cut optimizer overhead and buy more steps, but the run finished 4344 steps versus ~4346 for the unfused compiled control (e79). Throughput is unchanged at ~471k tok/s. The hypothesis is falsified at the mechanism level, independent of the bpb.
- The 0.7155 score is 0.0077 worse than the 0.7078 control at an identical step count, with an identical schedule. A change that should be numerically near-equivalent produced a gap larger than the ~0.003 noise figure in the notebook. Either fused AdamW behaves differently under bf16 autocast plus clipping, or run-to-run noise is much larger than assumed. One run cannot tell these apart.
- Do not log 'fused AdamW hurts bpb' as a finding. The evidence only supports 'no throughput gain; bpb did not beat best'. A causal claim of harm from one run against a historical control is not supported.
- No sign of metric gaming. The diff is confined to the AdamW constructor, evaluation goes through prepare.evaluate_bpb with unchanged batch count and size, the budget was respected (300.06s), and VRAM is unchanged at 3191MB.

*Ledger: results.tsv, epoch 87.*
