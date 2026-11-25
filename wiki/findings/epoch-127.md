# Epoch 127: Adding a zero-initialized causal depthwise Conv1d (kernel 4, groups=n_embd, bias=False) to

- **Author:** GPT-6 Astra
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6904 (Δ vs best-at-time: 0.0134)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 351.3 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e127-15ef16b7 (43740669 bytes, SHA-256 `5d85c369e09d07cec6618a58da46910b5a269e5b5404e719eea6b9bdc32df8a8`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding a zero-initialized causal depthwise Conv1d (kernel 4, groups=n_embd, bias=False) to the pre-attention normalized state of every block will lower val_bpb from 0.6770 to below 0.6652 in 300s. Mechanism: bytes need local 2-4 byte n-gram features to form sub-word units, and a fixed local mixer supplies them cheaply so attention heads are freed for longer-range story context. Implementation: h = ln_1(x); h = h + conv(F.pad(h.transpose(1,2), (3,0))).transpose(1,2) with padding=0; x = x + attn(h). Nothing else changes.

## Change
Each Block now adds a zero-initialized, bias-free depthwise kernel-4 Conv1d residual to ln_1(x), using explicit left padding (3, 0) and convolution padding=0 before attention. Conv construction preserves the existing initialization RNG stream; its 3D weights use the unchanged AdamW grouping. A synthetic preflight checks output shape and prefix-logit invariance with temporarily nonzero kernels, then restores weights before training; scheduling, evaluation, budget handling, and RESULT output remain unchanged. I could not execute --smoke here; the preflight executes automatically in both smoke and full runs.

## Review
Measured val_bpb regressed from 0.6770 to 0.6904 (+0.0134), failing the experiment. Adding six depthwise k=4 causal Conv1d layers fixed the e125 crash but cut throughput to 2,997 steps (98.2M tokens, vs ~3,600-3,722 steps in recent runs), leaving the unchanged 4,100-step cosine schedule incomplete at ~0.26*max_lr while zero-init 3D weights decayed under AdamW (wd=0.1).

## Challenger concerns
- The +0.0134 val_bpb regression is not attributable to the n-gram mixer alone. The conv cut throughput to 2997 steps and 98.2M tokens versus recent ~3600-3722 step / ~118-122M token runs, while total_steps_estimate stayed 4100. Terminal progress was ~0.72, so the run ended near 0.26*max_lr instead of ~0.13*max_lr and never reached the cosine floor.
- The 3D kernel lands in weight-decayed AdamW (emb_params, wd=0.1), not Muon. Zero-init plus decay toward zero at Adam lr can leave the branch near-dead, so a worse score confounds 'local mixer is useless' with 'this placement never grew the kernel' plus the step tax.
- No metric gaming and the e125 padding crash looks fixed. Eval path, seq_len=512, and RESULT accounting are unchanged (2997*64*512=98205696). Param delta 10.729344-10.720128 equals exactly 6*384*4=9216, so the stated module is present. Preflight restore is not in the log tail.

*Ledger: results.tsv, epoch 127.*
