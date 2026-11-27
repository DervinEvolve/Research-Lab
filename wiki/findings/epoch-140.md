# Epoch 140: Adding a depthwise-separable 1D convolutional stem (kernel size 3, 384 channels, zero-init

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6817 (Δ vs best-at-time: 0.0047)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 338.8 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e140-15ef16b7 (44297798 bytes, SHA-256 `fd4835d23602016c725364a5c01eca8540b7134799d880e063c91fe32080e6a9`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding a depthwise-separable 1D convolutional stem (kernel size 3, 384 channels, zero-initialized output projection, ~0.44M extra params) before the first transformer block will lower val_bpb from 0.6770 to <=0.6600 within the 300s budget, because byte-level GPTs otherwise spend early layers composing ASCII bytes into common subword patterns; a local conv captures these n-gram features with far fewer parameters than the 6.3M-param hash tables from e138/e139, isolating the inductive bias from capacity.

## Change
Added CausalConvStem (RMSNorm -> left-padded causal depthwise Conv1d with kernel_size=3 -> SiLU -> zero-initialized Linear projection) to GPT.transformer and applied it as a residual before the first transformer block in GPT.forward. Note that a true depthwise-separable 384-channel kernel-3 conv adds ~0.15M params (148,992) rather than 0.44M, keeping capacity minimal while testing the local n-gram inductive bias.

## Review
The run completed but measured 0.6817 val_bpb, which is 0.0047 worse than the 0.6770 best and far from the <=0.6600 target, so it cannot be kept. The regression is inside the 0.0077 run noise and is confounded by a ~7% step tax (3497 steps / 114.6M tokens vs ~3750 / ~123M) under a cosine still anchored at 4100, so this is a non-confirmation rather than a clean rejection of local-conv bias. The stem cost more throughput and VRAM (3609 MB) than the +6.3M-param hash tables of e138/e139 (3636-3680 steps, 3512 MB), which did reach 0.6639/0.6596.

## Challenger concerns
- The +0.0047 bpb regression is not attributable to the conv stem. 3497 steps / 114.6M tokens is ~7% below the 3746-3806-step, ~123M-token reference that produced 0.6770, and the cosine is still anchored at 4100 so the LR path differs too. A no-op with the same step tax would look like this.
- 0.6817 minus 0.6770 is 0.0047, inside the 0.0077 run noise. The success target <=0.6600 was missed, but the sign is not a conclusive rejection either.
- Cost profile shows an implementation tax, not a cheap bias test. +0.15M params dropped steps to 3497 and raised VRAM to 3609 MB, worse throughput than e138/e139 at +6.3M params (3636-3680 steps, 3512 MB). Zero-init output projection also blocks grad into the depthwise filters until the pointwise layer moves, and a 3D Conv1d weight is not in the Muon transformer.h. group.

*Ledger: results.tsv, epoch 140.*
