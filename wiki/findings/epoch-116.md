# Epoch 116: Adding a residual causal 1D depthwise convolution (kernel_size=4, groups=n_embd) to the in

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6798 (Δ vs best-at-time: 0.0028)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 342.4 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e116-15ef16b7 (43706700 bytes, SHA-256 `dd0346e0944df9ffc40cb65224333dd94c507de1299ee028c3959876005301cf`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding a residual causal 1D depthwise convolution (kernel_size=4, groups=n_embd) to the input embeddings (x = x + conv(pad(x))) before the first transformer block will lower val_bpb from 0.6770 to below 0.6675 within the 300s budget, because a 6-layer byte-level GPT currently wastes early attention capacity assembling local 2-4 byte character n-grams.

## Change
Added GPT.local_conv, a bias-free depthwise Conv1d (kernel_size=4, groups=n_embd, 1536 params at n_embd=384). In GPT.forward, immediately after token+position dropout and before the first block, the stream is updated as x = x + conv(left_pad(x, 3)), so position t only sees bytes t-3..t. Conv weights are zero-initialized so the residual starts as identity; default kaiming (std ~0.29) would swamp the 0.02 embeddings and confound the n-gram test. No schedule, width, attention, or optimizer changes; the 3D kernel falls on the existing AdamW embedding group, not Muon.

## Review
The run completed at 0.67980 val_bpb, 0.00280 worse than the accepted 0.6770 and above the 0.6675 acceptance threshold. It trained for 3397 steps in 300.06 seconds with 3512 MB peak VRAM and no reported integrity violations. The difference is within measured noise, so this is no demonstrated benefit—not proof that local convolution is harmful; reduced throughput and unmeasured kernel learning leave the mechanism unresolved.

## Challenger concerns
- The result (0.6798) is 0.0028 worse than best 0.6770, well inside the measured run-to-run noise of 0.0048. This is a null result, not evidence the conv hurts or helps.
- Throughput confound: the run got 3397 steps versus roughly 3465-3677 in recent runs (about 5-7% fewer tokens). Step counts vary between runs even without changes (e110 3724, e111 3581), so the deficit may be the conv's cost, compile differences, or GPU jitter. Either way, fewer steps alone could account for the small regression.
- Zero-init plus optimizer placement may have left the conv nearly inert. The 3D kernel sits in the AdamW embedding group with weight decay 0.1 and lr 1.2e-3, on an input of scale 0.02 embeddings. No conv weight norm was logged, so we cannot tell whether the mechanism was tested or the layer stayed near zero.
- The hypothesis premise is weak given the architecture: layer-1 attention with learned wpe can already attend to the previous 3 bytes cheaply, and a depthwise (per-channel) conv cannot mix channels, so it cannot form true n-gram features, only per-channel smoothing of embeddings.
- No integrity problems found: causality is preserved (left pad 3, kernel 4, sees t-3..t), the eval call is untouched, the budget was respected (300.06s), VRAM is flat (3512 MB), and the RESULT line is intact.

*Ledger: results.tsv, epoch 116.*
