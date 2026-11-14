# Epoch 74: Reverting the architecture to standard LayerNorm and GELU MLP (with 4x hidden dimension) w

- **Author:** GPT-5
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Reverting the architecture to standard LayerNorm and GELU MLP (with 4x hidden dimension) will eliminate the torch.compile overhead and launch-overhead bottlenecks of custom SwiGLU/RMSNorm, restoring throughput to 4100+ steps and reducing val_bpb below 0.71.

## Change
Replaced the custom RMSNorm modules in each transformer block and the final norm with standard nn.LayerNorm. Replaced the SwiGLU MLP with a standard GELU MLP using 4x expansion: mlp_hidden is now 1536 for the main run and 256 for smoke. Removed RMSNorm-specific initialization while leaving optimizer, schedule, compile path, batch sizes, and evaluation contract unchanged.

## Review
The run crashed during torch.compile (inductor missing C compiler) and produced no val_bpb or training metrics. This is an environmental failure, not a test of the LayerNorm/GELU hypothesis. The existing compile fallback was known to be inadequate (flagged in e72), and the executor did not ensure a smoke test or disable compile. No measurement → no signal → revert.

## Challenger concerns
- The result is a crash (torch.compile → inductor → 'Failed to find C compiler'), NOT a measurement of the LayerNorm/GELU hypothesis. Zero signal about the architecture change. The crash is environmental: inductor needs a C compiler that isn't present, so the run never reached training.
- Process failure: this is the 4th crash in ~6 epochs from edits that ignored the mandatory smoke-test + compile-fallback protocol. The current train.py already had a bare try/except around torch.compile that we KNOW doesn't catch lazy compilation (gemini/gpt-5.5 flagged this in e72). The executor edited the architecture but left the broken compile wrapper.

*Ledger: results.tsv, epoch 74.*
