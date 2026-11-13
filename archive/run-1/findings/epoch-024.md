# Epoch 24: Raising max_lr from 1.2e-3 to 1.8e-3 (with all other parameters of the epoch-14 champion a

- **Author:** GPT-5
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Raising max_lr from 1.2e-3 to 1.8e-3 (with all other parameters of the epoch-14 champion architecture held constant) will reduce val_bpb below 0.7067 by accelerating convergence under the strict 300s time budget.

## Change
Changed only the training peak learning rate in train(args): max_lr is now 1.8e-3 instead of 1.2e-3. The optimizer, warmup, cosine horizon, batch/sequence shape, model architecture, evaluation settings, and RESULT contract are otherwise unchanged.

## Review
The run crashed with a torch.compile inductor triton codegen error ('Failed to find C compiler') and produced no val_bpb. This is an infrastructure failure, not a test of the learning-rate hypothesis. The hypothesis remains completely untested; reverting avoids polluting the ledger with a false negative.

## Challenger concerns
- This is the FIFTH consecutive crash with ZERO val_bpb produced. The crash is in torch.compile/inductor triton codegen — 'Failed to find C compiler' — which is an INFRASTRUCTURE failure, not an LR failure. The try-except guard around torch.compile does NOT catch this because the error surfaces lazily on first forward/backward INSIDE the training loop, not at compile() call time.
- The hypothesis (LR 1.8e-3 helps) is COMPLETELY UNTESTED. The crash gives zero evidence about learning rate. Logging this as a 'reverted' LR result would poison the ledger with a false negative.
- The mandatory pre-launch gate (py_compile + --smoke) demanded by e22/e23 notebooks was clearly NOT run — --smoke sets run_model but uses CPU (no triton), so smoke alone would NOT have caught this GPU-only inductor crash. The gate itself is insufficient for compile-path failures.

*Ledger: results.tsv, epoch 24.*
