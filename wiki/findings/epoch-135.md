# Epoch 135: Removing gradient clipping at max-norm 1.0 will lower val_bpb from 0.6770 to <=0.6650 in 3

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6769 (Δ vs best-at-time: -0.0001)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.9 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e135-15ef16b7 (43699663 bytes, SHA-256 `63ee402a0af82fa8108027b574fc8945447c902dc2c297a9fadfbdfa30a00056`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Removing gradient clipping at max-norm 1.0 will lower val_bpb from 0.6770 to <=0.6650 in 300s. Muon updates are already norm-normalized by Newton-Schulz and the AdamW embedding group sees Adam's own scale-invariance, so clip_grad_norm_ only adds a global rescale that shrinks the Adam-group gradients, a throughput cost, and a distortion of the relative gradient scale between groups.

## Change
Removed exactly one line from the training loop: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after `loss.backward()` and before `optimizer.step()`. No other code, imports, schedules, or hyperparameters were changed. The RESULT line contract, TIME_BUDGET handling, and --smoke path remain intact.

## Review
Removing clip_grad_norm_(1.0) achieved 0.6769 val_bpb (0.676869) at 3,746 steps (3,488 MB VRAM, 122.7M tokens), an improvement of only 0.000131 bpb over the 0.6770 best. This is well within the 0.0077 measured noise and far short of the 0.0155 confirmation margin and the hypothesized <=0.6650 target, with no throughput gain over recent ~3,764-3,770 step runs. Reverting avoids dropping a gradient-stability safeguard for an indistinguishable null result.

## Challenger concerns
- The improvement is only 0.000131 bpb, far below the 0.0077 measured noise and 0.0155 confirmation margin. The stated <=0.6650 target was missed; this is a null result, not evidence that removing clipping improves validation performance.
- The proposed mechanism mischaracterizes global clipping: it applies the same scalar to all gradients and therefore preserves their instantaneous relative scale across groups. Time-varying clipping can affect momentum and Adam moments, so optimizer scale-invariance also does not establish that clipping is redundant.
- There is no matched baseline run or clipping telemetry, so neither the proposed throughput benefit nor actual clipping activity is established. The unchanged step-indexed schedule also couples any speed change to terminal learning rate.

*Ledger: results.tsv, epoch 135.*
