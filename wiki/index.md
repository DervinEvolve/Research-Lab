# Wiki Index

Knowledge base for the Research-Lab autonomous research org. This is the
institutional memory layer described in `program.md` -- every epoch's
verdict should leave a trace here, following the LLM-wiki pattern (agents
read and write markdown as their shared long-term memory across epochs).

## Structure

- `index.md` -- this file. Catalog of every page in the wiki, kept current.
- `open-questions.md` -- unresolved questions worth investigating, seeded at
  program start and appended to as new ones surface.
- `failed-approaches.md` -- reverted or negative-result hypotheses, so the
  org does not re-spend an epoch re-discovering the same dead end.
- `findings/epoch-NNN.md` -- one page per epoch of the current run, written
  by the lab after the verdict, whatever the outcome. Earlier runs' pages are
  in `../archive/run-N/findings/`. The lab never overwrites a page written by
  another experiment. Each findings page contains:
  1. The hypothesis as stated by the Planner.
  2. The result numbers (val_bpb before/after, train_steps, train_seconds,
     peak_vram_mb, params_m).
  3. The verdict (kept / reverted) and why.
  4. Links to the relevant epoch number(s) and the exact `results.tsv`
     row(s) it corresponds to.

## Conventions

- Every page cross-references epoch numbers and `results.tsv` rows -- a
  reader should always be able to go from a wiki claim to the exact
  measured row that backs it.
- Agents update the wiki **after** each epoch's verdict is reached, not
  before -- the wiki records what was measured, not what was hoped.
- If a new epoch's result **contradicts** an existing findings page, the
  contradiction is flagged (a new dated note appended to that page) rather
  than silently overwritten. Silent overwrites destroy the ability to
  detect noisy/non-reproducible effects.
- Findings pages are named by epoch number (`findings/epoch-014.md`), so
  the page and its ledger row share one key. Pages for a topic that spans
  several epochs may be added by hand alongside them.

## Current findings

_(none yet -- run 2 has not run an epoch)_

## Earlier runs

- [Run 1](../archive/run-1/README.md), 7 Jul to 13 Nov 2026: 49 epochs, 4
  keeps, best self-reported 0.7067 val_bpb at epoch 14 (never reproduced).
  Ledger: `archive/run-1/results.tsv` (as written) and
  `archive/run-1/results-dedup.tsv` (one row per epoch). Pages:
  `archive/run-1/findings/`.
