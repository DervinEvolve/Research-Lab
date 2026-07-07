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
- `findings/<topic>.md` -- one page per **validated** finding. A finding
  earns a page when a hypothesis is tested, accepted (or conclusively
  rejected with a clear negative result worth remembering), and logged to
  `results.tsv`. Each findings page must contain:
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
- Findings pages are named by topic/mechanism, not by epoch number, e.g.
  `findings/rmsnorm-vs-layernorm.md`, `findings/batch-size-512.md`.

## Current findings

_(none yet -- this program has not run an epoch)_
