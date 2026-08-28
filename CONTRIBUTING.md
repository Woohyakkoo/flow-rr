# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt ruff pytest
export FLOWRR_DATASETS=/path/to/corpora
python -m pytest -q tests        # no data or GPU needed
```

## Branches and review

- `main` is protected; work on `feature/<short-name>` or `fix/<short-name>`.
- One experiment per PR. State the question, the change, and the numbers it moved.
- CI runs `ruff check` and the data-free tests. Both must pass.

## Rules that exist because breaking them cost us results

**Never split by session.** COHFACE has 4 sessions per person, MAHNOB up to 20. A
session-level split puts the same face on both sides and inflates everything.

**Never select a checkpoint on anything a label touched.** The validation criterion is the
InfoNCE on held-out subjects. If you add a selection rule, it must be computable without a
reference respiration value, or the method stops being self-supervised.

**Never re-detect the face per frame.** The signal is sub-pixel vertical motion; per-frame
alignment removes it. The box is fixed per session by construction.

**Share the speed warp within a positive pair.** The warp changes the rate, which is the very
quantity the loss compares. Drawing it independently per view asks the model to call two
different rates identical.

**Report the seed spread.** Single-seed numbers have been misleading here by several bpm.
Run at least 4 seeds and report mean ± sd.

## Adding a corpus

Add `flowrr/datasets/<name>.py` exposing:

```python
sessions(**kw) -> {sid: meta}     # sid is "<CORPUS>_<subject>_<session>"; subject must
                                  # identify a PERSON, not a recording
frames(meta)   -> callable        # zero-argument, returns a fresh BGR frame iterator
gt_rr(meta)    -> float | None    # bpm, or None if the corpus has no reference
```

The `sid` format is load-bearing: `sid.split("_")[1]` is how every split, metric and leakage
check identifies a person.

Close any frame generator you do not exhaust (`gen.close()`), or its `VideoCapture` leaks —
this silently killed a whole extraction run once the frames got large.
