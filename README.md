# flow-rr

Self-supervised respiratory-rate estimation from facial video, with **no respiration labels
used at any point in training or model selection**.

A fixed face box is gridded into 8×8 patches, the vertical component of dense optical flow is
averaged per patch to give a spatio-temporal map, and a 2D CNN is trained with InfoNCE over
band-limited power spectra. The method follows CalibrationPhys (Akamatsu et al., *IEEE J-BHI*
2024) with one substitution: that paper builds positive pairs from two cameras filming the
same person simultaneously, which single-camera corpora cannot provide, so positives here are
two interleaved halves of one clip's ROI grid.

```
video ─▶ fixed face box ─▶ 128×128 gray ─▶ dense optical flow ─▶ vertical component
      ─▶ 8×8 patch means ─▶ ST-map (M=64, T) @30 Hz ─▶ decimate to 15 Hz
      ─▶ PhysNet2D (290,817 params) ─▶ waveform ─▶ band PSD 0.1–0.5 Hz ─▶ argmax ─▶ bpm
```

## Why the face box is fixed

The box is the median of 24 detections, expanded ×1.5, and then **held constant for the whole
session**. This is not an optimisation — per-frame tracking destroys the signal. What the
model measures is the sub-pixel vertical displacement of the head and torso driven by
breathing; re-centring every frame on its own detection subtracts exactly that quantity.

## Install

```bash
git clone <this repo> && cd flow-rr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # optional, so `import flowrr` works from anywhere
```

## Point it at your data

Only one variable is required. The corpora are licensed separately and are **not** in this
repository.

```bash
export FLOWRR_DATASETS=/path/to/corpora     # holds cohface/ MANHOB_HCI/ PURE/ UBFC-rPPG/ vv250/
export FLOWRR_DATA=/path/to/scratch         # optional; defaults to ./data
```

Expected layout under `$FLOWRR_DATASETS`:

| corpus | directory | reference signal | used for |
|---|---|---|---|
| COHFACE | `cohface/` | respiration belt (hdf5) | train + **evaluate** |
| MAHNOB-HCI | `MANHOB_HCI/emotion_elicitation/` | respiration belt (BDF `Resp`) | train + **evaluate** |
| PURE | `PURE/` | none | train only |
| UBFC-rPPG | `UBFC-rPPG/` | none | train only |
| VitalVideo | `vv250/` | none | train only |

Set `FLOWRR_COHFACE`, `FLOWRR_MAHNOB`, … individually if your layout differs.

## Run

```bash
bash scripts/00_build_stmaps.sh     # video -> ST-maps + manifest   (hours, CPU-bound)
bash scripts/01_make_splits.sh      # subject-level 7:1:2 splits     (seconds)
bash scripts/02_train_eval.sh       # train one model and score it   (~3 min on one GPU)
```

or directly:

```bash
python -m flowrr.preprocess.build_stmaps --jobs 24
python -m flowrr.splits
python -m flowrr.train --train cohface,pure --eval cohface,mahnob --seed 42 --tag demo
```

`--train` takes any comma-separated subset of the five corpora. A labelled corpus contributes
its train split only; an unlabelled one contributes everything except a seeded 10 % val slice.

## What the splits guarantee

Splits are **subject-level**, 7:1:2, seeded. This matters more than it sounds: COHFACE records
4 sessions per person and MAHNOB up to 20, so a session-level split would put the same face on
both sides of the boundary and every reported number would be inflated.

Checkpoints are selected on the **validation InfoNCE over held-out subjects** — a label-free
criterion. No reference respiration value is read until the final scoring pass.

## Reading the numbers

`flowrr.evaluate` reports MAE, RMSE, Pearson *r* and ICC(1,1) over sessions, with subjects
namespaced by dataset (COHFACE subject "10" and MAHNOB subject "10" are different people).

Four readouts are available (`--readout`). They fail differently and the default is chosen for
that reason:

- **`psd_sum`** (default) — sums the spectra of all sliding windows, then one argmax. It *can*
  return the band edge verbatim, which is a useful tell that a model has collapsed.
- `mean` / `median` — per-window argmax then average. Structurally cannot emit the band edge,
  so they hide collapse rather than showing it.
- `snr_mean` — as `mean`, weighted by each window's in-band peak-to-median ratio.

## One property of the objective you should know before changing it

`band_psd` slices 0.1–0.5 Hz and normalises the slice to sum 1, so **energy outside the band
cannot influence the loss**. This keeps the objective scale-free and rate-focused, but it also
means the loss cannot distinguish a working model from one whose output has left the band
entirely — and checkpoint selection on the validation InfoNCE inherits that blind spot. If you
add a regulariser, this is the reason one is worth adding.

## Layout

```
flowrr/
  config.py            paths and signal constants, all env-overridable
  datasets/            one reader per corpus: session discovery + frames() + reference value
  preprocess/
    frontend.py        face detection, fixed box, optical flow, patch means, resampling
    build_stmaps.py    corpus -> data/stmaps/*.npz + data/manifest.json
  splits.py            subject-level 7:1:2 splits; training-pool assembly
  model.py             PhysNet2D (CalibrationPhys Fig. 4)
  losses.py            band-limited PSD + InfoNCE
  readout.py           waveform -> bpm, four variants
  train.py             training loop, inference, CLI
  evaluate.py          MAE / RMSE / Pearson / ICC
scripts/               three shell entry points, in order
tests/                 data-free shape and behaviour guards
```

## Citation

The method reimplemented here:

```bibtex
@article{akamatsu2024calibrationphys,
  title   = {CalibrationPhys: Self-supervised Video-based Heart and Respiratory Rate
             Measurements by Calibrating Between Multiple Cameras},
  author  = {Akamatsu, Yusuke and Umematsu, Terumi and Imaoka, Hitoshi},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2024}
}
```
