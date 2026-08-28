"""Single source of truth for paths and signal constants.

Everything here is overridable from the environment so a teammate only has to point
FLOWRR_DATASETS at their copy of the corpora; nothing else needs editing, and no path in
this repository is specific to one machine.

  FLOWRR_DATASETS   parent directory holding cohface/ MANHOB_HCI/ PURE/ UBFC-rPPG/ vv250/
  FLOWRR_DATA       where ST-maps, splits, checkpoints and results are written
  FLOWRR_STMAPS     ST-map directory (defaults to $FLOWRR_DATA/stmaps)
"""
import os

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get("FLOWRR_ROOT", _here)
DATASETS = os.environ.get("FLOWRR_DATASETS", os.path.join(ROOT, "datasets"))
DATA = os.environ.get("FLOWRR_DATA", os.path.join(ROOT, "data"))

STMAPS = os.environ.get("FLOWRR_STMAPS", os.path.join(DATA, "stmaps"))
MANIFEST = os.path.join(DATA, "manifest.json")
SPLITS = os.path.join(DATA, "splits.json")
MODELS = os.path.join(DATA, "models")
PREDS = os.path.join(DATA, "preds")
RESULTS = os.path.join(DATA, "results")
ASSETS = os.path.join(ROOT, "assets")
YUNET = os.path.join(ASSETS, "yunet.onnx")

# corpus roots — override individually if your layout differs
COHFACE = os.environ.get("FLOWRR_COHFACE", os.path.join(DATASETS, "cohface"))
MAHNOB = os.environ.get("FLOWRR_MAHNOB",
                        os.path.join(DATASETS, "MANHOB_HCI", "emotion_elicitation"))
PURE = os.environ.get("FLOWRR_PURE", os.path.join(DATASETS, "PURE"))
UBFC = os.environ.get("FLOWRR_UBFC", os.path.join(DATASETS, "UBFC-rPPG"))
VITALVIDEO = os.environ.get("FLOWRR_VITALVIDEO", os.path.join(DATASETS, "vv250"))

# ── signal constants ─────────────────────────────────────────────────────────
FS = 30.0                 # rate the ST-map is stored at
FS_TRAIN = 15.0           # rate the trainer decimates to; see preprocess note in train.py
LO = float(os.environ.get("FLOWRR_LO", 0.1))    # respiratory band, Hz
HI = float(os.environ.get("FLOWRR_HI", 0.5))    # 0.1-0.5 Hz = 6-30 bpm
GRID = int(os.environ.get("FLOWRR_GRID", 8))    # GRID x GRID ROI blocks -> M = GRID**2
CROP = 128                # face crop, pixels
BOX_SCALE = 1.5           # expansion of the median detected face box

T_CLIP = 300              # training clip length in columns, at FS_TRAIN -> 20 s
TAU = 0.1                 # InfoNCE temperature
TAUG_M, TAUG_P = 30, 60   # speed-warp range: Lp ~ U[T-30, T+60] -> effective RR x0.90-x1.20

# per-dataset ROI grid is fixed; M is derived, never hardcoded
M = GRID * GRID


def ensure_dirs():
    for d in (DATA, STMAPS, MODELS, PREDS, RESULTS):
        os.makedirs(d, exist_ok=True)
