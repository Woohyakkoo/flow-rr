"""Vital Videos — 250 participants, 508 recordings, training only.

Layout: <root>/{vv100,vv250}/<GUID>_<scenario>.mp4 plus <GUID>.json per participant.
The GUID identifies the person; each has 2-4 scenario recordings. 1920x1200, 30 fps, 30 s.

The JSON carries ppg / hr / spo2 and nothing respiratory, so this is a training-only dataset
like PURE and UBFC: `gt_rr` returns None and no evaluation ever touches it.

Each file is ~1.9 GB and the corpus is 922 GB, so I/O dominates extraction — the front end's
normal two passes cost ~18 s per session here. Seeking to the 24 detection frames instead of
streaming a second time was tried and is WORSE: these are long-GOP mp4s, so each seek decodes
from the preceding keyframe and 24 scattered seeks took 31 s against 8 s for a full stream.
`probe_seek` is kept for reference but is not used.
"""
import os, glob
import numpy as np
import cv2

NAME = "VV"
from .. import config as _cfg
ROOT = os.environ.get("FLOWRR_VITALVIDEO", _cfg.VITALVIDEO)
SUBSETS = ("vv100", "vv250")


def sessions():
    """{sid: meta}. sid = VV_{guid}_{scenario} — the GUID is the person."""
    out = {}
    for sub in SUBSETS:
        for m in sorted(glob.glob(os.path.join(ROOT, sub, "*.mp4"))):
            stem = os.path.basename(m)[:-4]
            if "_" not in stem:
                continue
            guid, scen = stem.rsplit("_", 1)
            cap = cv2.VideoCapture(m)
            if not cap.isOpened():
                cap.release()
                continue
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if n < 2:
                continue
            out[f"{NAME}_{guid}_{scen}"] = {
                "subject": guid, "session": scen, "subset": sub,
                "video": m, "fps": float(fps), "n_frames": n,
            }
    return out


def frames(meta):
    """Zero-argument callable returning a fresh iterator of BGR uint8 frames."""
    path = meta["video"]

    def it():
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
        finally:
            cap.release()
    return it


def probe_seek(meta, n_probe=24):
    """Detection frames by seeking. NOT USED — measured at 31 s vs 8 s for a full stream,
    because seeking a long-GOP mp4 decodes from the preceding keyframe each time."""
    path, n = meta["video"], meta["n_frames"]
    idx = np.linspace(0, max(0, n - 1), min(n_probe, n)).astype(int)
    cap = cv2.VideoCapture(path)
    out = []
    try:
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                out.append(f)
    finally:
        cap.release()
    return out


def gt_rr(meta):
    """No respiratory reference — ppg / hr / spo2 only."""
    return None
