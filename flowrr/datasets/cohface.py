"""COHFACE reader (40 subjects x 4 sessions = 160 sessions).

Layout
------
    <COHFACE>/<subject>/<session>/{data.avi, data.mkv, data.hdf5}
    subject in "1".."40", session in "0".."3"  ->  sid = COHFACE_{subject}_{session}

Both the dataset root AND every subject directory contain non-session entries
(desktop.ini, protocols, py.py, README.rst, test.sh), so BOTH levels are filtered
with `str.isdecimal()` + `os.path.isdir`.

Why data.avi and not data.mkv
-----------------------------
data.avi is the ORIGINAL capture: 640x480, 20.0 fps on all 160 sessions (verified via
CAP_PROP_FPS).  data.mkv is a derived crf-0 re-encode that *claims* 30 fps but was
produced by frame DUPLICATION -- mkv[0..8] maps to avi[0,0,1,2,2,3,4,4,5] with a
mean-absolute difference of exactly 0.  Feeding those duplicated frames to Farneback
would emit zero-flow columns at a 1-in-3 cadence and destroy the respiratory signal,
so we read the AVI and report fps = 20.0.  The front-end resamples 20 Hz -> 30 Hz.

Ground-truth respiration
------------------------
data.hdf5 holds three flat float64 arrays ('pulse', 'respiration', 'time') on a
perfectly uniform 256 Hz grid (root attr 'sample-rate-hz' == 256, each attr stored as
a length-1 array).  'pulse' really is 256 Hz (run-length of identical consecutive
samples == 1), but 'respiration' is a 256 Hz zero-order hold of a 32 Hz sensor: the
modal AND median run length of identical consecutive values is exactly 8.  It is
therefore decimated with `resp[::RESP_DECIM]` and treated as fs = 32 Hz before the
INTERFACE.md Welch recipe.  (Running the recipe at the nominal 256 Hz is not even
possible: nperseg = int(256*60) = 15360 > nfft = 8192 and scipy raises.)
"""
import os

import cv2
import h5py
import numpy as np
from scipy import signal

from .. import config as paths

NAME = "COHFACE"

FPS = 20.0            # native rate of data.avi (verified 20.0 on all 160 sessions)
RESP_FS_RAW = 256.0   # rate the 'respiration' array is *stored* at
RESP_DECIM = 8        # zero-order-hold factor of the respiration channel
RESP_FS = RESP_FS_RAW / RESP_DECIM   # 32.0 Hz -- its TRUE rate

#: Illumination condition per session index, verified against the hdf5 root attribute
#: 'illumination' on all 160 sessions (40/40 for each cell).
ILLUMINATION = {"0": "lamp", "1": "lamp", "2": "natural", "3": "natural"}


# ── enumeration ────────────────────────────────────────────────────────────────

def _session_dirs(root):
    """Yield (subject, session, dir) for real session directories only, sorted."""
    try:
        subs = sorted((d for d in os.listdir(root) if d.isdecimal()), key=int)
    except OSError:
        return
    for sub in subs:
        sdir = os.path.join(root, sub)
        if not os.path.isdir(sdir):
            continue
        try:
            sess = sorted((d for d in os.listdir(sdir) if d.isdecimal()), key=int)
        except OSError:
            continue
        for ses in sess:
            d = os.path.join(sdir, ses)
            if os.path.isdir(d):
                yield sub, ses, d


def _probe_video(path):
    """(n_frames, fps) from the container header, or None if it will not open."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if n < 2:
        return None
    return n, fps


def _illumination(h5_path, ses):
    """Illumination from the hdf5 root attrs; falls back to the session-index map."""
    try:
        with h5py.File(h5_path, "r") as f:
            v = f.attrs["illumination"]
            v = v[0] if getattr(v, "shape", ()) else v      # attrs are length-1 arrays
            return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        return ILLUMINATION.get(ses)


def sessions(root=None):
    """{sid: meta} for every readable COHFACE session. Never raises; skips bad ones."""
    root = paths.COHFACE if root is None else root
    out = {}
    for sub, ses, d in _session_dirs(root):
        video = os.path.join(d, "data.avi")
        h5 = os.path.join(d, "data.hdf5")
        if not os.path.isfile(video):
            continue
        try:
            probe = _probe_video(video)
        except Exception:
            probe = None
        if probe is None:
            continue
        n_frames, hdr_fps = probe
        # Trust the header only when it agrees with the documented 20 fps capture.
        fps = float(hdr_fps) if abs(hdr_fps - FPS) < 0.5 else FPS
        sid = f"{NAME}_{sub}_{ses}"
        out[sid] = {
            "sid": sid,
            "subject": sub,
            "session": ses,
            "fps": fps,
            "video": video,
            "hdf5": h5 if os.path.isfile(h5) else None,
            "n_frames": n_frames,
            "duration_s": n_frames / fps,
            "illumination": _illumination(h5, ses) if os.path.isfile(h5)
                            else ILLUMINATION.get(ses),
            "resp_fs": RESP_FS,
        }
    return out


# ── frames ─────────────────────────────────────────────────────────────────────

def frames(meta):
    """Zero-argument callable -> fresh iterator of BGR uint8 (480, 640, 3) frames."""
    path = meta["video"]

    def _iter():
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            cap.release()          # also runs on GeneratorExit / early break

    return _iter


# ── ground truth ───────────────────────────────────────────────────────────────

def _rr_from_resp(resp, fs):
    """INTERFACE.md Welch recipe, verbatim."""
    x = signal.detrend(resp, type="linear")
    b, a = signal.butter(3, [paths.LO / (fs / 2), paths.HI / (fs / 2)], btype="band")
    y = signal.filtfilt(b, a, x)
    nper = min(len(y), int(fs * 60))
    f, P = signal.welch(y, fs=fs, nperseg=nper, noverlap=nper // 2, nfft=8192)
    m = (f >= paths.LO) & (f <= paths.HI)
    if not m.any():
        return None
    return float(f[m][np.argmax(P[m])] * 60.0)


def gt_rr(meta):
    """Respiration rate in bpm, or None. Never raises."""
    h5 = meta.get("hdf5")
    if not h5:
        return None
    try:
        with h5py.File(h5, "r") as f:
            resp = np.asarray(f["respiration"][:], dtype=np.float64)
    except Exception:
        return None
    resp = resp[::RESP_DECIM]                     # undo the 8x zero-order hold
    if resp.size < int(RESP_FS * 20) or not np.isfinite(resp).all():
        return None
    if np.ptp(resp) == 0:
        return None
    try:
        return _rr_from_resp(resp, RESP_FS)
    except Exception:
        return None
