"""MAHNOB-HCI (emotion elicitation) reader — frontal colour camera + BDF respiration belt.

Layout: ``Sessions/<N>/`` (1200 dirs, ids 1..3810, NOT dense). The directory number does
**not** encode the subject; the subject id lives in ``session.xml`` (``<subject id=...>``)
and is the only reliable source (cross-checked against the ``P{subj}-`` avi prefix and the
``Part_{subj}_`` bdf prefix, 0 mismatches over 1200 dirs). 30 subjects x 40 sessions.

Four things about this dataset drive the design below:

1. **Camera.** Each session ships 6 avis; only ``'*C1 trigger*.avi'`` (literal spaces) is
   the frontal COLOUR camera — it is the file that ``session.xml`` marks
   ``<track type="Video" color="1">`` in 1131/1131 sessions. The other five are grayscale
   side/BW views. 780x580 H.264-in-AVI, decodes cleanly under OpenCV 5.

2. **Frame rate.** ``cv2.CAP_PROP_FPS`` lies (61.0 / 61.00015860041236). The true rate is
   ``session.xml @vidRate = 60.9708`` for all 1200 sessions. We DECIMATE by 2, so
   ``frames()`` yields at 30.4854 fps — half the optical-flow cost and already on the
   front-end's 30 Hz target, so the resampling in ``frontend.resample_to_fs`` is a ~1.6%
   correction rather than a 2x decimation.

3. **Trial selection.** Every subject has 20 neutral ('N') and 20 emotional ('S') trials.
   The neutral stimulus spans only 5.3-22.2 s (measured, see below) — 1-2 breathing cycles
   at 0.1 Hz — so their belt RR label is not trustworthy. ``sessions()`` therefore returns
   only the 'S' trials by default; ``sessions(include_neutral=True)`` adds the 'N' ones.

4. **Video clipping.** The avi runs 1.14x-3.25x longer than the stimulus: after the clip
   ends the subject gives a spoken self-report, i.e. large speech/head motion that would
   swamp the sub-pixel respiratory motion. The stimulus span comes from the BDF ``Status``
   channel, whose low 16 bits take only the values {0, 268} and contain exactly 2 pulses
   (verified 1054/1054 here). The first rising edge sits at EXACTLY t = 30.000 s in every
   file (stimulus onset), the second is stimulus end. The colour avi starts at stimulus
   onset, so we keep source frames ``[0, stim_span_s * 60.9708)`` and score the belt over
   ``Resp[30.0*fs : t_trigger2*fs]``.

Belt ground truth: BDF channel ``'Resp'`` (label index 44, 256 Hz). The channel is padded
with the CONSTANT value 16 where the amplifier was idle; that step edge dumps its energy at
0.1 Hz and pins the Welch peak to the band edge, so the flat-16 runs are trimmed before the
spectral step. (Empirically the padding always lies OUTSIDE the stimulus window, so on this
dataset the trim is a no-op once the window is applied — it is kept as a guard, and it is
what makes the window-free numbers agree with the reference values.)

No fallbacks: a session whose colour avi, bdf, ``Resp`` channel or ``Status`` trigger is
missing/malformed is SKIPPED at enumeration time rather than approximated.
"""
import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import cv2
from scipy import signal

from .. import config as paths

NAME = "MAHNOB"

_ROOT = os.path.join(paths.MAHNOB, "Sessions")

VID_RATE      = 60.9708      # session.xml @vidRate; identical for all 1200 sessions
DECIMATE      = 2            # yield every 2nd frame -> 30.4854 fps
FPS           = VID_RATE / DECIMATE

BELT_LABEL    = "Resp"       # BDF label index 44, 256 Hz
STATUS_LABEL  = "Status"     # BDF trigger channel
FLAT_VALUE    = 16           # amplifier-idle padding value on the Resp channel
STIM_ONSET_S  = 30.0         # expected time of the 1st Status rising edge
ONSET_TOL_S   = 1.0          # reject a session whose 1st edge is not at STIM_ONSET_S
MIN_BELT_S    = 10.0         # shortest belt window we will put a number on

_CAM_GLOB     = "*C1 trigger*.avi"


# ───────────────────────────── BDF (BioSemi 24-bit) ─────────────────────────────
# The reference decoder in contrastive-rr-estimation/make_mahnob_belt.py is correct for
# these files but loops sample-by-sample in Python. The reshape decode below is ~50x
# faster and reads only the requested channel's byte columns via a memmap, which also
# avoids pulling the whole 4-40 MB record block through the page cache.

def _bdf_header(path):
    """Parse the BDF fixed + signal headers. Returns None if the file is not a BDF."""
    with open(path, "rb") as f:
        h = f.read(256)
        if len(h) < 256 or h[1:8] != b"BIOSEMI":
            return None
        try:
            n_rec    = int(h[236:244])
            dur_rec  = float(h[244:252])
            n_sig    = int(h[252:256])
        except ValueError:
            return None
        if n_sig <= 0 or dur_rec <= 0:
            return None
        lab = f.read(n_sig * 16)
        if len(lab) < n_sig * 16:
            return None
        labels = [lab[i * 16:(i + 1) * 16].decode("ascii", "replace").strip()
                  for i in range(n_sig)]
        # signal header block per channel: label 16 | transducer 80 | dim 8 | phys min 8 |
        # phys max 8 | dig min 8 | dig max 8 | prefilter 80 | n_samples 8 | reserved 32
        f.seek(256 + n_sig * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80))
        nsp = f.read(n_sig * 8)
        if len(nsp) < n_sig * 8:
            return None
        try:
            n_samp = [int(nsp[i * 8:(i + 1) * 8]) for i in range(n_sig)]
        except ValueError:
            return None
    return {"n_rec": n_rec, "dur_rec": dur_rec, "n_sig": n_sig,
            "labels": labels, "n_samp": n_samp}


def _bdf_channel(path, label):
    """(samples int64, fs Hz) for one channel by label, or (None, None)."""
    hd = _bdf_header(path)
    if hd is None or label not in hd["labels"]:
        return None, None
    idx    = hd["labels"].index(label)
    n_ch   = hd["n_samp"][idx]
    if n_ch <= 0:
        return None, None
    rec_b  = sum(hd["n_samp"]) * 3                      # bytes per data record
    off_b  = sum(hd["n_samp"][:idx]) * 3                # byte offset of this channel
    base   = 256 + hd["n_sig"] * 256                    # start of the data records
    avail  = os.path.getsize(path) - base
    n_rec  = min(hd["n_rec"], avail // rec_b if rec_b else 0)
    if n_rec <= 0:
        return None, None
    with open(path, "rb") as f:
        mm = np.memmap(f, dtype=np.uint8, mode="r", offset=base, shape=(n_rec, rec_b))
        raw = np.array(mm[:, off_b:off_b + n_ch * 3])   # copy out just this channel
        del mm
    b = raw.reshape(n_rec, n_ch, 3).astype(np.int64)
    v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)        # 24-bit little-endian
    v = np.where(v & 0x800000, v - 0x1000000, v).ravel()        # two's complement
    return v, n_ch / hd["dur_rec"]


def _stim_window(bdf):
    """(t_onset_s, t_end_s, n_pulses) from the Status trigger, or None."""
    st, fs = _bdf_channel(bdf, STATUS_LABEL)
    if st is None or fs <= 0 or len(st) < 2:
        return None
    on   = ((st & 0xFFFF) > 0).astype(np.int8)
    rise = np.flatnonzero(np.diff(on) > 0) + 1
    if len(rise) < 2:
        return None
    t1, t2 = rise[0] / fs, rise[1] / fs
    if abs(t1 - STIM_ONSET_S) > ONSET_TOL_S or t2 <= t1:
        return None
    return float(t1), float(t2), int(len(rise))


# ───────────────────────────────── enumeration ──────────────────────────────────

def _session_dirs():
    if not os.path.isdir(_ROOT):
        return []
    out = []
    for name in os.listdir(_ROOT):
        p = os.path.join(_ROOT, name)
        if os.path.isdir(p):
            out.append((int(name) if name.isdigit() else 1 << 40, name, p))
    return [(n, p) for _, n, p in sorted(out)]


def _trial_kind(bdf_name):
    """'S' (emotional) / 'N' (neutral) from 'Part_{subj}_{S|N}_Trial{k}_emotion.bdf'."""
    parts = os.path.basename(bdf_name).split("_")
    if len(parts) >= 3 and parts[2] in ("S", "N"):
        return parts[2]
    return None


def sessions(include_neutral=True):
    """{sid: meta} over the sessions that have BOTH a frontal colour avi and a usable bdf.

    Default is the FULL set: 527 emotional ('S') trials + 527 neutral ('N') trials from 27
    subjects. Pass ``include_neutral=False`` for the 'S' trials only.

    Caveat on the 'N' trials, kept here because it affects how their numbers should be read:
    their stimulus span is 5.3-22.2 s, i.e. roughly 1-4 breathing cycles at 0.1-0.5 Hz, so
    the belt reference for a neutral trial is a much noisier target than for an emotional
    one, and the shortest of them fall below the trainer's 370-column (12.3 s) minimum and
    are dropped from training anyway. `trial` in the meta records which kind each session is,
    so results can be broken down by trial type.
    """
    out = {}
    for name, d in _session_dirs():
        try:
            xml = os.path.join(d, "session.xml")
            if not os.path.exists(xml):
                continue
            root = ET.parse(xml).getroot()
            sub_el = root.find("subject")
            if sub_el is None:
                continue
            subject = sub_el.get("id")
            if not subject or "_" in subject:
                continue
            vid_rate = float(root.get("vidRate") or VID_RATE)
            if not np.isfinite(vid_rate) or vid_rate <= 0:
                continue

            avis = sorted(glob.glob(os.path.join(d, _CAM_GLOB)))
            bdfs = sorted(glob.glob(os.path.join(d, "*.bdf")))
            if len(avis) != 1 or not bdfs:
                continue                                   # no colour camera / no physio
            avi, bdf = avis[0], bdfs[0]

            kind = _trial_kind(bdf)
            if kind is None or (kind == "N" and not include_neutral):
                continue

            hd = _bdf_header(bdf)
            if hd is None or BELT_LABEL not in hd["labels"]:
                continue

            win = _stim_window(bdf)
            if win is None:
                continue
            t1, t2, n_pulses = win
            span = t2 - t1

            n_src = int(span * vid_rate)                   # source frames to keep
            n_out = (n_src + DECIMATE - 1) // DECIMATE     # frames actually yielded
            if n_out < 2:
                continue
            fps = vid_rate / DECIMATE

            sid = "%s_%s_%s" % (NAME, subject, name)
            out[sid] = {
                "subject":        str(subject),
                "session":        str(name),
                "fps":            float(fps),
                "video":          avi,
                "bdf":            bdf,
                "trial":          kind,
                "vid_rate":       float(vid_rate),
                "decimate":       int(DECIMATE),
                "n_src_frames":   int(n_src),
                "n_frames":       int(n_out),
                "duration_s":     float(n_out / fps),
                "stim_onset_s":   float(t1),
                "stim_end_s":     float(t2),
                "stim_span_s":    float(span),
                "n_trigger":      int(n_pulses),
                "belt_fs":        float(hd["n_samp"][hd["labels"].index(BELT_LABEL)]
                                        / hd["dur_rec"]),
            }
        except Exception:
            continue                                       # never raise on a bad session
    return dict(sorted(out.items()))


# ─────────────────────────────────── frames ─────────────────────────────────────

def frames(meta):
    """Zero-arg callable -> fresh iterator of BGR uint8 frames at meta['fps'].

    Clipped to the stimulus span and decimated by ``meta['decimate']``. Skipped frames are
    ``grab()``-ed (no colour conversion), which is materially cheaper than decoding them.
    """
    path = meta["video"]
    n_src = int(meta["n_src_frames"])
    step = max(1, int(meta.get("decimate", DECIMATE)))

    def _iter():
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return
            i = 0
            while i < n_src:
                if i % step == 0:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        return
                    yield frame
                else:
                    if not cap.grab():
                        return
                i += 1
        finally:
            cap.release()

    return _iter


# ───────────────────────────────── ground truth ─────────────────────────────────

def _drop_flat(x, value=FLAT_VALUE):
    """Trim the leading/trailing amplifier-idle runs (constant ``value``)."""
    live = np.flatnonzero(x != value)
    if live.size == 0:
        return x[:0]
    return x[live[0]:live[-1] + 1]


def _rr_bpm(y, fs):
    """The INTERFACE.md recipe, verbatim."""
    x = signal.detrend(np.asarray(y, float), type="linear")
    b, a = signal.butter(3, [paths.LO / (fs / 2), paths.HI / (fs / 2)], btype="band")
    y = signal.filtfilt(b, a, x)
    nper = min(len(y), int(fs * 60))
    f, P = signal.welch(y, fs=fs, nperseg=nper, noverlap=nper // 2, nfft=8192)
    m = (f >= paths.LO) & (f <= paths.HI)
    if not m.any():
        return None
    return float(f[m][np.argmax(P[m])] * 60.0)


def gt_rr(meta):
    """Belt RR (bpm) over the stimulus window — the same span the video is clipped to.

    The belt runs at 256 Hz; the mandated Welch call (``nfft=8192``) is only well posed at
    the 30 Hz analysis rate (``nperseg = fs*60`` would be 15360 > nfft at 256 Hz), so the
    window is resampled to ``paths.FS`` first and the recipe then applied verbatim. This
    also matches the reference belt cache in the upstream repo.
    """
    try:
        resp, fs = _bdf_channel(meta["bdf"], BELT_LABEL)
        if resp is None or fs <= 0:
            return None
        i0 = int(round(float(meta["stim_onset_s"]) * fs))
        i1 = int(round(float(meta["stim_end_s"]) * fs))
        w = _drop_flat(np.asarray(resp)[max(0, i0):max(0, i1)])
        if len(w) < int(MIN_BELT_S * fs):
            return None
        n30 = int(round(len(w) / fs * paths.FS))
        if n30 < 2:
            return None
        rr = _rr_bpm(signal.resample(w.astype(float), n30), paths.FS)
        if rr is None or not np.isfinite(rr):
            return None
        return float(rr)
    except Exception:
        return None
