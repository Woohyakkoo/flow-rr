"""PURE dataset reader (59 sessions, 10 subjects, 6 tasks).

Layout (verified exhaustively for all 59 sessions):

    <PURE>/<XX-YY>/<XX-YY>/Image<timestamp_ns>.png     640x480 uint8 BGR
    <PURE>/<XX-YY>/<XX-YY>.json                        CMS50E pulse-ox + frame timestamps

XX = subject 01..10, YY = task 01..06
    01 steady, 02 talking, 03 slow translation, 04 fast translation,
    05 small head rotation, 06 medium head rotation.
Session 06-02 does not exist, so the count is 59 and subject 06 has only 5 sessions.

The dataset root ALSO contains ``PURE_zip`` (the source archives) and
``Intra_PURE_rhythmformer`` (an unrelated rPPG-Toolbox preprocessing cache).  Both are
excluded by the ``^\\d{2}-\\d{2}$`` directory filter -- do not replace it with a glob.

FRAME TIMING.  Timestamps are strictly monotonic with no duplicates, but ~118 frames are
dropped across 27 of the 59 sessions (67 / 100 / 167 ms gaps instead of 33.3 ms).  We
report ``meta['fps'] = (n-1)/(t_last-t_first)`` per session (29.85-30.00 Hz), which makes
the *total duration* of the ST-map exact and therefore leaves the respiration-band
frequency estimate unbiased; what it leaves behind is a local time-warp of at most 250 ms
(worst session 07-01; median session 0.03 ms).  The raw timestamps are exposed as
``meta['timestamps_ns']`` so a later stage can do exact non-uniform resampling if wanted.

GROUND TRUTH.  The JSON has only '/FullPackage' (CMS50E: BVP waveform, pulseRate,
o2saturation + status flags) and '/Image' (frame timestamps).  There is NO respiration,
chest-belt, airflow or CO2 channel anywhere in the dataset -- verified by unioning the
Value keys over all 59 JSONs.  ``gt_rr`` therefore returns None for every session.  A
BVP-derived pseudo-label is deliberately NOT provided: RIIV and AM proxies were measured
disagreeing by up to 6 bpm on the same session, so such a label would be noise dressed up
as truth.
"""
import os
import re
import cv2

from .. import config as paths

NAME = "PURE"

_SESS_RE = re.compile(r"^(\d{2})-(\d{2})$")

TASKS = {
    "01": "steady",
    "02": "talking",
    "03": "slow_translation",
    "04": "fast_translation",
    "05": "small_head_rotation",
    "06": "medium_head_rotation",
}
# Tasks 03-06 ask the subject to move deliberately; the front-end holds ONE fixed crop box
# per session (by design -- tracking would cancel the breathing motion we measure), so on
# these the head drifts inside the box.
MOVING_TASKS = {"03", "04", "05", "06"}


def _timestamps(img_dir):
    """Sorted int64-safe list of frame timestamps (ns) parsed from the file names."""
    try:
        names = os.listdir(img_dir)
    except OSError:
        return []
    ts = []
    for n in names:
        if len(n) > 9 and n.startswith("Image") and n.endswith(".png"):
            try:
                ts.append(int(n[5:-4]))
            except ValueError:
                continue
    ts.sort()
    return ts


def sessions():
    """{sid: meta} for every readable PURE session; never raises, skips broken ones."""
    root = paths.PURE
    out = {}
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for name in entries:
        m = _SESS_RE.match(name)
        if not m:
            continue                                   # PURE_zip, Intra_PURE_rhythmformer, ...
        subj, task = m.group(1), m.group(2)
        base = os.path.join(root, name)
        img_dir = os.path.join(base, name)
        js = os.path.join(base, name + ".json")
        if not os.path.isdir(img_dir):
            continue
        ts = _timestamps(img_dir)
        if len(ts) < 2:
            continue
        span_s = (ts[-1] - ts[0]) / 1e9
        if span_s <= 0:
            continue
        fps = (len(ts) - 1) / span_s
        out["%s_%s_%s" % (NAME, subj, task)] = {
            "subject": subj,
            "session": task,
            "task": task,
            "task_name": TASKS.get(task, "unknown"),
            "head_motion": task in MOVING_TASKS,
            "fps": float(fps),
            "img_dir": img_dir,
            "json": js if os.path.isfile(js) else None,
            "n_frames": len(ts),
            "duration_s": float(span_s),
            "timestamps_ns": ts,          # raw, non-uniform; for exact resampling later
            "width": 640,
            "height": 480,
        }
    return out


def frame_paths(meta):
    """Absolute PNG paths in timestamp order (the file name is Image<ts>.png exactly)."""
    d = meta["img_dir"]
    return [os.path.join(d, "Image%d.png" % t) for t in meta["timestamps_ns"]]


def frames(meta):
    """Zero-arg callable -> fresh lazy iterator of BGR uint8 (480, 640, 3) frames."""
    d = meta["img_dir"]
    ts = list(meta["timestamps_ns"])

    def _iter():
        for t in ts:
            img = cv2.imread(os.path.join(d, "Image%d.png" % t), cv2.IMREAD_COLOR)
            if img is None:
                continue          # unreadable PNG: skip rather than kill the session
            yield img

    return _iter


def gt_rr(meta):
    """PURE ships no respiration reference of any kind -> always None."""
    return None
