"""Shared front-end: raw session frames -> fixed face crop -> vertical-flow ST-map @30 Hz.

Design notes (these matter for RR, and differ from a typical rPPG front-end):

1. FIXED crop box per session. The respiratory signal IS the sub-pixel vertical motion of
   the head/torso. Per-frame face tracking would motion-compensate exactly the component we
   want to measure, so we detect on a sample of frames, take the MEDIAN box, expand it by
   BOX_SCALE, and hold it constant for the whole session.

2. Flow is computed at the session's NATIVE frame rate, then the resulting (M, T) map is
   linearly resampled onto a uniform 30 Hz grid. The four datasets run at 20 / 30.5 / 30 /
   23-30 fps; the train/infer cores hardcode fs=30, so the resampling has to happen here or
   every frequency estimate is scaled by fps/30. Respiration is 0.1-0.5 Hz, far below any
   Nyquist limit involved, so resampling the flow signal (rather than the frames) is safe
   and avoids interpolation artefacts in the flow itself.
"""
import numpy as np, cv2
from .. import config as paths

cv2.setNumThreads(1)

_DET = None


def _detector(w, h):
    global _DET
    if _DET is None:
        _DET = cv2.FaceDetectorYN.create(paths.YUNET, "", (w, h), 0.6, 0.3, 5000)
    _DET.setInputSize((w, h))
    return _DET


def detect_box(frames_sample):
    """frames_sample: list of BGR uint8 frames -> (x, y, w, h) median face box, or None."""
    boxes = []
    for f in frames_sample:
        h, w = f.shape[:2]
        det = _detector(w, h)
        ok, faces = det.detect(f)
        if faces is not None and len(faces):
            faces = faces[np.argsort(-faces[:, -1])]      # highest score first
            boxes.append(faces[0][:4].astype(np.float64))
    if not boxes:
        return None
    return np.median(np.stack(boxes), axis=0)


def expand_box(box, W, H, scale=None):
    """Expand a face box about its centre and clamp to the frame; returns int (x0,y0,x1,y1)."""
    scale = paths.BOX_SCALE if scale is None else scale
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    s = max(w, h) * scale                       # square box -> no aspect distortion on resize
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = int(round(cx + s / 2)), int(round(cy + s / 2))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    return x0, y0, x1, y1


def crop_gray(frame, bx, size=None):
    """BGR frame + (x0,y0,x1,y1) -> (size,size) uint8 grayscale crop."""
    size = paths.CROP if size is None else size
    x0, y0, x1, y1 = bx
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def flow_map(gray_frames, grid=None):
    """Iterable of (S,S) uint8 gray crops -> (M=grid^2, T-1) float32 vertical-flow ST-map.

    Identical Farneback parameters and vertical-component/block-mean reduction as the
    upstream repo's common/stmap.py, so the ST-maps are drop-in for its train/infer cores.
    """
    grid = paths.GRID if grid is None else grid
    cols, prev = [], None
    for g in gray_frames:
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None,
                                              pyr_scale=0.5, levels=3, winsize=15,
                                              iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            v = fl[..., 1]                                       # vertical component
            bs = v.shape[0] // grid
            cols.append(v[:grid * bs, :grid * bs]
                        .reshape(grid, bs, grid, bs).mean(axis=(1, 3)).reshape(-1))
        prev = g
    if not cols:
        return None
    return np.stack(cols, axis=1).astype(np.float32)             # (M, T-1)


def resample_to_fs(m, src_fs, dst_fs=None):
    """(M, T) map at src_fs -> (M, T') at dst_fs by linear interpolation along time."""
    dst_fs = paths.FS if dst_fs is None else dst_fs
    if m is None or m.shape[1] < 2:
        return m
    if abs(src_fs - dst_fs) < 1e-6:
        return m
    T = m.shape[1]
    dur = (T - 1) / float(src_fs)
    T2 = int(np.floor(dur * dst_fs)) + 1
    if T2 < 2:
        return None
    src_t = np.arange(T, dtype=np.float64) / float(src_fs)
    dst_t = np.arange(T2, dtype=np.float64) / float(dst_fs)
    out = np.empty((m.shape[0], T2), np.float32)
    for i in range(m.shape[0]):
        out[i] = np.interp(dst_t, src_t, m[i])
    return out


def session_stmap(frame_iter, fps, n_probe=24, min_seconds=5.0, n_frames=None,
                  precropped=False, probe_frames=None):
    """Full front-end for one session.

    frame_iter: callable returning a fresh iterator of BGR uint8 frames. It is called twice
                (once to collect detection frames, once for the real pass), so it must be
                cheap to restart. Pass n_frames when the caller already knows the frame count
                (every dataset module does) to avoid a third, purely-counting pass — that
                matters for PURE, whose frames are 123k individual PNGs, and for UBFC, where
                each pass streams ~1.8 GB off disk.
    Returns (stmap (M, T) float32 @30 Hz, info dict) or (None, info) if unusable.
    """
    info = {}
    # -- pass 1: frames spread over the session, for face detection.
    # probe_frames lets a reader supply them by SEEKING instead, so a corpus of multi-GB files
    # is streamed once rather than twice (Vital Videos: 0.9 TB instead of 1.9 TB).
    sample, first, n = [], None, 0
    if probe_frames:
        sample = list(probe_frames)
        first = sample[0] if sample else None
        n = int(n_frames or 0)
        if n < 2 or first is None:
            info["reason"] = "no_frames"
            return None, info
        info["n_frames"] = n
        info["probe"] = "seek"
    elif n_frames and n_frames >= 2:
        n = int(n_frames)
        idx = set(np.linspace(0, n - 1, min(n_probe, n)).astype(int).tolist())
        seen = 0
        for i, f in enumerate(frame_iter()):
            seen = i + 1
            if first is None:
                first = f
            if i in idx:
                sample.append(f)
        n = seen                      # trust the real count, not the hint
    else:
        for f in frame_iter():
            if first is None:
                first = f
            n += 1
        if n >= 2:
            idx = set(np.linspace(0, n - 1, min(n_probe, n)).astype(int).tolist())
            for i, f in enumerate(frame_iter()):
                if i in idx:
                    sample.append(f)
    info["n_frames"] = n
    if n < 2 or first is None:
        info["reason"] = "no_frames"
        return None, info

    if precropped:
        # the frame IS the ROI (e.g. a kiosk that already emits a 128x128 face crop).
        # Detecting again inside it would cut away the periphery, which carries much of the
        # respiratory motion.
        info["box"] = [0, 0, int(first.shape[1]), int(first.shape[0])]
        info["precropped"] = True

        def grays_pre():
            for f in frame_iter():
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                if g.shape[0] != paths.CROP or g.shape[1] != paths.CROP:
                    g = cv2.resize(g, (paths.CROP, paths.CROP), interpolation=cv2.INTER_AREA)
                yield g

        m = flow_map(grays_pre())
        if m is None:
            info["reason"] = "no_flow"
            return None, info
        m = resample_to_fs(m, fps)
        if m is None or m.shape[1] < int(min_seconds * paths.FS):
            info["reason"] = "too_short"
            return None, info
        info["T"] = int(m.shape[1]); info["fps_src"] = float(fps)
        return m, info

    box = detect_box(sample)
    if box is None:
        info["reason"] = "no_face"
        return None, info
    H, W = first.shape[:2]
    bx = expand_box(box, W, H)
    info["box"] = [int(v) for v in bx]
    info["det_box"] = [float(v) for v in box]
    if (bx[2] - bx[0]) < 32 or (bx[3] - bx[1]) < 32:
        info["reason"] = "box_too_small"
        return None, info

    # -- pass 2: crop -> gray -> flow
    def grays():
        for f in frame_iter():
            g = crop_gray(f, bx)
            if g is not None:
                yield g

    m = flow_map(grays())
    if m is None:
        info["reason"] = "no_flow"
        return None, info
    m = resample_to_fs(m, fps)
    if m is None or m.shape[1] < int(min_seconds * paths.FS):
        info["reason"] = "too_short"
        info["T"] = 0 if m is None else int(m.shape[1])
        return None, info
    info["T"] = int(m.shape[1])
    info["fps_src"] = float(fps)
    return m, info
