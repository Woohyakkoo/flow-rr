"""UBFC-rPPG dataset reader (DATASET_1 + DATASET_2).

Why this module ships its own AVI decoder
-----------------------------------------
The videos are raw uncompressed ``'DIB '`` AVI (24 bpp BI_RGB, 640x480).  Under the
OpenCV 5.0.0 build in this venv ``cv2.VideoCapture`` *opens* them and reports correct
fps / frame-count / resolution, but SIGSEGVs on the very first ``.read()`` -- on every
file.  There is no ffmpeg / imageio / av / decord on this machine.  So frames are read
by parsing the RIFF container directly (logic verified in ``_ubfc_avi_reference.py``).

Three container facts that silently corrupt naive readers:

1. Rows are stored **bottom-up** (``biHeight > 0``), so every frame must be flipped
   vertically.  YuNet happily detects upside-down faces, so a successful detection is
   *not* evidence of correct orientation -- this was confirmed by eye instead.
2. Files larger than 2 GiB are **multi-RIFF OpenDML**: a second ``RIFF/AVIX`` segment
   holds the tail of the movi data.  Parsing only the first RIFF silently truncates
   them (``after-exercise`` would lose ~1000 of its 3368 frames).
3. **fps is neither 30 nor constant.**  It is read per file from the stream header
   (``dwRate/dwScale``): DATASET_1 is 28.638-28.672, DATASET_2 spans 23.207-29.976
   (subjects 25/26/27 run at ~23 fps).  Hardcoding 30 would scale every frequency
   estimate by fps/30.  The true value goes in ``meta['fps']``.

Subject identity (load-bearing for subject-disjoint splits)
-----------------------------------------------------------
The two parts use *independent* numbering: ``DATASET_1/5-gt`` and ``DATASET_2/subject5``
are different people.  Keying on the bare number would leak identity across splits, so
DATASET_1 subjects are namespaced ``a<n>`` and DATASET_2 subjects ``b<n>``.

There is exactly one *true* cross-part overlap, and the numbering does not signal it:
``DATASET_1/8-gt``, ``DATASET_1/10-gt``, ``DATASET_1/after-exercise`` and
``DATASET_2/subject42`` are all the same man (verified by face-embedding similarity and
visual inspection).  All four are mapped to subject ``b42``.

Result: 50 sessions, 47 distinct people.

Respiration ground truth
------------------------
There is **none**.  The dataset ships PPG/HR (DATASET_2: BVP + HR + timestamps;
DATASET_1: ``gtdump.xmp`` with timestamp/HR/SpO2/PPG) and nothing else -- no belt, no
airflow, no annotation file.  ``gt_rr`` therefore returns ``None`` unconditionally.  A
respiration pseudo-label is deliberately *not* derived from the PPG: a prior check found
several sessions pinning to exactly 6.0 bpm (the band edge) and three plausible
derivation routes disagreeing on the same session, so such a label would be noise
wearing the costume of ground truth.
"""
import os
import struct

import numpy as np

from .. import config as paths

NAME = "UBFC"

_D1 = "DATASET_1"
_D2 = "DATASET_2"

# The one verified cross-part identity overlap (see module docstring).
_D1_ALIASES = {"8-gt": "b42", "10-gt": "b42", "after-exercise": "b42"}


# ───────────────────────────── AVI container parsing ─────────────────────────────

def _avi_meta(path):
    """Parse the AVI headers -> dict(w, h, fps, n, bottom_up). Raises on anything odd."""
    with open(path, "rb") as f:
        d = f.read(8192)

    i = d.find(b"avih")
    if i < 0:
        raise ValueError("no avih chunk")
    a = d[i + 8: i + 8 + struct.unpack("<I", d[i + 4:i + 8])[0]]
    _, _, _, _, _, _, _, _, aw, ah = struct.unpack("<10I", a[:40])

    # First stream header must be the video stream.
    j = d.find(b"strh")
    if j < 0:
        raise ValueError("no strh chunk")
    s = d[j + 8: j + 8 + struct.unpack("<I", d[j + 4:j + 8])[0]]
    if s[:4] != b"vids":
        raise ValueError("first stream is not video: %r" % (s[:4],))
    scale, rate, _, length = struct.unpack("<4I", s[20:36])
    if scale == 0 or rate == 0:
        raise ValueError("degenerate dwScale/dwRate")

    k = d.find(b"strf")
    if k < 0:
        raise ValueError("no strf chunk")
    bi = d[k + 8: k + 8 + struct.unpack("<I", d[k + 4:k + 8])[0]]
    _, bw, bh, _, bits, comp, _ = struct.unpack("<IiiHHII", bi[:24])
    if bits != 24 or comp != 0:
        raise ValueError("expected raw BI_RGB 24bpp, got bits=%d comp=%d" % (bits, comp))

    w, h = int(bw), int(abs(bh))
    if (w, h) != (int(aw), int(ah)):
        raise ValueError("avih %dx%d disagrees with strf %dx%d" % (aw, ah, w, h))
    if w <= 0 or h <= 0:
        raise ValueError("degenerate frame size")

    return dict(w=w, h=h, fps=rate / float(scale), n=int(length), bottom_up=bh > 0)


def _chunk_id_ok(cid):
    """True if `cid` looks like a real FourCC rather than dead space.

    Every legitimate chunk id in these files is printable ASCII ('00db', 'idx1',
    'LIST', 'JUNK', 'ix00', ...).  A zeroed / dead region instead decodes as
    b'\\x00\\x00\\x00\\x00' with length 0, which would otherwise send the walker
    crawling forward 8 bytes at a time to EOF (99M iterations, ~40 s, on
    DATASET_2/subject37).  Testing for printability rather than allow-listing
    specific ids keeps the multi-RIFF path intact -- an allow-list that forgot
    'idx1' silently cut every >2 GiB file off at the 2 GiB boundary.
    """
    return len(cid) == 4 and all(0x20 <= b <= 0x7E for b in cid)


def _trailing_zero_run(path, off, nbytes):
    """Length of the run of zero bytes at the end of a frame payload."""
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(nbytes)
    a = np.frombuffer(buf, np.uint8)
    nz = np.flatnonzero(a)
    return len(a) if nz.size == 0 else len(a) - 1 - int(nz[-1])


def _frame_offsets(path, frame_bytes):
    """Byte offsets of every intact frame payload, in order.

    Walks *every* RIFF segment, so multi-RIFF OpenDML files (>2 GiB) are read whole.
    Stops at the first structurally invalid chunk header, which is how a file that was
    truncated-and-zero-padded announces itself, and drops a trailing frame whose payload
    runs into that dead zone rather than yielding a half-black frame.
    """
    offs = []
    size = os.path.getsize(path)
    clean = True
    with open(path, "rb") as f:
        pos = 0
        while pos < size - 8:
            f.seek(pos)
            hdr = f.read(12)
            if len(hdr) < 12 or hdr[:4] != b"RIFF":
                break
            rlen = struct.unpack("<I", hdr[4:8])[0]
            end, p = min(pos + 8 + rlen, size), pos + 12
            while p < end - 8:
                f.seek(p)
                ch = f.read(12)
                if len(ch) < 8 or not _chunk_id_ok(ch[:4]):
                    clean = False
                    break
                cid, clen = ch[:4], struct.unpack("<I", ch[4:8])[0]
                if cid == b"LIST" and ch[8:12] == b"movi":
                    q, mend = p + 12, min(p + 8 + clen, end)
                    while q < mend - 8:
                        f.seek(q)
                        fh = f.read(8)
                        if len(fh) < 8 or not _chunk_id_ok(fh[:4]):
                            clean = False          # dead / zeroed region -> stop here
                            break
                        flen = struct.unpack("<I", fh[4:8])[0]
                        # '##db' (uncompressed) / '##dc' (compressed) video chunks.
                        if fh[2:4] in (b"db", b"dc"):
                            if flen < frame_bytes or q + 8 + frame_bytes > size:
                                clean = False
                                break
                            offs.append(q + 8)
                        q += 8 + flen + (flen & 1)
                    if not clean:
                        break
                p += 8 + clen + (clen & 1)
            if not clean:
                break
            pos += 8 + rlen + (rlen & 1)

    # The frame straddling the start of a dead zone is itself partly zero-filled.
    if not clean and offs:
        if _trailing_zero_run(path, offs[-1], frame_bytes) >= frame_bytes // 8:
            offs.pop()
    return offs


_OFFSET_CACHE = {}


def _offsets_cached(path, frame_bytes):
    """The front-end iterates a session three times; scan the container only once.

    Offset lists are a few thousand ints per session, so caching all 50 is cheap.
    """
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    offs = _OFFSET_CACHE.get(key)
    if offs is None:
        offs = _frame_offsets(path, frame_bytes)
        _OFFSET_CACHE[key] = offs
    return offs


# ───────────────────────────────── public API ────────────────────────────────────

def sessions():
    """{sid: meta} for all readable UBFC sessions, deterministically ordered."""
    out = {}
    for part, dirname, subject, session in _layout():
        d = os.path.join(paths.UBFC, part, dirname)
        vid = os.path.join(d, "vid.avi")
        gt = os.path.join(d, "gtdump.xmp" if part == _D1 else "ground_truth.txt")
        if not os.path.isfile(vid):
            continue
        try:
            m = _avi_meta(vid)
            # Scan the container rather than trusting dwLength: DATASET_2/subject37 is
            # truncated and zero-padded, so its header over-reports by 863 frames.
            n = len(_offsets_cached(vid, m["w"] * m["h"] * 3))
        except Exception:
            continue                    # corrupt/unsupported -> skip, never raise
        if n < 2:
            continue
        sid = "%s_%s_%s" % (NAME, subject, session)
        out[sid] = dict(
            subject=subject,
            session=session,
            fps=float(m["fps"]),
            path=vid,
            gt_path=gt if os.path.isfile(gt) else None,
            gt_kind="gtdump" if part == _D1 else "ground_truth",
            part=part,
            dirname=dirname,
            width=int(m["w"]),
            height=int(m["h"]),
            bottom_up=bool(m["bottom_up"]),
            n_frames=int(n),
            n_frames_header=int(m["n"]),
            truncated=bool(n != m["n"]),
            duration_s=float(n) / float(m["fps"]),
        )
    return out


def _layout():
    """Yield (part, dirname, subject, session) in a stable order. No I/O errors escape."""
    root = paths.UBFC

    # -- DATASET_1: "<n>-gt" dirs plus the non-numeric "after-exercise".
    d1 = os.path.join(root, _D1)
    numeric, extra = [], []
    try:
        names = sorted(os.listdir(d1))
    except OSError:
        names = []
    for name in names:
        if not os.path.isdir(os.path.join(d1, name)):
            continue
        if name.endswith("-gt") and name[:-3].isdigit():
            numeric.append((int(name[:-3]), name))
        elif name == "after-exercise":
            extra.append(name)
    for n, name in sorted(numeric):
        alias = _D1_ALIASES.get(name)
        if alias is None:
            # Subject key already carries the number, so the session token need not.
            yield _D1, name, "a%d" % n, "d1gt"
        else:
            yield _D1, name, alias, "d1gt%d" % n
    for name in extra:
        # 'after-exercise' carries no number; its identity is only known from the alias map.
        yield _D1, name, _D1_ALIASES[name], "d1afterex"

    # -- DATASET_2: "subject<n>" dirs, one session each.
    d2 = os.path.join(root, _D2)
    subs = []
    try:
        names = sorted(os.listdir(d2))
    except OSError:
        names = []
    for name in names:
        if not os.path.isdir(os.path.join(d2, name)):
            continue
        if name.startswith("subject") and name[7:].isdigit():
            subs.append((int(name[7:]), name))
    for n, name in sorted(subs):
        yield _D2, name, "b%d" % n, "d2"


def frames(meta):
    """-> zero-arg callable producing a FRESH iterator of upright BGR uint8 frames."""
    path = meta["path"]
    w, h = int(meta["width"]), int(meta["height"])
    flip = bool(meta.get("bottom_up", True))
    nbytes = w * h * 3

    def make():
        offs = _offsets_cached(path, nbytes)
        f = open(path, "rb", buffering=0)
        try:
            for off in offs:
                f.seek(off)
                buf = f.read(nbytes)
                if len(buf) < nbytes:
                    break               # truncated tail: stop, never pad with fake pixels
                # BI_RGB 24bpp is already byte-order BGR; rows are bottom-up.
                img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                # .copy() both un-flips the stride and detaches from the read buffer,
                # so callers may retain frames across iterations safely.
                yield img[::-1].copy() if flip else img.copy()
        finally:
            f.close()

    return make


def gt_rr(meta):
    """Always None: UBFC-rPPG ships no respiration signal (see module docstring)."""
    return None
