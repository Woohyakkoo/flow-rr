"""Driver: raw sessions (4 datasets) -> data/stmaps/{sid}.npz + data/manifest.json.

Single pass per session: decode -> fixed face box -> 128x128 gray crop -> Farneback vertical
flow -> 8x8 block ST-map -> resample to 30 Hz. Only the ST-map is persisted (float16,
~128 B per frame), so the whole corpus is a few hundred MB rather than the ~110 GB the
intermediate face crops would cost.

  python -m flowrr.make_stmaps [--datasets cohface,mahnob,pure,ubfc] [--jobs 30] [--force]
"""
import os, json, argparse, traceback
import numpy as np
from multiprocessing import Pool

from .. import config as paths

DATASETS = ("cohface", "mahnob", "pure", "ubfc", "cudo", "vitalvideo")


def _mod(name):
    import importlib
    return importlib.import_module(f"flowrr.datasets.{name}")


# Datasets whose files live on the external drive: decoding several multi-GB long-GOP mp4s
# at once starves the decoders and they return zero frames. 249 of 507 Vital Videos sessions
# failed that way at 20 workers and every one of them decoded fine on a retry in isolation.
IO_HEAVY = {"vitalvideo"}
IO_HEAVY_JOBS = 6


def _work(item):
    ds, sid, meta, force = item
    out = f"{paths.STMAPS}/{sid}.npz"
    rec = {"sid": sid, "dataset": ds, "subject": str(meta.get("subject")),
           "fps_src": float(meta.get("fps", 0))}
    # carry a few per-session descriptors through to the manifest so results can be broken
    # down later (MAHNOB neutral vs emotional trials, PURE static vs head-motion tasks,
    # COHFACE lamp vs natural illumination) without re-opening the raw data
    for k in ("trial", "task", "task_name", "head_motion", "illumination", "part",
              "duration_s", "stim_span_s"):
        if k in meta:
            rec[k] = meta[k]
    try:
        D = _mod(ds)
        if os.path.exists(out) and not force:
            m = np.load(out)["flow"]
            rec.update(status="cached", T=int(m.shape[1]), M=int(m.shape[0]))
        else:
            from .frontend import session_stmap
            probe = D.probe(meta) if hasattr(D, "probe") else None
            m, info = session_stmap(D.frames(meta), meta["fps"], n_frames=meta.get("n_frames"),
                                    precropped=bool(meta.get("precropped")),
                                    probe_frames=probe)
            rec.update({k: v for k, v in info.items() if k != "reason"})
            if m is None and info.get("reason") == "no_frames":
                # retry once, serially: this failure mode is drive contention, not a bad file
                import time as _t
                _t.sleep(1.0 + (hash(sid) % 5))
                m, info = session_stmap(D.frames(meta), meta["fps"],
                                        n_frames=meta.get("n_frames"),
                                        precropped=bool(meta.get("precropped")),
                                        probe_frames=probe)
                info["retried"] = True
            if m is None:
                rec.update(status="skip", reason=info.get("reason", "unknown"))
                return rec
            np.savez_compressed(out, flow=m.astype(np.float16))
            rec.update(status="ok", T=int(m.shape[1]), M=int(m.shape[0]))
        g = D.gt_rr(meta)
        rec["gt_rr"] = None if g is None or not np.isfinite(g) else float(g)
    except Exception as e:
        rec.update(status="error", reason=f"{type(e).__name__}: {e}",
                   tb=traceback.format_exc()[-800:])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--retry-skipped", action="store_true",
                    help="re-attempt sessions the manifest records as skipped")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap sessions per dataset")
    a = ap.parse_args()

    items = []
    for ds in a.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        S = _mod(ds).sessions()
        keys = sorted(S)
        if a.retry_skipped:
            import json as _j
            mp0 = f"{paths.DATA}/manifest.json"
            old = _j.load(open(mp0)) if os.path.exists(mp0) else {}
            keys = [k for k in keys if old.get(k, {}).get("status") not in ("ok", "cached")]
        if a.limit:
            keys = keys[:a.limit]
        if ds in IO_HEAVY and a.jobs > IO_HEAVY_JOBS:
            print(f"[{ds}] capping workers {a.jobs} -> {IO_HEAVY_JOBS} (external-drive I/O)",
                  flush=True)
        print(f"[{ds}] {len(keys)} sessions", flush=True)
        items += [(ds, k, S[k], a.force) for k in keys]

    print(f"total {len(items)} sessions -> {paths.STMAPS}", flush=True)
    recs, stat = [], {}
    jobs = min(a.jobs, IO_HEAVY_JOBS) if any(i[0] in IO_HEAVY for i in items) else a.jobs
    with Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, items, chunksize=1)):
            recs.append(r)
            stat[r["status"]] = stat.get(r["status"], 0) + 1
            if (i + 1) % 50 == 0 or i + 1 == len(items):
                print(f"  [{i+1}/{len(items)}] {stat}", flush=True)
            if r["status"] == "error":
                print(f"  ERR {r['sid']}: {r['reason']}", flush=True)

    man = {r["sid"]: r for r in sorted(recs, key=lambda x: x["sid"])}
    mp = f"{paths.DATA}/manifest.json"
    if os.path.exists(mp) and not a.force and a.datasets != ",".join(DATASETS):
        old = json.load(open(mp))
        old.update(man)
        man = old
    json.dump(man, open(mp, "w"), indent=1)
    print(f"[done] {stat} -> {mp}", flush=True)

    ok = [r for r in man.values() if r["status"] in ("ok", "cached")]
    print(f"usable ST-maps: {len(ok)}")
    for ds in sorted({r['dataset'] for r in ok}):
        sub = [r for r in ok if r["dataset"] == ds]
        withgt = [r for r in sub if r.get("gt_rr") is not None]
        subj = {r["subject"] for r in sub}
        Ts = np.array([r["T"] for r in sub])
        print(f"  {ds:8s} n={len(sub):4d} subjects={len(subj):3d} gt={len(withgt):4d} "
              f"T[min/med/max]={Ts.min()}/{int(np.median(Ts))}/{Ts.max()}")


if __name__ == "__main__":
    main()
