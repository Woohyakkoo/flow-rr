"""Self-supervised training: clip-positive InfoNCE over band-limited PSDs.

THE POSITIVE PAIR. CalibrationPhys forms positives from two cameras filming the same person
at the same instant. We have one camera, so the pair is made by splitting ONE sampled clip's
ROI axis into two interleaved halves (even and odd flattened index, i.e. column parity on the
GRID x GRID map). Both halves see the same person, the same instant and the same breathing
cycle, and differ only in which patches of the face they average — which is exactly the
invariance the estimate should have.

THE SPEED WARP. A clip of Lp ~ U[T-30, T+60] columns is resampled to exactly T, scaling the
effective rate by Lp/T (x0.90 to x1.20). This is the paper's temporal augmentation and it is
the only thing that broadens the rate distribution the model sees.

  The warp is drawn ONCE per pair, not per view. It is periodicity-VARIANT: it changes the
  very quantity the loss compares. Drawing it independently would ask the model to call two
  genuinely different rates identical. Clip positives get this for free (one clip, split
  after warping); the guard matters if you add a positive rule that samples twice.

DECIMATION TO 15 Hz. Respiration occupies 0.1-0.5 Hz, so a 30 Hz map is ~30x oversampled.
Independent frequency resolution is set by the DURATION a clip spans (df = 1/seconds), not by
its sample count, so decimating buys real band bins at a fixed tensor width: 15 Hz with T=300
spans 20 s and yields 9 band bins -- the same as 30 Hz with T=600, at half the compute. The
anti-alias filter also removes cardiac motion near 1 Hz, which is noise for this task.

CHECKPOINT SELECTION is the validation InfoNCE, computed on held-out SUBJECTS. No label is
read at any point during training or model selection.
"""
import os, json, time, argparse
import numpy as np
import torch

from . import config, splits, readout
from .model import PhysNet2D, n_params
from .losses import band_psd, info_nce

MIN_T = config.T_CLIP + config.TAUG_P + 10


# ── clip sampling ────────────────────────────────────────────────────────────
def _resample_T(m, T):
    if m.shape[1] == T:
        return np.ascontiguousarray(m, dtype=np.float32)
    src = np.linspace(0, m.shape[1] - 1, T)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, m.shape[1] - 1)
    w = (src - i0).astype(np.float32)
    return np.ascontiguousarray(m[:, i0] * (1 - w) + m[:, i1] * w, dtype=np.float32)


def _draw_Lp(rng, T):
    lo = int(round(T * config.TAUG_M / config.T_CLIP))
    hi = int(round(T * config.TAUG_P / config.T_CLIP))
    return rng.integers(T - lo, T + hi + 1)


def _sample_clip(m, rng, T, Lp=None):
    Lp = _draw_Lp(rng, T) if Lp is None else int(Lp)
    s0 = rng.integers(0, m.shape[1] - Lp + 1)
    return _resample_T(m[:, s0:s0 + Lp], T)


def _two_views(clip, rng):
    """Even/odd ROI rows, each with light gaussian noise."""
    a, b = clip[0::2], clip[1::2]
    return (a + rng.normal(0, 0.05 * (a.std() + 1e-6), a.shape).astype(np.float32),
            b + rng.normal(0, 0.05 * (b.std() + 1e-6), b.shape).astype(np.float32))


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def _batch(maps, sids, idx, rng, dev, T):
    va, vb = [], []
    for i in idx:
        a, b = _two_views(_sample_clip(maps[sids[i]], rng, T), rng)
        va.append(_norm(a)); vb.append(_norm(b))
    return (torch.from_numpy(np.stack(va)[:, None]).to(dev),
            torch.from_numpy(np.stack(vb)[:, None]).to(dev))


# ── ST-map loading ───────────────────────────────────────────────────────────
def decimate_map(m, q):
    if q <= 1:
        return m
    from scipy.signal import decimate
    return np.ascontiguousarray(decimate(m, q, axis=1, ftype="fir", zero_phase=True),
                                dtype=np.float32)


def load_maps(sids, min_T=MIN_T, stmap_dir=None, q=1):
    """{sid: (M, T) float32} for every session whose map exists and is long enough."""
    sm = stmap_dir or config.STMAPS
    out, dropped = {}, []
    for sid in sids:
        p = os.path.join(sm, f"{sid}.npz")
        if not os.path.exists(p):
            dropped.append((sid, "missing")); continue
        m = np.load(p)["flow"].astype(np.float32)
        if q > 1:
            if m.shape[1] < min_T * q:
                dropped.append((sid, f"short:{m.shape[1]}")); continue
            m = decimate_map(m, q)
        if m.shape[1] < min_T:
            dropped.append((sid, f"short:{m.shape[1]}")); continue
        out[sid] = m
    return out, dropped


# ── inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, maps, dev, fs, win=None):
    """{sid: {readout: bpm}} over whole sessions."""
    win = win or config.T_CLIP
    model.eval()
    out = {}
    for sid, m in maps.items():
        x = torch.from_numpy(_norm(m)[None, None]).to(dev)
        w = model(x)[0].float().cpu().numpy()
        out[sid] = readout.all_readouts(w, min(win, len(w)), fs)
    return out


# ── training ─────────────────────────────────────────────────────────────────
def train(train_sids, out_path, val_sids=None, epochs=30, steps=60, bs=32, lr=1e-4,
          seed=42, t_clip=None, fs_train=None, device=None, val_batches=20, stmap_dir=None):
    t_clip = t_clip or config.T_CLIP
    fs_train = fs_train or config.FS_TRAIN
    q = int(round(config.FS / fs_train))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    min_T = t_clip + int(round(t_clip * config.TAUG_P / config.T_CLIP)) + 10
    tr, drop = load_maps(train_sids, min_T, stmap_dir, q)
    va, _ = load_maps(val_sids or [], min_T, stmap_dir, q)
    if len(tr) < 2:
        raise RuntimeError(f"only {len(tr)} usable training sessions ({len(drop)} dropped)")
    ktr, kva = sorted(tr), sorted(va)

    model = PhysNet2D(out_ch=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"{len(tr)} train / {len(va)} val sessions   T={t_clip} @ {fs_train} Hz   "
          f"{n_params(model)} params", flush=True)

    best, best_ep, t0 = float("inf"), -1, time.time()
    for ep in range(epochs):
        model.train(); tot = 0.0
        for _ in range(steps):
            pick = rng.choice(len(ktr), bs, replace=True)
            xa, xb = _batch(tr, ktr, pick, rng, dev, t_clip)
            loss = info_nce(band_psd(model(xa), fs_train),
                            band_psd(model(xb), fs_train))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        vl = float("nan")
        if kva:
            model.eval(); acc = []
            vr = np.random.default_rng(1234)
            with torch.no_grad():
                for _ in range(val_batches):
                    pick = vr.choice(len(kva), min(bs, len(kva)), replace=True)
                    xa, xb = _batch(va, kva, pick, vr, dev, t_clip)
                    acc.append(float(info_nce(band_psd(model(xa), fs_train),
                                              band_psd(model(xb), fs_train))))
            vl = float(np.mean(acc))
        sel = vl if kva else tot / steps
        if sel < best:
            best, best_ep = sel, ep
            torch.save(model.state_dict(), out_path)
        print(f"  ep{ep:02d} train {tot/steps:.4f}  val {vl:.4f}"
              f"{'  *' if best_ep == ep else ''}  {time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(torch.load(out_path, map_location=dev))
    return model, dict(best_epoch=best_ep, val_metric="infonce", n_train=len(tr),
                       t_clip=t_clip, fs_train=fs_train, seed=seed)


def main():
    ap = argparse.ArgumentParser(description="train one self-supervised RR model")
    ap.add_argument("--train", required=True,
                    help="comma-separated dataset names for the training pool, e.g. cohface,pure")
    ap.add_argument("--eval", default="cohface,mahnob",
                    help="datasets to score on their own test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--readout", default="psd_sum",
                    choices=["psd_sum", "mean", "median", "snr_mean"])
    a = ap.parse_args()

    config.ensure_dirs()
    sp = splits.load()
    pool = [d.strip() for d in a.train.split(",")]
    tr, va = splits.pool(sp, pool)

    name = f"{a.tag}_{'+'.join(pool)}_s{a.seed}"
    md = os.path.join(config.MODELS, f"{name}.pth")
    model, meta = train(tr, md, va, epochs=a.epochs, steps=a.steps, bs=a.bs, seed=a.seed)

    from .evaluate import evaluate
    man = json.load(open(config.MANIFEST))
    dev = next(model.parameters()).device
    want = sorted({s for d in a.eval.split(",") for s in sp[d.strip()]["test"]})
    maps, _ = load_maps(want, 60, None, int(round(config.FS / config.FS_TRAIN)))
    preds = predict(model, maps, dev, config.FS_TRAIN)

    pj = os.path.join(config.PREDS, f"{name}.json")
    json.dump({"_meta": dict(meta, train_pool=pool, tag=a.tag), "preds": preds},
              open(pj, "w"), indent=1)
    print()
    for d in a.eval.split(","):
        d = d.strip()
        flat = {k: v[a.readout] for k, v in preds.items()}
        m = evaluate(flat, sp[d]["test"], man)
        print(f"{d}:test  n={m['n']:4d}  MAE {m['MAE']:.2f}  RMSE {m['RMSE']:.2f}  "
              f"r {m['pearson']:+.3f}  ICC {m['ICC']:.3f}")
    print(f"\n[saved] {md}\n[saved] {pj}")


if __name__ == "__main__":
    main()
