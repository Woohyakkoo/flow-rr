"""Metrics + figures + RESULTS.md for every experiment cell.

Metrics follow the upstream repo exactly: MAE / RMSE / Pearson r over the sessions that have
a respiration reference, and ICC(1,1) (one-way random effects, single measure, absolute
agreement) over predictions grouped BY PERSON — the consistency measure the project actually
cares about. Unlike upstream's eval_subset, MAE/r and the ICC groups are built from the SAME
GT-intersected subset, so a partially-labelled test set cannot report the two over different
session sets.

  python -m flowrr.evaluate
"""
import os, json, argparse
import numpy as np

from . import config as paths
from . import splits


def icc11(groups):
    """ICC(1,1). groups: {subject: [values]}. Only subjects with >=2 sessions contribute."""
    G = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(G) < 2:
        return float("nan")
    m = len(G)
    N = sum(len(v) for v in G.values())
    grand = np.concatenate([np.asarray(v, float) for v in G.values()]).mean()
    SSB = sum(len(v) * (np.mean(v) - grand) ** 2 for v in G.values())
    SSW = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in G.values())
    MSB = SSB / (m - 1)
    MSW = SSW / (N - m)
    k0 = (N - sum(len(v) ** 2 for v in G.values()) / N) / (m - 1)
    return float((MSB - MSW) / (MSB + (k0 - 1) * MSW + 1e-12))


def evaluate(preds, sids, man):
    """Restrict preds to sids that have both a finite prediction and a GT, then score."""
    ss = [s for s in sids
          if s in preds and np.isfinite(preds[s])
          and man.get(s, {}).get("gt_rr") is not None]
    if not ss:
        return dict(n=0, subjects=0, MAE=float("nan"), RMSE=float("nan"),
                    pearson=float("nan"), ICC=float("nan"))
    p = np.array([preds[s] for s in ss], float)
    g = np.array([man[s]["gt_rr"] for s in ss], float)
    e = p - g
    # namespace the person key by dataset. COHFACE_10_0 and MAHNOB_10_1172 both carry the
    # subject string "10" but are different people; without the dataset name an eval set that
    # mixed datasets would fuse them into one ICC group. train.py already namespaces this way.
    P = {}
    for s in ss:
        P.setdefault(f'{man[s]["dataset"]}:{man[s]["subject"]}', []).append(preds[s])
    # ICC of the REFERENCE itself: the ceiling any predictor can reach on this test set.
    # Without it a prediction ICC is uninterpretable - MAHNOB subjects genuinely change
    # their breathing rate between emotional stimuli, so a low prediction ICC there can be
    # correct behaviour rather than a failure.
    G = {}
    for s in ss:
        G.setdefault(f'{man[s]["dataset"]}:{man[s]["subject"]}', []).append(man[s]["gt_rr"])
    ec = e - e.mean()                       # oracle bias removal - a diagnostic, not a score
    return dict(
        n=len(ss), subjects=len(P),
        MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
        pearson=float(np.corrcoef(p, g)[0, 1]) if len(p) > 1 else float("nan"),
        ICC=icc11(P), ICC_gt=icc11(G),
        MAE_debiased=float(np.mean(np.abs(ec))),
        RMSE_debiased=float(np.sqrt(np.mean(ec ** 2))),
        pred_mean=float(p.mean()), pred_std=float(p.std()),
        gt_mean=float(g.mean()), gt_std=float(g.std()),
        bias=float(e.mean()),
        _pred=p.tolist(), _gt=g.tolist(),
        _subj=[str(man[s]["subject"]) for s in ss])


def _scatter(cells, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [c for c in cells if c["m"]["n"] > 1]
    if not cells:
        return
    ncol = min(4, len(cells))
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, c in enumerate(cells):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        m = c["m"]
        g, p = np.array(m["_gt"]), np.array(m["_pred"])
        ax.scatter(g, p, s=14, alpha=0.55, edgecolors="none")
        lim = [3, 31]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        if len(g) > 1:
            k, b = np.polyfit(g, p, 1)
            xs = np.linspace(*lim, 10)
            ax.plot(xs, k * xs + b, "r-", lw=1.2, alpha=0.85)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("reference RR (bpm)"); ax.set_ylabel("predicted RR (bpm)")
        ax.set_title(f"{c['run']}\n{c['test']}  MAE {m['MAE']:.2f}  r {m['pearson']:+.2f}  "
                     f"ICC {m['ICC']:.3f}", fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


SUBGROUP = {"mahnob": "trial", "pure": "head_motion", "cohface": "illumination"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(paths.MANIFEST))
    sp = splits.load()
    runs = json.load(open(paths.RUNS))

    # training-free reference row, so a weak learned number can be read against what the
    # front-end alone already delivers
    bl = f"{paths.PREDS}/baseline_flowpsd.json"
    if os.path.exists(bl):
        runs = dict(runs)
        runs["baseline_flowpsd"] = {
            "setting": "E0_no_training", "preds": bl,
            "note": "training-free reference: SNR-weighted block-PSD peak of the flow ST-map",
            "n_train": 0, "n_val": 0, "n_train_used": 0, "best_epoch": None,
            "eval_on": [[ds, sub, sp[ds][sub]]
                        for ds in ("cohface", "mahnob") if ds in sp
                        for sub in ("test", "all")],
        }

    cells, sub_cells = [], []
    for name, r in runs.items():
        preds = json.load(open(r["preds"]))
        for ds, subset, sids in r["eval_on"]:
            m = evaluate(preds, sids, man)
            cells.append(dict(run=name, setting=r["setting"], test=f"{ds}:{subset}",
                              dataset=ds, subset=subset, m=m))
            # break the cell down by the descriptor that matters for this dataset, so a
            # heterogeneous test set cannot hide a subgroup behind the average
            key = SUBGROUP.get(ds)
            if not key:
                continue
            vals = sorted({str(man[s].get(key)) for s in sids
                           if s in man and man[s].get(key) is not None})
            if len(vals) < 2:
                continue
            for v in vals:
                ss = [s for s in sids if s in man and str(man[s].get(key)) == v]
                mm = evaluate(preds, ss, man)
                if mm["n"]:
                    sub_cells.append(dict(run=name, setting=r["setting"],
                                          test=f"{ds}:{subset}", group=f"{key}={v}", m=mm))

    out = {}
    for c in cells:
        out.setdefault(c["setting"], {}).setdefault(c["run"], {})[c["test"]] = \
            {k: v for k, v in c["m"].items() if not k.startswith("_")}
    for c in sub_cells:
        out.setdefault("_subgroups", {}).setdefault(c["run"], {}) \
           .setdefault(c["test"], {})[c["group"]] = \
            {k: v for k, v in c["m"].items() if not k.startswith("_")}
    json.dump(out, open(f"{paths.RESULTS}/metrics.json", "w"), indent=1)
    _scatter(cells, f"{paths.RESULTS}/scatter_all.png")

    # ── console + markdown table ──
    L = []
    L.append("| setting | run | test set | n | subj | MAE | RMSE | bias | MAE−bias | "
             "Pearson r | ICC(1,1) | ICC of GT |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        m = c["m"]
        L.append(f"| {c['setting']} | `{c['run']}` | {c['test']} | {m['n']} | {m['subjects']} | "
                 f"{m['MAE']:.2f} | {m['RMSE']:.2f} | {m['bias']:+.2f} | "
                 f"{m['MAE_debiased']:.2f} | {m['pearson']:+.3f} | {m['ICC']:.3f} | "
                 f"{m['ICC_gt']:.3f} |")
    table = "\n".join(L)
    print(table)

    sub_table = ""
    if sub_cells:
        SL = ["| run | test set | subgroup | n | subj | MAE | MAE−bias | Pearson r | "
              "ICC(1,1) | ICC of GT |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in sub_cells:
            m = c["m"]
            SL.append(f"| `{c['run']}` | {c['test']} | {c['group']} | {m['n']} | "
                      f"{m['subjects']} | {m['MAE']:.2f} | {m['MAE_debiased']:.2f} | "
                      f"{m['pearson']:+.3f} | {m['ICC']:.3f} | {m['ICC_gt']:.3f} |")
        sub_table = "\n".join(SL)
        print("\n" + sub_table)

    md = ["# RR estimation results — face-flow PSD-InfoNCE (contrastive-rr-estimation method)\n",
          "Self-supervised (label-free) training; the respiration reference is used for "
          "evaluation only.\n",
          "## Metrics\n", table, "\n## Breakdown by subgroup\n",
          (sub_table + "\n" if sub_table else "_(none)_\n"),
          "\nMAHNOB `trial=N` are the neutral trials (5.3-22.2 s of stimulus, i.e. only "
          "~1-4 breathing cycles at 0.1-0.5 Hz), so their belt reference is a noisier target "
          "than the emotional (`trial=S`) trials'.\n",
          "\n### Reading the columns\n\n"
          "- **bias** is the mean signed error. Where `MAE` and `bias` are nearly equal the "
          "error is almost entirely a constant offset, not a failure to see respiration.\n"
          "- **MAE−bias** removes that offset using the test set's own mean. It is an "
          "*oracle* correction and therefore a diagnostic, not a score you could obtain "
          "without labels in the target domain.\n"
          "- **ICC of GT** is the ICC of the reference values themselves on the same "
          "sessions — the ceiling any predictor can reach. A prediction ICC must be read "
          "against it: MAHNOB subjects really do change their breathing rate between "
          "emotional stimuli, so the low ICC there is largely a property of the data.\n",
          "\n## Runs\n"]
    for name, r in runs.items():
        md.append(f"- **`{name}`** ({r['setting']}) — {r['note']}  \n"
                  f"  train {r['n_train']} sessions (used {r.get('n_train_used')}), "
                  f"val {r['n_val']}, best epoch {r.get('best_epoch')}\n")
    md.append("\n## Corpus\n")
    md.append("| dataset | sessions | subjects | with RR reference | split (subj tr/va/te) |")
    md.append("|---|---:|---:|---:|---|")
    for ds, d in sp.items():
        if ds.startswith("_"):
            continue
        s = d["subjects"]
        md.append(f"| {ds.upper()} | {d['n_sessions']['all']} | {len(s['all'])} | "
                  f"{d['n_with_gt']} | {len(s['train'])}/{len(s['val'])}/{len(s['test'])} |")
    p = a.out or f"{paths.ROOT}/RESULTS.md"
    open(p, "w").write("\n".join(md) + "\n")
    print(f"\n[saved] {paths.RESULTS}/metrics.json, {paths.RESULTS}/scatter_all.png, {p}")


if __name__ == "__main__":
    main()
