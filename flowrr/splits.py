"""Subject-level 7:1:2 splits, seeded and materialised to data/splits.json.

Splits are built from data/manifest.json — i.e. from the sessions that actually produced a
usable ST-map — so the counts reported are the counts that get trained on. Partitioning is
at the SUBJECT level: no person appears in more than one of train/val/test, which is what
makes the held-out ICC meaningful (ICC groups predictions by person).
"""
import json, argparse
import numpy as np
from . import config as paths

RATIOS = (0.7, 0.1, 0.2)
SEED = 42

# Datasets with no respiratory reference are TRAINING-ONLY: nothing is ever evaluated on
# them, so a held-out split would be a split that never gets used. They are recorded with
# all sessions under "all" and empty train/val/test, and the pool builder takes "all".
TRAIN_ONLY = ("pure", "ubfc", "vitalvideo")

# CUDO gets a subject-disjoint split even though it has no respiratory reference, because the
# CUDO track needs a held-out CUDO test set. It is still kept out of the AUX training pool in
# experiments.py: it is proprietary, so a model that saw it may only be reported on CUDO.
NO_GT_SPLIT = ("cudo",)


def load_manifest():
    return json.load(open(paths.MANIFEST))


def variant_filter(variant=None):
    """Predicate deciding whether a manifest record belongs to this variant's corpus."""
    variant = variant or paths.VARIANT
    if variant == "full":
        return lambda r: True
    if variant == "sonly":
        # MAHNOB neutral trials carry only 5.3-22.2 s of stimulus (~1-4 breathing cycles at
        # 0.1-0.5 Hz), so both their belt reference and any prediction over them are weak.
        return lambda r: not (r["dataset"] == "mahnob" and r.get("trial") == "N")
    raise ValueError(f"unknown variant: {variant}")


def usable(man, dataset=None, require_gt=False, keep=None):
    keep = keep or variant_filter()
    out = {}
    for sid, r in man.items():
        if r.get("status") not in ("ok", "cached"):
            continue
        if not keep(r):
            continue
        if dataset and r["dataset"] != dataset:
            continue
        if require_gt and r.get("gt_rr") is None:
            continue
        out[sid] = r
    return out


def split_subjects(subjects, ratios=RATIOS, seed=SEED):
    """Deterministic subject-level partition. Returns (train, val, test) lists."""
    subs = sorted(set(map(str, subjects)))
    n = len(subs)
    perm = np.random.default_rng(seed).permutation(n)
    order = [subs[i] for i in perm]
    n_test = int(round(ratios[2] * n))
    n_val = int(round(ratios[1] * n))
    n_test = max(1, min(n_test, n - 2)) if n >= 3 else 0
    n_val = max(1, min(n_val, n - n_test - 1)) if n >= 3 else 0
    test = sorted(order[:n_test])
    val = sorted(order[n_test:n_test + n_val])
    train = sorted(order[n_test + n_val:])
    return train, val, test


def build(seed=SEED, ratios=RATIOS, variant=None):
    variant = variant or paths.VARIANT
    keep = variant_filter(variant)
    man = load_manifest()
    out = {"_meta": {"seed": seed, "ratios": list(ratios), "unit": "subject",
                     "variant": variant}}
    datasets = sorted({r["dataset"] for r in man.values()
                       if r.get("status") in ("ok", "cached") and keep(r)})
    for ds in datasets:
        S = usable(man, ds, keep=keep)
        by = {}
        for sid, r in S.items():
            by.setdefault(str(r["subject"]), []).append(sid)
        if ds in TRAIN_ONLY:
            out[ds] = {
                "all": sorted(S), "train": [], "val": [], "test": [],
                "subjects": {"all": sorted(by), "train": [], "val": [], "test": []},
                "n_sessions": {"all": len(S), "train": 0, "val": 0, "test": 0},
                "n_with_gt": 0, "role": "training-only (no respiratory reference)",
            }
            continue
        tr, va, te = split_subjects(by.keys(), ratios, seed)
        pick = lambda ss: sorted(s for u in ss for s in by[u])
        out[ds] = {
            "all": sorted(S), "train": pick(tr), "val": pick(va), "test": pick(te),
            "subjects": {"all": sorted(by), "train": tr, "val": va, "test": te},
            "n_sessions": {"all": len(S), "train": len(pick(tr)), "val": len(pick(va)),
                           "test": len(pick(te))},
            "n_with_gt": sum(1 for s in S if S[s].get("gt_rr") is not None),
            "role": "labelled (evaluated)",
        }
    json.dump(out, open(paths.SPLITS, "w"), indent=1)
    return out, paths.SPLITS


def load():
    return json.load(open(paths.SPLITS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    sp, p = build(seed=a.seed)
    print(f"[saved] {p}  (variant={paths.VARIANT})")
    for ds, d in sp.items():
        if ds.startswith("_"):
            continue
        s = d["subjects"]
        if ds in TRAIN_ONLY:
            print(f"  {ds:8s} training-only: {d['n_sessions']['all']} sessions, "
                  f"{len(s['all'])} subjects, no respiratory reference -> not split")
            continue
        print(f"  {ds:8s} subjects {len(s['train'])}/{len(s['val'])}/{len(s['test'])} "
              f"(train/val/test of {len(s['all'])})  sessions "
              f"{d['n_sessions']['train']}/{d['n_sessions']['val']}/{d['n_sessions']['test']} "
              f"of {d['n_sessions']['all']}  with_gt={d['n_with_gt']}")


# ── training pools ───────────────────────────────────────────────────────────
DATASETS = ("cohface", "mahnob", "pure", "ubfc", "vitalvideo")
VAL_FRACTION = 0.10          # subject-level val slice carved from an unlabelled corpus
VAL_SEED = 20240501


def _aux_val(sp, ds):
    """Seeded SUBJECT-level val slice for a corpus that has no split of its own.

    PURE, UBFC and VitalVideo carry no respiration reference, so they are never scored and
    have no test split. They still need a val slice, because checkpoint selection runs on the
    label-free InfoNCE and that must be measured on subjects the model did not train on.
    """
    by = {}
    for s in sp[ds]["all"]:
        by.setdefault(s.split("_")[1], []).append(s)
    subs = sorted(by)
    k = max(1, int(round(len(subs) * VAL_FRACTION)))
    rng = np.random.default_rng(VAL_SEED)
    pick = {subs[i] for i in rng.permutation(len(subs))[:k]}
    return sorted(s for u in pick for s in by[u])


def pool(sp, combo):
    """(train_sids, val_sids) for a list of dataset names.

    A labelled corpus contributes its own train/val splits; its test split is never touched.
    An unlabelled corpus contributes everything except a seeded val slice.
    """
    tr, va = [], []
    for ds in combo:
        if ds not in sp:
            raise KeyError(f"unknown dataset {ds!r}; known: {sorted(k for k in sp if not k.startswith('_'))}")
        if sp[ds]["test"]:
            tr += sp[ds]["train"]; va += sp[ds]["val"]
        else:
            v = _aux_val(sp, ds)
            va += v
            tr += sorted(set(sp[ds]["all"]) - set(v))
    return sorted(set(tr)), sorted(set(va))
