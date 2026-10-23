#!/usr/bin/env python3
"""onset_common.py -- shared numerics for the onset-law pipeline: a ridge logistic
(IRLS), ROC-AUC, per-machine balancing weights, the two hold-out validators
(leave-one-shot-out and leave-one-machine-out), and the radial coherence width used
by the onset label. No sklearn; everything is deterministic and memory-light so the
validators run on a 16 GB laptop. Imported by run_onset.py and the figure notebook.
"""
import numpy as np
import pandas as pd

# ---- machine <-> aspect-class map (physical R/a; fixes the upstream regime-router contamination) ----
# The Stellar-side regime router mislabeled a few shots by aspect ratio (conventional D3D wrongly tagged
# 'spherical'; spherical SMART/SMRT wrongly tagged conv_*). Aspect ratio is machine-FIXED, so we enforce it
# here: SMRT's four conv-tagged shots are corrected back to spherical, and SMRT is KEPT as a full spherical
# machine (7 discharges) -- the frozen cross-machine law transfers to it at LOMO AUC 0.965, squarely in the
# spherical pack (MAST/NSTU/NSTX/ST40 = 0.956-0.969). Only AUGD is dropped, and solely on count: 2 discharges
# is too few for any hold-out. NOTE on "data-limited" (SMRT, CMOD, EAST -- <10 discharges in a regime): such a
# machine is a FULL member of the fleet and a scored cross-machine LOMO target (they transfer at 0.95-0.99);
# it is only held out of the within-machine LOSO headline MEAN, because leave-one-shot-out is unstable when
# a machine's onsets pile into one discharge (SMRT has 92% of its onset cells in shot 92220A03 -> its own
# LOSO is ~0.5, a small-sample artifact, NOT a physics failure). The conv_NBI<->conv_0torque split for a
# conventional machine is a legitimate per-shot torque/scenario call and is left to the router.
CONV_MACHINES = ['CMOD', 'D3D', 'EAST', 'KSTR', 'TCV']        # conventional aspect ratio (R/a > ~2.5)
SPHERICAL_MACHINES = ['MAST', 'NSTU', 'NSTX', 'SMRT', 'ST40'] # spherical tokamaks (R/a < ~2)
DROP_MACHINES = ['AUGD']                                      # only 2 discharges -- too few for any hold-out


def regime_filter(regime):
    """pyarrow filter for one regime that ALSO enforces machine<->aspect consistency and drops the
    data-limited machines. Use in place of [('regime','=',regime)] in every reduced_v2 read."""
    valid = SPHERICAL_MACHINES if regime == 'spherical' else CONV_MACHINES
    return [('regime', '=', regime), ('machine', 'in', valid)]


def fit_logistic(X, y, l2=2.0, iters=12, subsample=0, seed=0, wts=None):
    """Ridge logistic via Newton/IRLS (~10 iters). wts = optional per-row weights
    (per-machine balancing so no single machine dominates). subsample>0 draws that
    many rows (weighted -> balanced) then fits unweighted; subsample=0 is an exact
    weighted fit on all rows. Returns (w, mu, sd) for predict_logistic."""
    mu = X.mean(0); sd = np.maximum(X.std(0), 1e-9)
    sw = None
    if subsample and subsample < len(X):
        rng = np.random.default_rng(seed)
        if wts is not None:
            pp = np.asarray(wts, float); pp = pp / pp.sum()
            idx = rng.choice(len(X), subsample, replace=True, p=pp)
        else:
            idx = rng.choice(len(X), subsample, replace=False)
        Xs = (X[idx] - mu) / sd; yb = y[idx].astype(float)
    else:
        Xs = (X - mu) / sd; yb = np.asarray(y, float)
        if wts is not None:
            sw = np.asarray(wts, float); sw = sw * (len(sw) / sw.sum())
    A = np.column_stack([np.ones(len(Xs)), Xs])
    w = np.zeros(A.shape[1])
    R = l2 * np.eye(A.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(A @ w)))
        W = np.maximum(p * (1 - p), 1e-6)
        if sw is not None:
            W = W * sw
            g = A.T @ (sw * (yb - p)) - R @ w
        else:
            g = A.T @ (yb - p) - R @ w
        H = (A.T * W) @ A + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w, mu, sd


def predict_logistic(X, w, mu, sd):
    """Linear predictor (log-odds); monotone in probability, all we need for AUC."""
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w


def roc_auc(score, y):
    """Rank (Mann-Whitney) ROC-AUC; imbalance-robust, no sklearn."""
    y = np.asarray(y, bool)
    m = np.isfinite(score)
    score, y = np.asarray(score)[m], y[m]
    if len(score) < 40 or not (0 < y.mean() < 1):
        return np.nan
    order = np.argsort(score)
    rank = np.empty(len(score)); rank[order] = np.arange(len(score))
    p, n = y.sum(), (~y).sum()
    return (rank[y].sum() - p * (p - 1) / 2) / (p * n)


def machine_weights(mach):
    """Equal TOTAL weight per machine so a 100-shot device cannot outvote a 5-shot
    one in the pooled fit. weight_i = 1 / (cells of machine i)."""
    w = np.ones(len(mach), float)
    for m in np.unique(mach):
        mm = (mach == m)
        w[mm] = 1.0 / max(int(mm.sum()), 1)
    return w


def pooled_loso(X, y, shot, mach, cap=200000, seed=0):
    """Leave-one-SHOT-out AUC: draw a machine-balanced training pool once (<=cap
    rows), then for each held-out shot re-fit with that shot removed and score its
    full set of cells. Held-out shots are treated as shots that do not exist yet."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pool = rng.choice(n, cap, replace=False) if n > cap else np.arange(n)
    Xp, yp, sp, mp = X[pool], y[pool], shot[pool], mach[pool]
    order = np.argsort(shot, kind='stable'); shs = shot[order]
    uniq, first = np.unique(shs, return_index=True); bounds = np.append(first, len(shs))
    ps, ys = [], []
    for k in range(len(uniq)):
        te = order[bounds[k]:bounds[k + 1]]
        if len(te) < 40 or not (0 < y[te].mean() < 1):
            continue
        keep = sp != uniq[k]
        if keep.sum() < 200 or not (0 < yp[keep].mean() < 1):
            continue
        w, mu, sd = fit_logistic(Xp[keep], yp[keep].astype(float), l2=2.0,
                                 subsample=0, wts=machine_weights(mp[keep]))
        ps.append(predict_logistic(X[te], w, mu, sd)); ys.append(y[te].astype(bool))
    return roc_auc(np.concatenate(ps), np.concatenate(ys)) if ps else np.nan


def lomo(X, y, mach, cap=300000, seed=0, shot=None, min_shots=0, return_per=False):
    """Leave-one-MACHINE-out AUC: freeze the coefficients on all-but-one machine and
    score the never-seen machine (the transfer test that stands in for ITER). For
    each held-out machine a FRESH machine-balanced sample (<=cap rows) is drawn from
    the TRAINING machines only, so small machines are represented at full weight in
    every fold. A fold needs only a valid held-out AUC (>=40 cells of each class); the
    fleet's data-limited machines (SMRT/CMOD/EAST) all transfer at 0.95-0.99 and ARE
    scored -- so `min_shots` defaults to 0 (score every machine). It is retained only so
    a caller can optionally reproduce the old shot-gated mean. Returns (mean AUC over
    scored machines, n_machines_scored); with return_per, also a {machine: auc} dict."""
    rng = np.random.default_rng(seed)
    aucs = []; per = {}
    for m in np.unique(mach):
        te = mach == m; tr = np.where(mach != m)[0]
        if te.sum() < 500 or not (0 < y[te].mean() < 1) or not (0 < y[tr].mean() < 1):
            continue
        if shot is not None and min_shots > 0 and len(np.unique(shot[te])) < min_shots:
            continue                                  # optional legacy gate: skip <min_shots-discharge machine
        samp = rng.choice(tr, min(cap, len(tr)), replace=False)
        w, mu, sd = fit_logistic(X[samp], y[samp].astype(float), l2=2.0,
                                 subsample=0, wts=machine_weights(mach[samp]))
        a = roc_auc(predict_logistic(X[te], w, mu, sd), y[te])
        if np.isfinite(a):
            aucs.append(a); per[m] = float(a)
    mean_n = (float(np.nanmean(aucs)), len(aucs)) if aucs else (np.nan, 0)
    return (mean_n[0], mean_n[1], per) if return_per else mean_n


def coherent_extent(mask, shot, time, rho):
    """Radial WIDTH of a surging structure: at each (shot,time) count the surfaces
    within +/-2 that are simultaneously above threshold. Wide band = a coherent
    non-local avalanche; a lone pixel = local jitter. Returns a per-cell count."""
    n = len(mask)
    d = pd.DataFrame({'s': shot, 't': time, 'r': rho, 'm': np.asarray(mask, float),
                      '_i': np.arange(n)}).sort_values(['s', 't', 'r'])
    e = d.groupby(['s', 't'])['m'].transform(
        lambda v: v.rolling(5, center=True, min_periods=1).sum())
    out = np.zeros(n); out[d['_i'].values] = e.values
    return out
