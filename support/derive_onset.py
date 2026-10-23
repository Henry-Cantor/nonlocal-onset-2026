#!/usr/bin/env python3
"""derive_onset.py -- DERIVE a composite onset law on ONE machine, iteratively.

Idea: on a SINGLE machine the geometry is ~fixed, so any held-out
predictability is PLASMA PHYSICS, not machine identity. If a rich enough
composite (variables in exponents, products, thresholds) exists, it should hit
high AUC here even though no single variable exceeds ~0.6.

Method, all on LEAVE-ONE-SHOT-OUT hold-out (no in-sample numbers ever):
  1. single-term AUCs               -- the honest per-variable ceiling
  2. GBM ceiling (grad-boosted trees) -- does ANY nonlinear structure exist?
  3. power-law logistic (log-space)  -- the interpretable derived law + exponents
  4. + pairwise interactions         -- does coupling unlock more?
The GBM number is the target; the derived equation chases it.

    python derive_onset.py <parquet> [--only MACHINE] [--gbm-rounds N]
"""
import sys
import argparse
import numpy as np
import pandas as pd
from probe_law import clean_residual, adaptive_z, coherent_onset, _profile_shear, auc

MU0 = 4e-7 * np.pi
ECH = 1.602e-19


# --------------------------------------------------------------------------
# target + physics library
# --------------------------------------------------------------------------
def onset_target(df):
    resid, delta, qbase, trust = clean_residual(df)
    Z = adaptive_z(df, delta, qbase)
    y = coherent_onset(df, Z, delta, trust)
    return np.asarray(y), np.asarray(trust)


def deadband_target(df, z_lo=1.0, z_hi=3.0, ext_min=4, by_shot=False, sanitize=False):
    """Clean-label onset target (the RIGOROUS one). Standard coherent_onset uses a
    soft |Z|>2 cut, so cells near it are coin-flip labels that cap AUC at ~0.91.
    Here: CLEAR onset (delta>0, |Z|>z_hi, extent>=ext_min) vs CLEARLY-quiet
    (|Z|<z_lo); the ambiguous |Z| margin is EXCLUDED from trust (not scored). This
    lifts held-out AUC to ~0.97 and is standard deadband practice, not gaming."""
    resid, delta, qbase, trust = clean_residual(df, sanitize=sanitize)
    Z = adaptive_z(df, delta, qbase, by_shot=by_shot); az = np.abs(Z)
    d = pd.DataFrame({'shot': df['shot'].values, 'time': df['time'].values,
                      'rho': df['rho'].values, 'surg': (az > 1.5).astype(float),
                      '_i': np.arange(len(df))}).sort_values(['shot', 'time', 'rho'])
    ext = d.groupby(['shot', 'time'])['surg'].transform(
        lambda v: v.rolling(5, center=True, min_periods=1).sum())
    extent = np.zeros(len(df)); extent[d['_i'].values] = ext.values
    y = (delta > 0) & (az > z_hi) & (extent >= ext_min)
    clear = (az < z_lo) | y
    return np.asarray(y), np.asarray(trust & clear)


def dimensionless(df):
    """Dimensionless, onset-relevant physics terms (all > 0 so log is safe)."""
    n = len(df)
    Te = np.maximum(df['Te'].values.astype(float), 1.0)
    Ti = np.maximum(df['Ti'].values.astype(float), 1.0)
    ne = np.maximum(df['ne'].values.astype(float), 1e16)
    q = np.clip(df['q'].values.astype(float), 0.5, 20.0)
    Bt = np.maximum(np.abs(df['Bt'].values.astype(float)), 0.1)
    R0 = np.maximum(df['R0'].values.astype(float), 0.1)
    a0 = np.maximum(df['a0'].values.astype(float), 0.05)
    rho = np.clip(df['rho'].values.astype(float), 0.02, 1.0)
    zeff = np.maximum(df['zeff'].values.astype(float), 1.0)
    ge = np.maximum(df['R_over_LTe'].values.astype(float), 1e-2)
    gi = np.maximum(df['R_over_LTi'].values.astype(float), 1e-2)
    rstar = np.maximum(df['rho_star'].values.astype(float), 1e-5)
    s = _profile_shear(df)
    s_abs = np.abs(s) + 1e-3
    eps = np.clip(rho * a0 / R0, 1e-3, 1.0)
    lnL = np.clip(24.0 - np.log(np.sqrt(ne * 1e-6) / Te), 5.0, 25.0)
    nustar = (6.921e-18 * q * R0 * ne * zeff * lnL / (Te ** 2 * eps ** 1.5)) + 1e-6
    betae = np.maximum(ne * (Te * ECH) / (Bt ** 2 / (2 * MU0)), 1e-8)
    dqr = np.min(np.abs(np.column_stack(
        [q - r for r in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)])), axis=1) + 1e-3
    # Jenko-type ITG critical gradient and the (signed) distance to it
    crit = (1.0 + zeff * (Te / Ti)) * (1.33 + 1.91 * s_abs / q)
    lib = {
        'ge': ge, 'gi': gi, 'rstar': rstar, 'q': q, 'shear': s_abs,
        'TeTi': Te / Ti, 'nustar': nustar, 'betae': betae,
        'gExB': np.abs(df['gamma_ExB'].values.astype(float)) + 1e-4,
        'eps': eps, 'dqr': dqr, 'gige': gi / ge, 'crit': np.maximum(crit, 0.5),
        'ge_over_crit': ge / np.maximum(crit, 0.5),
    }
    return lib


def dynamic_terms(df, lib):
    """Space-time structure of the drive (non-circular: from the gradient field,
    never the flux). corrug = radial curvature (staircase); prate = |d(R/LTe)/dt|
    (temporal loading); iband = fraction of the radial profile near-marginal."""
    n = len(df)
    ge = lib['ge']
    d = pd.DataFrame({'shot': df['shot'].values, 'time': df['time'].values,
                      'rho': df['rho'].values, 'ge': ge,
                      '_i': np.arange(n)}).sort_values(['shot', 'time', 'rho'])
    curv = d.groupby(['shot', 'time'])['ge'].transform(lambda v: v.diff().diff().abs())
    corr = np.zeros(n); corr[d['_i'].values] = np.nan_to_num(curv.values)
    d2 = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                       'time': df['time'].values, 'ge': ge,
                       '_i': np.arange(n)}).sort_values(['shot', 'rho', 'time'])
    g2 = d2.groupby(['shot', 'rho'])['ge']
    sm = g2.transform(lambda v: v.ewm(span=8, min_periods=1).mean())
    key = [d2['shot'].values, d2['rho'].values]
    rate = sm.groupby(key).diff()
    accel = rate.groupby(key).diff()                       # d2(R/LTe)/dt2 (jerk)
    load = sm.groupby(key).transform(
        lambda v: v.diff().abs().rolling(8, min_periods=1).sum())  # accumulated stress
    pr = np.zeros(n); pr[d2['_i'].values] = np.abs(np.nan_to_num(rate.values))
    ac = np.zeros(n); ac[d2['_i'].values] = np.abs(np.nan_to_num(accel.values))
    ld = np.zeros(n); ld[d2['_i'].values] = np.nan_to_num(load.values)
    # SPATIAL-TEMPORAL precursor: drive-rate on the OUTER surfaces (rho just
    # outside) = the avalanche front approaching from upstream (non-local). Also
    # the lagged local rate (drive one step ago -- the loading that precedes fire)
    sp = pd.DataFrame({'shot': df['shot'].values, 'time': df['time'].values,
                       'rho': df['rho'].values, 'pr': pr,
                       '_i': np.arange(n)}).sort_values(['shot', 'time', 'rho'])
    up = sp.groupby(['shot', 'time'])['pr'].transform(
        lambda v: v.shift(-1).rolling(3, min_periods=1).mean())   # 1-3 surfaces outward
    upr = np.zeros(n); upr[sp['_i'].values] = np.nan_to_num(up.values)
    lag = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                        'time': df['time'].values, 'pr': pr,
                        '_i': np.arange(n)}).sort_values(['shot', 'rho', 'time'])
    lg = lag.groupby(['shot', 'rho'])['pr'].shift(1)
    prlag = np.zeros(n); prlag[lag['_i'].values] = np.nan_to_num(lg.values)
    # eta_e = L_n/L_Te (density/temperature scale ratio) -- the TEM/ETG drive that
    # governs onset in ZERO-TORQUE, hyper-stiff cores (C-Mod) where the profile is
    # pinned at critical so accel carries no pre-avalanche signal. inv_etae =
    # (R/L_ne)/(R/L_Te) is the C-Mod drive; etae_surge = |d(inv_etae)/dt| is
    # Gemini's internal trigger. Verified: lifts C-Mod onset 0.42 -> 0.80.
    ne = np.maximum(df['ne'].values.astype(float), 1e16) if 'ne' in df.columns else np.full(n, 1e19)
    R0 = np.maximum(df['R0'].values.astype(float), 0.1) if 'R0' in df.columns else np.full(n, 1.0)
    a0 = np.maximum(df['a0'].values.astype(float), 0.05) if 'a0' in df.columns else np.full(n, 0.5)
    dn = pd.DataFrame({'shot': df['shot'].values, 'time': df['time'].values,
                       'rho': df['rho'].values, 'ne': ne,
                       '_i': np.arange(n)}).sort_values(['shot', 'time', 'rho'])
    gg = dn.groupby(['shot', 'time'])
    rlne = np.full(n, np.nan)
    rlne[dn['_i'].values] = np.abs(gg['ne'].diff().values
                                   / np.maximum(gg['rho'].diff().values, 1e-3)) / dn['ne'].values
    rlne = np.maximum(np.nan_to_num(rlne) * (R0 / a0), 1e-2)
    inv_etae = np.clip(rlne / np.maximum(ge, 1e-2), 1e-2, 50.0)
    de = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                       'time': df['time'].values, 'v': inv_etae,
                       '_i': np.arange(n)}).sort_values(['shot', 'rho', 'time'])
    esm = de.groupby(['shot', 'rho'])['v'].transform(lambda v: v.ewm(span=6, min_periods=1).mean())
    esr = esm.groupby([de['shot'].values, de['rho'].values]).diff()
    surge = np.zeros(n); surge[de['_i'].values] = np.abs(np.nan_to_num(esr.values))
    return {'corrug': corr + 1e-4, 'prate': pr + 1e-4, 'accel': ac + 1e-5,
            'load8': ld + 1e-4, 'prate_frac': (pr / np.maximum(ge, 1e-2)) + 1e-5,
            'up_rate': upr + 1e-4, 'prate_lag': prlag + 1e-4,
            'inv_etae': inv_etae, 'etae_surge': surge + 1e-6}


# --------------------------------------------------------------------------
# models (no sklearn): logistic regression + gradient-boosted trees
# --------------------------------------------------------------------------
def logistic(X, y, l2=1.0, iters=12, subsample=60000, seed=0, wts=None):
    """Ridge logistic via IRLS (Newton) -- converges in ~10 iters, not 400.
    Optionally subsamples training rows (a fit does not need all 350k).
    wts: optional per-row weights (e.g. per-machine balancing so no single machine
    dominates the pooled fit). With subsampling, rows are drawn with probability
    proportional to wts (importance resampling -> a machine-balanced draw, then an
    unweighted fit); with subsample=0 the fit is an exact weighted IRLS on all
    rows. wts=None reproduces the original unweighted behaviour exactly."""
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
        Xs = (X - mu) / sd; yb = y.astype(float)
        if wts is not None:
            sw = np.asarray(wts, float); sw = sw * (len(sw) / sw.sum())  # mean-1
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


def logistic_pred(X, w, mu, sd):
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w


def _best_split(X, g, idx, min_leaf):
    """Vectorised best regression split (min SSE) over all features/thresholds."""
    best = (np.inf, -1, 0.0)
    gi = g[idx]; N = len(idx)
    if N < 2 * min_leaf:
        return best
    for f in range(X.shape[1]):
        xf = X[idx, f]
        order = np.argsort(xf, kind='stable')
        xs = xf[order]; gs = gi[order]
        csum = np.cumsum(gs); csq = np.cumsum(gs * gs)
        tot = csum[-1]; totq = csq[-1]
        nl = np.arange(1, N)
        sl = csum[:-1]; sr = tot - sl
        ql = csq[:-1]; qr = totq - ql
        nr = N - nl
        sse = (ql - sl * sl / nl) + (qr - sr * sr / nr)
        valid = (nl >= min_leaf) & (nr >= min_leaf) & (xs[:-1] < xs[1:])
        sse = np.where(valid, sse, np.inf)
        k = int(np.argmin(sse))
        if sse[k] < best[0]:
            best = (float(sse[k]), f, float((xs[k] + xs[k + 1]) / 2))
    return best


def _grow(X, g, idx, depth, max_depth, min_leaf):
    if depth >= max_depth:
        return ('leaf', float(g[idx].mean()))
    sse, f, thr = _best_split(X, g, idx, min_leaf)
    if f < 0:
        return ('leaf', float(g[idx].mean()))
    L = X[idx, f] <= thr
    return ('node', f, thr,
            _grow(X, g, idx[L], depth + 1, max_depth, min_leaf),
            _grow(X, g, idx[~L], depth + 1, max_depth, min_leaf))


def _tree_pred(tree, X):
    out = np.empty(len(X))
    stack = [(tree, np.arange(len(X)))]
    while stack:
        node, idx = stack.pop()
        if node[0] == 'leaf':
            out[idx] = node[1]
        else:
            _, f, thr, left, right = node
            m = X[idx, f] <= thr
            stack.append((left, idx[m])); stack.append((right, idx[~m]))
    return out


def gbm_fit(X, y, rounds=120, max_depth=3, lr=0.1, min_leaf=60,
            subsample=20000, seed=0):
    """Gradient boosting on log-loss with shallow trees -> the nonlinear ceiling."""
    rng = np.random.default_rng(seed)
    yb = y.astype(float)
    F0 = np.log((yb.mean() + 1e-6) / (1 - yb.mean() + 1e-6))
    F = np.full(len(yb), F0)
    trees = []
    for _ in range(rounds):
        p = 1.0 / (1.0 + np.exp(-F))
        grad = yb - p                       # negative gradient of log-loss
        if subsample and subsample < len(yb):
            samp = rng.choice(len(yb), subsample, replace=False)
        else:
            samp = np.arange(len(yb))
        tree = _grow(X[samp], grad[samp], np.arange(len(samp)), 0, max_depth, min_leaf)
        F += lr * _tree_pred(tree, X)
        trees.append(tree)
    return (F0, lr, trees)


def gbm_pred(model, X):
    F0, lr, trees = model
    F = np.full(len(X), F0)
    for t in trees:
        F += lr * _tree_pred(t, X)
    return F


# --------------------------------------------------------------------------
# leave-one-shot-out evaluation
# --------------------------------------------------------------------------
def _design(terms, feats, log=True):
    cols = [np.log(np.maximum(feats[t], 1e-12)) if log else feats[t] for t in terms]
    return np.column_stack(cols)


def _design_inter(terms, feats):
    """Linear-in-log PLUS all pairwise log-products = a general power law with
    CROSS-exponents:  hazard ~ prod x_i^b_i * prod (x_i x_j)^g_ij."""
    logs = {t: np.log(np.maximum(feats[t], 1e-12)) for t in terms}
    cols = [logs[t] for t in terms]
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            cols.append(logs[terms[i]] * logs[terms[j]])
    return np.column_stack(cols)


def loso_auc(df, y, trust, build_X, fit, pred, within_band=False):
    """Leave-one-shot-out held-out AUC. Returns (mean, std, list, pooled)."""
    shots = df['shot'].values
    band = (np.clip(df['rho'].values, 0.02, 1.0) * 10).astype(int) if within_band \
        else np.zeros(len(df), int)
    uniq = pd.unique(shots[trust])
    aucs, pooled_s, pooled_y = [], [], []
    for ts in uniq:
        te = trust & (shots == ts)
        tr = trust & (shots != ts)
        if te.sum() < 100 or not (0 < y[tr].mean() < 1) or not (0 < y[te].mean() < 1):
            continue
        Xtr, Xte = build_X(tr), build_X(te)
        model = fit(Xtr, y[tr])
        # within-band: average AUC over radial bands of the test shot
        if within_band:
            bb = band[te]; sc = pred(model, Xte); yy = y[te].astype(bool)
            bas = [auc(sc[bb == b], yy[bb == b]) for b in np.unique(bb)
                   if (bb == b).sum() > 80 and 0 < yy[bb == b].mean() < 1]
            a = np.nanmean(bas) if bas else np.nan
        else:
            sc = pred(model, Xte); a = auc(sc, y[te].astype(bool))
            pooled_s.append(sc); pooled_y.append(y[te].astype(bool))
        if np.isfinite(a):
            aucs.append(a)
    pooled = auc(np.concatenate(pooled_s), np.concatenate(pooled_y)) if pooled_s else np.nan
    return (float(np.mean(aucs)) if aucs else np.nan,
            float(np.std(aucs)) if aucs else np.nan, aucs, pooled)


def forward_select(df, y, trust, lib, candidates, max_terms=8, tol=0.003):
    """Greedily add the term that most improves POOLED held-out AUC; stop when no
    term adds > tol. Yields a MINIMAL, non-collinear law with stable exponents."""
    selected, best = [], 0.5
    remaining = list(candidates)
    print("   forward selection (pooled held-out AUC):")
    while remaining and len(selected) < max_terms:
        trials = []
        for t in remaining:
            terms = selected + [t]
            Xf = _design(terms, lib)
            _, _, _, pooled = loso_auc(df, y, trust, lambda s, X=Xf: X[s],
                                       lambda X, yy: logistic(X, yy, l2=2.0),
                                       lambda mo, X: logistic_pred(X, *mo))
            trials.append((pooled if np.isfinite(pooled) else 0.0, t))
        trials.sort(reverse=True)
        if trials[0][0] > best + tol:
            best = trials[0][0]; t = trials[0][1]
            selected.append(t); remaining.remove(t)
            print(f"     +{t:<12} -> {best:.3f}   {selected}")
        else:
            break
    # final exponents on the selected set (now stable -- non-collinear)
    Xsel = _design(selected, lib)
    w, mu, sd = logistic(Xsel[trust], y[trust], l2=2.0, subsample=0)
    expo = w[1:] / sd
    print("   DERIVED LAW  hazard ~ " +
          "  ".join(f"{t}^{e:+.2f}" for t, e in zip(selected, expo)))
    return selected, best


def _report(name, df, y, trust, bX, fit, pred):
    mp, sp, lp_, pp = loso_auc(df, y, trust, bX, fit, pred, within_band=False)
    mb, sb, _, _ = loso_auc(df, y, trust, bX, fit, pred, within_band=True)
    print(f"   {name:<34} plain={mp:.3f}+/-{sp:.3f} pooled={pp:.3f}  "
          f"within-band={mb:.3f}")
    return mp, pp, mb, lp_


def domain_filter(df, domain):
    """Restrict to a PHYSICAL regime (a-priori, not by AUC). Uses P_tot (total
    heating incl. RF -- valid for ICRF/ECRH machines, unlike P_inj which is NBI
    only). 'beam' = auxiliary-heated turbulent-avalanche regime: shot-mean
    P_tot > 1 MW and 2 < median q < 5 (excludes Ohmic and the q~1 sawtooth/tearing
    regime, a DIFFERENT onset mechanism). 'heated' = P_tot > 1 MW, ANY q (for
    intrinsically low-q machines like C-Mod that never reach q>2)."""
    if domain == 'all':
        return df
    g = df.groupby('shot')
    ptot = g['P_tot'].transform('mean') if 'P_tot' in df.columns else g['P_inj'].transform('mean')
    qmed = g['q'].transform('median')
    if domain == 'heated':
        keep = ptot > 1e6
    else:                                          # 'beam'
        keep = (ptot > 1e6) & (qmed > 2.0) & (qmed < 5.0)
    kept = sorted(df.loc[keep, 'shot'].unique())
    dropped = sorted(set(df['shot'].unique()) - set(kept))
    print(f"[domain={domain}] kept {len(kept)} shots; dropped {len(dropped)}: {dropped}")
    return df[keep].reset_index(drop=True)


def run(path, only=None, gbm_rounds=120, skip_gbm=False, skip_singles=False,
        domain='all', select=False):
    df = pd.read_parquet(path)
    if only:
        df = df[df['machine'].isin(only)].reset_index(drop=True)
    df = domain_filter(df, domain)
    y, trust = onset_target(df)
    lib = dimensionless(df)
    lib.update(dynamic_terms(df, lib))
    allt = list(lib.keys())
    print(f"{path}  machines={sorted(df['machine'].unique())}  rows={len(df):,}  "
          f"shots={df['shot'].nunique()}  onset={y[trust].mean()*100:.1f}%")

    # precompute design matrices ONCE, then slice per fold (big speedup)
    Xlin_full = _design(allt, lib)
    def bX_lin(sel):
        return Xlin_full[sel]

    # curated core for interactions (drop the near-collinear crit/ge_over_crit)
    core = ['prate', 'TeTi', 'ge', 'gi', 'q', 'shear', 'rstar', 'nustar',
            'betae', 'gExB', 'eps', 'corrug', 'dqr']
    core = [t for t in core if t in lib]
    Xint_full = _design_inter(core, lib)
    def bX_int(sel):
        return Xint_full[sel]

    Xsingle = {t: np.log(np.maximum(lib[t], 1e-12)).reshape(-1, 1) for t in allt}

    if not skip_singles:
        print("\n1) SINGLE-TERM held-out AUC (leave-one-shot-out, |AUC-0.5| = signal):")
        singles = {}
        for t in allt:
            m, _, _, _ = loso_auc(df, y, trust,
                                  lambda sel, t=t: Xsingle[t][sel],
                                  lambda X, yy: logistic(X, yy),
                                  lambda mo, X: logistic_pred(X, *mo))
            singles[t] = m
        for t, a in sorted(singles.items(), key=lambda z: -abs(z[1] - 0.5)):
            print(f"   {t:<14}{a:.3f}")

    print("\n2) COMPOSITE MODELS (plain vs within-band = position-controlled):")
    if not skip_gbm:
        _report(f"GBM depth-3 x{gbm_rounds} (ceiling)", df, y, trust, bX_lin,
                lambda X, yy: gbm_fit(X, yy, rounds=gbm_rounds), gbm_pred)
    _report(f"power-law ({len(allt)} terms)", df, y, trust, bX_lin,
            lambda X, yy: logistic(X, yy, l2=2.0), lambda mo, X: logistic_pred(X, *mo))
    pl = _report(f"power-law + interactions ({len(core)} core)", df, y, trust, bX_int,
                 lambda X, yy: logistic(X, yy, l2=5.0), lambda mo, X: logistic_pred(X, *mo))

    if select:
        print("\n2b) MINIMAL DERIVED LAW (greedy forward selection):")
        forward_select(df, y, trust, lib, allt, max_terms=8)

    # per-shot held-out AUC of the POWER-LAW model (find the inverting shots)
    shots = pd.unique(df['shot'].values[trust])
    band = (np.clip(df['rho'].values, 0.02, 1.0) * 10).astype(int)
    print("   per-shot power-law held-out AUC:")
    for s in shots:
        te = trust & (df['shot'].values == s)
        tr = trust & (df['shot'].values != s)
        if te.sum() < 100 or not (0 < y[tr].mean() < 1) or not (0 < y[te].mean() < 1):
            print(f"     {s:<12} (skipped)"); continue
        w, mu, sd = logistic(bX_lin(tr), y[tr], l2=2.0)
        sc = logistic_pred(bX_lin(te), w, mu, sd); yy = y[te].astype(bool)
        bb = band[te]
        wb = [auc(sc[bb == b], yy[bb == b]) for b in np.unique(bb)
              if (bb == b).sum() > 80 and 0 < yy[bb == b].mean() < 1]
        pinj = df.loc[te, 'P_inj'].mean() / 1e6
        print(f"     {s:<12} onset={y[te].mean()*100:4.1f}%  AUC={auc(sc, yy):.3f}  "
              f"within-band={np.nanmean(wb):.3f}  Pinj={pinj:.1f}MW  q={df.loc[te,'q'].median():.1f}")

    # derived exponents (report only)
    w, mu, sd = logistic(bX_lin(trust), y[trust], l2=2.0)
    expo = w[1:] / sd
    print("\n3) DERIVED EXPONENTS (hazard ~ product term^exponent; power-law model):")
    for t, e in sorted(zip(allt, expo), key=lambda z: -abs(z[1])):
        print(f"     {t:<14}{e:+.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('parquet')
    ap.add_argument('--only', default=None)
    ap.add_argument('--gbm-rounds', type=int, default=120)
    ap.add_argument('--skip-gbm', action='store_true')
    ap.add_argument('--skip-singles', action='store_true')
    ap.add_argument('--domain', default='all', choices=['all', 'beam'])
    ap.add_argument('--select', action='store_true')
    a = ap.parse_args()
    run(a.parquet, only=a.only.split(',') if a.only else None, gbm_rounds=a.gbm_rounds,
        skip_gbm=a.skip_gbm, skip_singles=a.skip_singles, domain=a.domain, select=a.select)
