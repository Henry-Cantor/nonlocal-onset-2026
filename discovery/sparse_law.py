#!/usr/bin/env python3
"""sparse_law.py -- find the ELEGANT law: the most compact dimensionless power-law
scaling (a monomial, IPB98-style) whose held-out AUC matches the full 21-term power
law. Onset genuinely depends on several dimensionless groups, so a 1-2 term symbolic
expression cannot capture it (see sr_search.py); but a 5-7 term product of powers

    H/(1-H)  ~  prod_i  x_i ^ beta_i          (onset when this exceeds a threshold)

recovers almost all of it and IS the interpretable scaling law. Forward-selection adds
the driver that most improves within-machine LOSO until the gain plateaus; the final
logistic in log-space gives the exponents beta_i. Reported vs the full 21-term law and
the Theta drive/damping form, per regime and unified.

    python sparse_law.py
"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import onset_common from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from onset_common import pooled_loso, lomo, fit_logistic, coherent_extent, regime_filter

BASE = ['accel', 'prate_frac', 'inv_etae', 'etae_surge', 'betae', 'gExB_hat', 'ge',
        'TeTi', 'gExB', 'load8', 'rstar', 'corrug', 'overdrive', 'gmc', 'prox',
        'nustar', 'crit']
GEM = ['net_ratio', 'eff_drive', 'uni_drive', 'u_peak']
POWER = BASE + GEM
COLS = list(dict.fromkeys(['machine', 'shot', 'regime', 'q', 'time', 'rho', 'Q_cond_e',
                           'Q_GB', 'recon_resid_te', 'ring_te'] + POWER))
MAX_TERMS = 8
GAIN = 0.002          # stop when the best remaining feature adds less than this to LOMO


def load_regime(regime):
    sub = pq.read_table(os.path.join(ROOT_DIR, "reduced_v2.parquet"), columns=COLS, filters=regime_filter(regime)).to_pandas()
    if regime != 'conv_NBI':
        sub = sub[sub['q'].values > 1.5].reset_index(drop=True)
    r = np.clip(sub['Q_cond_e'].values / np.maximum(sub['Q_GB'].values, 1e-9), 0, 1e4)
    shot = pd.factorize(sub['shot'].values)[0]; mach = pd.factorize(sub['machine'].values)[0]
    c0 = pd.Series(r).groupby(mach).transform('median').values
    resid = r / np.maximum(2.0 * c0, 1e-6) - 1.0
    ext = coherent_extent(resid > 0.4, shot, sub['time'].values, sub['rho'].values)
    y = (resid > 1.0) & (ext >= 4)
    rr, rg = sub['recon_resid_te'].values, sub['ring_te'].values
    keep = (rr <= np.nanquantile(rr, 0.75)) & (rg <= np.nanquantile(rg, 0.75)) & ((resid < 0.4) | y)
    ki = np.where(keep)[0]
    rng = np.random.default_rng(0)
    if len(ki) > 3_000_000:
        ki = np.sort(rng.choice(ki, 3_000_000, replace=False))
    L = {f: np.log(np.maximum(np.abs(sub[f].values[ki].astype(float)), 1e-12)) for f in POWER}
    return L, y[ki].astype(bool), shot[ki], mach[ki]


def dsg(L, feats, idx=None):
    cols = [L[f] if idx is None else L[f][idx] for f in feats]
    return np.column_stack(cols)


def order_by_lomo(L, y, mc):
    """greedy ordering of drivers by cross-machine LOMO gain (add the best each step,
    up to MAX_TERMS, no early stop) -- gives a nested sequence of feature sets."""
    order, cur = [], 0.5
    while len(order) < MAX_TERMS:
        best = (cur, None)
        for f in POWER:
            if f in order:
                continue
            a = lomo(dsg(L, order + [f]), y, mc, cap=120000)[0]
            if np.isfinite(a) and a > best[0]:
                best = (a, f)
        if best[1] is None:
            break
        order.append(best[1]); cur = best[0]
    return order


def report(tag, L, y, sh, mc):
    order = order_by_lomo(L, y, mc)
    # among the nested prefixes, pick the size that maximizes min(LOSO, LOMO): the
    # smallest law that clears BOTH the within- and cross-machine bars.
    scored = []
    for k in range(1, len(order) + 1):
        Xk = dsg(L, order[:k])
        lo = pooled_loso(Xk, y, sh, mc, cap=150000); lm = lomo(Xk, y, mc, cap=300000)[0]
        scored.append((min(lo, lm), lo, lm, order[:k]))
        print(f"    [{k} terms {'*'.join(order[:k])}]  LOSO={lo:.3f} LOMO={lm:.3f}", flush=True)
    _, sp_loso, sp_lomo, chosen = max(scored, key=lambda z: z[0])
    w, mu, sd = fit_logistic(dsg(L, chosen), y.astype(float), l2=2.0)
    exps = w[1:] / sd                                   # power-law exponents in raw log-space
    Xf = dsg(L, POWER)
    pw_loso = pooled_loso(Xf, y, sh, mc, cap=150000); pw_lomo = lomo(Xf, y, mc, cap=300000)[0]
    print(f"\n===== {tag}: onset={y.mean()*100:.1f}% -- chosen {len(chosen)}-term law =====", flush=True)
    print("  H/(1-H) ~ " + " * ".join(f"{f}^{e:+.2f}" for f, e in zip(chosen, exps)), flush=True)
    print(f"  compact law ({len(chosen)} terms)  LOSO={sp_loso:.3f}  LOMO={sp_lomo:.3f}", flush=True)
    print(f"  full power law (21 terms)   LOSO={pw_loso:.3f}  LOMO={pw_lomo:.3f}", flush=True)
    return chosen


sel = {}
for REGIME in ['conv_NBI', 'spherical', 'conv_0torque']:
    L, y, sh, mc = load_regime(REGIME)
    sel[REGIME] = report(REGIME, L, y, sh, mc)
print("\nselected feature sets:", {k: v for k, v in sel.items()}, flush=True)
print("DONE", flush=True)
