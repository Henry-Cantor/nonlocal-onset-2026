#!/usr/bin/env python3
"""law_compare.py -- the definitive law comparison for the paper's "law" section.
Evaluates three nested dimensionless power laws with the SAME held-out machinery
(pooled leave-one-shot-out and leave-one-machine-out), per regime:

  full   : the 21-term power law (the optimally-fit monomial = the baseline/ceiling)
  phys5  : a 5-term physical scaling  rstar, eff_drive, betae, gExB_hat, nustar
           (scale, drive, electromagnetic, E×B damping, collisionality gate)
  dom2   : the 2 dominant cross-machine terms  rstar, eff_drive

and prints the phys5 exponents (the elegant scaling law). Reads reduced_v2.parquet.

    python law_compare.py
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
FULL = BASE + GEM
PHYS5 = ['rstar', 'eff_drive', 'betae', 'gExB_hat', 'nustar']
DOM2 = ['rstar', 'eff_drive']
LAWS = {'full(21)': FULL, 'phys5': PHYS5, 'dom2': DOM2}
COLS = list(dict.fromkeys(['machine', 'shot', 'regime', 'q', 'time', 'rho', 'Q_cond_e',
                           'Q_GB', 'recon_resid_te', 'ring_te'] + FULL))


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
    L = {f: np.log(np.maximum(np.abs(sub[f].values[ki].astype(float)), 1e-12)) for f in FULL}
    return L, y[ki].astype(bool), shot[ki], mach[ki]


def dsg(L, feats):
    return np.column_stack([L[f] for f in feats])


print(f"{'regime':<14}{'law':<10}{'pooledLOSO':>11}{'LOMO':>8}")
phys5_law = {}
for REGIME in ['conv_NBI', 'spherical', 'conv_0torque']:
    L, y, sh, mc = load_regime(REGIME)
    for name, feats in LAWS.items():
        X = dsg(L, feats)
        lo = pooled_loso(X, y, sh, mc, cap=150000); lm = lomo(X, y, mc, cap=300000)[0]
        print(f"{REGIME:<14}{name:<10}{lo:>11.3f}{lm:>8.3f}", flush=True)
    w, mu, sd = fit_logistic(dsg(L, PHYS5), y.astype(float), l2=2.0)
    exps = w[1:] / sd
    phys5_law[REGIME] = {f: float(e) for f, e in zip(PHYS5, exps)}
    print("   phys5 law: H/(1-H) ~ " + " * ".join(f"{f}^{e:+.2f}" for f, e in zip(PHYS5, exps)), flush=True)
print("\nphys5 exponents:", phys5_law, flush=True)
print("DONE", flush=True)
