#!/usr/bin/env python3
"""Per-machine within-machine LOSO for spherical + conv_0torque, to see whether the
regime-mean is dragged by one weak/degenerate-label machine (fixable) or is broadly
soft. Also prints onset rate and shot count per machine."""
import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import onset_common from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from onset_common import pooled_loso, coherent_extent, regime_filter

BASE = ['accel', 'prate_frac', 'inv_etae', 'etae_surge', 'betae', 'gExB_hat', 'ge',
        'TeTi', 'gExB', 'load8', 'rstar', 'corrug', 'overdrive', 'gmc', 'prox',
        'nustar', 'crit']
GEM = ['net_ratio', 'eff_drive', 'uni_drive', 'u_peak']
POWER = BASE + GEM
COLS = list(dict.fromkeys(['machine', 'shot', 'regime', 'q', 'time', 'rho', 'Q_cond_e',
                           'Q_GB', 'recon_resid_te', 'ring_te'] + POWER))

for REG in ['spherical', 'conv_0torque']:
    sub = pq.read_table(os.path.join(ROOT_DIR, "reduced_v2.parquet"), columns=COLS, filters=regime_filter(REG)).to_pandas()
    sub = sub[sub['q'].values > 1.5].reset_index(drop=True)
    r = np.clip(sub['Q_cond_e'].values / np.maximum(sub['Q_GB'].values, 1e-9), 0, 1e4)
    shot = pd.factorize(sub['shot'].values)[0]
    mach = pd.factorize(sub['machine'].values)[0]
    mnames = pd.factorize(sub['machine'].values)[1]
    c0 = pd.Series(r).groupby(mach).transform('median').values
    resid = r / np.maximum(2.0 * c0, 1e-6) - 1.0
    ext = coherent_extent(resid > 0.4, shot, sub['time'].values, sub['rho'].values)
    y = (resid > 1.0) & (ext >= 4)
    rr, rg = sub['recon_resid_te'].values, sub['ring_te'].values
    keep = (rr <= np.nanquantile(rr, 0.75)) & (rg <= np.nanquantile(rg, 0.75)) & ((resid < 0.4) | y)
    ki = np.where(keep)[0]
    rng = np.random.default_rng(0)
    if len(ki) > 6_000_000:
        ki = np.sort(rng.choice(ki, 6_000_000, replace=False))
    yk = y[ki].astype(bool); shk = shot[ki]; mck = mach[ki]
    X = np.column_stack([np.log(np.maximum(np.abs(sub[c].values[ki].astype(float)), 1e-12)) for c in POWER])
    print(f"\n===== {REG} =====")
    print(f"  {'machine':<8}{'shots':>6}{'cells':>9}{'onset%':>8}{'LOSO_within':>13}")
    for mm in np.unique(mck):
        s = mck == mm
        ns = len(np.unique(shk[s]))
        a = np.nan
        if s.sum() >= 3000 and yk[s].sum() >= 50 and ns >= 4:
            a = pooled_loso(X[s], yk[s], shk[s], mck[s], cap=150000)
        aS = f"{a:.3f}" if np.isfinite(a) else "  n/a"
        print(f"  {str(mnames[mm]):<8}{ns:>6}{s.sum():>9}{yk[s].mean()*100:>7.1f}%{aS:>13}")
print("\nDONE", flush=True)
