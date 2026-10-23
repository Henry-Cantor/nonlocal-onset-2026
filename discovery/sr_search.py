#!/usr/bin/env python3
"""sr_search.py -- SYMBOLIC REGRESSION for the onset law. Instead of committing to the
21-term power law or a hand-written equation, let genetic programming search the space
of closed-form expressions (products, ratios, powers, logs) over the physical variables
and keep the most predictive COMPACT form. Each discovered expression is then scored
with the same held-out LOSO/LOMO machinery as the power law, so the comparison is fair.

    python sr_search.py            # discover + validate per regime, vs power law + Theta
"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import onset_common from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from onset_common import pooled_loso, lomo, roc_auc, coherent_extent, regime_filter

# curated, physically-meaningful, non-redundant drivers (raw magnitudes)
CORE = ['accel', 'inv_etae', 'ge', 'crit', 'betae', 'gExB_hat', 'TeTi', 'nustar',
        'eff_drive', 'u_peak', 'rstar', 'prate_frac']
BASE = ['accel', 'prate_frac', 'inv_etae', 'etae_surge', 'betae', 'gExB_hat', 'ge',
        'TeTi', 'gExB', 'load8', 'rstar', 'corrug', 'overdrive', 'gmc', 'prox',
        'nustar', 'crit']
GEM = ['net_ratio', 'eff_drive', 'uni_drive', 'u_peak']
POWER = BASE + GEM
COLS = list(dict.fromkeys(['machine', 'shot', 'regime', 'q', 'time', 'rho', 'Q_cond_e',
                           'Q_GB', 'recon_resid_te', 'ring_te'] + POWER + CORE))


def _auc_fitness(y, y_pred, w):
    m = np.isfinite(y_pred)
    if m.sum() < 40:
        return 0.0
    a = roc_auc(y_pred[m], y[m].astype(bool))
    if not np.isfinite(a):
        return 0.5
    return max(a, 1.0 - a)                 # direction-agnostic: GP can build either sign


AUC = make_fitness(function=_auc_fitness, greater_is_better=True)


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
    if len(ki) > 4_000_000:
        ki = np.sort(rng.choice(ki, 4_000_000, replace=False))
    return sub, ki, y[ki].astype(bool), shot[ki], mach[ki]


def logcols(sub, ki, cols):
    return np.column_stack([np.log(np.maximum(np.abs(sub[c].values[ki].astype(float)), 1e-12)) for c in cols])


for REGIME in ['conv_NBI', 'spherical', 'conv_0torque']:
    sub, ki, y, sh, mc = load_regime(REGIME)
    # standardized log-features for the GP search (well-conditioned; form is what matters)
    Xcore = logcols(sub, ki, CORE)
    Xcore = (Xcore - Xcore.mean(0)) / np.maximum(Xcore.std(0), 1e-9)
    rng = np.random.default_rng(1)
    tr = rng.choice(len(y), min(120000, len(y)), replace=False)   # GP search subsample
    print(f"\n===== {REGIME}: {len(y):,} cells, onset={y.mean()*100:.1f}% =====", flush=True)
    est = SymbolicRegressor(
        population_size=1500, generations=18, tournament_size=25,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log'),
        metric=AUC, parsimony_coefficient=0.0015, const_range=(-3.0, 3.0),
        p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
        p_point_mutation=0.1, max_samples=0.8, feature_names=CORE,
        random_state=0, n_jobs=1, verbose=0)
    est.fit(Xcore[tr], y[tr].astype(float))
    prog = est._program
    sr_score = prog.execute(Xcore)                # the discovered law on all kept cells
    # score the discovered expression with the SAME held-out machinery as the power law
    Xsr = np.log(np.maximum(np.abs(sr_score), 1e-12))[:, None]
    sr_loso = pooled_loso(Xsr, y, sh, mc, cap=150000)
    sr_lomo = lomo(Xsr, y, mc, cap=300000)[0]
    # power-law reference (21 log-features)
    Xpw = logcols(sub, ki, POWER)
    pw_loso = pooled_loso(Xpw, y, sh, mc, cap=150000)
    pw_lomo = lomo(Xpw, y, mc, cap=300000)[0]
    print(f"  discovered law: {prog}", flush=True)
    print(f"  complexity (nodes)={prog.length_}", flush=True)
    print(f"  SR law        LOSO={sr_loso:.3f}  LOMO={sr_lomo:.3f}", flush=True)
    print(f"  power law(21) LOSO={pw_loso:.3f}  LOMO={pw_lomo:.3f}", flush=True)
print("\nDONE", flush=True)
