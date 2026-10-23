#!/usr/bin/env python3
"""law_report.py -- CLEAN reporting of the onset law: exact EXPONENTS + real AUC for dom2, phys5,
and the full 21-term law, per regime (incl torque-free conv_0torque) and per machine. Fixes the
sparse JSONs (they stored regimes/forecast but buried the exponents). Uses the EXACT Paper-1
label_and_clean from run_onset.py. Prints the algebraic substitution of dom2 so we see the true
physical drivers.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
from onset_common import pooled_loso, lomo, machine_weights, fit_logistic, predict_logistic, roc_auc, coherent_extent, regime_filter

AGM = "/Users/henrycantor/mentorship2/ACTUALLY_GOOD_MAYBE"
BASE = ['accel', 'prate_frac', 'inv_etae', 'etae_surge', 'betae', 'gExB_hat', 'ge', 'TeTi', 'gExB',
        'load8', 'rstar', 'corrug', 'overdrive', 'gmc', 'prox', 'nustar', 'crit']
GEM = ['net_ratio', 'eff_drive', 'uni_drive', 'u_peak']
POWER = BASE + GEM
DOM2 = ['rstar', 'eff_drive']
PHYS5 = ['rstar', 'eff_drive', 'betae', 'gExB_hat', 'nustar']
THETA = 2.0
GLOBAL_C0 = '--global' in sys.argv
COLS = list(dict.fromkeys(['machine', 'shot', 'regime', 'q', 'time', 'rho', 'Q_cond_e', 'Q_GB',
                           'recon_resid_te', 'ring_te'] + POWER))


def label_and_clean(sub):
    r = np.clip(sub['Q_cond_e'].values / np.maximum(sub['Q_GB'].values, 1e-9), 0, 1e4)
    shot = pd.factorize(sub['shot'].values)[0]; mach = pd.factorize(sub['machine'].values)[0]
    c0 = np.full(len(sub), np.median(r)) if GLOBAL_C0 else pd.Series(r).groupby(mach).transform('median').values
    resid = r / np.maximum(THETA * c0, 1e-6) - 1.0
    ext = coherent_extent(resid > 0.4, shot, sub['time'].values, sub['rho'].values)
    y = (resid > 1.0) & (ext >= 4)
    rr, rg = sub['recon_resid_te'].values, sub['ring_te'].values
    clean = (rr <= np.nanquantile(rr, 0.75)) & (rg <= np.nanquantile(rg, 0.75))
    keep = clean & ((resid < 0.4) | y)
    return y, keep, shot, mach


def design(sub, idx, cols):
    return np.column_stack([np.log(np.maximum(np.abs(sub[c].values[idx].astype(float)), 1e-12)) for c in cols])


def exps(sub, ki, yk, mck, cols):
    X = design(sub, ki, cols)
    w, mu, sd = fit_logistic(X, yk.astype(float), l2=2.0, wts=machine_weights(mck))
    return {cols[j]: float(w[j + 1] / sd[j]) for j in range(len(cols))}, X


def main():
    print(f"ONSET LAW -- exact exponents + AUC  ({'GLOBAL c0' if GLOBAL_C0 else 'per-machine c0'})\n")
    allrep = {}
    for regime in ['conv_NBI', 'spherical', 'conv_0torque']:
        # memory-safe: sample ~90 shots per regime, read only those (keeps shot structure for LOSO)
        idx = pq.read_table(f"{AGM}/reduced_v2.parquet", columns=['regime', 'shot'],
                            filters=regime_filter(regime)).to_pandas()
        ushot = idx['shot'].unique()
        rng0 = np.random.default_rng(0)
        keep_shots = ushot if len(ushot) <= 90 else rng0.choice(ushot, 90, replace=False)
        sub = pq.read_table(f"{AGM}/reduced_v2.parquet", columns=COLS,
                            filters=regime_filter(regime)+[('shot', 'in', list(keep_shots))]).to_pandas()
        del idx
        if regime != 'conv_NBI':
            sub = sub[sub['q'].values > 1.5].reset_index(drop=True)
        y, keep, shot, mach = label_and_clean(sub)
        names = pd.factorize(sub['machine'].values)[1]
        ki = np.where(keep)[0]
        rng = np.random.default_rng(0)
        if len(ki) > 4_000_000:
            ki = np.sort(rng.choice(ki, 4_000_000, replace=False))
        yk, shk, mck = y[ki].astype(bool), shot[ki], mach[ki]
        nmach = len(np.unique(mck))
        print(f"================  {regime}  ({nmach} machines, {len(ki):,} cells, onset {yk.mean()*100:.1f}%)  ================")
        rep = {'onset_pct': float(yk.mean() * 100), 'machines': int(nmach)}
        for name, cols in [('dom2', DOM2), ('phys5', PHYS5), ('full21', POWER)]:
            ex, X = exps(sub, ki, yk, mck, cols)
            loso = pooled_loso(X, yk, shk, mck, cap=150000)
            lm = lomo(X, yk, mck, cap=250000, shot=shk, min_shots=4)[0]    # >=4-discharge transfer targets (drops only EAST's degenerate 3-shot NBI fold)
            rep[name] = {'exponents': ex, 'loso_within': float(loso), 'lomo_cross': float(lm)}
            top = sorted(ex.items(), key=lambda kv: -abs(kv[1]))[:6]
            print(f"  {name:<7} LOSO {loso:.3f}  LOMO {lm:.3f}   exps: " +
                  "  ".join(f"{k} {v:+.2f}" for k, v in (ex.items() if name != 'full21' else top)))
        # dom2 substitution: onset ~ rstar^a * eff_drive^b ; rstar~sqrt(Ti)/(B a) (contains q), eff_drive=inv_etae*(1-ffast)
        a, b = rep['dom2']['exponents']['rstar'], rep['dom2']['exponents']['eff_drive']
        print(f"  -> dom2: onset ~ rho*^({a:+.2f}) * eff_drive^({b:+.2f})")
        print(f"     rho* {'NEGATIVE (high rho* = LESS onset: SOC/micro-turb washout; machine-fixed via B,a)' if a<0 else 'POSITIVE'};"
              f" eff_drive {'POSITIVE (reduce it to prevent -- via ffast^ + R/L_ne flatten, at FIXED R/L_Te)' if b>0 else 'NEGATIVE'}.")
        print()
        allrep[regime] = rep
    json.dump(allrep, open(f"{AGM}/law_report_{'global' if GLOBAL_C0 else 'permachine'}.json", "w"), indent=2)
    print(f"wrote law_report_{'global' if GLOBAL_C0 else 'permachine'}.json")


if __name__ == '__main__':
    main()
