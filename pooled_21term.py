"""Does a SINGLE 21-term universal law (one set of coefficients across ALL regimes + machines) work,
or does the collinear full law saturate? Fit the pooled 21-term (L2-regularized), report pooled LOSO
(leave-shot-out) and per-machine LOMO (leave-one-machine-out = the ITER transfer test), and check for
saturation (predictions pinned at 0/1). Compare to the pooled 2-term dom2 (rho*, eff_drive).
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, '/Users/henrycantor/mentorship2/ACTUALLY_GOOD_MAYBE')
import run_onset as R
from onset_common import fit_logistic, predict_logistic, roc_auc, pooled_loso, machine_weights, regime_filter

AGM = R.AGM


def lomo_folds(X, y, mach, names, l2, cap=300000, seed=0):
    """per-machine leave-one-out AUC + the saturation rate of the held-out predictions."""
    rng = np.random.default_rng(seed); rows = []
    for m in np.unique(mach):
        te = mach == m; tr = np.where(mach != m)[0]
        if te.sum() < 500 or not (0 < y[te].mean() < 1) or not (0 < y[tr].mean() < 1):
            continue
        samp = rng.choice(tr, min(cap, len(tr)), replace=False)
        w, mu, sd = fit_logistic(X[samp], y[samp].astype(float), l2=l2, wts=machine_weights(mach[samp]))
        p = predict_logistic(X[te], w, mu, sd)
        sat = float(np.mean((p < 1e-3) | (p > 1 - 1e-3)))          # fraction of pinned (saturated) predictions
        rows.append((str(names[m]), int(te.sum()), float(roc_auc(p, y[te])), sat))
    return rows


def main():
    Xs, Xis, ys, shots, machs = [], [], [], [], []
    for regime in ['conv_NBI', 'spherical', 'conv_0torque']:
        ush = pq.read_table(f'{AGM}/reduced_v2.parquet', columns=['regime', 'shot'],
                            filters=regime_filter(regime)).to_pandas()['shot'].unique()
        ks = ush if len(ush) <= 80 else np.random.default_rng(0).choice(ush, 80, replace=False)
        sub = pq.read_table(f'{AGM}/reduced_v2.parquet', columns=R.COLS,
                            filters=regime_filter(regime)+[('shot', 'in', list(ks))]).to_pandas()
        if regime != 'conv_NBI':
            sub = sub[sub['q'].values > 1.5].reset_index(drop=True)
        y, keep, shot, mach = R.label_and_clean(sub)
        ki = np.where(keep)[0]
        if len(ki) > 2_000_000:
            ki = np.sort(np.random.default_rng(0).choice(ki, 2_000_000, replace=False))
        Xs.append(R.design(sub, ki, R.POWER)); Xis.append(R.design(sub, ki, ['rstar', 'eff_drive']))
        ys.append(y[ki]); shots.append(sub['shot'].values[ki]); machs.append(sub['machine'].values[ki])
        print(f"  read {regime}: {len(ki):,} cells", flush=True)
    X = np.vstack(Xs); Xd = np.vstack(Xis); y = np.concatenate(ys).astype(bool)
    shot = pd.factorize(np.concatenate(shots))[0]
    mnames = np.concatenate(machs); mach = pd.factorize(mnames)[0]; names = pd.factorize(mnames)[1]
    print(f"\nPOOLED all-regime: {len(y):,} cells, {len(np.unique(mach))} machines, onset {y.mean()*100:.1f}%\n")

    out = {'n_cells': int(len(y)), 'n_machines': int(len(np.unique(mach))), 'onset_pct': float(y.mean()*100),
           'laws': {}}
    for tag, key, XX, l2 in [('21-term, L2=2 (standard)', 'pooled_21term_l2_2', X, 2.0),
                             ('21-term, L2=20 (heavy reg)', 'pooled_21term_l2_20', X, 20.0),
                             ('2-term dom2 (rho*,eff_drive)', 'pooled_dom2', Xd, 2.0)]:
        loso = pooled_loso(XX, y, shot, mach, cap=200000)
        folds = lomo_folds(XX, y, mach, names, l2=l2)
        aucs = [a for _, _, a, _ in folds]; sats = [s for _, _, _, s in folds]
        print(f"{tag}:")
        print(f"  pooled LOSO (leave-shot-out) = {loso:.3f}")
        print(f"  LOMO (leave-machine-out) mean = {np.mean(aucs):.3f}  spread [{min(aucs):.3f}-{max(aucs):.3f}]  "
              f"over {len(aucs)} machines; mean saturation {np.mean(sats)*100:.0f}%")
        bad = [f"{n}:{a:.2f}(sat{ss*100:.0f}%)" for n, _, a, ss in folds if a < 0.7 or ss > 0.2]
        if bad:
            print(f"  weak/saturating folds: {', '.join(bad)}")
        print()
        out['laws'][key] = {'label': tag, 'pooled_loso': float(loso),
                            'lomo_mean': float(np.mean(aucs)), 'lomo_min': float(min(aucs)),
                            'lomo_max': float(max(aucs)), 'lomo_saturation_mean': float(np.mean(sats)),
                            'per_machine': [{'machine': n, 'n': nn, 'lomo_auc': a, 'saturation': ss}
                                            for n, nn, a, ss in folds]}
    json.dump(out, open(f"{AGM}/pooled_21term_results.json", "w"), indent=2)
    print("wrote pooled_21term_results.json")


if __name__ == '__main__':
    main()
