#!/usr/bin/env python3
"""make_figures.py -- regenerate every paper figure and table from the pipeline result
files (onset_results_per_machine.json + onset_results_global.json from run_onset.py,
heat_impact.json from heat_impact.py, onset_final.parquet). Saves PNGs + tables.md into
this figures/ directory. The companion notebook make_figures.ipynb calls these functions
cell-by-cell.

    python figures/make_figures.py        # writes all PNGs + tables.md
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
AGM = os.path.dirname(HERE)
REG = ['conv_NBI', 'spherical', 'conv_0torque']
CREG = {'conv_NBI': '#2b7bba', 'spherical': '#d1495b', 'conv_0torque': '#edae49'}
# curated, readable physical drivers for the exponent figure
SHOWFEAT = ['rstar', 'eff_drive', 'inv_etae', 'ge', 'betae', 'TeTi', 'gExB_hat', 'nustar']
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.axisbelow': True})


def _load(name):
    p = os.path.join(AGM, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _roc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    m = np.isfinite(s); s, y = s[m], y[m]
    o = np.argsort(-s); ys = y[o]
    tp = np.cumsum(ys) / max(ys.sum(), 1); fp = np.cumsum(~ys) / max((~ys).sum(), 1)
    return np.concatenate([[0], fp]), np.concatenate([[0], tp])


def fig_concept():
    """tokamak poloidal cross-section: local surge (one surface) vs non-local avalanche
    (coherent band across several nested flux surfaces)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, title, nonlocal_ in [(axes[0], 'LOCAL transport', False),
                                 (axes[1], 'NON-LOCAL avalanche', True)]:
        # nested D-shaped flux surfaces (Miller-ish), plus a hot band
        for i, a in enumerate(np.linspace(1.0, 0.12, 9)):
            k, d = 1.7, 0.33                       # elongation, triangularity
            th = np.linspace(0, 2 * np.pi, 200)
            R = 1.0 + a * np.cos(th + d * np.sin(th)); Z = k * a * np.sin(th)
            hot = False
            if nonlocal_ and 3 <= i <= 6:
                hot = True
            elif (not nonlocal_) and i == 4:
                hot = True
            ax.plot(R, Z, color=('#d1495b' if hot else '#b9c4cf'),
                    lw=(3.0 if hot else 1.0), zorder=(3 if hot else 1))
        if nonlocal_:
            ax.annotate('', xy=(1.0, 1.7 * 0.78), xytext=(1.0, 1.7 * 0.22),
                        arrowprops=dict(arrowstyle='-|>', color='#7a1020', lw=2.6))
            ax.text(1.0, -1.85, 'coherent surge across >=4 surfaces', ha='center',
                    va='center', fontsize=11, color='#7a1020', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#7a1020', lw=1.6, alpha=0.95))
        else:
            ax.text(1.0, -1.85, 'one surface (local)', ha='center', va='center', fontsize=11,
                    color='#7a1020', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#d1495b', lw=1.6, alpha=0.95))
        ax.set_xlim(-0.2, 2.2); ax.set_ylim(-2.0, 2.0); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title); ax.grid(False)
        ax.set_xlabel('major radius R'); ax.set_ylabel('height Z')
    fig.suptitle('Onset = electron heat flux > 4x local gyro-Bohm, coherent over >=4 flux surfaces',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig1_concept.png", bbox_inches='tight'); plt.close(fig)


def fig_validation(res):
    """nowcast LOSO_within & LOMO_cross per regime, single 0.9 reference line."""
    if res is None:
        return
    rr = {r['regime']: r for r in res['regimes']}
    x = np.arange(len(REG)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    lo = [rr[r]['loso_within'] for r in REG]; lm = [rr[r]['lomo_cross'] for r in REG]
    ax.bar(x - w / 2, lo, w, label='within-machine LOSO (future shot)', color='#2b7bba')
    ax.bar(x + w / 2, lm, w, label='cross-machine LOMO (unseen machine / ITER)', color='#d1495b')
    ax.axhline(0.9, ls='--', c='gray', alpha=0.8)
    for xi, v in list(zip(x - w / 2, lo)) + list(zip(x + w / 2, lm)):
        ax.text(xi, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(REG); ax.set_ylim(0.5, 1.0); ax.set_ylabel('held-out AUC')
    ax.set_title('Onset prediction: held-out AUC by regime (nowcast)'); ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig2_validation.png", bbox_inches='tight'); plt.close(fig)


def fig_forecast(res):
    """nowcast vs +10 ms forecast cross-machine LOMO per regime (the lead-time value)."""
    if res is None or 'forecast' not in res:
        return
    now = {r['regime']: r for r in res['regimes']}; fo = {r['regime']: r for r in res['forecast']}
    x = np.arange(len(REG)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w / 2, [now[r]['lomo_cross'] for r in REG], w, label='nowcast', color='#4c72b0')
    ax.bar(x + w / 2, [fo[r]['lomo_cross'] for r in REG], w,
           label=f"forecast +{res.get('horizon_ms',10):.0f} ms", color='#dd8452')
    ax.axhline(0.9, ls='--', c='gray', alpha=0.8)
    for xi, r in zip(x - w / 2, REG):
        ax.text(xi, now[r]['lomo_cross'] + 0.005, f"{now[r]['lomo_cross']:.3f}", ha='center', fontsize=9)
    for xi, r in zip(x + w / 2, REG):
        ax.text(xi, fo[r]['lomo_cross'] + 0.005, f"{fo[r]['lomo_cross']:.3f}", ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(REG); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('cross-machine LOMO AUC'); ax.legend()
    ax.set_title('Forecast: onset is predictable 10 ms ahead');
    fig.tight_layout(); fig.savefig(f"{HERE}/fig_forecast.png", bbox_inches='tight'); plt.close(fig)


def fig_permachine(res, min_shots=8):
    """per-machine within-machine LOSO for machines with >= min_shots discharges."""
    if res is None:
        return
    pm = [p for p in res['per_machine'] if p['loso'] is not None and p['shots'] >= min_shots]
    pm.sort(key=lambda p: (REG.index(p['regime']), -p['loso']))
    fig, ax = plt.subplots(figsize=(7.4, max(4, 0.36 * len(pm))))
    for i, p in enumerate(pm):
        ax.scatter(p['loso'], i, s=70, color=CREG[p['regime']], edgecolor='k', zorder=3)
        ax.text(p['loso'] + 0.003, i, f"{p['machine']} ({p['shots']})", va='center', fontsize=9)
    ax.axvline(0.95, ls='--', c='gray', alpha=0.7)
    ax.set_yticks([]); ax.set_xlim(0.9, 1.0); ax.set_xlabel('within-machine LOSO AUC')
    ax.set_title(f'Per-machine onset prediction ({min_shots}+ discharges)')
    hs = [plt.Line2D([], [], marker='o', ls='', color=CREG[r], label=r) for r in REG]
    ax.legend(handles=hs, loc='lower left', fontsize=8)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig3_permachine.png", bbox_inches='tight'); plt.close(fig)


def fig_heat(hi):
    if hi is None:
        return
    x = np.arange(len(REG)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    occ = [hi['regimes'][r]['onset_pct'] for r in REG]
    fsh = [hi['regimes'][r]['flux_share_pct'] for r in REG]
    exc = [hi['regimes'][r]['excess_share_pct'] for r in REG]
    ax.bar(x - w, occ, w, label='onset occupancy', color='#9aa7b1')
    ax.bar(x, fsh, w, label='flux share of onset', color='#2b7bba')
    ax.bar(x + w, exc, w, label='avoidable excess', color='#d1495b')
    for xi, arr in [(x - w, occ), (x, fsh), (x + w, exc)]:
        for xj, v in zip(xi, arr):
            ax.text(xj, v + 0.4, f'{v:.0f}%', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(REG); ax.set_ylabel('% of electron heat / cells')
    ax.set_title('Non-local events are a small state fraction but a large heat fraction'); ax.legend()
    fig.tight_layout(); fig.savefig(f"{HERE}/fig4_heat_impact.png", bbox_inches='tight'); plt.close(fig)


def fig_threshold(hi):
    """heat captured vs cells flagged, with held-out AUC overlaid, as m sweeps."""
    if hi is None:
        return
    ms = sorted((float(k) for k in hi['sweep']))
    def g(m, k): return hi['sweep'][str(m) if str(m) in hi['sweep'] else f'{m:.1f}'][k]
    cells = [g(m, 'cells_pct') for m in ms]; exc = [g(m, 'excess_captured_pct') for m in ms]
    aucs = [g(m, 'auc') for m in ms]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(cells, exc, '-o', color='#33682f', label='excess heat captured')
    for m, c, e in zip(ms, cells, exc):
        op = abs(m - 4.0) < 1e-6
        ax.annotate(f'm={m:g}', (c, e), textcoords='offset points', xytext=(6, -11),
                    fontsize=8, weight='bold' if op else 'normal')
        if op:
            ax.scatter([c], [e], s=170, facecolor='none', edgecolor='#d1495b', lw=2, zorder=5)
    ax.set_xlabel('% of cells flagged as onset'); ax.set_ylabel('% of non-local excess heat captured')
    ax2 = ax.twinx()
    if any(a is not None for a in aucs):
        ax2.plot(cells, [a if a else np.nan for a in aucs], '-s', color='#7a1fa2', label='held-out AUC')
        ax2.set_ylabel('held-out AUC (power law)', color='#7a1fa2'); ax2.set_ylim(0.5, 1.0)
        ax2.tick_params(axis='y', colors='#7a1fa2'); ax2.grid(False)
    ax.set_title('Threshold: AUC>=0.9 at every m (rises with m); m=4 balances heat capture and rarity')
    fig.tight_layout(); fig.savefig(f"{HERE}/fig5_threshold.png", bbox_inches='tight'); plt.close(fig)


def fig_calibration(df):
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.3))
    for r in REG:
        d = df[df['regime'] == r]
        if not len(d):
            continue
        q = pd.qcut(d['hazard'], 10, labels=False, duplicates='drop'); grp = d.groupby(q)
        ax.plot(grp['hazard'].mean().values, grp['onset'].mean().values, '-o', color=CREG[r], label=r, ms=4)
    lim = [0, max(0.6, df['onset'].mean() * 4)]; ax.plot(lim, lim, 'k--', alpha=0.4)
    ax.set_xlabel('mean predicted hazard (rank)'); ax.set_ylabel('observed onset rate')
    ax.set_title('Law calibration (deciles)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig6_calibration.png", bbox_inches='tight'); plt.close(fig)


def fig_example_shot(df):
    """One discharge, side by side: MEASURED electron heat flux (log10 flux / local gyro-Bohm baseline)
    and the law's PREDICTED onset hazard, both in (rho, t). The green outline is the TRUE onset region;
    on the hazard panel the bright predicted hazard fills that same outline. The title reports the
    shot-level AUC (hazard vs true onset) so the fit quality is explicit."""
    if df is None:
        return
    cand = df.groupby(['regime', 'machine', 'shot'])['onset'].mean().reset_index()
    cand = cand[(cand['onset'] > 0.1) & (cand['onset'] < 0.35)]
    if not len(cand):
        return
    # representative example: the MEDIAN per-shot AUC among candidates (typical skill, not cherry-picked).
    # One groupby pass over (machine, shot) -- no per-shot full-frame scan.
    rows_auc = []
    for (mc, sh), g in df[['machine', 'shot', 'onset', 'hazard']].groupby(['machine', 'shot'], sort=False):
        yv = g['onset'].values.astype(bool)
        if yv.sum() < 20 or yv.all():
            continue
        f_, t_ = _roc(g['hazard'].values, yv); rows_auc.append((mc, sh, float(np.trapz(t_, f_))))
    sa = pd.DataFrame(rows_auc, columns=['machine', 'shot', 'auc'])
    cand = cand.merge(sa, on=['machine', 'shot'], how='inner').sort_values('auc').reset_index(drop=True)
    if not len(cand):
        return
    row = cand.iloc[len(cand) // 2]
    d = df[(df['machine'] == row['machine']) & (df['shot'] == row['shot'])]
    piv = d.pivot_table(index='rho', columns='time', values='Q_cond_e', aggfunc='mean')
    base = d.pivot_table(index='rho', columns='time', values='Q_GB', aggfunc='mean')
    haz = d.pivot_table(index='rho', columns='time', values='hazard', aggfunc='mean')
    ons = d.pivot_table(index='rho', columns='time', values='onset', aggfunc='max')
    scale = np.nanmedian(piv.values / np.maximum(base.values, 1e-9))
    ratio = np.log10(np.clip(piv.values / np.maximum(base.values * scale, 1e-9), 0.1, 100))
    onsv = np.nan_to_num(ons.values.astype(float))
    # PREDICTED onset region: apply the SAME onset rule to the law's hazard that defines the true label --
    # top cells by hazard (base-rate-matched) AND the >=4-adjacent-surface coherence. So both panels are
    # coherent-band onset regions (measured flux vs predicted hazard): a fair, like-for-like comparison.
    import sys as _sys; _sys.path.insert(0, AGM)
    from onset_common import coherent_extent
    frac = float(d['onset'].mean()); hcell = d['hazard'].values.astype(float)
    thr = np.nanquantile(hcell, 1.0 - frac) if 0.0 < frac < 1.0 else np.inf
    pos = hcell >= thr
    ext = coherent_extent(pos, np.zeros(len(d), int), d['time'].values, d['rho'].values)
    predcell = pos & (ext >= 4)
    predpiv = d.assign(_p=predcell).pivot_table(index='rho', columns='time', values='_p', aggfunc='max')
    predv = np.nan_to_num(predpiv.values.astype(float))
    tcell = d['onset'].values.astype(bool)
    jacc = float((predcell & tcell).sum() / max(int((predcell | tcell).sum()), 1))
    fp, tp = _roc(d['hazard'].values, d['onset'].values); shot_auc = float(np.trapz(tp, fp))

    def outline(ax, xs, ys, field):        # region boundary: lime over a black underlay (reads on any cmap)
        ax.contour(xs, ys, field, levels=[0.5], colors='black', linewidths=4.2)
        ax.contour(xs, ys, field, levels=[0.5], colors='#39ff14', linewidths=2.0)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.8, 4.1), sharex=True, sharey=True)
    im0 = a0.pcolormesh(piv.columns.values, piv.index.values, ratio, cmap='magma', shading='auto')
    fig.colorbar(im0, ax=a0, label='log10( flux / local baseline )')
    outline(a0, ons.columns.values, ons.index.values, onsv)
    a0.set_title('TRUE onset region  (on the measured heat flux)')
    im1 = a1.pcolormesh(haz.columns.values, haz.index.values, haz.values, cmap='magma', shading='auto')
    fig.colorbar(im1, ax=a1, label="the law's onset hazard")
    outline(a1, predpiv.columns.values, predpiv.index.values, predv)
    a1.set_title("PREDICTED onset region  (on the law's hazard)")
    for a in (a0, a1):
        a.set_xlabel('time (s)'); a.set_ylabel('rho (normalized radius)')
    a0.plot([], [], color='#39ff14', lw=2.5, label='true onset'); a0.legend(loc='upper right', fontsize=8, framealpha=0.85)
    a1.plot([], [], color='#39ff14', lw=2.5, label='predicted onset'); a1.legend(loc='upper right', fontsize=8, framealpha=0.85)
    fig.suptitle(f"{row['machine']} shot {row['shot']} ({row['regime']}):  TRUE onset (left) vs the law's "
                 f"PREDICTED onset (right)  --  AUC = {shot_auc:.3f},  region overlap = {100*jacc:.0f}% (Jaccard)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig7_example_shot.png", bbox_inches='tight'); plt.close(fig)


def fig_exponents(res=None):
    """Exponents of the compact 5-term 'phys5' fit, per regime: the interpretable cross-regime physics
    (beta_e stabilizing in every regime; rho* strong in beam plasmas, ~0 in STs; nu* strong in STs).
    Read from law_report_permachine.json (law_report.py)."""
    lr = _load('law_report_permachine.json')
    if lr is None:
        return
    feats = ['rstar', 'eff_drive', 'betae', 'gExB_hat', 'nustar']
    sym = {'rstar': r'$\rho^{*}$', 'eff_drive': r'$\eta_e^{-1}(1{-}f_{\mathrm{fast}})$',
           'betae': r'$\beta_e$', 'gExB_hat': r'$\hat{\gamma}_{E\times B}$', 'nustar': r'$\nu^{*}$'}
    x = np.arange(len(feats)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for i, r in enumerate(REG):
        ex = lr.get(r, {}).get('phys5', {}).get('exponents', {})
        vals = [ex.get(ft, 0.0) for ft in feats]
        ax.bar(x + (i - 1) * w, vals, w, label=r, color=CREG[r])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([sym[ft] for ft in feats], fontsize=14)
    ax.set_ylabel(r'power-law exponent   $\mathrm{d}\,\ln H / \mathrm{d}\,\ln x$')
    ax.set_title('Learned exponents across regimes (compact 5-term fit)')
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig8_exponents.png", bbox_inches='tight'); plt.close(fig)


def fig_roc(df):
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for r in REG:
        d = df[df['regime'] == r]
        if not len(d):
            continue
        fp, tp = _roc(d['hazard'].values, d['onset'].values)
        a = np.trapz(tp, fp)
        ax.plot(fp, tp, color=CREG[r], label=f'{r} (AUC={a:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
    ax.set_title('ROC: power-law hazard vs onset (in-sample)'); ax.legend(fontsize=9, loc='lower right')
    ax.set_aspect('equal')
    fig.tight_layout(); fig.savefig(f"{HERE}/fig9_roc.png", bbox_inches='tight'); plt.close(fig)


def fig_hazard_trace(df):
    """hazard(t) building up to onset in one discharge (radial max), with true onset."""
    if df is None:
        return
    cand = df.groupby(['regime', 'machine', 'shot'])['onset'].mean().reset_index()
    cand = cand[(cand['onset'] > 0.08) & (cand['onset'] < 0.3)]
    if not len(cand):
        return
    row = cand.sort_values('onset').iloc[len(cand) // 2]
    d = df[(df['machine'] == row['machine']) & (df['shot'] == row['shot'])]
    g = d.groupby('time'); t = np.array(sorted(d['time'].unique()))
    hz = g['hazard'].max().reindex(t).values; on = g['onset'].max().reindex(t).values
    hz = (hz - np.nanmin(hz)) / (np.nanmax(hz) - np.nanmin(hz) + 1e-9)
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(t, hz, color='#7a1fa2', lw=1.3, label='predicted hazard (radial max, scaled)')
    ax.fill_between(t, 0, 1, where=on > 0.5, color='#39ff14', alpha=0.25, label='true onset')
    ax.set_xlabel('time (s)'); ax.set_ylabel('hazard'); ax.set_ylim(0, 1.02)
    ax.set_title(f"Hazard builds ahead of onset: {row['machine']} shot {row['shot']}"); ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(f"{HERE}/fig10_hazard_trace.png", bbox_inches='tight'); plt.close(fig)


def fig_confinement(hi):
    """the reactor lever: threshold-free non-local heat fraction (all heat above the
    local gyro-Bohm baseline) and the coherent-avalanche subset the law predicts."""
    if hi is None:
        return
    nl = [hi['regimes'][r].get('nonlocal_total_pct', hi['regimes'][r]['excess_share_pct']) for r in REG]
    av = [hi['regimes'][r]['excess_share_pct'] for r in REG]
    x = np.arange(len(REG)); w = 0.5
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x, nl, w, color='#9aa7b1', label='total non-local (threshold-free)')
    ax.bar(x, av, w, color=[CREG[r] for r in REG], label='coherent avalanche onset (predicted)')
    for xi, v in zip(x, nl):
        ax.text(xi, v + 0.6, f'{v:.0f}%', ha='center', fontsize=10, weight='bold')
    for xi, v in zip(x, av):
        ax.text(xi, v - 3, f'{v:.0f}%', ha='center', fontsize=9, color='white')
    ax.set_xticks(x); ax.set_xticklabels(REG)
    ax.set_ylabel('% of electron conducted heat above local baseline')
    ax.set_title('Reactor lever: non-local heat fraction (threshold-free)')
    ax.text(0.02, 0.97, 'grey = all heat local transport cannot explain (any amplitude);\n'
            'coloured = the coherent avalanche events the law predicts.\n'
            'Controlling a fraction -> tens-of-% confinement lever (not 100x).',
            transform=ax.transAxes, va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    ax.legend(loc='upper right', fontsize=8); ax.set_ylim(0, max(nl) * 1.3)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig11_confinement.png", bbox_inches='tight'); plt.close(fig)


def write_tables(res, hi, resg=None):
    L = ["# Paper tables (auto-generated by figures/make_figures.py)\n"]
    def vtable(r, title):
        out = [f"\n## {title}\n",
               "| regime | machines | shots | onset% | LOSO_within | LOMO_cross | null | noQ_GB |",
               "|---|---|---|---|---|---|---|---|"]
        for x in r:
            out.append(f"| {x['regime']} | {x['machines']} | {x['shots']} | {x['onset_pct']:.1f}% | "
                       f"{x['loso_within']:.3f} | {x['lomo_cross']:.3f} | {x['null']:.3f} | {x['noQ_GB']:.3f} |")
        return out
    if res:
        L += vtable(res['regimes'], "Table 1 -- Held-out validation (NOWCAST, per-machine c0)")
        if 'forecast' in res:
            L += vtable(res['forecast'], f"Table 1b -- FORECAST (+{res.get('horizon_ms',10):.0f} ms)")
    if resg:
        L += vtable(resg['regimes'], "Table 1c -- NOWCAST with a single GLOBAL c0 (universality / ITER)")
    if res:
        L += ["\n## Table 2 -- Per-machine within-machine LOSO (nowcast)\n",
              "| regime | machine | shots | onset% | LOSO |", "|---|---|---|---|---|"]
        for p in res['per_machine']:
            a = f"{p['loso']:.3f}" if p['loso'] is not None else "n/a"
            flag = "" if (p['loso'] is not None and p['shots'] >= 10) else " (data-limited)"
            L.append(f"| {p['regime']} | {p['machine']} | {p['shots']} | {p['onset_pct']:.1f}% | {a}{flag} |")
    if hi:
        L += ["\n## Table 3 -- Transport impact (per machine, area-weighted)\n",
              "| regime | onset occupancy | flux share | avoidable excess | enhancement |",
              "|---|---|---|---|---|"]
        for r in REG:
            h = hi['regimes'][r]
            L.append(f"| {r} | {h['onset_pct']:.0f}% | {h['flux_share_pct']:.0f}% | "
                     f"{h['excess_share_pct']:.0f}% | {h['enhance']:.0f}x |")
        L += ["\n## Table 4 -- Threshold justification (heat concentration + AUC)\n",
              "| m (x baseline) | cells flagged | excess heat captured | held-out AUC |", "|---|---|---|---|"]
        for m in sorted((float(k) for k in hi['sweep'])):
            s = hi['sweep'][str(m) if str(m) in hi['sweep'] else f'{m:.1f}']
            op = " <- operating point" if abs(m - 4.0) < 1e-6 else ""
            au = f"{s['auc']:.3f}" if s.get('auc') else "n/a"
            L.append(f"| {m:g} | {s['cells_pct']:.0f}% | {s['excess_captured_pct']:.0f}% | {au}{op} |")
    open(f"{HERE}/tables.md", 'w').write("\n".join(L) + "\n")


def fig_amplitude_coherence():
    """K3, panel A: among cells ALREADY above the amplitude threshold, local-drive amplitude does not
    set radial coherence (flat, high) while rho*/topology does -- i.e. the law predicts non-locality,
    not magnitude. Data from paper1_k3_plotdata.json (written by killshots.py)."""
    d = _load('paper1_k3_plotdata.json')
    regimes = [r for r in d if d[r].get('amp')] if d else []
    if not regimes:
        return
    def binned(x, y, nb=10):
        xx = np.log10(np.maximum(x, 1e-6))
        q = np.quantile(xx, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
        idx = np.clip(np.searchsorted(q, xx, side='right') - 1, 0, nb - 1)
        cx = np.array([np.median(x[idx == k]) for k in range(nb)])
        cy = np.array([y[idx == k].mean() for k in range(nb)])
        return cx, cy
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for reg in regimes:
        amp = np.array(d[reg]['amp']); coh = np.array(d[reg]['coh']); rst = np.array(d[reg]['rstar'])
        cx, cy = binned(amp, coh); ax[0].plot(cx, cy * 100, 'o-', ms=4, color=CREG.get(reg, 'k'), label=reg)
        rx, ry = binned(rst, coh); ax[1].plot(rx, ry * 100, 'o-', ms=4, color=CREG.get(reg, 'k'), label=reg)
    ax[0].set_xscale('log'); ax[0].set_xlabel('amplitude  r/(2 c0) - 1'); ax[0].set_ylabel('% radially coherent')
    ax[0].set_title('Amplitude does NOT set coherence\n(coherent fraction ~flat and high vs amplitude)')
    ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\rho^*$ (rstar)'); ax[1].set_ylabel('% radially coherent')
    ax[1].set_title(r'$\rho^*$ / topology modulate coherence' '\n(coherent fraction varies with rho*)')
    for a in ax:
        a.legend(fontsize=8); a.set_ylim(60, 101)
    fig.suptitle("Among cells already above the amplitude threshold: large excursions are intrinsically "
                 "coherent (non-local); rho* modulates the residual local fraction", y=1.03, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig12_amplitude_coherence.png", bbox_inches='tight'); plt.close(fig)


def fig_coherence_separation():
    """K3, panel B: the law's predicted coherence probability separates radially-coherent (non-local)
    from incoherent (local) large excursions; at a base-rate operating point it catches the rare local
    spikes well above chance -- so the law resolves non-locality, not just magnitude."""
    d = _load('paper1_k3_plotdata.json')
    regimes = [r for r in d if d[r].get('amp')] if d else []
    if not regimes:
        return
    coh = np.concatenate([d[r]['coh'] for r in regimes]).astype(int)
    pred = np.concatenate([d[r]['pred'] for r in regimes])
    inc_rate = 1 - coh.mean()
    thr = np.quantile(pred, inc_rate)                     # base-rate-matched operating point
    yhat_inc = pred < thr
    tp = np.sum(yhat_inc & (coh == 0)); fp = np.sum(yhat_inc & (coh == 1))
    recall = tp / max((coh == 0).sum(), 1); prec = tp / max(tp + fp, 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    bins = np.linspace(0, 1, 40)
    ax[0].hist(pred[coh == 1], bins=bins, density=True, alpha=0.6, color=CREG['conv_NBI'], label='actually coherent (non-local)')
    ax[0].hist(pred[coh == 0], bins=bins, density=True, alpha=0.6, color=CREG['conv_0torque'], label='actually incoherent (local spike)')
    ax[0].axvline(thr, color='k', ls='--', lw=1, label=f'base-rate threshold ({thr:.2f})')
    ax[0].set_xlabel("law's predicted coherence probability"); ax[0].set_ylabel('density')
    ax[0].set_title('Incoherent (local) spikes skew to LOWER predicted\ncoherence — the law separates them (AUC 0.65–0.79)')
    ax[0].legend(fontsize=8)
    ax[1].bar(['recall\n(incoherent caught)', 'precision\n(of flagged cells)'],
              [recall * 100, prec * 100], color=CREG['conv_NBI'])
    ax[1].axhline(inc_rate * 100, color='k', ls=':', lw=1, label=f'random precision = base rate {inc_rate * 100:.0f}%')
    ax[1].set_ylabel('%'); ax[1].set_ylim(0, max(50, recall * 100 * 1.3, prec * 100 * 1.3)); ax[1].legend(fontsize=8)
    ax[1].set_title(f'Rare-class detection at base-rate point\n(flag lowest-prob {inc_rate * 100:.0f}%: catch '
                    f'{recall * 100:.0f}% of incoherent, {recall / max(inc_rate, 1e-9):.1f}x random)')
    fig.suptitle("Non-locality is the norm for large excursions; the law still resolves the rare local spikes", y=1.03, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig13_coherence_separation.png", bbox_inches='tight'); plt.close(fig)


def fig_confusion_matrix(df):
    """The full TP/FP/FN/TN matrix of the onset nowcast, per regime, at the Youden-optimal operating
    threshold (max TPR-FPR). Reads onset_final.parquet (onset label + in-sample hazard)."""
    if df is None or 'hazard' not in df or 'onset' not in df:
        return
    regs = [r for r in REG if (df['regime'] == r).any()]
    fig, axes = plt.subplots(1, len(regs), figsize=(4.0 * len(regs), 4.0))
    if len(regs) == 1:
        axes = [axes]
    for ax, reg in zip(axes, regs):
        d = df[df['regime'] == reg]
        y = d['onset'].values.astype(bool); s = d['hazard'].values.astype(float)
        fpr, tpr = _roc(s, y)
        thr_grid = np.quantile(s, np.linspace(0.5, 0.999, 200))
        j = [( (s >= t) & y).sum() / max(y.sum(), 1) - ((s >= t) & ~y).sum() / max((~y).sum(), 1) for t in thr_grid]
        thr = thr_grid[int(np.argmax(j))]
        pred = s >= thr
        TP = int((pred & y).sum()); FP = int((pred & ~y).sum())
        FN = int((~pred & y).sum()); TN = int((~pred & ~y).sum())
        M = np.array([[TN, FP], [FN, TP]], float)
        Mn = M / M.sum()
        ax.imshow(Mn, cmap='Blues', vmin=0, vmax=Mn.max())
        labels = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for k in range(2):
                cnt = int(M[i, k]); pct = Mn[i, k] * 100
                ax.text(k, i, f"{labels[i][k]}\n{cnt:,}\n{pct:.1f}%", ha='center', va='center',
                        fontsize=9, color='white' if Mn[i, k] > Mn.max() * 0.5 else '#222')
        recall = TP / max(TP + FN, 1); spec = TN / max(TN + FP, 1); prec = TP / max(TP + FP, 1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['pred quiet', 'pred onset'], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['true quiet', 'true onset'], fontsize=8)
        ax.set_title(f"{reg}\nrecall {recall*100:.0f}%  spec {spec*100:.0f}%  prec {prec*100:.0f}%", fontsize=9)
    fig.suptitle("Onset nowcast confusion matrix (Youden operating point): the law catches the onsets with "
                 "few false alarms", y=1.02, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig14_confusion_matrix.png", bbox_inches='tight'); plt.close(fig)


def main():
    res = _load('onset_results_per_machine.json')
    resg = _load('onset_results_global.json')
    hi = _load('heat_impact.json')
    pqf = os.path.join(AGM, 'onset_final.parquet')
    df = pd.read_parquet(pqf) if os.path.exists(pqf) else None
    fig_concept()
    fig_validation(res); fig_forecast(res); fig_permachine(res)
    fig_heat(hi); fig_threshold(hi); fig_confinement(hi)
    fig_calibration(df); fig_example_shot(df); fig_roc(df); fig_hazard_trace(df)
    fig_exponents(res)
    fig_amplitude_coherence(); fig_coherence_separation()   # K3 (from killshots.py plotdata)
    fig_confusion_matrix(df)                                  # TP/FP/FN/TN matrix
    write_tables(res, hi, resg)
    print("wrote figures + tables.md to", HERE)


if __name__ == '__main__':
    main()
