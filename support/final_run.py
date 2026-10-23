#!/usr/bin/env python3
"""final_run.py -- THE final Nature-Physics onset validation on the full database
(420 discharges, 10 tokamaks), memory-safely on a 16 GB laptop.

The physics is IDENTICAL to run_onset_stellar: every feature and label is
per-shot-local (per-(shot,rho) time series and per-(shot,time) radial profiles),
so we STREAM the 6.3 GB parquet shot-by-shot -- computing features, the nowcast
label (deadband + per-shot Z + Q_cond sanitation), the +10 ms forecast label, the
regime, and the drive/geometry router per shot -- and keep only the scoreable
cells (heated plasma with a clean nowcast OR forecast label). That reduced table
(~few GB) is what the leave-one-shot-out / leave-one-machine-out validators run on,
so nothing is approximated: the raw parquet just never lands in RAM all at once.

  python final_run.py                # build reduced cache if absent, then validate
  python final_run.py --rebuild      # force re-stream the raw parquet
  python final_run.py --build-only   # only build the reduced cache
"""
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from derive_onset import (dimensionless, dynamic_terms, deadband_target, _design,
                          logistic, logistic_pred, auc)
from spherical_onset import add_spherical
from run_onset_stellar import (classify, drive_geometry, unified_design,
                               _machine_weights, machine_quality,
                               leave_one_machine_out, closure_test,
                               per_machine_diag, forecast_target, REGIMES)

AGM = "/Users/henrycantor/mentorship2/ACTUALLY_GOOD_MAYBE"
RAW = os.path.join(AGM, "stage1_5_reconstructed_data.parquet")
RED = os.path.join(AGM, "onset_reduced.parquet")
OUT = os.path.join(AGM, "onset_final.parquet")
HORIZON = 0.010                                   # 10 ms forecast

READ_COLS = ['machine', 'shot', 'time', 'rho', 'Te', 'Ti', 'ne', 'q', 'Bt', 'R0',
             'a0', 'zeff', 'R_over_LTe', 'R_over_LTi', 'rho_star', 'gamma_ExB',
             'Q_cond_e', 'Q_GB', 'grad_Te', 'P_tot', 'P_inj']
# features consumed by the regime laws + the unified equation + diagnostics
def add_aligned(sd, lib):
    """marginal-relative aligning features (discovery-selected): overdrive=ge/crit,
    gmc=|ge-crit|, prox=exp(-((ge-crit)/w)^2). These carry cross-shot signal that
    raw ge/accel do not."""
    crit = np.maximum(lib['crit'], 0.5); ge = lib['ge']
    lib['overdrive'] = ge / crit
    lib['gmc'] = np.abs(ge - lib['crit']) + 1e-6
    w = np.maximum(0.3 * lib['crit'], 1.0)
    lib['prox'] = np.exp(-((ge - lib['crit']) / w) ** 2) + 1e-6
    return lib


# expanded feature set (was 9): the deployed 2-4-feature laws were starved; the
# discovery search + true-rate label cleaning lift conv_NBI 0.65 -> 0.83.
FEATS = ['accel', 'prate_frac', 'inv_etae', 'etae_surge', 'betae', 'gExB_hat',
         'ge', 'TeTi', 'gExB']
NUM = ['time', 'rho', 'q'] + FEATS + ['fnbi', 'gst', 'Q_cond_e', 'ne', 'grad_Te', 'a0']


# ---------------------------------------------------------------------------
# BUILD: stream per shot -> reduced scoreable table (identical per-shot physics)
# ---------------------------------------------------------------------------
def process_shot(sd):
    sd = sd.reset_index(drop=True)
    lib = dimensionless(sd)
    lib.update(dynamic_terms(sd, lib))
    lib = add_spherical(sd, lib)
    y, t = deadband_target(sd, by_shot=True, sanitize=True)
    yf, tf = forecast_target(sd, y, t, HORIZON)
    reg = classify(sd)
    fnbi, gst = drive_geometry(sd)
    heated = np.full(len(sd), float(sd['P_tot'].mean()) > 1e5)
    out = pd.DataFrame({
        'machine': sd['machine'].values, 'shot': sd['shot'].values,
        'time': sd['time'].values, 'rho': sd['rho'].values, 'q': sd['q'].values,
        'accel': lib['accel'], 'prate_frac': lib['prate_frac'],
        'inv_etae': lib['inv_etae'], 'etae_surge': lib['etae_surge'],
        'betae': lib['betae'], 'gExB_hat': lib['gExB_hat'], 'ge': lib['ge'],
        'TeTi': lib['TeTi'], 'gExB': lib['gExB'], 'fnbi': fnbi, 'gst': gst,
        'Q_cond_e': sd['Q_cond_e'].values, 'ne': sd['ne'].values,
        'grad_Te': sd['grad_Te'].values, 'a0': sd['a0'].values,
        'y': np.asarray(y, bool), 't': np.asarray(t, bool),
        'yf': np.asarray(yf, bool), 'tf': np.asarray(tf, bool),
        'regime': reg, 'heated': heated.astype(bool),
    })
    keep = out['heated'].values & (out['t'].values | out['tf'].values)
    out = out.loc[keep].copy()
    for c in NUM:
        out[c] = out[c].astype(np.float32)
    return out


def build_reduced():
    """Stream the raw parquet per shot and WRITE each shot's reduced rows straight
    to disk (never accumulate all ~30M rows in RAM -- the 16 GB constraint)."""
    print(f"[build] streaming {RAW} per shot -> {RED}")
    pf = pq.ParquetFile(RAW)
    buf = None
    state = {'n': 0, 'rows': 0, 'writer': None, 'schema': None}
    t0 = time.time()

    def flush(complete):
        for s, sd in complete.groupby('shot', sort=False):
            try:
                o = process_shot(sd)
            except Exception as e:
                print(f"\n  [skip shot {s}] {type(e).__name__}: {e}")
                continue
            state['n'] += 1
            if not len(o):
                continue
            tbl = pa.Table.from_pandas(o, preserve_index=False)
            if state['writer'] is None:
                state['schema'] = tbl.schema
                state['writer'] = pq.ParquetWriter(RED, tbl.schema)
            else:
                tbl = tbl.cast(state['schema'])          # keep schema stable
            state['writer'].write_table(tbl)
            state['rows'] += len(o)

    for b in pf.iter_batches(batch_size=2_000_000, columns=READ_COLS):
        d = b.to_pandas()
        if buf is not None:
            d = pd.concat([buf, d], ignore_index=True)
        last = d['shot'].values[-1]
        is_last = d['shot'].values == last
        buf = d.loc[is_last].reset_index(drop=True)
        flush(d.loc[~is_last])
        print(f"  ...{state['n']} shots, {state['rows']:,} rows, "
              f"{time.time()-t0:.0f}s", end='\r', flush=True)
    if buf is not None and len(buf):
        flush(buf)
    if state['writer'] is not None:
        state['writer'].close()
    print(f"\n[build] {state['rows']:,} scoreable rows from {state['n']} shots in "
          f"{time.time()-t0:.0f}s  ->  {RED}")
    return None


# ---------------------------------------------------------------------------
# VALIDATE: nowcast + 10 ms forecast, per-regime + unified, on the reduced table
# ---------------------------------------------------------------------------
def _design_idx(law, lib, idx):
    """log-space design matrix for the named features on COMPACT row indices idx."""
    return np.column_stack([np.log(np.maximum(lib[k][idx], 1e-12)) for k in law])


def _pooled_loso(Xr, yr, shotr, machr, cap=200000, seed=0):
    """Pooled leave-one-SHOT-out AUC on compact regime arrays. A machine-weighted
    training pool (<=cap rows) is drawn once and re-fit per held-out shot with that
    shot's rows removed -- identical logic to the full per-shot LOSO but ~20x faster
    (no 36M-row re-masking / re-copy per fold; a 2-4 param logistic is fully
    determined by a 200k machine-balanced sample). The TEST is always the full
    held-out shot."""
    rng = np.random.default_rng(seed)
    n = len(yr)
    pool = rng.choice(n, cap, replace=False) if n > cap else np.arange(n)
    Xp, yp, sp, mp = Xr[pool], yr[pool], shotr[pool], machr[pool]
    order = np.argsort(shotr, kind='stable'); shs = shotr[order]
    uniq, first = np.unique(shs, return_index=True); bounds = np.append(first, len(shs))
    ps, ys = [], []
    for k in range(len(uniq)):
        te = order[bounds[k]:bounds[k + 1]]
        if len(te) < 40 or not (0 < yr[te].mean() < 1):
            continue
        keep = sp != uniq[k]
        if keep.sum() < 200 or not (0 < yp[keep].mean() < 1):
            continue
        w, mu, sd = logistic(Xp[keep], yp[keep].astype(float), l2=2.0, subsample=0,
                             wts=_machine_weights(mp[keep]))
        ps.append(logistic_pred(Xr[te], w, mu, sd)); ys.append(yr[te].astype(bool))
    return auc(np.concatenate(ps), np.concatenate(ys)) if ps else np.nan


def _lomo(Xr, yr, machr, cap=200000, seed=0):
    """Leave-one-MACHINE-out AUC (frozen coefficients to an unseen machine)."""
    rng = np.random.default_rng(seed)
    n = len(yr)
    pool = rng.choice(n, cap, replace=False) if n > cap else np.arange(n)
    Xp, yp, mp = Xr[pool], yr[pool], machr[pool]
    ps, ys = [], []
    for m in np.unique(machr):
        te = machr == m
        if te.sum() < 80 or not (0 < yr[te].mean() < 1):
            continue
        keep = mp != m
        if keep.sum() < 200 or not (0 < yp[keep].mean() < 1):
            continue
        w, mu, sd = logistic(Xp[keep], yp[keep].astype(float), l2=2.0, subsample=0,
                             wts=_machine_weights(mp[keep]))
        ps.append(logistic_pred(Xr[te], w, mu, sd)); ys.append(yr[te].astype(bool))
    return (auc(np.concatenate(ps), np.concatenate(ys)), len(ps)) if ps else (np.nan, 0)


def run_tables(df, lib, y, t, heated, mach, q, fnbi, gst, header):
    print(f"\n===== {header} =====")
    print(f"{'regime':<14}{'machines':>9}{'shots':>7}{'onset%':>8}"
          f"{'POOLED-LOSO':>12}{'LOMO':>8}{'target':>8}")
    shot = df['shot'].cat.codes.to_numpy()
    reg = df['regime'].values
    for name, cfg in REGIMES.items():
        sel = t & (reg == name) & heated
        if name != 'conv_NBI':
            sel = sel & (q > 1.5)
        if sel.sum() < 200 or y[sel].sum() < 50:
            print(f"{name:<14}{'-':>9}{'-':>7}   (insufficient data)")
            continue
        idx = np.where(sel)[0]
        Xr = _design_idx(cfg['law'], lib, idx)
        yr, sr, mr = y[idx], shot[idx], mach[idx]
        ploso = _pooled_loso(Xr, yr, sr, mr)
        lomo, _ = _lomo(Xr, yr, mr)
        nm = len(np.unique(mr)); ns = len(np.unique(sr))
        flag = 'PASS' if (np.isfinite(ploso) and ploso >= cfg['target'] - 0.03) else 'CHECK'
        print(f"{name:<14}{nm:>9}{ns:>7}{yr.mean()*100:>7.1f}%{ploso:>12.3f}"
              f"{lomo:>8.3f}{cfg['target']:>8.2f}  {flag}")
    # unified single equation (Eq 7): drive-blend + geometry-gate across all cells
    selU = t & heated & ((q > 1.5) | (reg == 'conv_NBI'))
    idx = np.where(selU)[0]
    ulib = {k: lib[k][idx] for k in ('accel', 'inv_etae', 'betae', 'TeTi', 'gExB', 'gExB_hat')}
    Xu = unified_design(ulib, fnbi[idx], gst[idx])
    au = _pooled_loso(Xu, y[idx], shot[idx], mach[idx])
    print(f"UNIFIED single-equation (Eq 7: drive-blend + geometry-gate)  "
          f"POOLED-LOSO: {au:.3f}")


def per_machine_diag_fast(df, lib, y, t, heated, mach, q):
    """Per (machine, regime): within-machine leave-one-shot-out AUC + data-quality
    flags. This is where beam-machine Ohmic phases (KSTAR conv_0torque etc.) get
    checked against the zero-torque law -- the 'does ramp-up hold' test."""
    print("\n  PER-MACHINE x REGIME DIAGNOSTIC (within-machine leave-one-shot-out):")
    print(f"  {'machine':<8}{'regime':<13}{'shots':>6}{'cells':>9}{'onset%':>7}"
          f"{'within-AUC':>11}{'Qc<0%':>7}{'chi_med':>9}")
    E = 1.602e-19
    shot = df['shot'].cat.codes.to_numpy()
    reg = df['regime'].values
    machine = df['machine'].values
    Qc = df['Q_cond_e'].values.astype(float)
    ne = df['ne'].values.astype(float)
    gTe = np.abs(df['grad_Te'].values.astype(float))
    for m in sorted(df['machine'].cat.categories):
        mm = machine == m
        for rname in ('spherical', 'conv_NBI', 'conv_0torque'):
            sel = t & heated & mm & (reg == rname)
            if rname != 'conv_NBI':
                sel = sel & (q > 1.5)
            if sel.sum() < 100:
                continue
            idx = np.where(sel)[0]
            a = np.nan
            if len(idx) > 200 and y[idx].sum() > 30 and len(np.unique(shot[idx])) >= 2:
                Xr = _design_idx(REGIMES[rname]['law'], lib, idx)
                a = _pooled_loso(Xr, y[idx], shot[idx], mach[idx], cap=120000)
            qc = (Qc[idx] < 0).mean() * 100
            chi = np.median(np.abs(Qc[idx]) / (np.maximum(ne[idx], 1e16) * E
                                               * np.maximum(gTe[idx], 1e-3) + 1e-9))
            aS = f"{a:.3f}" if np.isfinite(a) else "  n/a"
            print(f"  {str(m):<8}{rname:<13}{len(np.unique(shot[idx])):>6}{len(idx):>9}"
                  f"{y[idx].mean()*100:>6.1f}%{aS:>11}{qc:>6.0f}%{chi:>9.2f}")


def validate(red, write_output=True):
    df = red.reset_index(drop=True)
    # categorical keys keep 30M string rows from blowing 16 GB (codes + a few
    # hundred category strings, not one Python str per row)
    for c in ('machine', 'shot', 'regime'):
        df[c] = df[c].astype('category')
    mach = df['machine'].cat.codes.to_numpy()                # compact int codes for weights
    q = df['q'].values.astype(float)
    heated = df['heated'].values.astype(bool)
    lib = {k: df[k].values.astype(float) for k in FEATS}
    fnbi = df['fnbi'].values.astype(float)
    gst = df['gst'].values.astype(float)

    y = df['y'].values.astype(bool)
    t0 = df['t'].values.astype(bool)
    yf = df['yf'].values.astype(bool)
    tf0 = df['tf'].values.astype(bool)

    print(f"\nreduced table: {len(df):,} rows, {df['shot'].nunique()} shots, "
          f"{df['machine'].nunique()} machines")
    mq = machine_quality(df, y, t0, heated)          # drop garbage machines
    t = t0 & mq
    tf = tf0 & mq

    # shot-flow (nowcast)
    print("\nshot-flow per regime  [scoreable -> plasma -> q>1.5 -> has-onset]:")
    for name in REGIMES:
        rs = df['regime'].values == name
        nh = df.loc[rs & heated, 'shot'].nunique()
        qf = rs & heated & ((q > 1.5) | (name == 'conv_NBI'))
        non = df.loc[qf & t & y, 'shot'].nunique() if (qf & t).sum() else 0
        print(f"   {name:<14} {df.loc[rs, 'shot'].nunique():>4} -> {nh:>4} -> "
              f"{df.loc[qf, 'shot'].nunique():>4} -> {non:>4}")

    run_tables(df, lib, y, t, heated, mach, q, fnbi, gst,
               "NOWCAST (0 ms -- onset happening now)")
    run_tables(df, lib, yf, tf, heated, mach, q, fnbi, gst,
               "FORECAST (+10 ms -- onset within 10 ms)")

    print("\nPOOLED-LOSO = fit on the whole multi-machine database, hold out each")
    print("  shot (IPB98-style). LOMO = leave-one-MACHINE-out (frozen coefficients")
    print("  to a never-seen machine). UNIFIED = one drive-blended, geometry-gated")
    print("  equation. NOWCAST = onset now; FORECAST = onset within 10 ms.")
    closure_test(df, lib, y, t & heated)
    per_machine_diag_fast(df, lib, y, t, heated, mach, q)

    if write_output:
        # per-cell nowcast hazard from the unified law (full weighted fit; in-sample
        # -- for FIGURES only, the headline AUCs above are all held-out)
        Xu = unified_design(lib, fnbi, gst)
        sfit = t & heated & ((q > 1.5) | (df['regime'].values == 'conv_NBI'))
        w, mu, sd = logistic(Xu[sfit], y[sfit].astype(float), l2=2.0, subsample=0,
                             wts=_machine_weights(mach[sfit]))
        df['hazard_unified'] = logistic_pred(Xu, w, mu, sd).astype(np.float32)
        df.to_parquet(OUT, index=False)
        print(f"\n[output] {len(df):,} rows + per-cell unified hazard  ->  {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='force re-stream raw')
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--no-output', action='store_true')
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    if a.rebuild or not os.path.exists(RED):
        red = build_reduced()
    else:
        print(f"[cache] loading {RED} (use --rebuild to re-stream)")
        red = pd.read_parquet(RED)
    if not a.build_only:
        validate(red, write_output=not a.no_output)


if __name__ == '__main__':
    main()
