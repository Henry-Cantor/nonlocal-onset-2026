#!/usr/bin/env python3
"""build_features.py -- stream the TRANSP parquet (stage1_5_v3.parquet) per shot and
build reduced_v2.parquet: the dimensionless drivers, the absolute onset label
(flux > 4x the per-machine local gyro-Bohm baseline, coherent over >=4 surfaces), the
regime router, Q_cond_e/Q_GB, and the non-circular reconstruction-quality columns.
Streams per shot so the 43 M-row table never lands in RAM at once. Feature-engineering
helpers live in support/.

    python build_features.py        # stage1_5_v3.parquet -> reduced_v2.parquet
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'support'))
from derive_onset import dimensionless, dynamic_terms, deadband_target
from spherical_onset import add_spherical
from probe_law import clean_residual, adaptive_z
from run_onset_stellar import classify, drive_geometry, forecast_target
from final_run import add_aligned

AGM = "/Users/henrycantor/mentorship2/ACTUALLY_GOOD_MAYBE"
RAW = os.path.join(AGM, "stage1_5_v3.parquet")
OUT = os.path.join(AGM, "reduced_v2.parquet")
QUAL = ['recon_resid_te', 'recon_resid_ne', 'ring_te', 'te_snr', 'ne_snr',
        'nubeam_ms', 'recon_tgap_ms']              # non-circular reconstruction quality
HORIZON = 0.010
R_HI, R_LO, EXT = 1.0, 0.4, 4     # abs-label: flux >2x local, coherent >=4 surfaces
ECH = 1.602e-19
M_I = 3.344e-27

# Option-A physics features (shaping / rotation / fast-ion), from the new columns:
#   kappa=elongation, tri=1.5+triangularity (signed, log-safe; TCV neg-delta matters),
#   mach=|omega|R0/c_s (toroidal Mach), gExB_rot=|d omega/d rho|/(c_s/a) (rotation
#   shear, the ST turbulence quench), nfast_frac=n_fast/n_e (fast-ion stabilisation).
M_E = 9.109e-31
PHYS = ['kappa', 'tri', 'mach', 'gExB_rot', 'nfast_frac']
# Gemini physics composites (unlock conv_NBI / conv_0torque):
GEM = ['net_ratio', 'eff_drive', 'uni_drive', 'u_peak']
FEATS = ['accel', 'prate_frac', 'inv_etae', 'etae_surge', 'betae', 'gExB_hat', 'ge',
         'TeTi', 'gExB', 'load8', 'rstar', 'corrug', 'overdrive', 'gmc', 'prox',
         'nustar', 'crit'] + PHYS + GEM


def add_gemini(sd, lib):
    """physics composites:
      net_ratio = gamma_drive/gExB  (drive vs E×B shear; onset when >1)
      eff_drive = eta_e*(1-f_fast)  (fast-ion-diluted drive)
      uni_drive = w(nu*)eta_e + (1-w)eta_i, w=1/(1+nu*^2)  (collisionality gate: TEM
                  low-nu* vs ITG high-nu* -- resolves the eta_e cross-machine sign flip)
      u_peak    = q * R/L_ne  (density-peaking x q)"""
    Te = np.maximum(sd['Te'].values.astype(float), 1.0)
    R0 = np.maximum(sd['R0'].values.astype(float), 0.1)
    vthe = np.sqrt(Te * ECH / M_E)
    ge = lib['ge']; crit = np.maximum(lib['crit'], 0.5)
    gamma_drive = (vthe / R0) * np.maximum(ge - crit, 0.0)
    lib['net_ratio'] = gamma_drive / np.maximum(lib['gExB'], 1e2) + 1e-4
    ffast = np.clip(np.nan_to_num(sd['n_fast_frac'].values.astype(float)), 0.0, 0.95)
    lib['eff_drive'] = lib['inv_etae'] * (1.0 - ffast) + 1e-3
    nu = lib['nustar']; w = 1.0 / (1.0 + nu ** 2)
    rlne = lib['inv_etae'] * ge
    etae_i = rlne / np.maximum(lib['gi'], 1e-2)
    lib['uni_drive'] = w * lib['inv_etae'] + (1.0 - w) * etae_i + 1e-3
    lib['u_peak'] = lib['q'] * rlne + 1e-3
    # signed radial spreading curvature d2(R/LTe)/drho2 -- the non-locality kernel
    # (heat jumping across flux surfaces; Gemini's Garbet-Diamond spreading term)
    nrow = len(sd)
    dd = pd.DataFrame({'shot': sd['shot'].values, 'time': sd['time'].values,
                       'rho': sd['rho'].values, 'ge': ge,
                       '_i': np.arange(nrow)}).sort_values(['shot', 'time', 'rho'])
    curv = dd.groupby(['shot', 'time'])['ge'].transform(lambda v: v.diff().diff())
    spread = np.zeros(nrow); spread[dd['_i'].values] = np.nan_to_num(curv.values)
    lib['spread'] = spread
    # Gemini onset order parameter (first-principles): theta_onset > 0 => nonlocal
    # avalanche. drive-over-(fast-ion-modified)-critical  -  E×B shear^2  +  spreading.
    g0 = vthe / R0
    crit_f = crit * (1.0 + 2.0 * ffast)
    lib['theta_onset'] = ((ge - crit_f) - (lib['gExB'] / np.maximum(g0, 1e2)) ** 2
                          + 5.0 * lib['rstar'] ** 2 * spread)
    return lib
NUM = ['time', 'rho', 'q'] + FEATS + ['theta_onset', 'spread', 'fnbi', 'gst', 'Z',
                                      'resid', 'Q_cond_e', 'Q_GB', 'ne', 'grad_Te', 'a0'] + QUAL
READ = ['machine', 'shot', 'time', 'rho', 'Te', 'Ti', 'ne', 'q', 'Bt', 'R0', 'a0',
        'zeff', 'R_over_LTe', 'R_over_LTi', 'rho_star', 'gamma_ExB', 'Q_cond_e',
        'Q_GB', 'grad_Te', 'P_tot', 'P_inj', 'kappa', 'delta', 'omega_tor',
        'n_fast', 'n_fast_frac'] + QUAL


def add_physics(sd, lib):
    """shaping / rotation / fast-ion features from the Option-A columns."""
    n = len(sd)
    Te = np.maximum(sd['Te'].values.astype(float), 1.0)
    a0 = np.maximum(sd['a0'].values.astype(float), 0.05)
    R0 = np.maximum(sd['R0'].values.astype(float), 0.1)
    cs = np.sqrt(Te * ECH / M_I)
    lib['kappa'] = np.maximum(np.abs(sd['kappa'].values.astype(float)), 1e-3)
    lib['tri'] = np.clip(1.5 + np.nan_to_num(sd['delta'].values.astype(float)), 0.1, 3.0)
    om = np.nan_to_num(sd['omega_tor'].values.astype(float))
    lib['mach'] = np.abs(om) * R0 / np.maximum(cs, 1e3) + 1e-6
    d = pd.DataFrame({'shot': sd['shot'].values, 'time': sd['time'].values,
                      'rho': sd['rho'].values, 'om': om,
                      '_i': np.arange(n)}).sort_values(['shot', 'time', 'rho'])
    g = d.groupby(['shot', 'time'])
    dom = g['om'].diff().values
    drho = g['rho'].diff().values
    sh = np.abs(dom / np.where(np.abs(drho) > 1e-6, drho, np.nan))
    rot = np.zeros(n); rot[d['_i'].values] = np.nan_to_num(sh)
    lib['gExB_rot'] = rot / (cs / a0) + 1e-6
    lib['nfast_frac'] = np.maximum(np.nan_to_num(sd['n_fast_frac'].values.astype(float)), 0.0) + 1e-9
    return lib


def abs_label(sd, resid, trust):
    """absolute physical onset: resid>R_HI, coherent over >=EXT adjacent surfaces."""
    surge = (resid > R_LO).astype(float)
    d = pd.DataFrame({'shot': sd['shot'].values, 'time': sd['time'].values,
                      'rho': sd['rho'].values, 's': surge,
                      '_i': np.arange(len(sd))}).sort_values(['shot', 'time', 'rho'])
    ext = d.groupby(['shot', 'time'])['s'].transform(
        lambda v: v.rolling(5, center=True, min_periods=1).sum())
    extent = np.zeros(len(sd)); extent[d['_i'].values] = ext.values
    y = (resid > R_HI) & (extent >= EXT) & trust
    tt = trust & ((resid < R_LO) | y)                 # clear onset or clear quiet
    return np.asarray(y, bool), np.asarray(tt, bool)


def process_shot(sd):
    sd = sd.reset_index(drop=True)
    lib = dimensionless(sd)
    lib.update(dynamic_terms(sd, lib))
    lib = add_spherical(sd, lib)
    lib = add_aligned(sd, lib)
    lib = add_physics(sd, lib)
    lib = add_gemini(sd, lib)
    y, t = deadband_target(sd, by_shot=True, sanitize=True)
    resid, delta, qbase, trust = clean_residual(sd, sanitize=True)
    Z = adaptive_z(sd, delta, qbase, by_shot=True)
    ya, ta = abs_label(sd, np.nan_to_num(resid), trust)
    yf, tf = forecast_target(sd, np.asarray(y, bool), np.asarray(t, bool), HORIZON)
    reg = classify(sd); fnbi, gst = drive_geometry(sd)
    heated = float(sd['P_tot'].mean()) > 1e5
    y = np.asarray(y, bool); t = np.asarray(t, bool)
    # keep ALL physically-valid cells (not just the deadband) so the onset threshold
    # can be re-derived downstream -- resid is stored per cell.
    keep = np.where(heated & np.asarray(trust, bool))[0]
    if len(keep) == 0:
        return None
    out = {'machine': sd['machine'].values[keep], 'shot': sd['shot'].values[keep],
           'y': y[keep], 't': t[keep], 'yf': yf[keep], 'tf': tf[keep],
           'y_abs': ya[keep], 't_abs': ta[keep], 'regime': reg[keep]}
    src = {'Z': Z, 'resid': np.nan_to_num(resid), 'fnbi': fnbi, 'gst': gst,
           'Q_cond_e': sd['Q_cond_e'].values, 'Q_GB': sd['Q_GB'].values, 'ne': sd['ne'].values,
           'grad_Te': sd['grad_Te'].values, 'a0': sd['a0'].values,
           'time': sd['time'].values, 'rho': sd['rho'].values, 'q': sd['q'].values}
    for qc in QUAL:
        src[qc] = sd[qc].values
    for k in NUM:
        v = lib[k] if k in lib else src[k]
        out[k] = v[keep].astype(np.float32)
    return pd.DataFrame(out)


def main():
    pf = pq.ParquetFile(RAW)
    buf = None; w = None; sch = None; n = 0; rows = 0; t0 = time.time()

    def flush(comp):
        nonlocal w, sch, n, rows
        for s, sd in comp.groupby('shot', sort=False):
            try:
                o = process_shot(sd)
            except Exception as e:
                print(f"\n  [skip {s}] {type(e).__name__}: {e}", flush=True); continue
            n += 1
            if o is None or not len(o):
                continue
            tbl = pa.Table.from_pandas(o, preserve_index=False)
            if w is None:
                sch = tbl.schema; w = pq.ParquetWriter(OUT, sch)
            else:
                tbl = tbl.cast(sch)
            w.write_table(tbl); rows += len(o)

    for b in pf.iter_batches(batch_size=2_000_000, columns=READ):
        d = b.to_pandas()
        if buf is not None:
            d = pd.concat([buf, d], ignore_index=True)
        last = d['shot'].values[-1]; is_last = d['shot'].values == last
        buf = d.loc[is_last].reset_index(drop=True)
        flush(d.loc[~is_last])
        print(f"  ...{n} shots, {rows:,} rows, {time.time()-t0:.0f}s", end='\r', flush=True)
    if buf is not None and len(buf):
        flush(buf)
    if w is not None:
        w.close()
    print(f"\n[build_features] {rows:,} rows, {n} shots in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
