#!/usr/bin/env python3
"""spherical_onset.py -- derive the SPHERICAL-tokamak onset branch (NSTX) and
test the ST-specific physics (Gemini): strong E×B shear damping, beta_e-gated
MTM/ETG drive, aspect-ratio geometry. NSTX is high-NBI (torque-driven, f_NBI~1),
so the drive is accel -- but STs damp it far harder via supersonic rotation shear.

ST feature adaptations:
  betae      = 2*mu0*ne*Te / B^2                 (electron beta; EM/MTM gate)
  gExB_hat   = gamma_ExB / (c_s/a)               (shear normalised to sound rate)
  shear_damped_drive = accel / (1 + a*gExB_hat)  (drive over ST shear barrier)
  betae_etae = inv_etae * betae                  (beta-gated TEM->MTM drive)

    python spherical_onset.py <allmachines.parquet> [--machine NSTX]
"""
import sys
import argparse
import numpy as np
import pandas as pd
from derive_onset import (domain_filter, deadband_target, dimensionless,
                          dynamic_terms, forward_select, _design, logistic,
                          logistic_pred, auc)

M_I = 3.344e-27
ECH = 1.602e-19
MU0 = 4e-7 * np.pi


def add_spherical(d, lib):
    ne = np.maximum(d['ne'].values.astype(float), 1e16)
    Te = np.maximum(d['Te'].values.astype(float), 1.0)
    Bt = np.maximum(np.abs(d['Bt'].values.astype(float)), 0.1)
    a0 = np.maximum(d['a0'].values.astype(float), 0.05)
    cs = np.sqrt(Te * ECH / M_I)
    lib['betae'] = np.maximum(ne * Te * ECH / (Bt ** 2 / (2 * MU0)), 1e-8)
    gExB = np.abs(lib.get('gExB', np.full(len(d), 0.05)))
    lib['gExB_hat'] = gExB / (cs / a0) + 1e-6                 # shear / sound rate
    lib['betae_etae'] = lib['inv_etae'] * lib['betae'] + 1e-9  # beta-gated TEM drive
    lib['ft'] = np.sqrt(np.clip(d['rho'].values * a0 / np.maximum(d['R0'].values, 0.1), 1e-3, 0.99))
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parquet'); ap.add_argument('--machine', default='NSTX')
    a = ap.parse_args()
    allm = pd.read_parquet(a.parquet)
    d = domain_filter(allm[allm['machine'] == a.machine].reset_index(drop=True), 'heated')
    y, t = deadband_target(d, by_shot=True)
    lib = dimensionless(d); lib.update(dynamic_terms(d, lib)); lib = add_spherical(d, lib)
    fnbi = np.median(np.clip(d['P_inj'].values / np.maximum(d['P_tot'].values, 1e3), 0, 1))
    print(f"{a.machine}: {d['shot'].nunique()} shots, {len(d):,} rows, onset={y[t].mean()*100:.1f}%, "
          f"n_pos={int(y[t].sum())}, f_NBI={fnbi:.2f}, R/a={np.median(d['R0']/d['a0']):.1f}")

    print("\nper-term single AUC (does the ST physics predict onset?):")
    for k in ['accel', 'prate_frac', 'gExB', 'gExB_hat', 'inv_etae', 'betae',
              'betae_etae', 'TeTi', 'nustar', 'shear']:
        if k in lib:
            print(f"   {k:<14} {auc(np.log(np.maximum(lib[k][t], 1e-12)), y[t].astype(bool)):.3f}")

    # shear-damped drive (Gemini): accel / (1 + alpha*gExB_hat); scan alpha
    print("\nshear-damped drive  accel/(1+alpha*gExB_hat), single-feature AUC:")
    for alpha in (0.0, 1.0, 5.0, 20.0):
        sd = np.log(np.maximum(lib['accel'], 1e-12)) - np.log1p(alpha * lib['gExB_hat'])
        print(f"   alpha={alpha:<5} {auc(sd[t], y[t].astype(bool)):.3f}")

    print("\nforward-select SPHERICAL law (LOSO across shots):")
    forward_select(d, y, t, lib, list(lib.keys()), max_terms=6)


if __name__ == '__main__':
    main()
