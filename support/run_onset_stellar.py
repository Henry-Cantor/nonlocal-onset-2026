#!/usr/bin/env python3
"""run_onset_stellar.py -- ONE-AND-DONE validation of the complete 3-regime onset
law on the full Stellar shot set. Classifies every CELL by two knobs -- a fixed
PRE-SHOT geometry (aspect ratio eps=a/R) and the INSTANTANEOUS external drive
fraction f_NBI(t)=P_NBI(t)/P_tot(t) -- applies the regime's law, and validates
LEAVE-ONE-MACHINE-OUT (each regime now has enough machines to constrain its
coefficients -- the test that could not be done on the local set). The pooled fit
is per-machine weighted so no single large machine sets the coefficients.

Regimes and their derived drivers (see DERIVATION_ONSET.md):
  spherical (eps>0.5)                 : beta_e law   (MTM / finite-beta EM)  ~0.91
  conventional + NBI (f_NBI>0.5)      : accel law    (ITG, torque-driven)    ~0.96
  conventional + zero-torque          : eta_e law    (TEM, pinned profile)   ~0.80

Every number is deadband target + per-shot Z + Q_cond sanitation, held out.

    python run_onset_stellar.py <450shot>.parquet
"""
import sys
import numpy as np
import pandas as pd
from derive_onset import (deadband_target, dimensionless, dynamic_terms, _design,
                          logistic, logistic_pred, auc)
from probe_law import youden_threshold
from spherical_onset import add_spherical

ECH = 1.602e-19


def closure_test(df, lib, y, t):
    """tau_r CLOSURE / universal critical value: onset should fire when the regime's
    DIMENSIONLESS order parameter crosses ONE constant on every machine. For the
    torque-driven regime accel is made dimensionless with the turbulent relaxation
    time tau_r = a^2/chi_eff (chi_eff = |Q_cond|/(n_e e |grad Te|)); eta_e and
    beta_e are already dimensionless. Report the cross-machine spread of the per-
    machine Youden critical value: tight spread = a falsifiable universal law, not
    just a good predictor."""
    ne = np.maximum(df['ne'].values.astype(float), 1e16)
    gTe = np.abs(df['grad_Te'].values.astype(float))
    a0 = np.maximum(df['a0'].values.astype(float), 0.05)
    chi = np.clip(np.abs(df['Q_cond_e'].values.astype(float))
                  / (ne * ECH * np.maximum(gTe, 1e-3) + 1e-9), 1e-2, 1e3)   # m^2/s
    tau_r = a0 ** 2 / chi
    omega = {'conv_NBI': lib['accel'] * tau_r ** 2,      # dimensionless via tau_r
             'conv_0torque': lib['inv_etae'],            # already dimensionless
             'spherical': lib['betae']}                  # already dimensionless
    print(f"\n  tau_r-CLOSURE universal critical value (spread of Omega_crit across machines):")
    for regime, om in omega.items():
        sel = t & (df['regime'].values == regime)
        crits = []
        for m in pd.unique(df['machine'].values[sel]):
            ms = sel & (df['machine'].values == m); yy = y[ms].astype(bool)
            if yy.sum() < 20 or (~yy).sum() < 20:
                continue
            thr = youden_threshold(om[ms], yy)
            if np.isfinite(thr) and thr > 0:
                crits.append(thr)
        if len(crits) >= 2:
            cv = float(np.exp(np.std(np.log(crits))) - 1.0)
            verdict = 'UNIVERSAL constant' if cv < 0.35 else 'partial' if cv < 0.8 else 'machine-dependent'
            print(f"     {regime:<14} {len(crits)} machines, spread {cv*100:>4.0f}%  [{verdict}]")
        else:
            print(f"     {regime:<14} <2 machines with events here (need Stellar)")

REGIMES = {
    'spherical':   dict(law=['betae', 'ge', 'TeTi', 'gExB_hat'],           target=0.91),
    'conv_NBI':    dict(law=['accel', 'prate_frac'],                        target=0.95),
    'conv_0torque': dict(law=['inv_etae', 'TeTi', 'gExB', 'etae_surge'],    target=0.80),
}


def machine_quality(df, y, t, heated):
    """PER-MACHINE data-quality gate. Drops whole machines whose power balance is
    garbage -- median chi_eff outside a sane thermal-diffusivity range, or a large
    fraction of Q_cond<0 -- e.g. the RFP MST (chi~3000, 54% negative). Keeps the
    physics machines; does NOT touch individual cells within a good machine (the
    per-cell chi gate wrongly deleted good edge cells and hurt AUC)."""
    E = 1.602e-19
    good = np.ones(len(df), bool)
    dropped = []
    for m in df['machine'].unique():
        mm = df['machine'].values == m
        gTe = np.abs(df.loc[mm, 'grad_Te'].values); ne = np.maximum(df.loc[mm, 'ne'].values, 1e16)
        chi = np.median(np.abs(df.loc[mm, 'Q_cond_e'].values) / (ne * E * np.maximum(gTe, 1e-3) + 1e-9))
        qc = (df.loc[mm, 'Q_cond_e'].values < 0).mean()
        if not (0.05 < chi < 50.0) or qc > 0.40:
            good &= ~mm; dropped.append(f"{m}(chi={chi:.1f},Qc<0={qc*100:.0f}%)")
    if dropped:
        print(f"[data-quality] dropped garbage machines: {dropped}")
    return good


def drive_geometry(df):
    """The two continuous router fields, used BOTH for the discrete regime labels
    and for the unified blended equation:
      eps = a/R                     : machine geometry (shot-constant, known t<0).
            Rendered as a SOFT gate g_ST = sigma((eps-0.5)/0.05) in [0,1] (~1
            spherical). g_ST>0.5 <=> eps>0.5, so the discrete spherical cut is
            unchanged; the softness only matters for the blended equation.
      f_NBI(t) = P_NBI(t)/P_tot(t)  : the beam-power fraction AT TIME t. Resolved in
            time ON PURPOSE -- a beam machine (e.g. KSTAR) is Ohmic (zero-torque)
            during its ramp and torque-driven once beams fire, so the SAME shot
            routes to conv_0torque early and conv_NBI later. Causal (present-only
            drive, no future) and non-circular (the drive is the external FORCING
            that causes avalanches, not a response to them), so it is a legal
            per-cell input. Time-resolving it is what lets the Ohmic phases of
            otherwise NBI-dominated machines populate the data-starved zero-torque
            regime. NB: the shot-integrated energy fraction sum(P_NBI)/sum(P_tot)
            stays the OPERATIONAL pre-shot router for the control-framing (Paper 2);
            the physics here routes on the drive actually present at each cell."""
    g = df.groupby('shot')
    eps = (g['a0'].transform('median') / np.maximum(g['R0'].transform('median'), 1e-3)).values
    pinj = np.nan_to_num(df['P_inj'].values.astype(float), nan=0.0)
    ptot = np.nan_to_num(df['P_tot'].values.astype(float), nan=0.0)
    fnbi = np.clip(pinj / np.maximum(ptot, 1e5), 0.0, 1.0)     # instantaneous beam fraction
    gst = 1.0 / (1.0 + np.exp(-(eps - 0.5) / 0.05))            # soft geometry gate
    return fnbi, gst


def classify(df):
    """Discrete regime labels for the per-regime tables (organisation only -- the
    unified_design blend below is the single continuous law). See drive_geometry
    for the physics of the two routing fields."""
    fnbi, gst = drive_geometry(df)
    return np.where(gst > 0.5, 'spherical',
           np.where(fnbi > 0.5, 'conv_NBI', 'conv_0torque'))


def unified_design(lib, fnbi, gst):
    """Log-linearised form of the SINGLE master equation (Eq 7 in DERIVATION_ONSET):
    one drive-blended, geometry-gated driver
        M = (1-g_ST)*[ f_NBI*log(accel) + (1-f_NBI)*log(eta_e) ] + g_ST*log(beta_e)
    plus the shared secondaries (Te/Ti and E×B shear). Fitting ONE logistic on this
    across ALL cells tests whether the 3 regimes are really ONE continuous law
    (drive-blends accel<->eta_e as beams turn on; geometry-gates to beta_e) rather
    than three separate fits -- and it is the ONLY test that scores the transitional
    f_NBI~0.5 cells the hard-routed fits never look at. g_ST is ~0/1 for real
    machines, so the additive blend equals log(Eq-7 bracket) except at the rare
    geometry boundary. The single logistic coefficient on M is the exponent 'a'."""
    L = lambda x: np.log(np.maximum(x, 1e-12))
    M = ((1 - gst) * (fnbi * L(lib['accel']) + (1 - fnbi) * L(lib['inv_etae']))
         + gst * L(lib['betae']))
    shear = (1 - gst) * L(lib['gExB']) + gst * L(lib['gExB_hat'])
    return np.column_stack([M, L(lib['TeTi']), shear])


def _machine_weights(mach):
    """Per-machine balancing weights for the pooled fit: every machine gets equal
    TOTAL weight regardless of how many cells it contributes, so a 100-shot machine
    cannot dominate the coefficients over a 5-shot one (the multi-machine-scaling
    standard, and the fix for the shots/machine imbalance). weight_i = 1 / (cells of
    machine i in this training set)."""
    w = np.ones(len(mach), float)
    for m in np.unique(mach):
        mm = (mach == m)
        w[mm] = 1.0 / max(int(mm.sum()), 1)
    return w


def leave_one_machine_out(df, y, t, X):
    mach = df['machine'].values
    ps, ys = [], []
    for m in pd.unique(mach[t]):
        te = t & (mach == m); tr = t & (mach != m)
        if te.sum() < 80 or tr.sum() < 200 or not (0 < y[tr].mean() < 1) or not (0 < y[te].mean() < 1):
            continue
        # balance the TRAINING machines so the frozen coefficients aren't set by
        # whichever remaining machine happens to be largest
        w, mu, sd = logistic(X[tr], y[tr].astype(float), l2=2.0, subsample=0,
                             wts=_machine_weights(mach[tr]))
        ps.append(logistic_pred(X[te], w, mu, sd)); ys.append(y[te].astype(bool))
    return (auc(np.concatenate(ps), np.concatenate(ys)), len(ps)) if ps else (np.nan, 0)


def per_machine_diag(df, lib, y, t, heated):
    """Split 'is it the physics or a few bad machines?'. Because the drive router is
    time-resolved, ONE machine can now populate more than one regime (its NBI phase
    in conv_NBI, its Ohmic phase in conv_0torque), so this reports a row per
    (machine, regime) it actually fills. That is exactly how you SEE whether e.g.
    KSTAR's Ohmic slices obey the zero-torque eta_e law -- the 'does ramp-up hold'
    check that decides if beam-machine Ohmic phases can rescue the thin regime. A
    machine+regime with high within-AUC but that tanks LOMO = feature-scale issue;
    low within-AUC or weird Q_cond<0 / onset% = bad data dragging the regime."""
    print(f"\n  PER-MACHINE x REGIME DIAGNOSTIC (within-machine leave-one-shot-out):")
    print(f"  {'machine':<8}{'regime':<13}{'shots':>6}{'cells':>8}{'onset%':>7}"
          f"{'within-AUC':>11}{'Qc<0%':>7}{'chi_med':>9}")
    E = 1.602e-19
    q = df['q'].values
    for m in sorted(df['machine'].unique()):
        mm = (df['machine'].values == m)
        for reg in ('spherical', 'conv_NBI', 'conv_0torque'):
            sel = t & heated & mm & (df['regime'].values == reg)
            if reg != 'conv_NBI':
                sel = sel & (q > 1.5)                   # drop sawtooth surfaces
            if sel.sum() < 100:
                continue
            law = REGIMES[reg]['law']
            shots = df.loc[sel, 'shot'].unique()
            # within-machine leave-one-shot-out on THIS regime's cells + law
            a = np.nan
            if sel.sum() > 200 and y[sel].sum() > 30 and len(shots) >= 2:
                X = _design(law, lib); ps, ys = [], []
                for s in shots:
                    te = sel & (df['shot'].values == s); tr = sel & (df['shot'].values != s)
                    if te.sum() < 40 or tr.sum() < 100 or not (0 < y[tr].mean() < 1) or not (0 < y[te].mean() < 1):
                        continue
                    w, mu, sd = logistic(X[tr], y[tr].astype(float), l2=2.0)
                    ps.append(logistic_pred(X[te], w, mu, sd)); ys.append(y[te].astype(bool))
                if ps:
                    a = auc(np.concatenate(ps), np.concatenate(ys))
            qc = (df.loc[sel, 'Q_cond_e'].values < 0).mean() * 100
            gTe = np.abs(df.loc[sel, 'grad_Te'].values); ne = np.maximum(df.loc[sel, 'ne'].values, 1e16)
            chi = np.median(np.abs(df.loc[sel, 'Q_cond_e'].values) / (ne * E * np.maximum(gTe, 1e-3) + 1e-9))
            aS = f"{a:.3f}" if np.isfinite(a) else "  n/a"
            print(f"  {m:<8}{reg:<13}{len(shots):>6}{sel.sum():>8}{y[sel].mean()*100:>6.1f}%"
                  f"{aS:>11}{qc:>6.0f}%{chi:>9.2f}")


def forecast_target(df, y, trust, horizon):
    """Shift the onset LABEL forward by `horizon` seconds within each (shot, rho)
    series: predict FUTURE onset from PRESENT features (features are UNCHANGED --
    only the label moves). A cell is scored only if (a) a cell ~horizon ahead exists
    in its series with a clean deadband label (matched within 2 ms of t+horizon),
    and (b) it is NOT already in clear onset now -- so this is a genuine early-
    warning transition test, not persistence of an ongoing burst. horizon<=0 returns
    the nowcast unchanged. 'hazard' is forward-looking, so this is its native use."""
    if horizon <= 0:
        return y, trust
    n = len(df)
    base = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                         'time': df['time'].values.astype(float), '_i': np.arange(n)})
    base['key'] = base['time'] + float(horizon)
    fut = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                        'ftime': df['time'].values.astype(float),
                        'fy': y.astype(bool), 'ftr': trust.astype(bool)})
    b = base.sort_values('key'); f = fut.sort_values('ftime')
    m = pd.merge_asof(b, f, left_on='key', right_on='ftime', by=['shot', 'rho'],
                      direction='forward', tolerance=2e-3).sort_values('_i')
    matched = m['ftime'].notna().values
    y_fwd = np.where(matched, m['fy'].fillna(False).values, False).astype(bool)
    ftr = np.where(matched, m['ftr'].fillna(False).values, False).astype(bool)
    trust_fwd = matched & ftr & (~y.astype(bool))       # exclude cells already onset now
    return y_fwd, trust_fwd


def regime_table(df, lib, y, t, heated, mach, q, header):
    """Per-regime POOLED-LOSO + LOMO table and the unified single-equation line, for
    ONE label horizon. Features are identical across horizons; only y/t (the label
    and its trust mask) change, so nowcast and forecast are directly comparable."""
    print(f"\n===== {header} =====")
    print(f"{'regime':<14}{'machines':>9}{'shots':>7}{'onset%':>8}{'POOLED-LOSO':>12}{'LOMO':>8}{'target':>8}")
    for name, cfg in REGIMES.items():
        sel = t & (df['regime'].values == name) & heated
        if name != 'conv_NBI':
            sel = sel & (q > 1.5)                       # drop sawtooth surfaces
        if sel.sum() < 200 or y[sel].sum() < 50:
            print(f"{name:<14}{'-':>9}{'-':>7}   (insufficient data)")
            continue
        X = _design(cfg['law'], lib)
        # POOLED leave-one-SHOT-out (IPB98-style multi-machine validation), fit
        # per-machine weighted so no one machine sets the coefficients
        shots = pd.unique(df['shot'].values[sel]); ps, ys = [], []
        for s in shots:
            te = sel & (df['shot'].values == s); tr = sel & (df['shot'].values != s)
            if te.sum() < 40 or tr.sum() < 200 or not (0 < y[tr].mean() < 1) or not (0 < y[te].mean() < 1):
                continue
            w, mu, sd = logistic(X[tr], y[tr].astype(float), l2=2.0,
                                 wts=_machine_weights(mach[tr]))
            ps.append(logistic_pred(X[te], w, mu, sd)); ys.append(y[te].astype(bool))
        ploso = auc(np.concatenate(ps), np.concatenate(ys)) if ps else np.nan
        lomo, _ = leave_one_machine_out(df, y, sel, X)   # stringent cross-machine
        nm = df.loc[sel, 'machine'].nunique(); ns = df.loc[sel, 'shot'].nunique()
        flag = 'PASS' if (np.isfinite(ploso) and ploso >= cfg['target'] - 0.03) else 'CHECK'
        print(f"{name:<14}{nm:>9}{ns:>7}{y[sel].mean()*100:>7.1f}%{ploso:>12.3f}"
              f"{lomo:>8.3f}{cfg['target']:>8.2f}  {flag}")
    # UNIFIED single-equation (Eq 7): one drive-blended, geometry-gated law across
    # all regimes -- the only test that also scores the transitional f_NBI~0.5 cells
    fnbi, gst = drive_geometry(df)
    Xu = unified_design(lib, fnbi, gst)
    selU = t & heated & ((q > 1.5) | (df['regime'].values == 'conv_NBI'))
    psU, ysU = [], []
    for s in pd.unique(df['shot'].values[selU]):
        te = selU & (df['shot'].values == s); tr = selU & (df['shot'].values != s)
        if te.sum() < 40 or tr.sum() < 200 or not (0 < y[tr].mean() < 1) or not (0 < y[te].mean() < 1):
            continue
        w, mu, sd = logistic(Xu[tr], y[tr].astype(float), l2=2.0, wts=_machine_weights(mach[tr]))
        psU.append(logistic_pred(Xu[te], w, mu, sd)); ysU.append(y[te].astype(bool))
    if psU:
        au = auc(np.concatenate(psU), np.concatenate(ysU))
        print(f"UNIFIED single-equation (Eq 7: drive-blend + geometry-gate)  POOLED-LOSO: {au:.3f}")


def main():
    df = pd.read_parquet(sys.argv[1])
    df['regime'] = classify(df)
    # domain: heated (P_tot>1MW) + drop sawtooth surfaces q<=1.5 handled per-regime
    y, t = deadband_target(df, by_shot=True, sanitize=True)
    lib = dimensionless(df); lib.update(dynamic_terms(df, lib)); lib = add_spherical(df, lib)
    # MINIMAL plasma gate only (P_tot>0.1MW = a real discharge). We do NOT drop
    # Ohmic shots -- Ohmic is zero-torque and routes to conv_0torque (the eta_e
    # regime); the old P_tot>1MW gate silently deleted every shot whose mean power
    # was Ohmic-dominated, i.e. most of the database.
    heated = df.groupby('shot')['P_tot'].transform('mean').values > 1e5
    t = t & machine_quality(df, y, t, heated)         # drop garbage machines (per-machine)
    q = df['q'].values
    mach = df['machine'].values                       # for per-machine fit weighting

    print(f"{sys.argv[1]}: {len(df):,} rows, {df['shot'].nunique()} shots, "
          f"{df['machine'].nunique()} machines")
    # SHOT-FLOW: where do shots go / get lost? (regime routes per CELL by the
    # instantaneous drive, so a single shot's Ohmic phase and NBI phase land in
    # DIFFERENT regimes -- a shot can appear in two regimes, which is why the
    # per-regime shot counts can sum to > n_shots.)
    print("\nshot-flow per regime  [classified -> plasma(P_tot>0.1MW) -> q>1.5 -> has-onset]:")
    for name in REGIMES:
        rs = df['regime'].values == name
        nh = df.loc[rs & heated, 'shot'].nunique()
        qf = rs & heated & ((q > 1.5) | (name == 'conv_NBI'))
        non = df.loc[qf & t & (y > 0), 'shot'].nunique() if (qf & t).sum() else 0
        print(f"   {name:<14} {df.loc[rs,'shot'].nunique():>4} -> {nh:>4} -> "
              f"{df.loc[qf,'shot'].nunique():>4} -> {non:>4}")
    # NOWCAST (0 ms) and +10 ms FORECAST -- identical features, the label shifted
    # forward in time. The forecast excludes cells already in onset now, so it is a
    # genuine early-warning transition test (not persistence of an ongoing burst).
    regime_table(df, lib, y, t, heated, mach, q, "NOWCAST (0 ms -- onset happening now)")
    yf, tf = forecast_target(df, y, t, 0.010)
    regime_table(df, lib, yf, tf, heated, mach, q, "FORECAST (+10 ms -- onset within 10 ms)")

    print("\nPOOLED-LOSO = fit the law on the whole multi-machine database, hold out")
    print("  each shot (the ITER-scaling-style validation -- does the law predict")
    print("  onset on unseen shots across all machines). LOMO = leave-one-MACHINE-")
    print("  out (frozen coefficients to a never-seen machine -- the stringent test).")
    print("UNIFIED = one drive-blended, geometry-gated equation (Eq 7); close to the")
    print("  per-regime rows means the 3 limbs are ONE continuous law. NOWCAST = is a")
    print("  surface avalanching now; FORECAST = will it within 10 ms (hazard is")
    print("  forward-looking, so the forecast is its native use).")
    # tau_r closure: is the onset critical value ONE universal constant per regime?
    closure_test(df, lib, y, t & heated)
    # split "physics vs a few bad machines" (nowcast labels)
    per_machine_diag(df, lib, y, t, heated)


if __name__ == '__main__':
    main()
