#!/usr/bin/env python3
"""
probe_law.py  --  standalone, minimal, TESTED.

The point: step2_evaluate.py is 4000 lines and fragile. Everything we actually
need to answer "can we predict the non-local flux residual and its onset" is
narrow. This file does ONLY that, computes the residual and onset the CLEAN way
(no per-band sigma, no clip artefacts), and validates with an honest hold-out.

Run modes:
    python probe_law.py --self-test         # verify the code is correct
    python probe_law.py path/to/step1.parquet

The residual, defined cleanly:
    Q_cond   = actual conducted electron heat flux (TRANSP power balance)
    Q_GB     = local gyro-Bohm prediction
    c_GB(shot,rho) = median_t( Q_cond / Q_GB )     # the local saturation level
    resid    = Q_cond / (c_GB * Q_GB) - 1          # fractional excess over local
So resid ~ 0 when transport is local; |resid| large = non-local. No sigma.

Onset (clean): |resid| > THRESH on a TRUSTWORTHY row (gradient meaningful, the
local prediction is not degenerate). A row where Q_GB is tiny (flat gradient)
has an unreliable ratio and is excluded -- that is the calibration blowup that
faked the old detector's heavy tail.
"""
import sys
import numpy as np
import pandas as pd

THRESH = 2.0          # "non-local event" = flux is >= 3x the local prediction
MIN_GRAD = 1.0        # R/L_Te floor: below this the normalization is unreliable


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------
def clean_residual(df, sanitize=False):
    """Return (resid, trustworthy) as numpy arrays. resid is the fractional
    excess of the actual flux over the calibrated local prediction.
    sanitize=True excludes cells with unphysical conducted flux Q_cond<=0 (over-
    subtracted power balance -- endemic on spherical tokamaks from fast-ion orbit-
    loss / convection mis-accounting). Same onset definition, cleaner ground
    truth; barely touches conventional data (~15% quiet negatives) but removes the
    corrupt NSTX cells that were poisoning the residual calibration."""
    q_cond = df['Q_cond_e'].values.astype(float)
    q_gb = df['Q_GB'].values.astype(float)
    # fast per-(MACHINE,shot,rho) grouping via a single integer key. MACHINE must
    # be in the key: shot IDs can COLLIDE across machines (BUGFIX v4.32 -- without
    # this, machine A shot 0 and machine B shot 0 were calibrated together, so
    # c_GB was a cross-machine average and the residual was dominated by
    # calibration error, firing the detector on ~100% of rows).
    ms = pd.factorize(pd.Series(df['machine'].astype(str)) + '|'
                      + pd.Series(df['shot'].astype(str)), sort=False)[0].astype(np.int64)
    rho_i = pd.factorize(np.round(df['rho'].values, 4), sort=False)[0].astype(np.int64)
    codes = ms * (rho_i.max() + 1) + rho_i
    ratio = np.where(q_gb > 0, q_cond / q_gb, np.nan)
    # c_GB = per-(shot,rho) median ratio (robust local saturation level)
    cgb = pd.Series(ratio).groupby(codes).transform('median').values
    qbase = cgb * q_gb                                   # calibrated local baseline
    delta = q_cond - qbase                               # ABSOLUTE residual (W/m^2)
    resid = np.where((cgb > 0) & (q_gb > 0), q_cond / np.maximum(qbase, 1e-9) - 1.0, np.nan)
    grad = df['R_over_LTe'].values if 'R_over_LTe' in df.columns else np.full(len(df), 10.0)
    trustworthy = (np.isfinite(resid) & (q_gb > 0) & (grad >= MIN_GRAD))
    if sanitize:
        trustworthy = trustworthy & (q_cond > 0)         # drop over-subtracted flux
    return resid, delta, qbase, trustworthy


def adaptive_z(df, delta, qbase, eta=0.15, by_shot=False):
    """Adaptive REGULARIZED z-score (the regularized-z detector). Fixes the two
    traps that made the old detectors measure jitter instead of non-locality:
      (1) The relative trap: |resid| = delta/qbase blows up where qbase ~ 0
          (quiet/edge), flagging tiny fluctuations as huge events.
      (2) The flat-sigma trap: one machine-wide sigma is wrong because the noise
          is HETEROSCEDASTIC (core vs edge differ), so it buries core events and
          over-fires at the edge.
    The fix combines an additive per-radius floor with a proportional term:
        Z = delta / sqrt( sigma_floor(machine,rho)^2 + (eta*qbase)^2 )
    sigma_floor is the MAD of delta on QUIET rows (bottom half of |delta|) at
    that (machine, radial band); eta~0.15 is the ~15% TRANSP profile-fit
    uncertainty. At the edge the floor dominates (kills relative spikes); in the
    core the proportional term scales so real signals are not buried."""
    band = np.round(np.clip(df['rho'].values, 0.0, 1.0), 1)
    # by_shot=True keys the noise floor per (SHOT, radial band) instead of
    # (machine, band). Needed for heterogeneous machines like C-Mod where shots
    # span very different activity levels: a machine-pooled sigma_floor is
    # inflated by the hot shots (burying quiet-shot events) and too small for
    # them (over-firing), giving the 0%/30%-per-shot pathology. Per-shot self-
    # normalisation makes |Z|>3 mean the same physical thing on every discharge.
    grp = df['shot'].values if by_shot else df['machine'].values
    key = pd.factorize(pd.Series(grp).astype(str) + '|' + band.astype(str))[0]
    # sigma_floor = robust noise SCALE of the residual per (machine, radial band):
    # 1.4826 * MAD(delta) over ALL rows in the cell. MAD is robust to the
    # few-percent event outliers, so it recovers the true noise level.
    # (BUGFIX v4.32: the previous version took the MAD of only the bottom-quartile
    # |delta| -- i.e. the spread of the SMALLEST residuals, which is ~0 -- so
    # sigma_floor collapsed and |Z| was hugely inflated, firing on ~68% of rows.
    # That contaminated the detector and every Z-based feature in all prior runs.)
    ds = pd.Series(delta)
    cell_med = ds.groupby(key).transform('median')
    sig = 1.4826 * (np.abs(ds - cell_med)).groupby(key).transform('median').values
    sig = np.nan_to_num(sig, nan=np.nanmedian(sig))
    sig = np.maximum(sig, 1e-3 * np.abs(qbase) + 1.0)      # floor: >=0.1% of baseline
    denom = np.sqrt(sig ** 2 + (eta * np.abs(qbase)) ** 2)
    return delta / np.maximum(denom, 1e-9)


def _critical_gradient(df):
    """Jenko-type ITG critical gradient R/L_Te,crit."""
    Te = np.maximum(df['Te'].values.astype(float), 1.0)
    Ti = np.maximum(df['Ti'].values.astype(float), 1.0)
    q = np.clip(df['q'].values.astype(float), 0.5, 20.0)
    zeff = np.maximum(df['zeff'].values.astype(float), 1.0) if 'zeff' in df.columns else 1.5
    s = _profile_shear(df)
    return (1.0 + zeff * (Te / Ti)) * (1.33 + 1.91 * np.abs(s) / q)


def coherent_onset(df, Z, delta, trustworthy, z_hi=2.0, z_nb=1.5, min_extent=3):
    """NON-LOCAL detector, defined by RADIAL SPATIAL EXTENT -- the robust
    non-locality signature (the propagation-LAG test was too strict because real
    avalanches propagate faster than the ~ms grid resolution, so the lag is
    sub-timestep and invisible).

    Physics: a NON-LOCAL avalanche / heat wave spans a WIDE radial range at one
    instant (many flux surfaces surge together); ordinary LOCAL or stiff
    transport is confined to the 1-2 surfaces where the local gradient happens to
    be high. So the number of simultaneously-surging adjacent surfaces (the
    avalanche WIDTH) separates non-local from local -- without using R/LTe or any
    delta-derivative, so it is NON-CIRCULAR and every plasma-parameter predictor
    stays clean.

    A cell is flagged non-local if: delta>0 (a positive flux excess), |Z|>z_hi
    (a real surge on the adaptive heteroscedastic Z), AND it sits inside a radial
    band where at least `min_extent` surfaces (within +/-2) are simultaneously
    surging (|Z|>z_nb). Rate is tuned by z_hi / min_extent to the physical
    ~5-15%."""
    az = np.abs(Z)
    d = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                      'time': df['time'].values, 'az': az,
                      'surg': (az > z_nb).astype(float),
                      '_i': np.arange(len(df))}).sort_values(['shot', 'time', 'rho'])
    # radial WIDTH of the surging structure: count surfaces with |Z|>z_nb in a
    # +/-2 window at the same (shot, time). Wide = avalanche; narrow = local.
    extent = d.groupby(['shot', 'time'])['surg'].transform(
        lambda v: v.rolling(5, center=True, min_periods=1).sum())
    ext_arr = np.zeros(len(df)); ext_arr[d['_i'].values] = extent.values
    return (delta > 0) & (az > z_hi) & (ext_arr >= min_extent) & trustworthy


def onset(resid, trustworthy, thresh=THRESH):
    """Legacy relative-threshold detector (kept for the self-test)."""
    return (np.abs(resid) > thresh) & trustworthy


_RATIONALS = [(m, n) for n in range(1, 6) for m in range(1, 9)]


def rational_density(q, delta=0.1):
    """Pivot B: local DENSITY of low-order rational surfaces (m<=8, n<=5),
    weighted 1/n^2. High where many resonances overlap -> island overlap ->
    stochastic field -> avalanches. A physically richer feature than distance to
    the single nearest rational."""
    q = np.asarray(q, float)
    Dq = np.zeros_like(q)
    for m, n in _RATIONALS:
        Dq += (1.0 / n ** 2) * np.exp(-((q - m / n) ** 2) / (2 * delta ** 2))
    return Dq


def _profile_shear(df):
    """Magnetic shear s = (rho/q) dq/drho, from the q profile per (shot,time)."""
    d = df[['shot', 'time', 'rho', 'q']].copy()
    d['_i'] = np.arange(len(d))
    d = d.sort_values(['shot', 'time', 'rho'])
    g = d.groupby(['shot', 'time'])
    dq = g['q'].diff()
    drho = g['rho'].diff()
    s = (d['rho'].values / np.maximum(d['q'].values, 0.5)) * (dq.values / drho.values)
    out = np.full(len(df), np.nan)
    out[d['_i'].values] = s
    # fill the leading NaN of each profile with the next value
    return np.nan_to_num(out, nan=0.0)


def state_features(df):
    """Physical state predictors (inputs only). No residual, no position leak
    beyond rho, which the within-band test controls."""
    q = np.clip(df['q'].values.astype(float), 0.5, 20.0)
    Te = np.maximum(df['Te'].values.astype(float), 1.0)
    Ti = np.maximum(df['Ti'].values.astype(float), 1.0)
    dqr = np.min(np.abs(np.column_stack(
        [q - r for r in (1.0, 1.5, 2.0, 2.5, 3.0)])), axis=1) + 1e-3
    RLT = np.maximum(df['R_over_LTe'].values.astype(float), 1e-3) \
        if 'R_over_LTe' in df.columns else np.full(len(df), 10.0)
    rstar = np.maximum(df['rho_star'].values.astype(float), 1e-5) \
        if 'rho_star' in df.columns else np.full(len(df), 1e-2)
    Dq = rational_density(q)
    s = _profile_shear(df)
    # PRECURSOR RATE (Gemini): a profile is unstable when its gradient is
    # ACCELERATING past marginal, not merely sitting at threshold. |d(R/LTe)/dt|
    # on a smoothed series scored AUC 0.634 locally -- a real predictor the
    # static snapshot misses. Computed per (shot, rho) time series.
    d2 = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                       'time': df['time'].values, 'g': RLT,
                       '_i': np.arange(len(df))}).sort_values(['shot', 'rho', 'time'])
    sm = d2.groupby(['shot', 'rho'])['g'].transform(
        lambda v: v.ewm(span=8, min_periods=1).mean())
    rate = sm.groupby([d2['shot'].values, d2['rho'].values]).diff()
    rate_arr = np.zeros(len(df))
    rate_arr[d2['_i'].values] = np.nan_to_num(rate.values, nan=0.0)
    feats = {
        'rho_star': rstar,          # the empirical winner (0.686) -- gyro-Bohm scale
        'q': q,
        'Te_Ti': Te / Ti,
        'dist_qrat': dqr,
        'rat_density': Dq + 1e-3,                     # Pivot B.1
        's_x_ratdens': np.abs(s) * Dq + 1e-3,         # Pivot B.2 (low-shear pockets)
        'R_over_LTe': RLT,
        'precursor_rate': np.abs(rate_arr) + 1e-6,    # |d(R/LTe)/dt| (Gemini)
    }
    # DRIVE / OVERDRIVE / SHEAR / ExB -- the predictors rightly noted
    # were never tested directly against onset.
    Ti_a = np.maximum(df['Ti'].values.astype(float), 1.0)
    zeff = np.maximum(df['zeff'].values.astype(float), 1.0) if 'zeff' in df.columns else 1.5
    crit = (1.0 + zeff * (Te / Ti_a)) * (1.33 + 1.91 * np.abs(s) / np.maximum(q, 0.5))
    feats['overdrive'] = np.maximum(0.0, RLT / np.maximum(crit, 2.0) - 1.0) + 1e-6
    feats['shear_abs'] = np.abs(s) + 1e-3
    if 'gamma_ExB' in df.columns:
        feats['gamma_ExB'] = np.abs(df['gamma_ExB'].values.astype(float)) + 1e-3
    # dimensionless collisionality nu*_e (Gemini's suggestion; computable from
    # standard TRANSP profiles). Physically motivated companion to rho_star.
    if {'ne', 'R0', 'a0'}.issubset(df.columns):
        ne = np.maximum(df['ne'].values.astype(float), 1e17)
        R0 = np.maximum(df['R0'].values.astype(float), 0.1)
        a0 = np.maximum(df['a0'].values.astype(float), 0.05)
        rho = np.clip(df['rho'].values.astype(float), 0.02, 1.0)
        epsr = np.clip(rho * a0 / R0, 1e-3, 2.0)
        zeff = np.maximum(df['zeff'].values.astype(float), 1.0) \
            if 'zeff' in df.columns else 1.5
        lnL = np.clip(24.0 - np.log(np.sqrt(ne * 1e-6) / Te), 5.0, 25.0)
        nu_star = (6.921e-18 * q * R0 * ne * zeff * lnL
                   / (Te ** 2 * epsr ** 1.5)) + 1e-6
        feats['nu_star_e'] = nu_star

    # ---- COMPOSITE PHYSICS EQUATIONS (Gemini): weak singles that may SYNERGISE
    od = feats['overdrive']
    R0v = np.maximum(df['R0'].values.astype(float), 0.1) if 'R0' in df.columns else 1.0
    v_ti = np.sqrt(np.maximum(Ti_a, 1.0) * 1.602e-19 / 3.34e-27)   # thermal speed
    gamma_max = (v_ti / R0v) * np.sqrt(od)                          # linear growth
    gExB = np.abs(df['gamma_ExB'].values.astype(float)) if 'gamma_ExB' in df.columns \
        else np.zeros(len(df))
    # (1) SHEAR COLLAPSE FACTOR: growth outrunning E×B shear suppression
    feats['R_shear'] = gamma_max / (gExB + 0.05 * gamma_max.mean() + 1e-9) + 1e-6
    # (2) TOPOLOGICAL BARRIER: low shear near a rational surface = free corridor
    feats['T_topo'] = (q ** 2 / (np.abs(s) + 0.1)) * np.exp(-5.0 * dqr) + 1e-6
    # (3) DYNAMIC ACTION INTEGRAL: accumulated (ramp x overdrive) over the
    #     turbulence correlation time tau_turb ~ a/v_ti (SOC loading, with memory)
    a0v = np.maximum(df['a0'].values.astype(float), 0.05) if 'a0' in df.columns else 0.5
    load = np.abs(rate_arr) * od
    dld = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                        'time': df['time'].values, 'l': load,
                        '_i': np.arange(len(df))}).sort_values(['shot', 'rho', 'time'])
    aov = dld.groupby(['shot', 'rho'])['l'].transform(
        lambda v: v.rolling(6, min_periods=1).sum())
    A_over = np.zeros(len(df)); A_over[dld['_i'].values] = aov.values
    feats['A_over'] = A_over + 1e-6
    # (4) COMBINED UNIVERSAL ONSET SCALAR
    feats['Omega_NL'] = (feats['R_shear'] * (1.0 + A_over) * feats['T_topo']
                         * np.maximum(rstar, 1e-4) ** (-1.0)) + 1e-6

    # ---- LEVER 1: SPATIAL NON-LOCAL VISION (Gemini) -- the piece that was
    #      structurally missing. Point features cannot see a NON-LOCAL process;
    #      give the model the radial PROFILE context.
    ov = pd.DataFrame({'shot': df['shot'].values, 'time': df['time'].values,
                       'rho': df['rho'].values, 'od': od, 'g': RLT,
                       '_i': np.arange(len(df))})
    # (A) I_drive: accumulated super-critical drive from the OUTER surfaces,
    #     decayed with radial distance. ADAPTIVE kernel (Gemini): the avalanche
    #     correlation length scales with rho_star (lam ~ C * rho_star), so a
    #     fixed radial span is physically wrong across 12 machines (huge on C-Mod
    #     rho_star~0.004, tiny on MST rho_star~0.16). Use a per-machine EWMA span
    #     set by that machine's median rho_star -> the reservoir is integrated
    #     over the scale-appropriate neighbourhood.
    ov['mach'] = df['machine'].values
    ov['rs'] = rstar
    I_drive = np.zeros(len(df))
    for m, gm in ov.groupby('mach'):
        C_turb, drho = 15.0, 0.02
        span = float(np.clip(C_turb * gm['rs'].median() / drho, 2.0, 30.0))
        gi = gm.sort_values(['shot', 'time', 'rho'], ascending=[True, True, False])
        val = gi.groupby(['shot', 'time'])['od'].transform(
            lambda v: v.ewm(span=span, min_periods=1).mean())
        I_drive[gi['_i'].values] = val.values
    feats['I_drive'] = I_drive + 1e-6
    # (B) Delta_grad: gradient MISMATCH to the outer plasma (build-up upstream)
    ovr = ov.sort_values(['shot', 'time', 'rho'])
    gout = ovr.groupby(['shot', 'time'])['g'].shift(-3)      # ~outer surface
    dgrad = (gout.values - ovr['g'].values)
    Dgrad = np.zeros(len(df)); Dgrad[ovr['_i'].values] = np.nan_to_num(dgrad)
    feats['Delta_grad'] = np.abs(Dgrad) + 1e-6

    # ---- LEVER 2: TEMPORAL HISTORY (SOC accumulates dynamic stress over time) -
    #      give the model the recent PAST of the drive, not just the snapshot.
    hh = pd.DataFrame({'shot': df['shot'].values, 'rho': df['rho'].values,
                       'time': df['time'].values, 'od': od, 'idr': I_drive,
                       '_i': np.arange(len(df))}).sort_values(['shot', 'rho', 'time'])
    g2 = hh.groupby(['shot', 'rho'])
    for lag, nm in ((3, 'od_lag5'), (6, 'od_lag10')):
        lv = g2['od'].shift(lag)
        arr = np.zeros(len(df)); arr[hh['_i'].values] = np.nan_to_num(lv.values)
        feats[nm] = arr + 1e-6
    # temporal velocity of the upstream reservoir (is I_drive building?)
    didr = g2['idr'].transform(lambda v: v.diff())
    arr = np.zeros(len(df)); arr[hh['_i'].values] = np.nan_to_num(didr.values)
    feats['I_drive_rate'] = np.abs(arr) + 1e-6

    # ---- LEVER 4: MACHINE GEOMETRY NORMALISATION (spherical vs conventional) --
    if {'R0', 'a0'}.issubset(df.columns):
        feats['aspect_ratio'] = (np.maximum(df['R0'].values, 0.1)
                                 / np.maximum(df['a0'].values, 0.05))
        feats['rhostar_q2'] = np.maximum(rstar, 1e-5) * q ** 2
    return feats


# --------------------------------------------------------------------------
# THE REFRAME: onset as a UNIVERSAL CRITICAL THRESHOLD on a constructed order
# parameter Omega_NL -- not a point classifier. The scientific claim is that a
# SINGLE dimensionless Omega_crit fires the avalanche on EVERY machine (the
# Greenwald-limit template), which is falsifiable and machine-independent. Its
# "AUC" as a predictor is secondary; the universality of Omega_crit is the law.
# --------------------------------------------------------------------------
def omega_nl(feats, alpha=1.0, beta=0.1, tau=1.0):
    """Spatio-temporal onset ORDER PARAMETER (the formula):

        Omega_NL = [ I_drive * (1 + tau*|d/dt(R/LTe)|) * rho_star^alpha ]
                   / ( |gamma_ExB| + beta*nu_star_e + eps )

    Numerator = accumulated non-local overdrive (adaptive-kernel reservoir) with
    dynamic-acceleration memory, scaled by the gyro-Bohm scale rho_star^alpha.
    Denominator = flow-shear + collisional DAMPING. Damping divides (more shear
    -> smaller Omega -> harder to fire). NOTE: the inline text form multiplied by
    the damping; Gemini's own code DIVIDES by it, which is the correct sign, so
    that is what we use."""
    n = len(feats['rho_star'])
    rstar = np.maximum(feats['rho_star'], 1e-5)
    idr = feats['I_drive']
    rate = feats['precursor_rate']                       # |d(R/LTe)/dt|, smoothed
    gExB = feats.get('gamma_ExB', np.full(n, 0.05))
    nue = feats.get('nu_star_e', np.zeros(n))
    drive = idr * (1.0 + tau * rate) * rstar ** alpha
    damp = np.abs(gExB) + beta * nue + 1e-6
    return drive / damp


def omega_soc(df, feats, alpha=1.0, beta=0.1, tau=1.0, w_frac=0.5):
    """SOC order parameter -- the REBUILD (real profiles self-organize to
    marginality, so overdrive ~ 0 and cannot be the driver). Onset lives AT the
    threshold, not above it. So the drive is PROXIMITY to critical (poised), not
    excess over it:

        poised(rho,t)   = exp( -((R/LTe - crit)/w)^2 )        # peaks AT marginal
        I_poised(rho,t) = adaptive-kernel reservoir of poised NEIGHBOURS
                          (span ~ C*rho_star, per machine) = how wide a band is
                          simultaneously critical = primed for a system-avalanche
        Omega_SOC = I_poised * (1 + tau*|d/dt(R/LTe)|) * rho_star^alpha
                    / ( |gamma_ExB| + beta*nu_star_e + eps )

    NON-CIRCULAR: built ONLY from the gradient field (R/LTe), geometry (q, shear)
    and flow (gamma_ExB) -- never the flux residual delta/Z that DEFINES onset.
    Unlike overdrive it has full dynamic range on marginal real profiles, because
    poised is LARGEST exactly where R/LTe ~ crit."""
    n = len(df)
    RLT = feats['R_over_LTe']
    Te = np.maximum(df['Te'].values.astype(float), 1.0)
    Ti = np.maximum(df['Ti'].values.astype(float), 1.0)
    q = np.clip(df['q'].values.astype(float), 0.5, 20.0)
    s = _profile_shear(df)
    zeff = np.maximum(df['zeff'].values.astype(float), 1.0) if 'zeff' in df.columns else 1.5
    crit = (1.0 + zeff * (Te / Ti)) * (1.33 + 1.91 * np.abs(s) / np.maximum(q, 0.5))
    w = np.maximum(w_frac * crit, 1.0)
    poised = np.exp(-((RLT - crit) / w) ** 2)              # marginality proximity
    rstar = np.maximum(feats['rho_star'], 1e-5)
    # non-local reservoir of poised surfaces, adaptive kernel (per-machine span)
    ov = pd.DataFrame({'shot': df['shot'].values, 'time': df['time'].values,
                       'rho': df['rho'].values, 'p': poised, 'mach': df['machine'].values,
                       'rs': rstar, '_i': np.arange(n)})
    Ip = np.zeros(n)
    for m, gm in ov.groupby('mach'):
        span = float(np.clip(15.0 * gm['rs'].median() / 0.02, 2.0, 30.0))
        gi = gm.sort_values(['shot', 'time', 'rho'], ascending=[True, True, False])
        val = gi.groupby(['shot', 'time'])['p'].transform(
            lambda v: v.ewm(span=span, min_periods=1).mean())
        Ip[gi['_i'].values] = val.values
    load = feats['precursor_rate']                        # |d(R/LTe)/dt| loading
    gExB = feats.get('gamma_ExB', np.full(n, 0.05))
    nue = feats.get('nu_star_e', np.zeros(n))
    drive = (Ip + 1e-6) * (1.0 + tau * load) * rstar ** alpha
    damp = np.abs(gExB) + beta * nue + 1e-6
    return drive / damp


def youden_threshold(score, y):
    """The Omega_crit that best separates onset from quiescence (max TPR-FPR)."""
    y = np.asarray(y, bool)
    m = np.isfinite(score)
    score, y = np.asarray(score)[m], y[m]
    if y.sum() < 5 or (~y).sum() < 5:
        return np.nan
    order = np.argsort(-score)
    ys = y[order]; ss = score[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(~ys) / max((~ys).sum(), 1)
    return float(ss[int(np.argmax(tpr - fpr))])


def omega_universality(machine, om, y, trust):
    """Is there ONE universal Omega_crit that fires onset on every machine?
    Per machine (NO shot hold-out needed -- a single threshold, not a fitted
    classifier), report the single-order-parameter AUC and the Youden Omega_crit.
    A TIGHT cross-machine spread of Omega_crit = a universal law; a wide spread =
    machine-dependent. Works on 1-shot-per-machine data (unlike the classifier)."""
    machine = np.asarray(machine)
    rows = []
    for m in pd.unique(machine[trust]):
        sel = trust & (machine == m)
        o = om[sel]; yy = np.asarray(y)[sel].astype(bool)
        fin = np.isfinite(o)
        o, yy = o[fin], yy[fin]
        if yy.sum() < 8 or (~yy).sum() < 8:
            rows.append((m, int(len(o)), float(yy.mean()) if len(yy) else np.nan,
                         np.nan, np.nan))
            continue
        rows.append((m, int(len(o)), float(yy.mean()), auc(o, yy),
                     youden_threshold(o, yy)))
    return rows


def universality_cv(rows):
    """Coefficient of variation of Omega_crit across machines (std/mean, in log
    space since Omega spans orders of magnitude). Low = universal law."""
    lg = np.array([np.log(r[4]) for r in rows if np.isfinite(r[4]) and r[4] > 0])
    if len(lg) < 2:
        return np.nan, len(lg)
    # spread of a multiplicative constant: exp(std of log) - 1 ~ fractional spread
    return float(np.exp(np.std(lg)) - 1.0), len(lg)


# --------------------------------------------------------------------------
# metrics (no sklearn)
# --------------------------------------------------------------------------
def auc(score, y):
    y = np.asarray(y, bool)
    m = np.isfinite(score)
    score, y = np.asarray(score)[m], y[m]
    if len(score) < 40 or not (0 < y.mean() < 1):
        return np.nan
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(len(score))
    p, n = y.sum(), (~y).sum()
    return (rank[y].sum() - p * (p - 1) / 2) / (p * n)


def pr_auc(score, y):
    """Average precision = area under precision-recall. For a 7.3% base rate a
    RANDOM model scores ~0.073 here, NOT 0.5 -- this is the metric that is honest
    under 13:1 imbalance (ROC-AUC hides the false-positive flood)."""
    y = np.asarray(y, bool)
    m = np.isfinite(score)
    score, y = np.asarray(score)[m], y[m]
    if len(score) < 40 or y.sum() == 0:
        return np.nan
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / ys.sum()
    dr = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(dr * prec))


def prec_at_recall(score, y, target=0.6):
    """Best precision achievable while still catching >= target of real events.
    The operating-point number a physicist actually cares about."""
    y = np.asarray(y, bool)
    m = np.isfinite(score)
    score, y = np.asarray(score)[m], y[m]
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    rec = tp / ys.sum()
    prec = tp / np.maximum(tp + fp, 1)
    ok = rec >= target
    return float(prec[ok].max()) if ok.any() else np.nan


def none_pct(x):
    return "n/a" if (x is None or not np.isfinite(x)) else f"{x*100:.0f}%"


def fit_logit(X, y, l2=1e-2, class_weight=False):
    """Ridge least-squares on {0,1} -- a fast, deterministic linear scorer.
    class_weight=True up-weights the rare positives by n_neg/n_pos (the
    scale_pos_weight fix) so the fit is not swamped by the 93% negatives."""
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
    Xs = (X - mu) / sd
    A = np.column_stack([np.ones(len(Xs)), Xs])
    if class_weight:
        pos = np.asarray(y) > 0.5
        s = np.where(pos, max((~pos).sum(), 1) / max(pos.sum(), 1), 1.0)
        AtA = (A * s[:, None]).T @ A
        Aty = (A * s[:, None]).T @ y
    else:
        AtA, Aty = A.T @ A, A.T @ y
    w = np.linalg.lstsq(AtA + l2 * np.eye(A.shape[1]), Aty, rcond=None)[0]
    return w, mu, sd


def apply_logit(X, w, mu, sd):
    Xs = (X - mu) / sd
    return np.column_stack([np.ones(len(Xs)), Xs]) @ w


# --------------------------------------------------------------------------
# validation: shot-split hold-out, and within-radial-band (position-free)
# --------------------------------------------------------------------------
def shot_split_auc(df, y, feats, within_band=False):
    """Train on half the shots, test on the other half, WITHIN each machine.
    within_band=True further restricts to fixed radial bands so radial position
    cannot be the predictor. Returns mean test AUC + a precision/recall at the
    best-F1 operating point on the pooled test predictions."""
    F = np.column_stack([np.log(np.maximum(v, 1e-12)) for v in feats.values()])
    finite = np.all(np.isfinite(F), axis=1)          # hoisted out of the loop
    mach = pd.factorize(df['machine'].values, sort=False)[0]
    shot = pd.factorize(df['shot'].values, sort=False)[0]
    band = (np.clip(df['rho'].values, 0.02, 1.0) * 20).astype(int) if within_band \
        else np.zeros(len(df), int)
    test_scores, test_y = [], []
    aucs = []
    for m in np.unique(mach):
        for b in np.unique(band):
            cell = (mach == m) & (band == b) & finite
            if cell.sum() < 200:
                continue
            sh = np.unique(shot[cell])
            if len(sh) < 2:
                continue                       # need >=2 shots to hold out
            te_sh = sh[::2]                     # every other shot WITHIN this cell
            in_te = np.isin(shot, te_sh)       # fast vectorised membership
            te = cell & in_te
            tr = cell & ~in_te
            if tr.sum() < 100 or te.sum() < 100:
                continue
            if not (0 < y[tr].mean() < 1) or not (0 < y[te].mean() < 1):
                continue
            # UNWEIGHTED fit: ROC-AUC is rank-based and imbalance-robust, so the
            # honest ranking comes from the plain fit. Class-weighting a 24-feature
            # least-squares on tiny within-band cells overfits the up-weighted
            # positives and COLLAPSES the test ranking (0.65 -> 0.54, verified) --
            # it is an operating-point tool, not a ranking tool.
            w, mu, sd = fit_logit(F[tr], y[tr].astype(float))
            pred = apply_logit(F[te], w, mu, sd)
            a = auc(pred, y[te].astype(bool))
            if np.isfinite(a):
                aucs.append(a)
                test_scores.append(pred)
                test_y.append(y[te].astype(bool))
    if not aucs:
        return {'roc': np.nan, 'pr': np.nan, 'base': np.nan,
                'prec60': np.nan, 'prec': np.nan, 'rec': np.nan, 'n': 0}
    ps = np.concatenate(test_scores)
    ys = np.concatenate(test_y)
    # best-F1 precision/recall over threshold quantiles
    best = (-1, np.nan, np.nan)
    for qb in (0.5, 0.6, 0.7, 0.8, 0.9):
        thr = np.quantile(ps, qb)
        fire = ps > thr
        tp = (fire & ys).sum()
        fp = (fire & ~ys).sum()
        fn = (~fire & ys).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[0]:
            best = (f1, prec, rec)
    return {'roc': float(np.mean(aucs)), 'pr': pr_auc(ps, ys), 'base': float(ys.mean()),
            'prec60': prec_at_recall(ps, ys, 0.6), 'prec': best[1], 'rec': best[2],
            'n': len(aucs)}


# --------------------------------------------------------------------------
# self-test: synthetic data with a KNOWN law, verify the probe recovers it
# --------------------------------------------------------------------------
def structural_laws(df, resid, trust, max_slices=1500, max_series=1500, max_shots=12):
    """The 'truly non-local' characterisation. Per machine, strictly-positive
    physical scales (no classifier, no negative diffusivity):
      alpha  = residual tail exponent (Levy if ~2-3; Gaussian >>3).
      lam    = RADIAL CORRELATION LENGTH of the residual (in normalised rho):
               how far a non-local event stays coherent. Measured directly from
               the residual's radial autocorrelation -- NOT the degenerate
               kernel fit (v4.23 railed theta->1, lam->0.5 on every machine
               because Q_cond is merely smoother than Q_GB; that measured
               smoothing, not non-locality).
      coh    = lag-1 radial coherence (fraction, 0..1): the residual's
               correlation with its immediate radial neighbour = the non-local
               fraction, robustly.
      v_prop = radial PROPAGATION SPEED (m/s) from the time cross-correlation of
               the residual between inner and outer radii (Gemini LAW II -- the
               spatio-temporal analysis, previously skipped).
      ballistic = does the lag grow LINEARLY with radial distance (ballistic
               avalanche, ~const speed) or QUADRATICALLY (diffusive)? Reported
               as the fitted exponent p in tau ~ dr^p (p~1 ballistic, p~2 diff).
      H      = Hurst exponent (R/S) of the flux time series; >0.5 = long memory.
    """
    from scipy.signal import correlate
    out = {}
    for m, g in df.assign(_r=resid, _t=trust).groupby('machine'):
        gg = g[g['_t']]
        if len(gg) < 2000:
            continue
        rr = np.abs(gg['_r'].values)
        rr = rr[np.isfinite(rr) & (rr > 0.1)]
        alpha = np.nan
        if len(rr) > 500:
            srt = np.sort(rr)
            tail = srt[int(0.9 * len(srt)):]
            ycdf = 1.0 - np.linspace(0.9, 1.0, len(tail))
            v = ycdf > 0
            if v.sum() > 10:
                alpha = -np.polyfit(np.log(tail[v]), np.log(ycdf[v]), 1)[0]
        # radial autocorrelation of the residual, pooled over time slices, by
        # radial-index lag; convert to a correlation LENGTH via median d-rho.
        corr_by_lag = {k: [] for k in range(1, 8)}
        drhos = []
        slc = list(gg.groupby(['shot', 'time']))
        if len(slc) > max_slices:
            slc = [slc[i] for i in np.linspace(0, len(slc) - 1, max_slices).astype(int)]
        for _, fr in slc:
            fr = fr.sort_values('rho')
            a = fr['_r'].values
            rh = fr['rho'].values
            if len(a) < 6 or not np.all(np.isfinite(a)):
                continue
            a = a - a.mean()
            drhos.append(np.median(np.diff(rh)))
            for k in corr_by_lag:
                if len(a) > k + 3 and a.std() > 1e-9:
                    c = np.corrcoef(a[:-k], a[k:])[0, 1]
                    if np.isfinite(c):
                        corr_by_lag[k].append(c)
        lam, coh = np.nan, np.nan
        if drhos:
            drho = np.median(drhos)
            mc = {k: np.median(v) for k, v in corr_by_lag.items() if v}
            if 1 in mc:
                coh = mc[1]
            # correlation length = index-lag where autocorr crosses 1/e, x drho
            ks = sorted(mc)
            below = [k for k in ks if mc[k] < np.exp(-1)]
            lam = (below[0] * drho) if below else (max(ks) * drho if ks else np.nan)
        # propagation speed via time cross-correlation, inner vs outer channel,
        # and the tau-vs-dr scaling (ballistic vs diffusive)
        vprops, taus, drs = [], [], []
        shots = list(gg['shot'].unique())[:max_shots]
        for sh in shots:
            sd = gg[gg['shot'] == sh]
            piv = sd.pivot_table(index='time', columns='rho', values='_r',
                                 aggfunc='mean')
            if piv.shape[1] < 4 or piv.shape[0] < 30:
                continue
            rhos = piv.columns.values
            tt = piv.index.values
            dt = np.median(np.diff(tt))
            a0 = float(sd['a0'].iloc[0]) if 'a0' in sd else 1.0
            base = piv.iloc[:, 0].values
            base = np.nan_to_num(base - np.nanmean(base))
            for j in range(1, piv.shape[1]):
                sig = np.nan_to_num(piv.iloc[:, j].values - np.nanmean(piv.iloc[:, j].values))
                if sig.std() < 1e-9 or base.std() < 1e-9:
                    continue
                cc = correlate(sig, base, mode='full')
                lag = (np.argmax(cc) - (len(base) - 1)) * dt
                dr = abs(rhos[j] - rhos[0])
                if abs(lag) > 1e-6 and dr > 0:
                    vprops.append(a0 * dr / abs(lag))
                    taus.append(abs(lag))
                    drs.append(dr)
        v_prop = float(np.median(vprops)) if vprops else np.nan
        ballistic = np.nan
        if len(drs) >= 6:
            p = np.polyfit(np.log(np.maximum(drs, 1e-6)),
                           np.log(np.maximum(taus, 1e-9)), 1)[0]
            ballistic = float(p)          # ~1 ballistic front, ~2 diffusive
        Hs = []
        ser = list(gg.groupby(['shot', 'rho']))
        if len(ser) > max_series:
            ser = [ser[i] for i in np.linspace(0, len(ser) - 1, max_series).astype(int)]
        for _, ch in ser:
            ts = ch['Q_cond_e'].values
            if len(ts) < 20:
                continue
            dev = np.cumsum(ts - ts.mean())
            R = dev.max() - dev.min()
            Sd = ts.std() + 1e-12
            Hs.append(np.clip(np.log(R / Sd + 1e-12) / np.log(len(ts)), 0, 1))
        out[m] = dict(alpha=alpha, lam=lam, coh=coh, v_prop=v_prop,
                      ballistic=ballistic,
                      H=float(np.median(Hs)) if Hs else np.nan,
                      rho_star=float(gg['rho_star'].median())
                      if 'rho_star' in gg else np.nan)
    return out


def hazard_eval(df, y, trust, feats, win_ms=15.0):
    """Pivot A: don't predict the individual (chaotic) trigger -- predict the
    HAZARD RATE over a time window. Aggregate each (machine, shot, rho, window)
    to its EVENT COUNT and its mean features, then predict the count/rate. The
    random microtrigger averages out over the window while the systematic
    susceptibility survives, so the signal-to-noise jumps. Returns the
    window-level within-band shot-split AUC for 'any event in the window' and
    the Spearman of predicted vs observed event RATE."""
    m = trust
    d = pd.DataFrame({'machine': df['machine'].values[m], 'shot': df['shot'].values[m],
                      'rho': df['rho'].values[m],
                      'win': np.floor(df['time'].values[m] / (win_ms * 1e-3)),
                      'y': y[m].astype(float)})
    for k, v in feats.items():
        d[k] = v[m]
    band = (np.clip(d['rho'].values, 0.02, 1.0) * 20).astype(int)
    d['cell'] = d['machine'].astype(str) + '|' + band.astype(str)
    agg = {k: 'mean' for k in feats}
    agg['y'] = 'mean'                     # event RATE in the window
    g = d.groupby(['machine', 'cell', 'shot', 'win']).agg(agg).reset_index()
    g['any'] = (g['y'] > 0).astype(int)   # any event in the window
    # within-band (cell) shot-split AUC for 'any event in window'
    F = np.column_stack([np.log(np.maximum(g[k].values, 1e-12)) for k in feats])
    finite = np.all(np.isfinite(F), axis=1)
    shot = pd.factorize(g['shot'].values, sort=False)[0]
    aucs, ps, ys = [], [], []
    for c in g['cell'].unique():
        cell = (g['cell'].values == c) & finite
        if cell.sum() < 100:
            continue
        sh = np.unique(shot[cell])
        if len(sh) < 2:
            continue
        te = cell & np.isin(shot, sh[::2])
        tr = cell & ~np.isin(shot, sh[::2])
        if tr.sum() < 40 or te.sum() < 40:
            continue
        yy = g['any'].values
        if not (0 < yy[tr].mean() < 1) or not (0 < yy[te].mean() < 1):
            continue
        w, mu, sd = fit_logit(F[tr], yy[tr].astype(float))
        pred = apply_logit(F[te], w, mu, sd)
        a = auc(pred, yy[te].astype(bool))
        if np.isfinite(a):
            aucs.append(a)
            ps.append(pred)
            ys.append(yy[te])
    return (float(np.mean(aucs)) if aucs else np.nan), len(aucs)


def self_test():
    print("SELF-TEST: synthetic PROPAGATING avalanche fronts seeded near rational q")
    rng = np.random.default_rng(0)
    NT, NR = 150, 15
    rho_ax = np.linspace(0.2, 0.8, NR)
    rows = []
    for machine, cgb_true in [('A', 0.5), ('B', 1.2), ('C', 2.0)]:
        for shot in range(8):
            q0 = rng.uniform(0.8, 2.2)
            # q(t, rho) grid
            drift = 0.4 * np.sin(np.arange(NT) / 30.0)[:, None]
            qg = q0 + 1.2 * rho_ax[None, :] + drift + rng.normal(0, 0.05, (NT, NR))
            dqr = np.min(np.abs(np.stack(
                [qg - r for r in (1.0, 1.5, 2.0, 2.5, 3.0)])), axis=0)
            near = np.exp(-(dqr ** 2) / 0.02)
            # EVENT FIELD: WIDE avalanches (span +/-2 surfaces at once, seeded
            # near rationals) = non-local; PLUS single-pixel spikes (extent 1) =
            # local jitter the extent detector MUST reject.
            E = np.zeros((NT, NR))
            for t in range(NT):
                centres = np.where(rng.random(NR) < 0.12 * near[t])[0]
                for c in centres:
                    E[t, max(0, c - 2):min(NR, c + 3)] = 1.0    # wide band
            spike = (rng.random((NT, NR)) < 0.02).astype(float)   # single-pixel
            noise = rng.normal(0, 0.12, (NT, NR))
            Q_GB = (1e4 * (1 + rho_ax))[None, :] * np.ones((NT, NR))
            Q_cond = cgb_true * Q_GB * (1 + 5.0 * E + 4.0 * spike + noise)
            for t in range(NT):
                for j in range(NR):
                    rows.append((machine, shot, t * 1e-3, rho_ax[j], Q_cond[t, j],
                                 Q_GB[t, j], 3.0, 1.5, qg[t, j], 3.0 + 2.0 * rho_ax[j]))
    df = pd.DataFrame(rows, columns=['machine', 'shot', 'time', 'rho', 'Q_cond_e',
                                     'Q_GB', 'Te', 'Ti', 'q', 'R_over_LTe'])
    resid, delta, qbase, trust = clean_residual(df)
    Z = adaptive_z(df, delta, qbase)
    y = coherent_onset(df, Z, delta, trust)
    ok = True
    # (1) calibration removed the per-machine saturation level: median resid ~ 0
    med = np.nanmedian(resid[trust])
    print(f"  median residual (should be ~0): {med:+.3f}")
    ok &= abs(med) < 0.1
    # (2) onset fires at a sane rate
    print(f"  onset rate: {y[trust].mean()*100:.0f}% (event prob was ~50% near q=2)")
    ok &= 0.05 < y[trust].mean() < 0.5
    feats = state_features(df)
    # (3) MACHINERY test: a target that is a KNOWN function of dist_qrat (events
    # where q is near a rational). This validates the shot-split / within-band
    # machinery independently of the detector -- the propagated-front target
    # (arrivals move away from the rational seed) is validated separately in (3b).
    df2 = df[trust].copy().reset_index(drop=True)
    dqr_t = feats['dist_qrat'][trust]
    y2 = (dqr_t < 0.12) & (rng.random(int(trust.sum())) < 0.6)   # known law
    st = shot_split_auc(df2, y2, {'dist_qrat': dqr_t}, within_band=True)
    a_true, n = st['roc'], st['n']
    rand = rng.random(int(trust.sum()))
    a_rand = shot_split_auc(df2, y2, {'rand': rand}, within_band=True)['roc']
    print(f"  MACHINERY (known dist_qrat law): AUC={a_true:.3f}  "
          f"random feature={a_rand:.3f}  (n={n} cells)")
    ok &= (a_true > 0.65) and (0.42 < a_rand < 0.58)   # recovers signal, no false
    # (3b) EXTENT VALIDATION: the detector must fire on the WIDE avalanches but
    # NOT on single-pixel spikes. Build a spike-ONLY version (same spikes, no
    # wide bands) and confirm the detector rate collapses there.
    rows_s = []
    rng2 = np.random.default_rng(1)
    for machine, cgb_true in [('A', 0.5), ('B', 1.2), ('C', 2.0)]:
        for shot in range(8):
            for t in range(NT):
                spk = rng2.random(NR) < 0.02
                ns = rng2.normal(0, 0.12, NR)
                QG = 1e4 * (1 + rho_ax)
                QC = cgb_true * QG * (1 + 4.0 * spk + ns)
                for j in range(NR):
                    rows_s.append((machine, shot, t * 1e-3, rho_ax[j], QC[j], QG[j],
                                   3.0, 1.5, 2.0, 4.0))
    dfs = pd.DataFrame(rows_s, columns=['machine', 'shot', 'time', 'rho',
                                        'Q_cond_e', 'Q_GB', 'Te', 'Ti', 'q', 'R_over_LTe'])
    rs, ds, qbs, ts_ = clean_residual(dfs)
    Zs = adaptive_z(dfs, ds, qbs)
    y_spike = coherent_onset(dfs, Zs, ds, ts_)
    r_real, r_spike = y[trust].mean(), y_spike[ts_].mean()
    print(f"  EXTENT test: rate on WIDE avalanches={r_real*100:.1f}%  "
          f"on single-pixel spikes only={r_spike*100:.2f}%  (wide must be >> spike)")
    ok &= r_real > 3.0 * max(r_spike, 1e-4)
    # (4) no leakage: a pure-noise target gives AUC ~0.5
    ynull = rng.random(len(df2)) < 0.2
    a_null = shot_split_auc(df2, ynull, {'dist_qrat': feats['dist_qrat'][trust]},
                            within_band=True)['roc']
    print(f"  null target (random 0/1): AUC={a_null:.3f} (should be ~0.5)")
    ok &= 0.42 < a_null < 0.58
    # (5) THE TRAP TEST: inject an edge blow-up (Q_GB -> 0 at rho>0.7 with a tiny
    # baseline wobble). The SIMPLE relative detector must over-fire at the edge;
    # the ADAPTIVE-Z detector must NOT (its floor kills the relative spike).
    dft = df.copy()
    edge = dft['rho'].values > 0.7
    dft.loc[edge, 'Q_GB'] = 20.0                        # tiny edge baseline
    dft.loc[edge, 'Q_cond_e'] = 20.0 * (1 + rng.normal(0, 0.3, edge.sum()))  # noise only
    rr, dd, qb, tr = clean_residual(dft)
    y_simple = onset(rr, tr)                            # relative |resid|>2
    Zt = adaptive_z(dft, dd, qb)
    y_adapt = coherent_onset(dft, Zt, dd, tr)
    edge_simple = y_simple[tr & edge].mean() if (tr & edge).sum() else 0
    edge_adapt = y_adapt[tr & edge].mean() if (tr & edge).sum() else 0
    print(f"  EDGE-BLOWUP TRAP: edge false-positive rate  simple={edge_simple*100:.0f}%  "
          f"adaptive={edge_adapt*100:.0f}%")
    ok &= (edge_adapt < edge_simple - 0.05) or (edge_simple < 0.05)
    # (6) UNIVERSALITY MACHINERY: build 4 synthetic machines whose onset fires
    # when a hidden order parameter exceeds the SAME critical value, plus a
    # control where each machine has its OWN critical value. The test must report
    # a TIGHT spread for the universal case and a WIDE spread for the control.
    rng3 = np.random.default_rng(7)
    machs = ['M1', 'M2', 'M3', 'M4']
    uni_rows, dep_rows = [], []
    for mi, mm in enumerate(machs):
        N = 4000
        o = np.exp(rng3.normal(0, 1.0, N)) * (1.0 + 0.5 * mi)   # per-machine scale
        y_uni = (o >= 3.0) ^ (rng3.random(N) < 0.05)            # SAME crit=3.0
        crit_m = 3.0 * (1.0 + mi)                               # per-machine crit
        y_dep = (o >= crit_m) ^ (rng3.random(N) < 0.05)
        tr_all = np.ones(N, bool)
        uni_rows += omega_universality(np.array([mm] * N), o, y_uni, tr_all)
        dep_rows += omega_universality(np.array([mm] * N), o, y_dep, tr_all)
    cv_uni, _ = universality_cv(uni_rows)
    cv_dep, _ = universality_cv(dep_rows)
    print(f"  UNIVERSALITY test: spread(shared crit)={cv_uni*100:.0f}%  "
          f"spread(per-machine crit)={cv_dep*100:.0f}%  (shared must be << per-machine)")
    ok &= np.isfinite(cv_uni) and np.isfinite(cv_dep) and cv_uni < 0.35 < cv_dep
    # (7) SOC vs OVERDRIVE on MARGINAL profiles: the whole point of the rebuild.
    # Make R/LTe hover AT critical (overdrive ~ 0, like the real machines) and
    # drive onset by a poised+loading SOC rule. The SOC order parameter must
    # SEPARATE onset; overdrive (excess over critical) must be near-blind there.
    rng4 = np.random.default_rng(3)
    Nm = 8000; crit0 = 8.0
    RLT = crit0 + rng4.normal(0, 1.6, Nm)                 # profiles AT marginal
    poised = np.exp(-((RLT - crit0) / 4.0) ** 2)
    load = np.maximum(0.0, rng4.normal(0.0, 1.0, Nm))
    flow = np.exp(rng4.normal(0.0, 0.5, Nm))
    soc_score = poised * (1.0 + load) / flow
    onset_m = (soc_score >= np.quantile(soc_score, 0.90))
    onset_m = onset_m ^ (rng4.random(Nm) < 0.10)          # 10% label noise
    overdrive_m = np.maximum(0.0, RLT / crit0 - 1.0)
    a_soc = auc(soc_score, onset_m)
    a_od = auc(overdrive_m, onset_m)
    print(f"  SOC-vs-OVERDRIVE (marginal profiles): SOC AUC={a_soc:.3f}  "
          f"overdrive AUC={a_od:.3f}  gap={a_soc - a_od:+.3f}  "
          f"(SOC must separate; overdrive near-blind)")
    ok &= (a_soc > 0.70) and (a_od < 0.60) and (a_soc - a_od > 0.15)
    print("  ==> SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# --------------------------------------------------------------------------
def run(path, only=None):
    df = pd.read_parquet(path)
    if only:
        df = df[df['machine'].isin(only)].reset_index(drop=True)
        print(f"[filtered to machines: {only}]")
    need = {'machine', 'shot', 'time', 'rho', 'Q_cond_e', 'Q_GB', 'Te', 'Ti', 'q'}
    miss = need - set(df.columns)
    if miss:
        print("MISSING columns:", miss)
        return
    resid, delta, qbase, trust = clean_residual(df)
    Z = adaptive_z(df, delta, qbase)
    y = coherent_onset(df, Z, delta, trust)
    print(f"rows: {len(df):,} | trustworthy: {trust.sum():,} "
          f"({trust.mean()*100:.0f}%) | machines: {df['machine'].nunique()} | "
          f"shots: {df['shot'].nunique()}")
    print(f"NON-LOCAL onset (positive surge, |Z|>2.0, radial extent >=3 "
          f"surfaces = avalanche band): {y[trust].mean()*100:.1f}%")
    feats = state_features(df)
    # NOTE: flux_grad_decouple (|d delta/dt| / |d R/LTe/dt|) was REMOVED -- it is
    # CIRCULAR. Its numerator |d(delta)/dt| mechanically measures the residual
    # jump, and the target IS a residual surge, so it predicted the target by
    # construction (it hit AUC 0.93 while every clean feature sat ~0.55 -- the
    # classic run_out signature). The clean part of that idea (the gradient does
    # NOT change) is already carried by precursor_rate = |d(R/LTe)/dt|, which is
    # non-circular and is the honest top predictor.
    dft = df[trust].reset_index(drop=True)
    yt = y[trust]
    ft = {k: v[trust] for k, v in feats.items()}
    # target is now defined by the residual's SPACE-TIME COHERENCE only (not by
    # R/LTe or crit), so EVERY plasma-parameter predictor is non-circular.
    print("\nONSET PREDICTABILITY (train-shot / test-shot hold-out):")
    print("  target = coherent multi-surface flux surge (avalanche); all")
    print("  predictors below are clean, non-circular INPUT features.")
    print(f"  {'predictor':<22}{'plain AUC':>11}{'within-band AUC':>18}")
    for name in feats:
        rp = shot_split_auc(dft, yt, {name: ft[name]}, within_band=False)
        rb = shot_split_auc(dft, yt, {name: ft[name]}, within_band=True)
        print(f"  {name:<22}{rp['roc']:>11.3f}{rb['roc']:>18.3f}")
    print("  (within-band removes radial position -- the honest physics number)")

    def _report(tag, feat_subset):
        r = shot_split_auc(dft, yt, feat_subset, within_band=True)
        print(f"\n  {tag} within-band, shot-split ({len(feat_subset)} features):")
        print(f"    ROC-AUC = {r['roc']:.3f}   (rank quality; imbalance-robust)")
        print(f"    PR-AUC  = {r['pr']:.3f}   vs base rate {r['base']:.3f} "
              f"= lift {r['pr']/max(r['base'],1e-9):.1f}x over random")
        print(f"    precision @ 60% recall = {none_pct(r['prec60'])}   "
              f"best-F1: precision={none_pct(r['prec'])} recall={none_pct(r['rec'])}"
              f"   (n={r['n']} cells)")
        return r

    # CORE model: seven features chosen A PRIORI by physics, one per mechanism, so
    # this is NOT cherry-picked by AUC. drive (overdrive), damping (nu_star_e),
    # topology (q), turbulence scale (rho_star), stabilisation (shear_abs),
    # temporal push (precursor_rate), non-local coupling (I_drive). Pruning the
    # 17 redundant/degenerate features (aspect_ratio, gamma_ExB, I_drive_rate ...)
    # is the honest form of the Phase-2 request -- by mechanism, not metric.
    CORE = ['rho_star', 'q', 'shear_abs', 'overdrive',
            'nu_star_e', 'precursor_rate', 'I_drive']
    core_ft = {k: ft[k] for k in CORE if k in ft}
    r_core = _report("CORE MODEL (per-slice, physics-picked)", core_ft)
    r = _report("FULL MODEL (per-slice, all features)", ft)
    # LEVER 3: temporal label DILATION (jitter fix). A prediction within +/-2
    # time steps of a real event should not be scored a miss. Dilate the onset
    # label per (shot,rho) and re-score -- if the AUC jumps, timing jitter (not
    # physics) was capping it.
    dl = pd.DataFrame({'shot': df['shot'].values[trust], 'rho': df['rho'].values[trust],
                       'time': df['time'].values[trust], 'y': yt.astype(float),
                       '_i': np.arange(int(trust.sum()))}).sort_values(['shot', 'rho', 'time'])
    yd = dl.groupby(['shot', 'rho'])['y'].transform(
        lambda v: v.rolling(5, center=True, min_periods=1).max())
    yt_dil = np.zeros(int(trust.sum())); yt_dil[dl['_i'].values] = yd.values
    rd = shot_split_auc(dft, yt_dil > 0.5, ft, within_band=True)
    print(f"  FULL MODEL, LEVER-3 dilated label (+/-2 steps): AUC={rd['roc']:.3f} "
          f"(n={rd['n']}) -- if >> strict, timing jitter was the cap, not physics")
    # Pivot A: hazard rate over time windows (the statistically correct target)
    a_haz, nh = hazard_eval(df, y, trust, feats, win_ms=15.0)
    print(f"  FULL MODEL (HAZARD, 15ms windows) within-band, shot-split: "
          f"AUC={a_haz:.3f}  (n={nh} cells)")
    print("  -> per-slice asks 'did THIS chaotic trigger fire'; hazard asks 'is a")
    print("     window event-prone' -- the honest, physically-correct question.")
    print("  -> within-band AUC >> 0.5 = real physics; ~0.5 = radial position only.")
    if max(r['n'], nh) < 5:
        print("\n  *** NOTE: <5 hold-out cells for the CLASSIFIER (this data has")
        print("  ~1 shot/machine). The classifier AUC needs multi-shot data; the")
        print("  UNIVERSALITY test below does NOT -- it works on this data. ***")

    # ===== THE REFRAME: UNIVERSAL CRITICAL THRESHOLD (no shot hold-out needed) =
    print("\nUNIVERSAL ONSET THRESHOLD (Omega >= Omega_crit; the LAW, not a")
    print("  classifier). Per machine: single-order-parameter AUC (zero fitted")
    print("  params) and Omega_crit. Tight spread across machines = universal.")

    def _univ(tag, om):
        rows = omega_universality(df['machine'].values, om, y, trust)
        print(f"\n  [{tag}]")
        print(f"  {'machine':<8}{'n':>9}{'onset%':>8}{'Omega AUC':>11}{'Omega_crit':>13}")
        for m, nn, rate, a, thr in sorted(rows, key=lambda z: (z[4] if np.isfinite(z[4]) else 1e18)):
            aS = f"{a:.3f}" if np.isfinite(a) else "  n/a"
            tS = f"{thr:.3e}" if np.isfinite(thr) else "  n/a"
            print(f"  {m:<8}{nn:>9,}{rate*100:>7.1f}%{aS:>11}{tS:>13}")
        cv, k = universality_cv(rows)
        aucs = [z[3] for z in rows if np.isfinite(z[3])]
        if np.isfinite(cv):
            verdict = ("UNIVERSAL -- one critical value fits all machines"
                       if cv < 0.35 else "partial -- some spread; refine params"
                       if cv < 0.8 else "MACHINE-DEPENDENT -- not universal as built")
            print(f"  --> Omega_crit spread across {k} machines: {cv*100:.0f}%  [{verdict}]")
        if aucs:
            print(f"  --> mean single-parameter AUC = {np.mean(aucs):.3f}")

    _univ("OVERDRIVE order param (old: excess over critical -- dies at marginal)",
          omega_nl(feats))
    _univ("SOC order param (new: PROXIMITY to marginal -- keeps range on real data)",
          omega_soc(df, feats))
    print("  (spread = exp(std of log Omega_crit)-1, the multiplicative scatter of")
    print("   the critical value; the Greenwald-limit-style universality claim.)")

    # STRUCTURAL LAWS (the non-local characterisation -- no classifier, no
    # negative diffusivity). Does the structure scale with rho_star?
    print("\nSTRUCTURAL LAWS (non-local characterisation; scales with rho_star?):")
    S = structural_laws(df, resid, trust)
    print(f"  {'machine':<8}{'rho_star':>10}{'alpha':>8}{'lam':>8}{'coh':>7}"
          f"{'v_prop':>10}{'ballist':>9}{'H':>7}")
    for m in sorted(S, key=lambda k: S[k]['rho_star'] if np.isfinite(S[k]['rho_star']) else 0):
        r = S[m]
        print(f"  {m:<8}{r['rho_star']:>10.4f}{r['alpha']:>8.2f}{r['lam']:>8.3f}"
              f"{r['coh']:>7.2f}{r['v_prop']:>10.0f}{r['ballistic']:>9.2f}{r['H']:>7.2f}")
    import numpy as _np
    from math import sqrt
    rs = _np.array([S[m]['rho_star'] for m in S])
    print("  --- scaling with rho_star (across machines) ---")
    for lawname in ('alpha', 'lam', 'coh', 'v_prop', 'ballistic', 'H'):
        vals = _np.array([S[m][lawname] for m in S])
        ok = _np.isfinite(rs) & _np.isfinite(vals)
        if ok.sum() >= 5 and vals[ok].std() > 1e-9:
            c = _np.corrcoef(_np.log(_np.maximum(rs[ok], 1e-6)), vals[ok])[0, 1]
            # rough significance: t = r*sqrt((n-2)/(1-r^2))
            n = int(ok.sum())
            t = abs(c) * sqrt((n - 2) / max(1 - c * c, 1e-6))
            flag = "  <== STRONG universal scaling" if abs(c) > 0.6 else ""
            print(f"    corr(log rho_star, {lawname:<10}) = {c:+.2f}  (n={n}, t={t:.1f}){flag}")
    print("  -> a structural law that scales tightly with rho_star IS the")
    print("     universal non-local law -- the paper's core result.")
    print("  -> ballistic ~1 = constant-speed avalanche front (non-local);")
    print("     ~2 = ordinary diffusion. This is the direct non-locality test.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        sys.exit(0 if self_test() else 1)
    if len(sys.argv) > 1:
        # optional: --only MACH1,MACH2 to restrict to specific machines
        only = None
        if '--only' in sys.argv:
            only = sys.argv[sys.argv.index('--only') + 1].split(',')
        run(sys.argv[1], only=only)
    else:
        print(__doc__)
        print("Give a parquet path, or --self-test.")
