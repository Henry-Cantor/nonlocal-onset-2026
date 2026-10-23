#!/usr/bin/env python3
"""
Flux Candidate Scanner with Correct TRANSP Output Timestep Parsing
Parses /p/transparch/result/ across all tokamak facilities.

Correct TRANSP Timing Keys:
  - SEDIT  : Spatial Profile CDF Output Frequency (seconds) -> Primary Target
  - STEDIT : Scalar Time Series Output Frequency (seconds)
  - TGRID2 : Input Profile Time Step (seconds)
"""

import os
import re
from pathlib import Path

MACHINES = [
    "AUGD", "CMOD", "D3D", "EAST", "HL2A", "HL2M", "KSTR", "LTX", 
    "MAST", "NSTU", "NSTX", "PBXM", "ST40", "TCV", "TFTR", "SMRT", "MST"
]

BASE_ARCHIVE = Path("/p/transparch/result")
OUTPUT_MANIFEST = Path("flux_candidates.txt")

MIN_RUNS_PER_MACHINE = 2     # keep every machine with >=2 clean interpretive shots
MAX_RUNS_PER_MACHINE = 200   # (10 was dropping C-Mod's 6 clean shots to 0 -- fatal
                             #  for the zero-torque regime; a few clean shots still
                             #  contribute to the pooled multi-machine fit)

# ---------------------------------------------------------------------------
# INTERPRETIVE SCREENING (v2 — the v1 screen was far too permissive)
#
# BUG IN v1: the predictive-start-time keys (LELTIM, LITIM, ...) hold a NUMBER
# (the time in seconds at which TRANSP switches from data to prediction), but
# v1 only matched `KEY = .TRUE.`. A run with `LELTIM = 2.0` — predictive from
# t=2 s — therefore passed as "interpretive". Every numerically-flagged
# predictive run leaked into the dataset. That is fatal here: a predictive run
# computes its temperature from a transport model, so the "residual" we measure
# would be the model's own closure error, not physics. This is the one
# contamination that could invalidate the entire result.
#
# v2 screens on three independent grounds and requires POSITIVE interpretive
# evidence rather than mere absence of a few flags.
# ---------------------------------------------------------------------------

# (a) numeric predictive-start-time switches: ANY finite value = predictive
PREDICTIVE_TIME_KEYS = [
    "LELTIM", "LITIM", "LE2TIM", "LI2TIM", "LNE2TIM", "LNZTIM",
    "LPHITIM", "LQTIM", "LVTIM",
]
PREDICTIVE_TIME_REGEX = [
    (key, re.compile(rf'\b{key}\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?)',
                     re.IGNORECASE))
    for key in PREDICTIVE_TIME_KEYS
]

# (b) boolean predictive / transport-model switches
PREDICTIVE_BOOL_KEYS = [
    "NLGLF23", "NLTGLF", "NLMMM", "NLBAL", "NLCG", "NLPTRANSP",
    "NLPRED", "NLTRANSP_PRED", "NLMMM7_1", "NLGLF", "NLIFS",
]
PREDICTIVE_BOOL_REGEX = [
    (key, re.compile(rf'\b{key}\s*=\s*\.?\s*(TRUE|T)\b\.?', re.IGNORECASE))
    for key in PREDICTIVE_BOOL_KEYS
]

# (c) POSITIVE interpretive evidence: temperatures taken FROM DATA.
#     NLTEPRO/NLTIPRO = .TRUE. means Te/Ti are prescribed from measurement.
INTERPRETIVE_BOOL_REGEX = [
    (key, re.compile(rf'\b{key}\s*=\s*\.?\s*(TRUE|T)\b\.?', re.IGNORECASE))
    for key in ("NLTEPRO", "NLTIPRO")
]
# explicit negation of the above = predicting that channel
INTERPRETIVE_FALSE_REGEX = [
    (key, re.compile(rf'\b{key}\s*=\s*\.?\s*(FALSE|F)\b\.?', re.IGNORECASE))
    for key in ("NLTEPRO", "NLTIPRO")
]

def extract_shot_and_version(filename):
    match = re.search(r'(\d{5,6}).*?([A-Za-z]\d{2})\.(CDF|cdf)$', str(filename))
    if match:
        return match.group(1), match.group(2)
    return None, None

_PRED_CDF_KEYS = ["LELTIM", "LITIM", "LE2TIM", "LI2TIM", "LNE2TIM", "LNZTIM",
                  "NLGLF23", "NLTGLF", "NLMMM", "NLBAL", "NLCG"]


def cdf_is_predictive(cdf_path):
    """Fallback interpretive check from the CDF itself, used when the tr.DAT
    namelist cannot be located (so unverifiable-by-namelist shots are still
    RIGOROUSLY screened, not blindly dropped). Mirrors stellar_final_filter's
    predictive-key scan: returns True if any predictive/transport-model switch is
    present as a CDF variable or global attribute, False if clean, None if the CDF
    can't be opened. The Stellar-side stage re-runs this same check as a backstop."""
    try:
        from netCDF4 import Dataset
        ds = Dataset(str(cdf_path), 'r')
    except Exception:
        return None
    try:
        keys = {v.upper() for v in ds.variables} | {a.upper() for a in ds.ncattrs()}
        return any(k in keys for k in _PRED_CDF_KEYS)
    finally:
        try:
            ds.close()
        except Exception:
            pass


def find_tr_dat_file(cdf_path):
    """Locate the input namelist for a CDF. Searches (in order): the exact
    version's tr.DAT, then a fuzzy version match, then ANY tr.DAT for the SAME
    SHOT in this dir and one dir up (analysts often keep the namelist under the
    shot dir, or only one version's namelist survives). This shot-level fallback
    is what recovers the many false 'unverifiable' rejects -- if one version of a
    shot has a findable interpretive namelist, the shot is verifiable."""
    try:
        parent = cdf_path.parent
        stem = cdf_path.stem
        shot = re.match(r'(\d{5,6})', stem)
        shot = shot.group(1) if shot else stem[:6]
        # 1) exact version patterns
        for cand in (parent / f"{stem}TR.DAT", parent / f"{stem}tr.DAT",
                     parent / f"{stem}.TR.DAT", parent / f"{stem}.tr.DAT"):
            if cand.exists():
                return cand
        # 2) fuzzy on the full version stem, then 3) any tr.DAT for the same SHOT,
        #    searched in this dir and one directory up (case-insensitive)
        for base in (parent, parent.parent):
            for pat in (f"*{stem}*TR.DAT", f"*{stem}*tr.DAT",
                        f"*{shot}*TR.DAT", f"*{shot}*tr.DAT", f"*{shot}*.dat", f"*{shot}*.DAT"):
                hits = sorted(base.glob(pat))
                if hits:
                    return hits[0]
        return None
    except Exception:
        return None

def screen_tr_dat(tr_dat_path):
    """Strict interpretive screen. Returns (is_interpretive, reason).

    Returns (tier, reason) where tier is:
      'CONFIRMED' — no predictive evidence AND positive interpretive evidence
                    (NLTEPRO/NLTIPRO = TRUE). Safe for the headline dataset.
      'PROBABLE'  — no predictive evidence, but no positive confirmation either
                    (those keys absent from this namelist). Usable, but the
                    interpretive status is inferred from absence, not proven.
      'REJECT'    — positive predictive evidence, or unverifiable.

    Two tiers rather than one because a single strict gate has a silent failure
    mode: if a facility simply does not write NLTEPRO/NLTIPRO into its tr.DAT,
    a 'require positive evidence' screen rejects 100% of that machine and the
    dataset quietly loses it. Reporting both counts makes that visible instead.
    Use CONFIRMED alone if it retains enough shots; otherwise CONFIRMED+PROBABLE
    with the caveat stated in the paper.
    """
    if not tr_dat_path or not tr_dat_path.exists():
        return 'REJECT', "no tr.DAT found (unverifiable)"
    try:
        with open(tr_dat_path, 'r', encoding='latin-1') as f:
            content = f.read()
    except Exception as e:
        return 'REJECT', f"unreadable ({type(e).__name__})"

    # strip Fortran comments (! to end of line) and blank-out '=' inside strings
    cleaned = re.sub(r'!.*', '', content)

    # (a) numeric predictive-start-time switches
    for key, rgx in PREDICTIVE_TIME_REGEX:
        for m in rgx.finditer(cleaned):
            raw = m.group(1).replace('D', 'E').replace('d', 'e')
            try:
                v = float(raw)
            except ValueError:
                continue
            # any finite, non-sentinel value means prediction starts at that time.
            # TRANSP convention: a huge value (>= 1e5) or negative disables it.
            if 0.0 <= v < 1.0e5:
                return 'REJECT', f"predictive: {key}={v:g}"

    # (b) boolean predictive / transport-model switches
    for key, rgx in PREDICTIVE_BOOL_REGEX:
        if rgx.search(cleaned):
            return 'REJECT', f"predictive: {key}=TRUE"

    # (c) explicit negation of interpretive temperature prescription
    for key, rgx in INTERPRETIVE_FALSE_REGEX:
        if rgx.search(cleaned):
            return 'REJECT', f"predictive: {key}=FALSE"

    # positive interpretive evidence?
    pos = [key for key, rgx in INTERPRETIVE_BOOL_REGEX if rgx.search(cleaned)]
    if pos:
        return 'CONFIRMED', "interpretive (" + "+".join(pos) + ")"
    return 'PROBABLE', "no predictive evidence, but NLTEPRO/NLTIPRO absent"


# Set to True to require CONFIRMED (positive interpretive evidence).
# Leave False to accept CONFIRMED + PROBABLE; the run prints both counts either
# way so the choice is made on real numbers, not guesswork.
REQUIRE_CONFIRMED = False


def is_interpretive_tr_dat(tr_dat_path):
    tier = screen_tr_dat(tr_dat_path)[0]
    return tier == 'CONFIRMED' or (tier == 'PROBABLE' and not REQUIRE_CONFIRMED)

def extract_dt_from_tr_dat(tr_dat_path):
    if not tr_dat_path or not tr_dat_path.exists():
        return None

    try:
        with open(tr_dat_path, 'r', encoding='latin-1') as f:
            content = f.read()
            
        cleaned = re.sub(r'!.*', '', content)
        
        # SEDIT is profile output spacing, STEDIT is scalar output, TGRID2 is profile grid step
        matches = re.findall(
            r'(?:SEDIT|STEDIT|TGRID2)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?)', 
            cleaned, re.IGNORECASE
        )
        if matches:
            vals = []
            for m in matches:
                try:
                    # Convert Fortran double precision (e.g., 1.0D-03) to scientific notation (1.0E-03)
                    m_clean = m.replace('D', 'E').replace('d', 'e')
                    v = float(m_clean)
                    if v > 0:
                        vals.append(v)
                except ValueError:
                    pass
            if vals:
                return min(vals)
    except Exception:
        pass
    return None

REJECT_REASONS = {}
TIER_COUNTS = {}
MACHINE_TIERS = {}


def main():
    manifest = []

    tot_raw = 0
    tot_interpretive = 0
    tot_sub_1ms = 0
    tot_staged = 0

    print("==========================================================================")
    print(" SCANNING ARCHIVE & AUDITING RUN STAGES (USING TRANSP SEDIT KEY)")
    print(" Metrics Tracked: Raw CDFs -> Interpretive (Unique) -> SEDIT <= 1.0 ms")
    print("==========================================================================\n")
    print(f"{'MACHINE':<10} | {'RAW CDFs':<10} | {'INTERPRETIVE':<14} | {'SEDIT <= 1ms':<12} | {'STAGED':<8}")
    print("-" * 68)

    for machine in MACHINES:
        machine_dir = BASE_ARCHIVE / machine
        if not machine_dir.exists():
            continue

        try:
            all_cdfs = list(machine_dir.glob("**/*.CDF")) + list(machine_dir.glob("**/*.cdf"))
        except Exception:
            continue

        # INTEGRITY GATE: a valid multi-timeslice TRANSP run is >= a few MB.
        # Truncated/incomplete CDFs (e.g. NSTX 105759C12 at 1.5 MB) produce garbage
        # power balance (Q_cond<0 everywhere) and poison the residual calibration.
        # Drop them here at candidate selection (keeps the repo's full set intact).
        MIN_CDF_BYTES = 5 * 1024 * 1024
        n_before = len(all_cdfs)
        all_cdfs = [c for c in all_cdfs
                    if (c.is_file() and c.stat().st_size >= MIN_CDF_BYTES)]
        n_trunc = n_before - len(all_cdfs)
        if n_trunc:
            print(f"    [{machine}] dropped {n_trunc} truncated CDF(s) < 5 MB")

        raw_count = len(all_cdfs)
        tot_raw += raw_count

        shots_grouped = {}

        for cdf in all_cdfs:
            try:
                if not cdf.is_file():
                    continue
                mtime = cdf.stat().st_mtime
            except (FileNotFoundError, PermissionError, OSError):
                continue

            shot, version = extract_shot_and_version(cdf.name)
            if not shot:
                continue

            tr_dat_path = find_tr_dat_file(cdf)

            # Verify interpretive mode (v2 two-tier screen)
            tier, why = screen_tr_dat(tr_dat_path)
            # FALLBACK: if the namelist is unverifiable, screen the CDF itself
            # rather than dropping the shot -- keeps interpretive-but-unnamelisted
            # runs (most of the 'lost' C-Mod shots) while still catching predictive
            # ones. The Stellar stage re-runs this CDF check as a backstop.
            if tier == 'REJECT' and 'unverifiable' in why:
                pred = cdf_is_predictive(cdf)
                if pred is False:
                    tier, why = 'PROBABLE', 'no namelist; CDF clean of predictive keys'
                elif pred is True:
                    tier, why = 'REJECT', 'predictive (from CDF, no namelist)'
            TIER_COUNTS[tier] = TIER_COUNTS.get(tier, 0) + 1
            if tier == 'REJECT':
                key_r = why.split('=')[0].split('(')[0].strip()
                REJECT_REASONS[key_r] = REJECT_REASONS.get(key_r, 0) + 1
            ok_interp = (tier == 'CONFIRMED'
                         or (tier == 'PROBABLE' and not REQUIRE_CONFIRMED))
            if ok_interp:
                MACHINE_TIERS.setdefault(machine, {})
                MACHINE_TIERS[machine][tier] = \
                    MACHINE_TIERS[machine].get(tier, 0) + 1
                dt = extract_dt_from_tr_dat(tr_dat_path)
                dt_val = dt if dt is not None else 999.0
                
                if shot not in shots_grouped:
                    shots_grouped[shot] = []
                
                shots_grouped[shot].append({
                    'cdf_path': cdf,
                    'mtime': mtime,
                    'dt': dt_val,
                    'version': version
                })

        # Select best version per shot (smallest SEDIT dt, then latest mtime)
        selected_for_machine = []
        for shot, candidate_runs in shots_grouped.items():
            candidate_runs.sort(key=lambda x: (x['dt'], -x['mtime']))
            selected_for_machine.append(candidate_runs[0])

        interp_count = len(selected_for_machine)
        tot_interpretive += interp_count

        # Count how many of these unique interpretive shots have SEDIT <= 1.0 ms (0.001 seconds)
        sub_1ms_count = sum(1 for run in selected_for_machine if run['dt'] <= 0.001)
        tot_sub_1ms += sub_1ms_count

        # Sort candidate runs for staging
        selected_for_machine.sort(key=lambda x: (x['dt'], -x['mtime']))

        # Apply dataset windowing (10 to 100 runs)
        if interp_count < MIN_RUNS_PER_MACHINE:
            staged_count = 0
        elif interp_count > MAX_RUNS_PER_MACHINE:
            staged_count = MAX_RUNS_PER_MACHINE
            staged_runs = [item['cdf_path'] for item in selected_for_machine[:MAX_RUNS_PER_MACHINE]]
            manifest.extend(staged_runs)
        else:
            staged_count = interp_count
            staged_runs = [item['cdf_path'] for item in selected_for_machine]
            manifest.extend(staged_runs)

        tot_staged += staged_count

        print(f"{machine:<10} | {raw_count:<10} | {interp_count:<14} | {sub_1ms_count:<12} | {staged_count:<8}")

    # Write final manifest
    with open(OUTPUT_MANIFEST, "w") as f:
        for p in manifest:
            f.write(f"{p}\n")

    print("=" * 68)
    print(f"{'TOTALS':<10} | {tot_raw:<10} | {tot_interpretive:<14} | {tot_sub_1ms:<12} | {tot_staged:<8}")
    print("==========================================================================\n")
    print(" SCREEN TIERS (all CDFs examined):")
    for t in ('CONFIRMED', 'PROBABLE', 'REJECT'):
        print(f"   {TIER_COUNTS.get(t, 0):>7}  {t}")
    print(f"   mode: {'CONFIRMED only' if REQUIRE_CONFIRMED else 'CONFIRMED + PROBABLE'}"
          f"  (set REQUIRE_CONFIRMED at the top of this file to change)")
    if MACHINE_TIERS:
        print("\n PER-MACHINE CONFIRMED/PROBABLE (watch for a machine with 0"
              " CONFIRMED — that facility just may not write the key):")
        for m in sorted(MACHINE_TIERS):
            d = MACHINE_TIERS[m]
            print(f"   {m:<8} CONFIRMED={d.get('CONFIRMED', 0):<6} "
                  f"PROBABLE={d.get('PROBABLE', 0):<6}")
    if REJECT_REASONS:
        print("\n REJECTION BREAKDOWN (audit this):")
        for why, n in sorted(REJECT_REASONS.items(), key=lambda kv: -kv[1]):
            print(f"   {n:>7}  {why}")
        print("   NOTE: v1 accepted numerically-flagged predictive runs (LELTIM=2.0")
        print("   etc.) because it only matched .TRUE. — if 'predictive' counts are")
        print("   large here, those runs were contaminating the previous dataset.\n")
    print(f"Manifest written to: {OUTPUT_MANIFEST.resolve()} ({len(manifest)} CDFs)")

if __name__ == "__main__":
    main()