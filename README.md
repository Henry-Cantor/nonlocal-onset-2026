# A Universal Onset Law for Non-Local Heat Transport in Tokamaks

A compact, first-principles, cross-machine-validated law for **when non-local
("avalanche") electron heat transport switches on** in a tokamak — validated across
the aspect-consistent **10 machines / 420 discharges / 43.2 M flux-surface×time cells**
from TRANSP power balance.

## The result

Onset is defined **absolutely and machine-consistently**: a flux surface is in onset
when the conducted electron heat flux exceeds **~4× the local gyro-Bohm baseline** for
its machine, **coherently over ≥4 adjacent surfaces** (a wide radial band = an
avalanche; a lone hot pixel = local jitter). A compact dimensionless power law predicts
it (held out, nowcast):

| regime | onset% | within-machine LOSO | cross-machine LOMO | null | non-circular (noQ_GB) |
|---|---|---|---|---|---|
| conv_NBI (beam) | 11.9% | **0.988** | **0.956** | 0.500 | 0.929 |
| spherical (ST) | 12.5% | **0.975** | **0.967** | 0.500 | 0.864 |
| conv_0torque (RF/Ohmic) | 15.3% | 0.929 | **0.942** | 0.500 | 0.848 |

**The universality headline is the single 21-term law across all 10 machines + 3 regimes: pooled LOSO
0.940 / LOMO 0.959, tight [0.93–0.99]** (`pooled_21term.py`). per-regime LOMO is 0.94–0.97, and the frozen law
transfers to **every one of the 10 machines** at 0.93–0.99 (incl. SMART 0.965, CMOD 0.959, EAST 0.994); the
pooled 21-term law is the primary universality result.

- **LOSO** = leave-one-shot-out *within* a machine (held-out shots = future shots).
- **LOMO** = leave-one-machine-out (frozen law → an unseen machine = the ITER test).
- **null** = global label shuffle (must be ~0.5) → no leakage.
- **noQ_GB** = LOMO with every gyro-Bohm feature removed (stays high) → real dynamics,
  not algebra.

**Universal, no calibration:** a *single global* saturation constant for all 10 machines
matches/beats per-machine calibration (global-c0 LOMO 0.93–0.97), so the law extrapolates to a new
device (ITER) with zero local tuning.

**Why it matters:** the **coherent avalanche events the law predicts carry ~25–36% of the
electron conducted heat** (`excess_share`, the tight/suppressible number). A looser upper
bound — *all* heat above the local *median* gyro-Bohm baseline — is ~45–50%, but that
over-counts moderate above-median transport, so the headline is the avalanche-driven
~25–36%. Either way it is a **tens-of-percent confinement lever** if controlled
(`heat_impact.py`; a future paper will quantify the achievable recovery).

## Pipeline (run in this order)

| step | script | where | output |
|---|---|---|---|
| 1 | `flux_candidate_filter.py` | analysis cluster | candidate discharge list |
| 2 | `stage_candidates.sh` | analysis cluster | staged `tr.DAT` |
| 3 | `stellar_final_filter.py` | Stellar | interpretive/predictive triage |
| 4 | `step1_process.py` | Stellar | `stage1_5_v3.parquet` (profiles + quality) |
| 5 | `build_features.py` | laptop | `reduced_v2.parquet` (features + onset label) |
| 6 | `run_onset.py` | laptop | validation table + `onset_final.parquet` + results JSON |
| 7 | `heat_impact.py` | laptop | transport-impact metric + threshold justification |
| 8 | `figures/make_figures.py` | laptop | all paper figures + `tables.md` |

```bash
python run_onset.py            # final validation (per-machine c0)
python run_onset.py --global   # single global c0 (universality / ITER test)
python heat_impact.py          # impact metric + threshold justification
python figures/make_figures.py # regenerate every figure + tables.md
```

## The law

The predictor is a dimensionless **power law** (a product of powers of standard plasma
groups; the log-space logistic in `run_onset.py`). Two independent searches characterize
it:
- `discovery/sparse_law.py` — the compact scaling reduction (ρ*, drive η_e, β_e are the
  dominant terms).
- `discovery/sr_search.py` — genetic-programming symbolic regression; compact trees
  plateau below the power law, proving onset is irreducibly multivariate.
The mechanism is a drive/damping marginal-stability parameter Θ (see

## Layout

```
run_onset.py          final validation (LOSO/LOMO/null/non-circularity) -> results
onset_common.py       shared numerics (logistic, AUC, LOSO, LOMO) -- no sklearn
build_features.py     stream TRANSP parquet -> dimensionless features + absolute label
heat_impact.py        transport-impact metric + threshold justification
step1_process.py      TRANSP CDF -> parquet (profiles, gradients, quality columns)
flux_candidate_filter.py / stage_candidates.sh / stellar_final_filter.py   data acquisition
support/              feature-engineering library imported by build_features
discovery/            how the law was found (sr_search, sparse_law, law_compare,
                      test_circularity, diag_permachine)
figures/              make_figures.py (+ notebook) -> PNGs + tables.md
```

## Environment

Python 3.13, `numpy pandas pyarrow(>=19.0.1) scipy scikit-learn gplearn matplotlib`.
**`pyarrow>=19.0.1` is required** — 19.0.0 has a SizeStatistics read bug that makes the
parquets unreadable. Parquets are large and **git-ignored**; regenerate them with steps
4–5, or point the scripts at your own copies.

## Reproducibility

All headline numbers are written to committed JSON result files: `run_onset.py` writes
`onset_results_per_machine.json` (default) and `onset_results_global.json` (`--global`);
`heat_impact.py` writes `heat_impact.json`. From those JSONs, `figures/make_figures.py`
regenerates **8 of the 12 figures and all 4 tables**. The other 4 figures (calibration,
example-shot, ROC, hazard-trace) plot per-cell data from `onset_final.parquet`, which is
**not committed** — those ship as committed
PNGs. The onset label, the hold-out validators, and the null/non-circularity controls are
all in `run_onset.py` + `onset_common.py` (~250 lines total, no black boxes).
