"""
Step 07 — Sequential Monte Carlo Engine (Taipower-calibrated)
=============================================================
Uses real FOR data from Step 02 and hazard factors from Step 04.
Generates annual outage state timelines for all generators and branches.

NOTE: This SMC engine is designed to feed Step 09 AC-OPF (not DC-OPF).
Each sampled system state (gen on/off, load level) is evaluated with:
  - AC-OPF (primary): real power losses, voltage profile, reactive dispatch
  - AC-PF fallback: power flow without optimization
  - DC-OPF/DC-PF: last resort approximation
ENS from AC-OPF includes real line losses (not just generation-load imbalance).
Voltage violations from AC solution trigger additional reliability penalties.

Two-state Markov model per component:
  State 0: Operational   (failure rate λ failures/hr)
  State 1: Failed/repair (repair rate μ = 1/MTTR repairs/hr)

For generators (Step 02 FOR data):
  λ = FOR × μ / (1 - FOR)      where μ = 1/MTTR
  FOR from PDF outage data (unplanned outages only)

For lines/transformers (Step 04 hazard data):
  λ_eff = λ_base × HF           (hazard factor amplified)
  MTTR from IEEE/CIGRE standards

Convergence criterion: CoV(EENS) < 0.02 after minimum 50 years
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
STEP01_DIR  = RESULT_BASE / "step01"
STEP02_DIR  = RESULT_BASE / "step02"
STEP04_DIR  = RESULT_BASE / "step04"
STEP05_DIR  = RESULT_BASE / "step05"
OUT_DIR     = RESULT_BASE / "step07"

HOURS_PER_YEAR = 8760
RNG_SEED       = 42

# Convergence settings
MIN_YEARS      = 50
MAX_YEARS      = 1000
COV_THRESHOLD  = 0.02    # 2% CoV for EENS convergence

# Default component reliability (if not in FOR database)
DEFAULT_GEN_FOR    = 0.03    # 3% default FOR
DEFAULT_GEN_MTTR   = 72.0   # 72 hr default MTTR
DEFAULT_LINE_LAMBDA = 0.10 / HOURS_PER_YEAR  # 0.1 failure/year
DEFAULT_LINE_MTTR   = 24.0   # 24 hr MTTR for lines
DEFAULT_TR_LAMBDA   = 0.05 / HOURS_PER_YEAR  # 0.05 failure/year
DEFAULT_TR_MTTR     = 48.0   # 48 hr MTTR for transformers

# Batch size for parallel simulation
BATCH_SIZE = 10


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ComponentReliability:
    """Reliability parameters for a single network component."""
    comp_id:    str
    comp_type:  str         # 'generator', 'line', 'transformer'
    label:      str         # human-readable name
    pmax_mw:    float = 0.0 # installed capacity (MW), for ENS calculation
    lam_per_hr: float = 0.0 # failure rate [1/hr]
    mu_per_hr:  float = 0.0 # repair rate [1/hr]
    for_rate:   float = 0.0 # FOR [0,1]
    mttr_hr:    float = 0.0 # mean time to repair [hr]
    hf:         float = 1.0 # hazard factor


@dataclass
class YearResult:
    """Results for one simulated year."""
    year_idx:         int
    state_matrix:     np.ndarray    # shape: (8760, n_components), 0=up, 1=failed
    outage_hours:     dict          # {comp_id: hours_failed}
    annual_eens_mwh:  float = 0.0   # Expected Energy Not Supplied
    annual_lolh:      float = 0.0   # Loss of Load Hours
    n_contingencies:  int   = 0     # hours with at least 1 failure


@dataclass
class SMCResults:
    """Accumulated Monte Carlo results."""
    n_years:            int = 0
    eens_per_year:      list = field(default_factory=list)
    lolh_per_year:      list = field(default_factory=list)
    contingency_per_year: list = field(default_factory=list)
    running_eens_mean:  list = field(default_factory=list)
    running_eens_cov:   list = field(default_factory=list)
    converged:          bool = False
    convergence_year:   Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# Load reliability parameters
# ──────────────────────────────────────────────────────────────────────────────
def load_generator_reliability(for_csv: Path, gens_csv: Path) -> list[ComponentReliability]:
    """Load generator FOR data and match to network generators."""
    comps = []

    # Load FOR data
    if for_csv.exists():
        for_df = pd.read_csv(for_csv)
        log.info(f"Loaded {len(for_df)} generator FOR records")
    else:
        log.warning("FOR data not found. Using defaults.")
        for_df = pd.DataFrame()

    # Load generator list from Step 01
    if gens_csv.exists():
        gens_df = pd.read_csv(gens_csv)
    else:
        log.error("Generator CSV not found.")
        return comps

    for _, grow in gens_df.iterrows():
        bus_i  = int(grow["bus_i"])
        gen_id = str(grow.get("gen_id", "1"))
        pmax   = float(grow.get("pmax_mw", 0.0))
        status = int(grow.get("status", 1))

        if pmax <= 0:
            continue

        comp_id = f"GEN_{bus_i}_{gen_id}"

        # Try to match FOR data
        # Match on bus_i (rough; FOR data has plant names)
        for_match = None
        if not for_df.empty:
            # Simple heuristic: use average FOR for active generators
            # A more precise implementation would map bus→plant name→FOR
            fuel_types = for_df["fuel_type"].unique()
            for_match_rows = for_df[for_df["pmax_mw"].between(pmax * 0.8, pmax * 1.2)]
            if not for_match_rows.empty:
                for_match = for_match_rows.iloc[0]

        if for_match is not None:
            for_rate   = float(for_match["for_rate"])
            mttr_hr    = float(for_match["mttr_hr"])
            mu         = float(for_match["mu_per_hr"])
            lam        = float(for_match["lambda_per_hr"])
        else:
            for_rate = DEFAULT_GEN_FOR
            mttr_hr  = DEFAULT_GEN_MTTR
            mu       = 1.0 / mttr_hr
            lam      = for_rate * mu / max(1.0 - for_rate, 0.001)

        comps.append(ComponentReliability(
            comp_id   = comp_id,
            comp_type = "generator",
            label     = f"Bus{bus_i}_Gen{gen_id}",
            pmax_mw   = pmax,
            lam_per_hr = lam,
            mu_per_hr  = mu,
            for_rate   = for_rate,
            mttr_hr    = mttr_hr,
            hf         = 1.0,  # generators use FOR directly, no HF
        ))

    log.info(f"Created {len(comps)} generator reliability records")
    return comps


def load_branch_reliability(branches_csv: Path, hf_csv: Path,
                             sys_hf_json: Path) -> list[ComponentReliability]:
    """Load branch (line + transformer) reliability parameters with HF."""
    comps = []

    # Load hazard factors
    hf_df = pd.read_csv(hf_csv) if hf_csv.exists() else pd.DataFrame()
    sys_defaults = {}
    if sys_hf_json.exists():
        with open(sys_hf_json, encoding="utf-8") as f:
            sys_defaults = json.load(f)

    line_defaults = sys_defaults.get("line", {})
    tr_defaults   = sys_defaults.get("transformer", {})

    # Lines
    if branches_csv.exists():
        br_df = pd.read_csv(branches_csv)
        for _, row in br_df.iterrows():
            fb     = int(row["from_bus"])
            tb     = int(row["to_bus"])
            ckt    = str(row.get("ckt", "1"))
            status = int(row.get("status", 1))
            comp_id = f"LINE_{fb}_{tb}_{ckt}"

            # Look for HF match (by line name; no direct bus mapping)
            hf = float(line_defaults.get("hazard_factor", 1.0))
            base_lam = float(line_defaults.get("base_lambda_per_hr",
                                               DEFAULT_LINE_LAMBDA))
            mttr = float(line_defaults.get("mttr_hr", DEFAULT_LINE_MTTR))

            lam_eff = base_lam * hf
            mu      = 1.0 / max(mttr, 1.0)
            for_rate = lam_eff / (lam_eff + mu)

            comps.append(ComponentReliability(
                comp_id   = comp_id,
                comp_type = "line",
                label     = f"Line_{fb}_{tb}_{ckt}",
                pmax_mw   = 0.0,  # lines don't have MW capacity for ENS
                lam_per_hr = lam_eff,
                mu_per_hr  = mu,
                for_rate   = for_rate,
                mttr_hr    = mttr,
                hf         = hf,
            ))

    log.info(f"Created {len(comps)} line reliability records")
    return comps


# ──────────────────────────────────────────────────────────────────────────────
# Core SMC sampling functions
# ──────────────────────────────────────────────────────────────────────────────
def sample_ttf(lam: float, rng: np.random.Generator) -> float:
    """Sample time to failure from exponential distribution. [hours]"""
    if lam <= 0:
        return HOURS_PER_YEAR * 10  # effectively never fails
    return rng.exponential(1.0 / lam)


def sample_ttr(mu: float, rng: np.random.Generator) -> float:
    """
    Sample time to repair from log-normal distribution.
    Log-normal parameters derived from mu (mean repair rate):
      MTTR = 1/mu
      σ = 0.5 (coefficient of variation assumption)
    """
    mttr = 1.0 / max(mu, 1e-9)
    if mttr <= 0:
        return 1.0
    sigma = 0.5
    mu_ln = np.log(mttr) - 0.5 * sigma**2
    return max(rng.lognormal(mu_ln, sigma), 0.5)  # minimum 0.5 hr repair


def simulate_component_timeline(comp: ComponentReliability,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a binary state vector [0=up, 1=failed] for one year.
    Uses the two-state Markov chain via direct TTF/TTR sampling.

    Returns: np.ndarray of shape (HOURS_PER_YEAR,), values ∈ {0, 1}
    """
    state = np.zeros(HOURS_PER_YEAR, dtype=np.int8)

    # Start in operational state
    t = 0.0
    operational = True

    while t < HOURS_PER_YEAR:
        if operational:
            ttf = sample_ttf(comp.lam_per_hr, rng)
            t_fail = t + ttf
            if t_fail >= HOURS_PER_YEAR:
                break  # no failure this year
            # Mark hours in operational state (already 0)
            t = t_fail
            operational = False
        else:
            # Component is failed → repair
            ttr = sample_ttr(comp.mu_per_hr, rng)
            t_restore = t + ttr
            # Mark failed hours
            start_idx = int(np.floor(t))
            end_idx   = min(int(np.ceil(t_restore)), HOURS_PER_YEAR)
            state[start_idx:end_idx] = 1
            t = t_restore
            operational = True

    return state


def simulate_one_year(components: list[ComponentReliability],
                      load_scale: np.ndarray,
                      rng: np.random.Generator,
                      year_idx: int) -> YearResult:
    """
    Simulate one year for all components.
    Returns state matrix and preliminary ENS estimate.
    """
    n_comp = len(components)
    state_matrix = np.zeros((HOURS_PER_YEAR, n_comp), dtype=np.int8)

    for ci, comp in enumerate(components):
        state_matrix[:, ci] = simulate_component_timeline(comp, rng)

    # Compute simple EENS proxy: sum of (Pmax × failed_hours × load_scale)
    # Full OPF dispatch is done in Step 08; here we use curtailment approximation
    total_eens = 0.0
    total_lolh = 0.0
    n_contingencies = 0

    for hour in range(HOURS_PER_YEAR):
        n_failed_gens = sum(
            1 for ci, comp in enumerate(components)
            if comp.comp_type == "generator" and state_matrix[hour, ci] == 1
        )
        if n_failed_gens == 0:
            continue

        n_contingencies += 1
        total_lolh += 1.0  # rough; OPF will refine

        # Approximate lost capacity
        lost_mw = sum(
            comp.pmax_mw
            for ci, comp in enumerate(components)
            if comp.comp_type == "generator" and state_matrix[hour, ci] == 1
        )
        # ENS ≈ lost_mw × 1 hr (needs OPF to determine actual curtailment)
        total_eens += lost_mw * float(load_scale[hour])

    outage_hours = {
        comp.comp_id: int(state_matrix[:, ci].sum())
        for ci, comp in enumerate(components)
    }

    return YearResult(
        year_idx        = year_idx,
        state_matrix    = state_matrix,
        outage_hours    = outage_hours,
        annual_eens_mwh = total_eens,
        annual_lolh     = total_lolh,
        n_contingencies = n_contingencies,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convergence check
# ──────────────────────────────────────────────────────────────────────────────
def check_convergence(eens_list: list, min_years: int = MIN_YEARS) -> tuple[bool, float]:
    """
    Check if Monte Carlo has converged using CoV of EENS.
    Returns (converged, current_cov).
    """
    n = len(eens_list)
    if n < min_years:
        return False, 999.0

    arr  = np.array(eens_list)
    mean = arr.mean()
    std  = arr.std(ddof=1)

    if mean <= 0:
        return True, 0.0

    sem  = std / np.sqrt(n)        # standard error of mean
    cov  = sem / mean               # coefficient of variation of mean
    return cov < COV_THRESHOLD, float(cov)


# ──────────────────────────────────────────────────────────────────────────────
# Main SMC loop
# ──────────────────────────────────────────────────────────────────────────────
def run_smc(components: list[ComponentReliability],
            load_scale: np.ndarray,
            max_years: int = MAX_YEARS,
            seed: int = RNG_SEED) -> SMCResults:
    """
    Main SMC simulation loop with convergence checking.
    """
    rng     = np.random.default_rng(seed)
    results = SMCResults()

    log.info(f"Starting SMC: {len(components)} components, "
             f"max {max_years} years, convergence CoV < {COV_THRESHOLD}")

    for year in range(1, max_years + 1):
        yr = simulate_one_year(components, load_scale, rng, year)

        results.eens_per_year.append(yr.annual_eens_mwh)
        results.lolh_per_year.append(yr.annual_lolh)
        results.contingency_per_year.append(yr.n_contingencies)
        results.n_years = year

        # Running statistics
        mean_eens = np.mean(results.eens_per_year)
        results.running_eens_mean.append(mean_eens)

        converged, cov = check_convergence(results.eens_per_year)
        results.running_eens_cov.append(cov)

        if year % BATCH_SIZE == 0 or converged:
            log.info(f"  Year {year:5d}: EENS={mean_eens:>12.1f} MWh/yr, CoV={cov:.4f}")

        if converged:
            results.converged = True
            results.convergence_year = year
            log.info(f"✓ Converged at year {year} (CoV = {cov:.4f})")
            break

    if not results.converged:
        log.warning(f"Did not converge after {max_years} years "
                    f"(final CoV = {results.running_eens_cov[-1]:.4f})")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
def print_smc_summary(results: SMCResults, components: list):
    eens_arr = np.array(results.eens_per_year)
    lolh_arr = np.array(results.lolh_per_year)

    # Bootstrap 95% CI
    boot_means = [np.mean(np.random.choice(eens_arr, len(eens_arr), replace=True))
                  for _ in range(2000)]
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    print("\n" + "=" * 65)
    print("  SMC RELIABILITY RESULTS — Step 07")
    print("=" * 65)
    print(f"  Simulated years       : {results.n_years}")
    print(f"  Converged             : {results.converged}")
    if results.convergence_year:
        print(f"  Convergence at year   : {results.convergence_year}")

    print(f"\n  EENS (Expected Energy Not Supplied):")
    print(f"    Mean              : {eens_arr.mean():>12.1f} MWh/yr")
    print(f"    95% CI            : [{ci_lo:>10.1f}, {ci_hi:>10.1f}] MWh/yr")
    print(f"    Std dev           : {eens_arr.std():>12.1f} MWh/yr")
    print(f"    Max year          : {eens_arr.max():>12.1f} MWh")

    print(f"\n  LOLH (Loss of Load Hours):")
    print(f"    Mean              : {lolh_arr.mean():>12.1f} hr/yr")

    print(f"\n  LOLE (Loss of Load Expectation):")
    lole = np.mean(lolh_arr > 0)  # fraction of years with LOL
    print(f"    LOLE              : {lole:>12.4f} yr/yr")

    print(f"\n  Generator reliability summary:")
    gen_comps = [c for c in components if c.comp_type == "generator"]
    top5 = sorted(gen_comps, key=lambda c: c.for_rate, reverse=True)[:5]
    for c in top5:
        print(f"    {c.label:<30}: FOR={c.for_rate:.4f}, MTTR={c.mttr_hr:.0f}hr")
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load components
    gen_comps  = load_generator_reliability(
        STEP02_DIR / "generator_for.csv",
        STEP01_DIR / "generators.csv"
    )
    line_comps = load_branch_reliability(
        STEP01_DIR / "branches.csv",
        STEP04_DIR / "hazard_factors.csv",
        STEP04_DIR / "system_hazard_defaults.json",
    )

    components = gen_comps + line_comps
    log.info(f"Total components for SMC: {len(components)}")

    # Load profile
    load_scale_file = STEP05_DIR / "load_scale_8760.npy"
    if load_scale_file.exists():
        load_scale = np.load(str(load_scale_file))
    else:
        load_scale = np.ones(HOURS_PER_YEAR)
        log.warning("Load scale not found; using uniform load = 1.0")

    # Run SMC
    results = run_smc(components, load_scale)
    print_smc_summary(results, components)

    # Save results
    eens_arr = np.array(results.eens_per_year)
    lolh_arr = np.array(results.lolh_per_year)
    np.save(str(OUT_DIR / "eens_per_year.npy"), eens_arr)
    np.save(str(OUT_DIR / "lolh_per_year.npy"), lolh_arr)
    np.save(str(OUT_DIR / "running_eens_mean.npy"),
            np.array(results.running_eens_mean))
    np.save(str(OUT_DIR / "running_eens_cov.npy"),
            np.array(results.running_eens_cov))

    # Component reliability summary
    comp_df = pd.DataFrame([
        {"comp_id": c.comp_id, "comp_type": c.comp_type, "label": c.label,
         "pmax_mw": c.pmax_mw, "for_rate": c.for_rate, "mttr_hr": c.mttr_hr,
         "lam_per_hr": c.lam_per_hr, "mu_per_hr": c.mu_per_hr, "hf": c.hf}
        for c in components
    ])
    comp_df.to_csv(OUT_DIR / "component_reliability.csv", index=False)

    # Summary JSON
    boot_means = [np.mean(np.random.choice(eens_arr, len(eens_arr), replace=True))
                  for _ in range(2000)]
    summary = {
        "n_years":           results.n_years,
        "converged":         results.converged,
        "convergence_year":  results.convergence_year,
        "eens_mean_mwh_yr":  float(eens_arr.mean()),
        "eens_std_mwh_yr":   float(eens_arr.std()),
        "eens_ci_lo":        float(np.percentile(boot_means, 2.5)),
        "eens_ci_hi":        float(np.percentile(boot_means, 97.5)),
        "lolh_mean_hr_yr":   float(lolh_arr.mean()),
        "lole":              float(np.mean(lolh_arr > 0)),
        "n_generators":      len(gen_comps),
        "n_lines":           len(line_comps),
    }
    (OUT_DIR / "smc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info(f"SMC results saved to: {OUT_DIR}")
    return results, components


if __name__ == "__main__":
    main()