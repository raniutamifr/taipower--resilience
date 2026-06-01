"""
Step 3 — AC-OPF Convergence Test (THE MAIN GOAL)
==================================================

Goal tonight: prove that the Taipower 1970-bus network converges in
Julia PowerModels + IPOPT via pp.runpm_ac_opf (the reXplan default path).

Strategy:
  A. Load network via reXplan Network()
  B. Warmup Julia with case5 (small, always converges) — one-time ~30s
  C. Attempt pp.runpm_ac_opf on full Taipower network
  D. If fails: cascade to pp.runopp (PYPOWER fallback)
  E. Print convergence status, cost, ENS, voltage profile

This does NOT use Hazard/Fragility/SMC yet. Just raw OPF on the base net.
Tonight's milestone: "AC-OPF converges without hazards".

Run order:
  1. python 1_convert_to_rexplan_xlsx.py
  2. python 2_test_network_load.py        (must pass)
  3. python 3_test_ac_opf.py              (this script)

Output:
  - acopf_result.json  — convergence flag, cost, ENS, timing
  - acopf_debug.log    — full log for troubleshooting
"""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

# Suppress noisy "Missing scaling field" warnings from reXplan profile allocation
warnings.filterwarnings("ignore", message='Missing "scaling" field')
warnings.filterwarnings("ignore", message='Input parameter')

import numpy as np
import pandapower as pp

OUT_DIR = Path(r"C:\reXplan-repo\Project Taipower\file\output\taipower")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUT_DIR / "acopf_result.json"
LOG_FILE    = OUT_DIR / "acopf_debug.log"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)


def section(title):
    log("")
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


def save_log():
    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")


def save_result(res):
    RESULT_JSON.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Julia warmup — case5
# ─────────────────────────────────────────────────────────────────────────────
def warmup_julia():
    section("B. Julia warmup (one-time, ~30-60s)")
    from pandapower.networks import case5
    net = case5()

    # Minimum cost so runpm_ac_opf works
    for gi in net.gen.index:
        if not ((net.poly_cost["et"] == "gen") & (net.poly_cost["element"] == gi)).any():
            pp.create_poly_cost(net, gi, "gen",
                                cp2_eur_per_mw2=0.01,
                                cp1_eur_per_mw=10.0, cp0_eur=0.0)
    for egi in net.ext_grid.index:
        if not ((net.poly_cost["et"] == "ext_grid") & (net.poly_cost["element"] == egi)).any():
            pp.create_poly_cost(net, egi, "ext_grid",
                                cp2_eur_per_mw2=0.0,
                                cp1_eur_per_mw=100.0, cp0_eur=0.0)

    t0 = time.perf_counter()
    try:
        pp.runpm_ac_opf(net)
        elapsed = time.perf_counter() - t0
        if net.OPF_converged:
            log(f"  ✓ Julia+IPOPT warmup: {elapsed:.1f}s (case5 converged)")
            return True
        else:
            log(f"  ✗ case5 did not converge")
            return False
    except Exception as e:
        elapsed = time.perf_counter() - t0
        log(f"  ✗ Julia warmup FAILED ({elapsed:.1f}s)")
        log(f"    {type(e).__name__}: {e}")
        log(f"\n    Likely cause:")
        log(f"      - PandaModels package not installed in Julia")
        log(f"        Fix: julia -e 'using Pkg; Pkg.add(PackageSpec(name=\"PandaModels\", version=\"0.7.1\"))'")
        log(f"      - Julia not in PATH")
        log(f"        Fix: add C:\\Users\\user\\AppData\\Local\\Programs\\Julia-1.10.9\\bin to PATH")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Taipower AC-OPF attempts
# ─────────────────────────────────────────────────────────────────────────────
def try_powermodels(net) -> dict:
    """Attempt 1: pp.runpm_ac_opf (the reXplan-native path)."""
    log("\n  Attempt 1: pp.runpm_ac_opf (Julia+IPOPT)")
    t0 = time.perf_counter()
    try:
        pp.runpm_ac_opf(net)
        elapsed = time.perf_counter() - t0
        if net.OPF_converged:
            log(f"    ✓ Converged in {elapsed:.1f}s")
            return {"converged": True, "solver": "pm_ac_opf", "time_s": elapsed}
        else:
            log(f"    ✗ Did not converge ({elapsed:.1f}s)")
            return {"converged": False, "solver": "pm_ac_opf", "time_s": elapsed}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        log(f"    ✗ EXCEPTION ({elapsed:.1f}s): {type(e).__name__}: {e}")
        return {"converged": False, "solver": "pm_ac_opf", "time_s": elapsed,
                "error": str(e)}


def try_pandapower_opf(net) -> dict:
    """Attempt 2: pp.runopp (PYPOWER fallback)."""
    log("\n  Attempt 2: pp.runopp (PYPOWER fallback)")
    t0 = time.perf_counter()
    try:
        pp.runpp(net, numba=False, calculate_voltage_angles=True,
                 check_connectivity=False)
        pp.runopp(net, verbose=False, numba=False,
                  calculate_voltage_angles=True,
                  check_connectivity=False,
                  init="pf", max_iteration=300)
        elapsed = time.perf_counter() - t0
        if net.OPF_converged:
            log(f"    ✓ Converged in {elapsed:.1f}s")
            return {"converged": True, "solver": "pp_runopp", "time_s": elapsed}
        else:
            log(f"    ✗ Did not converge ({elapsed:.1f}s)")
            return {"converged": False, "solver": "pp_runopp", "time_s": elapsed}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        log(f"    ✗ EXCEPTION ({elapsed:.1f}s): {type(e).__name__}: {e}")
        return {"converged": False, "solver": "pp_runopp", "time_s": elapsed,
                "error": str(e)}


def extract_results(net, convergence_info) -> dict:
    """Extract summary metrics from converged OPF."""
    res = {**convergence_info}

    try:
        res["total_cost_ntd_hr"] = float(net.res_cost)
    except Exception:
        res["total_cost_ntd_hr"] = None

    res["total_gen_mw"]  = float(net.res_gen["p_mw"].sum())
    res["total_ext_mw"]  = float(net.res_ext_grid["p_mw"].sum())
    res["total_load_mw"] = float(net.load.loc[net.load["in_service"], "p_mw"].sum())
    res["total_loss_mw"] = (res["total_gen_mw"] + res["total_ext_mw"]
                             - res["total_load_mw"])

    res["vm_min_pu"] = float(net.res_bus["vm_pu"].min())
    res["vm_max_pu"] = float(net.res_bus["vm_pu"].max())
    res["vm_mean_pu"] = float(net.res_bus["vm_pu"].mean())

    res["n_voltage_violations"] = int(
        ((net.res_bus["vm_pu"] < 0.94) | (net.res_bus["vm_pu"] > 1.06)).sum()
    )
    res["n_line_overloads"] = int(
        (net.res_line["loading_percent"] > 100.0).sum()
    )
    res["max_line_loading_pct"] = float(
        net.res_line["loading_percent"].max()
    )

    # ENS via load-shed sgens (if any)
    if len(net.sgen) > 0:
        shed_mask = net.sgen["name"].astype(str).str.startswith("LoadShed_sgen_")
        if shed_mask.any():
            try:
                ens = float(net.res_sgen.loc[shed_mask, "p_mw"].clip(lower=0).sum())
                res["ens_mw"] = ens
            except Exception:
                res["ens_mw"] = None

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    section("A. Load Taipower network via reXplan")

    try:
        import reXplan
        from reXplan.network import Network
        # FIX: reXplan uses relative paths by default. Override to absolute path.
        import reXplan.config as cfg
        cfg.path.inputFolder = r"C:\reXplan-repo\file\input"
    except ImportError:
        log("  ✗ reXplan not importable. Run: pip install -e C:\\reXplan-repo")
        save_log()
        sys.exit(1)

    try:
        net_obj = Network("taipower")
        net = net_obj.pp_network
        log(f"  ✓ Network loaded. Buses={len(net.bus)}, Gens={len(net.gen)}, "
            f"Lines={len(net.line)}, Trafos={len(net.trafo)}, Loads={len(net.load)}")
    except Exception as e:
        log(f"  ✗ Network() failed: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        save_log()
        sys.exit(1)

    # Warmup
    warmup_ok = warmup_julia()

    # Taipower AC-OPF
    section("C. Taipower AC-OPF attempts")

    result = None
    if warmup_ok:
        result = try_powermodels(net)

    if result is None or not result["converged"]:
        result = try_pandapower_opf(net)

    # Final status
    section("D. Final result")
    if result and result["converged"]:
        result = extract_results(net, result)
        log(f"\n  ✓ AC-OPF CONVERGED")
        log(f"    Solver        : {result['solver']}")
        log(f"    Time          : {result['time_s']:.1f} s")
        log(f"    Total cost    : {result.get('total_cost_ntd_hr'):,.0f} NT$/hr" if result.get('total_cost_ntd_hr') else "    Cost: N/A")
        log(f"    Total gen     : {result['total_gen_mw']:.0f} MW")
        log(f"    Total ext_grid: {result['total_ext_mw']:.0f} MW")
        log(f"    Total load    : {result['total_load_mw']:.0f} MW")
        log(f"    Total loss    : {result['total_loss_mw']:.1f} MW")
        log(f"    Vmin / Vmax   : {result['vm_min_pu']:.4f} / {result['vm_max_pu']:.4f} pu")
        log(f"    V violations  : {result['n_voltage_violations']}")
        log(f"    Line overloads: {result['n_line_overloads']} (max {result['max_line_loading_pct']:.1f}%)")
        if "ens_mw" in result:
            log(f"    ENS (shed)    : {result['ens_mw']:.2f} MW")
        log("\n  MILESTONE ACHIEVED: AC-OPF works. Ready for hazard integration.")
    else:
        log(f"\n  ✗ AC-OPF FAILED")
        log(f"    Both PowerModels and PYPOWER failed.")
        log(f"    Check acopf_debug.log for error details.")
        log(f"\n  Likely root causes (in order of probability):")
        log(f"    1. Network has infeasibility (bad slack config, bad limits)")
        log(f"    2. Cost function not set up properly (check poly_cost table)")
        log(f"    3. Voltage bounds too tight for base case")
        log(f"    4. ext_grid not providing slack correctly")

    save_result(result)
    save_log()
    log(f"\n  Results: {RESULT_JSON}")
    log(f"  Log: {LOG_FILE}")


if __name__ == "__main__":
    main()