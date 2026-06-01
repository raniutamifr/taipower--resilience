"""
Step 2 — Preflight: Load network via reXplan Network() class
=============================================================

Goal: prove that the converted network.xlsx is structurally compatible
with reXplan BEFORE attempting any OPF. Fail fast at this stage is cheap;
failing during pp.runpm_ac_opf is expensive (Julia warmup ~30s wasted).

This script tries:
  1. `import reXplan`                  → reXplan package installed?
  2. `Network("taipower")`             → sheets parse correctly?
  3. `net.pp_network`                  → pandapower net was built?
  4. Basic sanity checks               → gen/load/bus/line counts reasonable?
  5. `pp.runpp()` AC-PF on pp_network  → does base case converge at all?

If all 5 pass → you're ready for AC-OPF.
If any fails → the error message tells you exactly what's wrong.
"""

import sys
import traceback
from pathlib import Path

import pandapower as pp

# ─────────────────────────────────────────────────────────────────────────────
# FIXED: Use absolute paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\reXplan-repo\Project Taipower")
REXPLAN_ROOT = Path(r"C:\reXplan-repo")
INPUT_ROOT = Path(r"C:\reXplan-repo\file\input")

# Ensure the network.xlsx is accessible
NETWORK_XLSX = INPUT_ROOT / "taipower" / "network.xlsx"


def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check(label, ok, detail=""):
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    section("PREFLIGHT — reXplan network load test")

    # Step 1: reXplan importable?
    section("1. reXplan package import")
    try:
        import reXplan
        check("import reXplan", True, f"from {reXplan.__file__}")
        from reXplan.network import Network
        from reXplan.simulation import Sim
        check("import Network, Sim classes", True)
    except ImportError as e:
        check("import reXplan", False, str(e))
        print("\n→ Fix: cd to reXplan-repo directory, run: pip install -e .")
        sys.exit(1)

    # Step 2: Override config path to use absolute path
    section("2. Setting up network path")
    import reXplan.config as cfg
    
    # FIX: Override the input folder to absolute path
    cfg.path.inputFolder = str(INPUT_ROOT)
    simulation_name = "taipower"
    xlsx_path = cfg.path.networkFile(simulation_name)
    check("networkFile path", True, xlsx_path)

    if not Path(xlsx_path).exists():
        check("network.xlsx exists", False,
              f"Expected at: {xlsx_path}")
        print("\n→ Copy network.xlsx to the correct location:")
        print(f"  mkdir -Force {INPUT_ROOT / 'taipower'}")
        print(f"  Copy-Item '{NETWORK_XLSX}' '{xlsx_path}'")
        print(f"\n  Current working directory: {Path.cwd()}")
        sys.exit(1)
    else:
        check("network.xlsx exists", True, xlsx_path)

    # Step 3: Network() constructor works?
    section("3. Network() instantiation")
    try:
        net = Network(simulation_name)
        check("Network() constructor succeeded", True)
    except Exception as e:
        check("Network() constructor", False, type(e).__name__)
        print(f"\n  Full error:\n{traceback.format_exc()}")
        print("\n→ The xlsx likely has a column mismatch with reXplan's const.py.")
        print("  Check conversion_report.txt in your output folder.")
        sys.exit(1)

    # Step 4: pp_network is populated?
    section("4. pandapower network structure")
    pp_net = net.pp_network
    check("pp_network attribute exists", pp_net is not None)
    if pp_net is None:
        sys.exit(1)

    n_bus  = len(pp_net.bus)
    n_line = len(pp_net.line)
    n_trf  = len(pp_net.trafo)
    n_ld   = len(pp_net.load)
    n_gen  = len(pp_net.gen)
    n_ext  = len(pp_net.ext_grid)
    n_cost = len(pp_net.poly_cost)

    check("buses", n_bus > 100, f"{n_bus}")
    check("lines", n_line > 100, f"{n_line}")
    check("trafos", n_trf > 10, f"{n_trf}")
    check("loads", n_ld > 10, f"{n_ld}")
    check("gens", n_gen > 10, f"{n_gen}")
    check("ext_grid", n_ext >= 1, f"{n_ext}")
    check("poly_cost", n_cost >= n_gen, f"{n_cost} (need >= n_gen + n_ext)")

    # Step 5: Basic AC-PF
    section("5. AC-PF base case (sanity check)")
    
    # Scale load to 85% for base case
    load_scale = 0.85
    pp_net.load["p_mw"] *= load_scale
    pp_net.load["q_mvar"] *= load_scale
    
    # Scale load shedding capacity
    shed_mask = pp_net.sgen["name"].str.startswith("LoadShed_sgen_") if len(pp_net.sgen) > 0 else []
    if len(shed_mask) > 0:
        pp_net.sgen.loc[shed_mask, "max_p_mw"] *= load_scale
    
    try:
        pp.runpp(pp_net, numba=False, calculate_voltage_angles=True,
                 check_connectivity=False, max_iteration=100, verbose=False)
        if pp_net.converged:
            vm_min = pp_net.res_bus["vm_pu"].min()
            vm_max = pp_net.res_bus["vm_pu"].max()
            max_ll = pp_net.res_line["loading_percent"].max() if len(pp_net.res_line) > 0 else 0
            total_load = pp_net.load["p_mw"].sum()
            total_gen = pp_net.gen["max_p_mw"].sum()
            check("AC-PF converged", True,
                  f"Vmin={vm_min:.4f}, Vmax={vm_max:.4f}, MaxLine={max_ll:.1f}%, Load={total_load:.0f}MW, GenCap={total_gen:.0f}MW")
        else:
            check("AC-PF converged", False, "runpp did not reach convergence")
            sys.exit(1)
    except Exception as e:
        check("AC-PF", False, str(e))
        print(f"\n  Full error:\n{traceback.format_exc()}")
        sys.exit(1)

    # All passed
    section("✓✓✓ PREFLIGHT PASSED ✓✓✓")
    print(f"""
  Network loaded via reXplan successfully!
  
   Network Statistics:
  ─────────────────────────────────────────────────
    Buses        : {n_bus}
    Lines        : {n_line}
    Transformers : {n_trf}
    Loads        : {n_ld}
    Generators   : {n_gen}
    External Grid: {n_ext}
    Cost entries : {n_cost}
  
   Power Flow Results:
  ─────────────────────────────────────────────────
    Load scale   : {load_scale:.0%}
    Total load   : {pp_net.load['p_mw'].sum():.0f} MW
    Gen capacity : {pp_net.gen['max_p_mw'].sum():.0f} MW
    Vmin         : {pp_net.res_bus['vm_pu'].min():.4f} pu
    Vmax         : {pp_net.res_bus['vm_pu'].max():.4f} pu
  
   Base case AC-PF: CONVERGED

   Next step:
    python 3_test_ac_opf.py   → try pp.runpm_ac_opf (Julia + IPOPT)
""")


if __name__ == "__main__":
    main()