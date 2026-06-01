"""
fix_cost_function.py
====================
Diagnose and fix cost function issues causing AC-OPF convergence failure
in the Taipower pandapower network.

Root cause summary (from diagnostic history):
  - LoadShed sgens  : cp1 = 50,000 NT$/MWh  [correct, already set]
  - Real generators : cp1 = 1,000 NT$/MWh   [default fallback — MAIN ISSUE]
  - Scaling gap     : 1,000 vs 50,000 → IPOPT objective ill-conditioned
  - Sentinel bounds : some gens have max_p_mw = 9,999 (placeholder value)
  - LoadShed bounds : max_p_mw may be anomalously large or zero

Fixes applied:
  1. Full audit of all existing poly_cost entries
  2. Rebuild poly_cost from scratch with fuel-aware marginal costs
  3. Fix sentinel generators (max_p_mw = 9999 → realistic capacity)
  4. Fix LoadShed sgen bounds (max_p_mw capped to actual bus load)
  5. Validate IPOPT objective scaling (VOLL / mean_gen_cost ratio)
  6. Run two-tier OPF convergence test on the patched network
  7. Save the patched network to a new JSON file
"""

import copy
import warnings
warnings.filterwarnings("ignore")

import pandapower as pp
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────────────────────────────────────
NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
OUT_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network_fixed.json")

# ─────────────────────────────────────────────────────────────────────────────
# Fuel-type marginal cost table  (NT$/MWh)
# Reference: Taipower public cost reports 2023-2024
# ─────────────────────────────────────────────────────────────────────────────
MARGINAL_COST: dict[str, float] = {
    "nuclear" :   200.0,
    "hydro"   :   100.0,
    "coal"    :   800.0,
    "gas"     : 1_500.0,
    "lng"     : 1_600.0,
    "oil"     : 2_500.0,
    "solar"   :     0.0,   # zero marginal cost (renewable)
    "wind"    :     0.0,   # zero marginal cost (renewable)
    "default" : 1_200.0,   # fallback for unrecognised generators
}

# Value of Lost Load — last-resort cost for load shedding sgens
VOLL_NTD_PER_MWH: float = 50_000.0

# Generator name → fuel type keyword mapping (Traditional Chinese + English)
FUEL_KEYWORDS: dict[str, str] = {
    # Nuclear
    "核"  : "nuclear",
    "nuclear": "nuclear",
    # Coal
    "煤"  : "coal",
    "林口": "coal",
    "台中": "coal",
    "大林": "coal",
    "麥寮": "coal",
    "coal": "coal",
    # Natural gas / LNG
    "大潭": "gas",
    "通霄": "gas",
    "興達": "gas",
    "南部": "gas",
    "gas" : "gas",
    "lng" : "gas",
    # Oil
    "協和": "oil",
    "oil" : "oil",
    # Hydro
    "水"  : "hydro",
    "hydro": "hydro",
    # Renewables
    "solar": "solar",
    "太陽" : "solar",
    "wind" : "wind",
    "風"   : "wind",
}

# Capacity assigned to sentinel generators (placeholder max_p_mw = 9,999)
SENTINEL_REPLACEMENT_MW: float = 300.0


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def detect_fuel(name: str) -> str:
    """Return fuel type string for a generator name via keyword matching."""
    for keyword, fuel in FUEL_KEYWORDS.items():
        if keyword.lower() in str(name).lower():
            return fuel
    return "default"


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Load network and audit current cost state
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 1 : COST FUNCTION AUDIT")

net = pp.from_json(str(NET_FILE))
print(f"\n  Network loaded  |  buses={len(net.bus)}  lines={len(net.line)}"
      f"  trafos={len(net.trafo)}  gens={len(net.gen)}  sgens={len(net.sgen)}")

poly = net.poly_cost.copy()
print(f"\n  poly_cost entries total   : {len(poly)}")
print(f"    et = 'gen'              : {(poly.et == 'gen').sum()}")
print(f"    et = 'sgen'             : {(poly.et == 'sgen').sum()}")
print(f"    et = 'ext_grid'         : {(poly.et == 'ext_grid').sum()}")
print(f"    et = 'load'             : {(poly.et == 'load').sum()}")

# ── Generator cost distribution ──────────────────────────────────────────────
gen_audit = poly[poly.et == "gen"][["element", "cp1_eur_per_mw"]].copy()
gen_audit["gen_name"] = gen_audit["element"].map(net.gen["name"].to_dict())
gen_audit["fuel"]     = gen_audit["gen_name"].apply(detect_fuel)

print("\n  --- Generator cp1 distribution ---")
print(f"  cp1 == 1,000  (default fallback) : {(gen_audit.cp1_eur_per_mw == 1_000).sum()} gens")
print(f"  cp1 == 0                         : {(gen_audit.cp1_eur_per_mw == 0).sum()} gens")
print(f"  cp1 in [100, 900]                : {gen_audit.cp1_eur_per_mw.between(100, 900).sum()} gens")
print(f"  cp1 in [900, 2,000]              : {gen_audit.cp1_eur_per_mw.between(900, 2_000).sum()} gens")
print(f"  cp1 > 5,000 (possible VOLL leak) : {(gen_audit.cp1_eur_per_mw > 5_000).sum()} gens")
print(f"\n  cp1 statistics:")
print(f"    min  = {gen_audit.cp1_eur_per_mw.min():>10.1f}")
print(f"    mean = {gen_audit.cp1_eur_per_mw.mean():>10.1f}")
print(f"    max  = {gen_audit.cp1_eur_per_mw.max():>10.1f}")
print(f"\n  Detected fuel distribution:")
print(gen_audit["fuel"].value_counts().to_string())

gens_with_cost    = set(poly[poly.et == "gen"]["element"])
gens_missing_cost = set(net.gen.index) - gens_with_cost
print(f"\n  Generators without a cost entry : {len(gens_missing_cost)}")
if gens_missing_cost:
    print("  Examples :", list(gens_missing_cost)[:10])

# ── LoadShed sgen cost audit ─────────────────────────────────────────────────
sgen_audit = poly[poly.et == "sgen"][["element", "cp1_eur_per_mw"]].copy()
sgen_audit["sgen_name"] = sgen_audit["element"].map(net.sgen["name"].to_dict())
shed_cost_mask = sgen_audit["sgen_name"].astype(str).str.startswith("LoadShed_")

print("\n  --- LoadShed sgen cost ---")
print(f"  LoadShed sgens with cost entry   : {shed_cost_mask.sum()}")
print(f"  Non-LoadShed sgens with cost     : {(~shed_cost_mask).sum()}")
if shed_cost_mask.any():
    ls_cp1 = sgen_audit.loc[shed_cost_mask, "cp1_eur_per_mw"]
    print(f"  LoadShed cp1 range               : [{ls_cp1.min():.0f}, {ls_cp1.max():.0f}]")

# ── Generator bound audit ────────────────────────────────────────────────────
sentinel_gens = net.gen[net.gen.max_p_mw >= 9_998].copy()
print(f"\n  --- Generator bound anomalies ---")
print(f"  Gens with max_p_mw >= 9,998 (sentinel) : {len(sentinel_gens)}")
if not sentinel_gens.empty:
    print(sentinel_gens[["name", "p_mw", "min_p_mw", "max_p_mw"]].head(10).to_string())

# ── LoadShed sgen bound audit ────────────────────────────────────────────────
shed_sgen = net.sgen[net.sgen["name"].astype(str).str.startswith("LoadShed_")].copy()
print(f"\n  --- LoadShed sgen bound anomalies ---")
print(f"  Total LoadShed sgens             : {len(shed_sgen)}")
print(f"  max_p_mw > 10,000 (anomalous)    : {(shed_sgen.max_p_mw > 10_000).sum()}")
print(f"  max_p_mw == 0     (ineffective)  : {(shed_sgen.max_p_mw == 0).sum()}")
if not shed_sgen.empty:
    print(f"  max_p_mw range                   : "
          f"[{shed_sgen.max_p_mw.min():.1f}, {shed_sgen.max_p_mw.max():.1f}]")
if (shed_sgen.max_p_mw > 10_000).sum() > 0:
    print("\n  WARNING: Oversized LoadShed max_p_mw detected — likely VOLL confusion.")
    print(shed_sgen[shed_sgen.max_p_mw > 10_000][["name", "max_p_mw"]].head(5).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Apply fixes
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 2 : APPLYING FIXES")

# Fix A: Clear all stale poly_cost rows and rebuild from scratch
print("\n  [Fix A] Clearing existing poly_cost table for full rebuild ...")
net.poly_cost = net.poly_cost.iloc[0:0].copy()

# Fix B: Real generators — fuel-aware marginal cost
print("  [Fix B] Assigning fuel-aware marginal cost to all generators ...")
n_gen_ok       = 0
n_gen_sentinel = 0

for gen_idx, gen in net.gen.iterrows():
    pmax = float(gen.get("max_p_mw", 0.0))

    # Repair sentinel capacity value
    if pmax >= 9_998.0:
        n_gen_sentinel += 1
        net.gen.at[gen_idx, "max_p_mw"] = SENTINEL_REPLACEMENT_MW
        net.gen.at[gen_idx, "min_p_mw"] = max(0.0, float(gen.get("min_p_mw", 0.0)))

    fuel = detect_fuel(str(gen.get("name", "")))
    cp1  = MARGINAL_COST[fuel]

    pp.create_poly_cost(
        net,
        element         = gen_idx,
        et              = "gen",
        cp0_eur         = 0.0,
        cp1_eur_per_mw  = cp1,
        cp2_eur_per_mw2 = 0.0,
    )
    n_gen_ok += 1

print(f"  [Fix B] Generator cost entries created : {n_gen_ok}")
print(f"  [Fix B] Sentinel generators repaired   : {n_gen_sentinel}  "
      f"(max_p_mw set to {SENTINEL_REPLACEMENT_MW} MW)")

# Fix C: External grids — priced as LNG imports
print("  [Fix C] Assigning marginal cost to external grids (LNG import rate) ...")
for ext_idx in net.ext_grid.index:
    pp.create_poly_cost(
        net,
        element         = ext_idx,
        et              = "ext_grid",
        cp0_eur         = 0.0,
        cp1_eur_per_mw  = MARGINAL_COST["lng"],
        cp2_eur_per_mw2 = 0.0,
    )
print(f"  [Fix C] External grid cost entries     : {len(net.ext_grid)}")

# Fix D: LoadShed sgens — VOLL cost + bounds repair
print("  [Fix D] Assigning VoLL cost to LoadShed sgens + repairing bounds ...")
shed_mask  = net.sgen["name"].astype(str).str.startswith("LoadShed_")
n_shed_ok  = 0
n_pmax_fix = 0

for sgen_idx in net.sgen.index[shed_mask]:
    cur_pmax = float(net.sgen.at[sgen_idx, "max_p_mw"])

    # Repair oversized or zero max_p_mw — cap to actual bus load
    if cur_pmax > 5_000.0 or cur_pmax <= 0.0:
        bus_id   = net.sgen.at[sgen_idx, "bus"]
        bus_load = net.load[net.load.bus == bus_id]["p_mw"].sum()
        safe_cap = max(float(bus_load), 1.0) if bus_load > 0 else 50.0
        net.sgen.at[sgen_idx, "max_p_mw"] = safe_cap
        n_pmax_fix += 1

    pp.create_poly_cost(
        net,
        element         = sgen_idx,
        et              = "sgen",
        cp0_eur         = 0.0,
        cp1_eur_per_mw  = VOLL_NTD_PER_MWH,
        cp2_eur_per_mw2 = 0.0,
    )
    n_shed_ok += 1

print(f"  [Fix D] LoadShed cost entries created  : {n_shed_ok}")
print(f"  [Fix D] LoadShed max_p_mw repaired     : {n_pmax_fix}")

# Fix E: Non-shed sgens (wind, solar, storage) — fuel-aware cost
print("  [Fix E] Assigning marginal cost to non-LoadShed sgens ...")
n_nonshed = 0
for sgen_idx in net.sgen.index[~shed_mask]:
    fuel = detect_fuel(str(net.sgen.at[sgen_idx, "name"]))
    cp1  = MARGINAL_COST.get(fuel, 0.0)
    pp.create_poly_cost(
        net,
        element         = sgen_idx,
        et              = "sgen",
        cp0_eur         = 0.0,
        cp1_eur_per_mw  = cp1,
        cp2_eur_per_mw2 = 0.0,
    )
    n_nonshed += 1
print(f"  [Fix E] Non-shed sgen cost entries     : {n_nonshed}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Verify patched cost table
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 3 : POST-FIX VERIFICATION")

poly_new  = net.poly_cost.copy()
gen_cp1   = poly_new[poly_new.et == "gen"]["cp1_eur_per_mw"]
sgen_cp1  = poly_new[poly_new.et == "sgen"]["cp1_eur_per_mw"]

print(f"\n  Total poly_cost entries after fix  : {len(poly_new)}")
print(f"    et = 'gen'                       : {(poly_new.et == 'gen').sum()}")
print(f"    et = 'sgen'                      : {(poly_new.et == 'sgen').sum()}")
print(f"    et = 'ext_grid'                  : {(poly_new.et == 'ext_grid').sum()}")

print(f"\n  Generator cp1  [{gen_cp1.min():.0f} – {gen_cp1.max():.0f}]  "
      f"mean = {gen_cp1.mean():.0f}  NT$/MWh")
print(f"  Sgen      cp1  [{sgen_cp1.min():.0f} – {sgen_cp1.max():.0f}]  "
      f"mean = {sgen_cp1.mean():.0f}  NT$/MWh")

# IPOPT objective scaling check
ratio = VOLL_NTD_PER_MWH / gen_cp1.mean()
print(f"\n  IPOPT scaling check:")
print(f"    VoLL / mean_gen_cost ratio = {ratio:.1f}x")
if ratio > 100:
    print("  WARNING: ratio > 100 — IPOPT may struggle with objective scaling.")
    print("  Consider scaling all costs down by a common factor (e.g. 1/1000).")
else:
    print("  Scaling ratio is acceptable for IPOPT  ✓")

# Missing cost entries
missing_gen  = set(net.gen.index)  - set(poly_new[poly_new.et == "gen"]["element"])
missing_sgen = set(net.sgen.index) - set(poly_new[poly_new.et == "sgen"]["element"])
print(f"\n  Generators missing cost entry  : {len(missing_gen)}")
print(f"  Sgens missing cost entry       : {len(missing_sgen)}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Two-tier OPF convergence test
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 4 : OPF CONVERGENCE TEST")

def run_opf_test(label: str, net_in: pp.pandapowerNet) -> bool:
    """Warm-start AC power flow, then run AC-OPF. Return True if converged."""
    try:
        pp.runpp(net_in, algorithm="nr", init="dc", max_iteration=100,
                 check_connectivity=False, enforce_q_lims=False, numba=False)
        pf_status = "converged" if net_in.converged else "not converged"
        v_range   = (f"V=[{net_in.res_bus.vm_pu.min():.3f},"
                     f"{net_in.res_bus.vm_pu.max():.3f}]"
                     if net_in.converged else "N/A")
        print(f"    AC power flow (warm-start) : {pf_status}  {v_range}")
    except Exception as exc:
        print(f"    AC power flow failed       : {exc!s:.80}")

    try:
        pp.runopp(net_in, init="pf", numba=False,
                  check_connectivity=False, verbose=False)
        if net_in.converged:
            cost  = float(net_in.res_cost)
            slack = float(net_in.res_ext_grid.p_mw.iloc[0])
            vmin  = net_in.res_bus.vm_pu.min()
            vmax  = net_in.res_bus.vm_pu.max()
            print(f"    AC-OPF                     : CONVERGED ✓")
            print(f"      cost  = {cost:,.0f} NT$/hr")
            print(f"      slack = {slack:+.1f} MW")
            print(f"      V     = [{vmin:.4f}, {vmax:.4f}] pu")
            return True
        print(f"    AC-OPF                     : net.converged = False")
        return False
    except Exception as exc:
        print(f"    AC-OPF                     : FAILED — {type(exc).__name__}: {exc!s:.90}")
        return False


# Test 1: Relaxed bounds — isolates cost/topology issues from tight constraints
print("\n  Test 1: Relaxed bounds  [0.80 – 1.20 pu]  |  no thermal limit")
nt1 = copy.deepcopy(net)
nt1.bus["min_vm_pu"]             = 0.80
nt1.bus["max_vm_pu"]             = 1.20
nt1.sgen.loc[shed_mask, "controllable"] = True
nt1.line["max_loading_percent"]  = 1e6
nt1.trafo["max_loading_percent"] = 1e6
result_t1 = run_opf_test("Relaxed", nt1)

# Test 2: Taipower operating standard — full constraint set
print("\n  Test 2: Taipower standard  [0.90 – 1.10 pu]  |  thermal limit 100 %")
nt2 = copy.deepcopy(net)
nt2.bus["min_vm_pu"]             = 0.90
nt2.bus["max_vm_pu"]             = 1.10
nt2.sgen.loc[shed_mask, "controllable"] = True
nt2.line["max_loading_percent"]  = 100.0
nt2.trafo["max_loading_percent"] = 100.0
result_t2 = run_opf_test("Taipower", nt2)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Save patched network
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 5 : SAVE PATCHED NETWORK")

net.to_json(str(OUT_FILE))
print(f"\n  Patched network saved to:")
print(f"    {OUT_FILE}")
print(f"\n  To use in step09 / step10, update the network path:")
print(f"    NET_FILE = Path(r'{OUT_FILE}')")


# ─────────────────────────────────────────────────────────────────────────────
# Summary and interpretation guide
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")

print(f"\n  Test 1 (relaxed bounds)  : {'PASSED ✓' if result_t1 else 'FAILED ✗'}")
print(f"  Test 2 (Taipower bounds) : {'PASSED ✓' if result_t2 else 'FAILED ✗'}")
print("""
  Interpretation:
    Both tests PASS   → Cost function fixed; network ready for step09/step10.

    Test 1 PASS only  → Cost function is OK; constraint-driven infeasibility.
                         Check voltage profile with check_buses.py.
                         Consider relaxing V bounds to [0.90, 1.10] first.

    Test 1 FAIL       → Fundamental topology or impedance issue remains.
                         Run operating_point.py for deeper diagnosis.

    IPOPT ratio > 100 → Scale all cost values down by 1/1000 and retry.
                         (e.g. NT$/MWh → kNT$/MWh; VOLL = 50 instead of 50,000)
""")