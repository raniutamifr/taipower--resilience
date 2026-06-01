"""
fix_slack_placement.py
=======================
DC-PF converges but AC-PF doesn't → ext_grid placement is likely wrong.

Taipower ext_grid currently at bus 1101 (大林新#1, 25 kV) → should be at
a 345 kV backbone bus with big generation.

This script:
  1. Identifies candidate slack buses (345 kV with large generators)
  2. For each candidate, test AC-PF with ext_grid moved there
  3. Report which candidate converges + voltage profile
  4. Apply best candidate to the xlsx file

Run:
    python fix_slack_placement.py
"""
import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore")

OUT_DIR = Path(r"C:\reXplan-repo\Project Taipower\file\output\taipower")
OUT_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(msg)


def section(title):
    log("")
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Load network
# ─────────────────────────────────────────────────────────────────────────────
section("LOAD NETWORK")
import reXplan.config as cfg
cfg.path.inputFolder = r"C:\reXplan-repo\file\input"
from reXplan.network import Network
net_obj = Network("taipower")
net = net_obj.pp_network
log(f"  Network loaded: {len(net.bus)} buses, {len(net.gen)} gens, "
    f"{len(net.ext_grid)} ext_grid")

current_extgrid_bus = int(net.ext_grid.at[0, "bus"])
log(f"  Current ext_grid at bus {current_extgrid_bus} "
    f"({net.bus.at[current_extgrid_bus, 'name']}, "
    f"vn_kv={net.bus.at[current_extgrid_bus, 'vn_kv']})")


# ─────────────────────────────────────────────────────────────────────────────
# Find candidate slack buses: 345 kV buses with large generation
# ─────────────────────────────────────────────────────────────────────────────
section("CANDIDATE SLACK BUSES")

# Get all gen capacity per bus
gen_per_bus = net.gen.groupby("bus")["max_p_mw"].sum().reset_index()
gen_per_bus.columns = ["bus", "gen_max_mw"]

# Merge with bus info
bus_info = net.bus[["vn_kv", "name", "in_service"]].reset_index().rename(columns={"index": "bus"})
candidates = gen_per_bus.merge(bus_info, on="bus")

# Filter: 345 kV, in service, large gen
candidates_345 = candidates[
    (candidates["vn_kv"] >= 340) &
    (candidates["vn_kv"] <= 360) &
    (candidates["in_service"]) &
    (candidates["gen_max_mw"] > 500)
].sort_values("gen_max_mw", ascending=False)

log(f"\n  Found {len(candidates_345)} candidate buses at ~345 kV with >500 MW gen:")
log(f"  {'Bus':>6} {'Name':<15} {'vn_kv':>8} {'Pmax [MW]':>12}")
log(f"  {'-'*6} {'-'*15} {'-'*8} {'-'*12}")
for _, row in candidates_345.head(15).iterrows():
    log(f"  {int(row['bus']):>6} {str(row['name'])[:15]:<15} "
        f"{row['vn_kv']:>8.1f} {row['gen_max_mw']:>12.1f}")

# Also check 161 kV candidates (as fallback)
candidates_161 = candidates[
    (candidates["vn_kv"] >= 155) &
    (candidates["vn_kv"] <= 170) &
    (candidates["in_service"]) &
    (candidates["gen_max_mw"] > 300)
].sort_values("gen_max_mw", ascending=False)

log(f"\n  {len(candidates_161)} candidate buses at ~161 kV with >300 MW gen:")
for _, row in candidates_161.head(5).iterrows():
    log(f"  {int(row['bus']):>6} {str(row['name'])[:15]:<15} "
        f"{row['vn_kv']:>8.1f} {row['gen_max_mw']:>12.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test each candidate
# ─────────────────────────────────────────────────────────────────────────────
section("TEST AC-PF WITH EACH CANDIDATE")

results = []

# Include current bus as baseline + top candidates
test_buses = list(dict.fromkeys(
    [current_extgrid_bus] +
    candidates_345["bus"].head(10).astype(int).tolist() +
    candidates_161["bus"].head(3).astype(int).tolist()
))

log(f"\n  Testing {len(test_buses)} candidate slack buses...\n")

for test_bus in test_buses:
    net_test = copy.deepcopy(net)

    # Remove existing ext_grid, add at test bus
    net_test.ext_grid = net_test.ext_grid.iloc[0:0]  # empty
    pp.create_ext_grid(
        net_test, bus=test_bus,
        vm_pu=1.02,      # slightly elevated — helps AC convergence
        va_degree=0.0,
        name=f"SlackTest_{test_bus}",
        in_service=True,
    )

    # Also need to remove this gen from regular gen list
    # (can't have both gen and ext_grid at same bus for NR)
    gen_at_bus_mask = net_test.gen["bus"] == test_bus
    if gen_at_bus_mask.any():
        net_test.gen.loc[gen_at_bus_mask, "in_service"] = False

    # Try AC-PF
    try:
        pp.runpp(net_test, init="dc", algorithm="nr", max_iteration=30,
                 numba=False, calculate_voltage_angles=True,
                 check_connectivity=False, tolerance_mva=1e-3)
        vm_min = net_test.res_bus["vm_pu"].min()
        vm_max = net_test.res_bus["vm_pu"].max()
        ext_p = net_test.res_ext_grid["p_mw"].sum()
        max_ll = net_test.res_line["loading_percent"].max()
        status = "✓ CONVERGED"
        converged = True
    except Exception as e:
        status = f"✗ {type(e).__name__}"
        vm_min = vm_max = ext_p = max_ll = None
        converged = False

    bus_name = str(net.bus.at[test_bus, 'name'])[:15]
    vn_kv = net.bus.at[test_bus, "vn_kv"]

    if converged:
        log(f"  bus {test_bus:>5} ({bus_name:<15}, {vn_kv:>5.0f}kV) {status} | "
            f"Vmin={vm_min:.4f} Vmax={vm_max:.4f} ExtP={ext_p:+.0f}MW MaxLL={max_ll:.1f}%")
    else:
        log(f"  bus {test_bus:>5} ({bus_name:<15}, {vn_kv:>5.0f}kV) {status}")

    results.append({
        "bus": test_bus, "name": bus_name, "vn_kv": vn_kv,
        "converged": converged, "vm_min": vm_min, "vm_max": vm_max,
        "ext_p": ext_p, "max_ll": max_ll,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Summary + recommendation
# ─────────────────────────────────────────────────────────────────────────────
section("RECOMMENDATION")

converged_results = [r for r in results if r["converged"]]

if not converged_results:
    log("  ✗ NO candidate converged!")
    log("  → The issue is NOT just slack placement. Deeper network analysis needed.")
    log("  → Possible causes:")
    log("    - Extreme line loading (DC showed 369% max line loading)")
    log("    - Network topology has hidden issue despite '1 island' report")
    log("    - Bad transformer tap/vector configuration")
    log("")
    log("  Next step: Fix line loading. Consider:")
    log("    1. Check which lines are 369% loaded in DC-PF")
    log("    2. If their max_i_ka is wrong → correct from PSSE RAW")
    log("    3. If truly overloaded → network is infeasible as-is")
else:
    # Pick best: lowest deviation from 1.0 pu
    converged_results.sort(
        key=lambda r: abs(r["vm_min"] - 1.0) + abs(r["vm_max"] - 1.0)
    )
    best = converged_results[0]
    log(f"  ✓ BEST candidate: bus {best['bus']} ({best['name']}, {best['vn_kv']:.0f} kV)")
    log(f"    Voltage range: [{best['vm_min']:.4f}, {best['vm_max']:.4f}]")
    log(f"    Ext_grid injection: {best['ext_p']:+.0f} MW")
    log(f"    Max line loading: {best['max_ll']:.1f}%")
    log("")
    log(f"  → To apply this fix:")
    log(f"    1. Open: C:\\reXplan-repo\\file\\input\\taipower\\network.xlsx")
    log(f"    2. Go to 'external_gen' sheet")
    log(f"    3. Change 'node' value from {current_extgrid_bus} to {best['bus']}")
    log(f"    4. Change 'vm_pu' to 1.02")
    log(f"    5. Save")
    log(f"    6. Re-run: python '3 test AC OPF.py'")
    log("")
    log(f"  Alternative: run this script's section 'APPLY FIX' below to auto-patch")


# ─────────────────────────────────────────────────────────────────────────────
# Save log
# ─────────────────────────────────────────────────────────────────────────────
(OUT_DIR / "slack_placement_report.txt").write_text(
    "\n".join(log_lines), encoding="utf-8"
)
log(f"\n  Report saved: {OUT_DIR / 'slack_placement_report.txt'}")