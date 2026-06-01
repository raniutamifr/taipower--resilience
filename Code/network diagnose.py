"""
diagnose_network.py
====================
AC-PF fails → network has a fundamental issue. This script systematically
identifies WHICH issue:

  1. Topology: islands, disconnected components
  2. Slack: ext_grid config, PV/slack placement
  3. Load vs generation balance: is there enough generation?
  4. Voltage bounds feasibility
  5. Line impedance: any extreme values (shorts, opens)?
  6. Transformer config issues
  7. Try DC power flow (if DC converges but AC doesn't → Q/voltage issue)
  8. Try AC-PF with progressively relaxed settings

Run:
    python diagnose_network.py

Output:
  - Console report with specific findings
  - diagnose_report.txt saved
"""
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top

warnings.filterwarnings("ignore")

OUT_DIR = Path(r"C:\reXplan-repo\Project Taipower\file\output\taipower")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "diagnose_report.txt"

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
try:
    import reXplan.config as cfg
    cfg.path.inputFolder = r"C:\reXplan-repo\file\input"
    from reXplan.network import Network
    net_obj = Network("taipower")
    net = net_obj.pp_network
    log(f"  ✓ Network loaded: {len(net.bus)} buses, {len(net.line)} lines, "
        f"{len(net.trafo)} trafos, {len(net.gen)} gens, {len(net.load)} loads, "
        f"{len(net.ext_grid)} ext_grid")
except Exception as e:
    log(f"  ✗ {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Topology: Islands
# ─────────────────────────────────────────────────────────────────────────────
section("1. TOPOLOGY — ISLAND DETECTION")

mg = top.create_nxgraph(net, respect_switches=False, include_out_of_service=False)
islands = list(top.connected_components(mg))
log(f"  Total connected components: {len(islands)}")

island_sizes = sorted([len(i) for i in islands], reverse=True)
log(f"  Top 10 island sizes: {island_sizes[:10]}")

# Find which island has slack
slack_buses = set(net.ext_grid.loc[net.ext_grid["in_service"], "bus"])
if "slack" in net.gen.columns:
    slack_buses |= set(net.gen.loc[net.gen["in_service"] & net.gen["slack"].fillna(False), "bus"])

log(f"  Slack buses: {slack_buses}")

islands_with_slack = [i for i in islands if i & slack_buses]
islands_orphan    = [i for i in islands if not (i & slack_buses)]

log(f"  Islands WITH slack:   {len(islands_with_slack)}")
log(f"  Islands WITHOUT slack (orphan): {len(islands_orphan)}")
if islands_orphan:
    orphan_sizes = sorted([len(i) for i in islands_orphan], reverse=True)
    log(f"  Orphan island sizes: {orphan_sizes[:10]}")
    total_orphan = sum(len(i) for i in islands_orphan)
    log(f"  Total buses in orphan islands: {total_orphan}")

main_island = max(islands, key=len)
log(f"  Main island size: {len(main_island)} buses")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generation vs Load balance (in main island)
# ─────────────────────────────────────────────────────────────────────────────
section("2. POWER BALANCE")

total_load_p = net.load.loc[net.load["in_service"], "p_mw"].sum()
total_load_q = net.load.loc[net.load["in_service"], "q_mvar"].sum()
total_gen_max = net.gen.loc[net.gen["in_service"], "max_p_mw"].sum()
total_gen_p_set = net.gen.loc[net.gen["in_service"], "p_mw"].sum()
total_extgrid = len(net.ext_grid[net.ext_grid["in_service"]])

log(f"  Total active load:         {total_load_p:>10.1f} MW")
log(f"  Total reactive load:       {total_load_q:>10.1f} MVAr")
log(f"  Total gen Pmax:            {total_gen_max:>10.1f} MW")
log(f"  Total gen initial P:       {total_gen_p_set:>10.1f} MW")
log(f"  ext_grid count (unlimited):{total_extgrid:>10d}")

if total_gen_max < total_load_p:
    log(f"\n  ⚠ Gen max < load — need ext_grid to supply deficit of "
        f"{total_load_p - total_gen_max:.0f} MW")
else:
    log(f"\n  ✓ Gen max ({total_gen_max:.0f}) >= load ({total_load_p:.0f})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Slack / ext_grid config
# ─────────────────────────────────────────────────────────────────────────────
section("3. SLACK BUS CONFIG")

if len(net.ext_grid) == 0:
    log("  ✗ NO ext_grid! This is the problem.")
    log("  → Need at least one slack bus for power flow to work.")
else:
    for _, eg in net.ext_grid.iterrows():
        bus = int(eg["bus"])
        vm = eg["vm_pu"]
        log(f"  ext_grid at bus {bus}: vm_pu={vm}, in_service={eg['in_service']}")
        log(f"    Bus vn_kv: {net.bus.at[bus, 'vn_kv']}")
        log(f"    Bus name: {net.bus.at[bus, 'name']}")

        # Check if this bus has gens that might conflict
        gens_here = net.gen[net.gen["bus"] == bus]
        log(f"    Generators at same bus: {len(gens_here)}")

        # Check voltage setpoint
        if vm < 0.9 or vm > 1.1:
            log(f"    ⚠ vm_pu={vm} is outside [0.9, 1.1] — may cause infeasibility")

        # Check if in main island
        if bus in main_island:
            log(f"    ✓ In main island")
        else:
            log(f"    ✗ NOT in main island!")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Voltage bounds
# ─────────────────────────────────────────────────────────────────────────────
section("4. VOLTAGE BOUNDS")

vmin_range = (net.bus["min_vm_pu"].min(), net.bus["min_vm_pu"].max())
vmax_range = (net.bus["max_vm_pu"].min(), net.bus["max_vm_pu"].max())
log(f"  min_vm_pu range: {vmin_range}")
log(f"  max_vm_pu range: {vmax_range}")

buses_tight = ((net.bus["max_vm_pu"] - net.bus["min_vm_pu"]) < 0.1).sum()
log(f"  Buses with tight V-band (<0.1): {buses_tight}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Line impedance check
# ─────────────────────────────────────────────────────────────────────────────
section("5. LINE IMPEDANCE — look for pathological values")

# Compute per-unit impedance
net.line["r_pu"] = net.line["r_ohm_per_km"] * net.line["length_km"]
net.line["x_pu"] = net.line["x_ohm_per_km"] * net.line["length_km"]
log(f"  r_ohm_per_km: min={net.line['r_ohm_per_km'].min():.4f}, "
    f"max={net.line['r_ohm_per_km'].max():.4f}, "
    f"median={net.line['r_ohm_per_km'].median():.4f}")
log(f"  x_ohm_per_km: min={net.line['x_ohm_per_km'].min():.4f}, "
    f"max={net.line['x_ohm_per_km'].max():.4f}, "
    f"median={net.line['x_ohm_per_km'].median():.4f}")

# Zero impedance lines (very dangerous)
zero_x = (net.line["x_ohm_per_km"] == 0).sum()
log(f"  Lines with x=0: {zero_x}")

# Very high impedance (effectively open)
very_high_x = (net.line["x_ohm_per_km"] > 100).sum()
log(f"  Lines with x>100 ohm/km: {very_high_x}")

# max_i_ka issues
log(f"  max_i_ka: min={net.line['max_i_ka'].min():.4f}, "
    f"max={net.line['max_i_ka'].max():.4f}")
zero_i = (net.line["max_i_ka"] == 0).sum()
log(f"  Lines with max_i_ka=0 (impossible): {zero_i}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Transformer check
# ─────────────────────────────────────────────────────────────────────────────
section("6. TRANSFORMER SANITY")

log(f"  vk_percent: min={net.trafo['vk_percent'].min():.4f}, "
    f"max={net.trafo['vk_percent'].max():.4f}, "
    f"median={net.trafo['vk_percent'].median():.4f}")
log(f"  vkr_percent: min={net.trafo['vkr_percent'].min():.4f}, "
    f"max={net.trafo['vkr_percent'].max():.4f}")
log(f"  sn_mva: min={net.trafo['sn_mva'].min():.1f}, "
    f"max={net.trafo['sn_mva'].max():.1f}")

zero_vk = (net.trafo["vk_percent"] == 0).sum()
tiny_vk = ((net.trafo["vk_percent"] > 0) & (net.trafo["vk_percent"] < 0.1)).sum()
log(f"  Trafos with vk=0 (impossible): {zero_vk}")
log(f"  Trafos with vk<0.1% (too low, numerical issues): {tiny_vk}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Try DC power flow (more robust)
# ─────────────────────────────────────────────────────────────────────────────
section("7. DC POWER FLOW TEST (more robust, linear)")

try:
    pp.rundcpp(net, check_connectivity=False)
    log("  ✓ DC-PF converged!")
    log(f"    ext_grid injection: {net.res_ext_grid['p_mw'].sum():.1f} MW")
    log(f"    Total gen: {net.res_gen['p_mw'].sum():.1f} MW")
    log(f"    Total load: {net.load.loc[net.load['in_service'], 'p_mw'].sum():.1f} MW")
    log(f"    Max line loading: {net.res_line['loading_percent'].max():.1f}%")
    log(f"    → Network topology + power balance OK")
    log(f"    → Problem likely in voltage/reactive domain (AC-specific)")
except Exception as e:
    log(f"  ✗ DC-PF failed: {type(e).__name__}: {e}")
    log(f"    → Fundamental topology/config issue")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Try AC-PF with progressive relaxation
# ─────────────────────────────────────────────────────────────────────────────
section("8. AC-PF WITH PROGRESSIVE RELAXATION")

# Attempt 1: DC-initialization
log("\n  Attempt 1: init='dc', nr algorithm, max_iter=30")
try:
    pp.runpp(net, init="dc", algorithm="nr", max_iteration=30,
             numba=False, calculate_voltage_angles=True,
             check_connectivity=False, tolerance_mva=1e-4)
    log(f"    ✓ Converged! Vmin={net.res_bus['vm_pu'].min():.4f}, "
        f"Vmax={net.res_bus['vm_pu'].max():.4f}")
except Exception as e:
    log(f"    ✗ {type(e).__name__}: {str(e)[:100]}")

# Attempt 2: Flat start with tolerance relaxation
log("\n  Attempt 2: init='flat', tolerance_mva=1e-3, max_iter=50")
try:
    pp.runpp(net, init="flat", algorithm="nr", max_iteration=50,
             numba=False, calculate_voltage_angles=True,
             check_connectivity=False, tolerance_mva=1e-3)
    log(f"    ✓ Converged! Vmin={net.res_bus['vm_pu'].min():.4f}, "
        f"Vmax={net.res_bus['vm_pu'].max():.4f}")
except Exception as e:
    log(f"    ✗ {type(e).__name__}: {str(e)[:100]}")

# Attempt 3: Scale load to 50%
log("\n  Attempt 3: scale load to 50%, init='dc'")
import copy
net_half = copy.deepcopy(net)
net_half.load["p_mw"] *= 0.5
net_half.load["q_mvar"] *= 0.5
try:
    pp.runpp(net_half, init="dc", algorithm="nr", max_iteration=30,
             numba=False, calculate_voltage_angles=True,
             check_connectivity=False, tolerance_mva=1e-3)
    log(f"    ✓ Converged at 50% load! "
        f"Vmin={net_half.res_bus['vm_pu'].min():.4f}, "
        f"Vmax={net_half.res_bus['vm_pu'].max():.4f}")
    log(f"    → Original load level may cause voltage collapse")
except Exception as e:
    log(f"    ✗ {type(e).__name__}: {str(e)[:100]}")

# Attempt 4: Scale load to 20%
log("\n  Attempt 4: scale load to 20%, init='flat'")
net_minimal = copy.deepcopy(net)
net_minimal.load["p_mw"] *= 0.2
net_minimal.load["q_mvar"] *= 0.2
try:
    pp.runpp(net_minimal, init="flat", algorithm="nr", max_iteration=30,
             numba=False, calculate_voltage_angles=True,
             check_connectivity=False, tolerance_mva=1e-3)
    log(f"    ✓ Converged at 20% load!")
    log(f"    → Network only converges at very low load — "
        f"major reactive power issue or bad topology")
except Exception as e:
    log(f"    ✗ {type(e).__name__}: {str(e)[:100]}")

# Attempt 5: Only main island
log("\n  Attempt 5: isolate MAIN island, 100% load")
net_main = copy.deepcopy(net)
off_buses = [b for b in net.bus.index if b not in main_island]
log(f"    Deactivating {len(off_buses)} buses outside main island")
for b in off_buses:
    net_main.bus.at[b, "in_service"] = False
for _, l in net.line.iterrows():
    if l["from_bus"] not in main_island or l["to_bus"] not in main_island:
        net_main.line.at[l.name, "in_service"] = False
for _, t in net.trafo.iterrows():
    if t["hv_bus"] not in main_island or t["lv_bus"] not in main_island:
        net_main.trafo.at[t.name, "in_service"] = False
for _, ld in net.load.iterrows():
    if ld["bus"] not in main_island:
        net_main.load.at[ld.name, "in_service"] = False
for _, g in net.gen.iterrows():
    if g["bus"] not in main_island:
        net_main.gen.at[g.name, "in_service"] = False

try:
    pp.runpp(net_main, init="dc", algorithm="nr", max_iteration=30,
             numba=False, calculate_voltage_angles=True,
             check_connectivity=False, tolerance_mva=1e-3)
    log(f"    ✓ Converged on main island only!")
    log(f"    → Problem was non-main islands. Solution: deactivate them.")
except Exception as e:
    log(f"    ✗ {type(e).__name__}: {str(e)[:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# Save report
# ─────────────────────────────────────────────────────────────────────────────
section("REPORT SAVED")
REPORT.write_text("\n".join(log_lines), encoding="utf-8")
log(f"  {REPORT}")