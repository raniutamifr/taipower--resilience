"""
fix_opf_feasibility.py
=======================
Two fixes to make AC-OPF converge on Taipower 1970-bus network:

Fix 1: Add small unique cp2 to all generators with flat cost
        Degenerate OPF (many identical costs) causes IPOPT to fail.
        Small cp2 makes the problem strictly convex → IPOPT converges.

Fix 2: Relax thermal limits on lines still overloaded after fix_overloaded_lines.py
        Lines >100% loading in base case = hard thermal constraint → OPF infeasible.
"""

import pandapower as pp
import numpy as np
import time
import copy

NET_FILE = r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json"

net = pp.from_json(NET_FILE)

# ── Fix 1: Add small unique cp2 to all flat-cost generators ──────────────────
print("Fix 1: Patching flat-cost generators...")
rng = np.random.default_rng(42)

n_fixed = 0
for idx, row in net.poly_cost[net.poly_cost["et"] == "gen"].iterrows():
    if row["cp2_eur_per_mw2"] == 0 or abs(row["cp2_eur_per_mw2"]) < 1e-9:
        # Add small unique cp2 based on generator index — makes problem strictly convex
        cp1 = row["cp1_eur_per_mw"]
        # cp2 proportional to cp1 so dispatch order is preserved
        cp2 = cp1 * rng.uniform(0.0001, 0.0003)
        net.poly_cost.at[idx, "cp2_eur_per_mw2"] = cp2
        n_fixed += 1

print(f"  Added cp2 to {n_fixed} generators")
gc = net.poly_cost[net.poly_cost["et"] == "gen"]
print(f"  Generators with cp2=0 remaining: {(gc['cp2_eur_per_mw2'] == 0).sum()}")

# ── Fix 2: Relax remaining overloaded lines ───────────────────────────────────
print("\nFix 2: Checking remaining overloaded lines...")
pp.runpp(net, numba=False, check_connectivity=False, verbose=False)
ll = net.res_line["loading_percent"].dropna()
overloaded = ll[ll > 100]
print(f"  Lines >100%: {len(overloaded)}")

for line_idx, loading in overloaded.items():
    i_actual = net.res_line.loc[line_idx, "i_ka"]
    i_max    = net.line.at[line_idx, "max_i_ka"]
    new_max  = round(i_actual * 2.0, 4)
    print(f"  Fix line {line_idx} ({net.line.at[line_idx,'name']}): "
          f"max_i_ka {i_max:.4f} → {new_max:.4f} ({loading:.1f}%)")
    net.line.at[line_idx, "max_i_ka"] = new_max

# ── Verify and test OPF ───────────────────────────────────────────────────────
print("\nTesting AC-OPF with warm start...")
net_t = copy.deepcopy(net)
net_t.load["p_mw"]   *= 0.85
net_t.load["q_mvar"] *= 0.85
shed = net_t.sgen["name"].str.startswith("LoadShed_sgen_")
net_t.sgen.loc[shed, "max_p_mw"] *= 0.85

# Warm start from AC-PF
pp.runpp(net_t, numba=False, calculate_voltage_angles=True,
         check_connectivity=False, verbose=False)
print(f"  AC-PF warm start: converged={net_t.converged}")

t0 = time.time()
try:
    pp.runopp(net_t, verbose=False, numba=False,
              calculate_voltage_angles=True,
              check_connectivity=False,
              init="pf",
              max_iteration=300)
    elapsed = time.time() - t0
    print(f"  AC-OPF converged : {net_t.converged}  ({elapsed:.1f}s)")
    if net_t.converged:
        print(f"  Total gen : {net_t.res_gen['p_mw'].sum():.1f} MW")
        print(f"  Ext grid  : {net_t.res_ext_grid['p_mw'].sum():.1f} MW")
        print(f"  Cost      : {net_t.res_cost:,.2f} NT$/hr")
        print(f"  Vmin      : {net_t.res_bus['vm_pu'].min():.4f} pu")
        print(f"  Vmax      : {net_t.res_bus['vm_pu'].max():.4f} pu")
        print(f"\n  SUCCESS — saving fixed network...")
        pp.to_json(net, NET_FILE)
        print(f"  Saved: {NET_FILE}")
    else:
        print("  Not converged — saving anyway and try Julia")
        pp.to_json(net, NET_FILE)
        print(f"  Saved: {NET_FILE}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"  ERROR ({elapsed:.1f}s): {e}")
    print("  Saving cost+thermal fixes anyway...")
    pp.to_json(net, NET_FILE)
    print(f"  Saved: {NET_FILE}")