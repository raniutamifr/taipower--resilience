"""
make_opf_converge.py
====================
Final convergence fix for the Taipower AC-OPF, based on the SECTION 1-9
deep diagnostic.

Diagnostic evidence:
  - AC-PF converges at scale 0.85 (V=[0.959,1.030]) but FAILS at 0.50-0.80.
    Non-monotonic convergence => generation dispatch is FIXED and does not
    track load. DC slack absorbs -6,914 MW at 0.85 rising to -21,740 MW at
    0.50, confirming ~42,920 MW of generation pinned regardless of load.
  - Native DC-OPF also fails (a linear problem) => objective is ill-posed.
  - Ext_grid cost row: cp1=0.0, cp2=1.0  => purely quadratic penalty on the
    slack power. With the slack forced to +/-20 GW the gradient explodes.
  - AC-OPF aborts with "res_bus index != bus index" => init='pf' warm-start
    from a stale result table left behind after bus fusing.

Root causes (ranked):
  1. Generators not controllable -> OPF cannot redispatch, slack overloads.
  2. Ext_grid quadratic-only cost (cp2=1.0) -> ill-conditioned objective.
  3. Stale result table used as OPF warm start.
  4. Slack sited on a single bus that physically cannot sink 20 GW.

Fixes:
  A. Set every generator controllable=True (let OPF co-optimise P).
  B. Replace ext_grid cost with a linear marginal price (cp2=0).
  C. Run OPF from a clean init='flat' (no stale res_* tables).
  D. Ensure generation can balance load; report the active-power picture.
  E. Save the OPF-ready network with the correct pandapower API.
"""

import copy
import warnings
warnings.filterwarnings("ignore")

import pandapower as pp
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
OUT_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network_opf_ready.json")

SHED_PREFIX = "LoadShed_"

# Marginal cost table (NT$/MWh)
MARGINAL_COST = {
    "nuclear":   200.0, "hydro": 100.0, "coal": 800.0,
    "gas":     1_500.0, "lng": 1_600.0, "oil": 2_500.0,
    "solar":       0.0, "wind":   0.0,  "default": 1_200.0,
}
VOLL              = 50_000.0   # NT$/MWh — Value of Lost Load
EXT_GRID_PRICE    = 1_600.0    # NT$/MWh — slack/import priced as LNG


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def flag(label: str, ok: bool, detail: str = "") -> None:
    status = "OK  " if ok else "FIX "
    print(f"  [{status}] {label}" + (f"  — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Confirm the three root causes
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 1 : CONFIRM ROOT CAUSES")

net = pp.from_json(str(NET_FILE))
shed_mask = net.sgen["name"].astype(str).str.startswith(SHED_PREFIX)

# Cause 1: generator controllability
if "controllable" in net.gen.columns:
    n_ctrl = net.gen["controllable"].eq(True).sum()
else:
    n_ctrl = 0
print("\n  Cause 1 — Generator controllability")
print(f"    Generators total          : {len(net.gen)}")
print(f"    controllable == True      : {n_ctrl}")
flag("All generators controllable", n_ctrl == len(net.gen),
     f"{len(net.gen) - n_ctrl} gens fixed -> slack must absorb imbalance")

# Cause 2: ext_grid cost shape
print("\n  Cause 2 — External grid cost shape")
ext_cost = net.poly_cost[net.poly_cost.et == "ext_grid"]
if not ext_cost.empty:
    cp1 = float(ext_cost["cp1_eur_per_mw"].iloc[0])
    cp2 = float(ext_cost["cp2_eur_per_mw2"].iloc[0])
    print(f"    ext_grid cp1 = {cp1:.1f}   cp2 = {cp2:.3f}")
    flag("Ext_grid cost is linear (cp2 ~ 0)", abs(cp2) < 1e-6,
         f"cp2={cp2:.3f} -> quadratic penalty on slack, ill-conditioned")
else:
    print("    No ext_grid cost row found.")

# Cause 3: fixed-dispatch confirmation (gen total vs load)
print("\n  Cause 3 — Active-power dispatch vs load")
total_load = net.load.loc[net.load.in_service, "p_mw"].sum()
total_pset = net.gen.loc[net.gen.in_service, "p_mw"].sum()
total_pmax = net.gen.loc[net.gen.in_service, "max_p_mw"].sum()
print(f"    Total load (nominal)      : {total_load:>10,.0f} MW")
print(f"    Sum of gen p_mw setpoint  : {total_pset:>10,.0f} MW")
print(f"    Sum of gen max_p_mw       : {total_pmax:>10,.0f} MW")
print(f"    Fixed surplus (Pset-load) : {total_pset - total_load:>+10,.0f} MW")
flag("Gen setpoint near load", abs(total_pset - total_load) < 0.1 * total_load,
     f"surplus {total_pset - total_load:+,.0f} MW dumped on slack at low load")
flag("Pmax can cover load", total_pmax >= total_load,
     f"margin {total_pmax - total_load:+,.0f} MW")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Apply fixes on a fresh copy
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 2 : APPLY FIXES")

net2 = pp.from_json(str(NET_FILE))
shed_mask2 = net2.sgen["name"].astype(str).str.startswith(SHED_PREFIX)

# Fix A — generators controllable
print("\n  [Fix A] Set all generators controllable=True ...")
net2.gen["controllable"] = True
net2.sgen.loc[shed_mask2, "controllable"] = True
print(f"    {len(net2.gen)} gens + {shed_mask2.sum()} LoadShed sgens are controllable.")

# Fix B — rebuild poly_cost with LINEAR costs everywhere (kills cp2=1.0 bug)
print("  [Fix B] Rebuild poly_cost — linear marginal costs, ext_grid cp2=0 ...")

FUEL_KEYWORDS = {
    "核": "nuclear", "nuclear": "nuclear",
    "煤": "coal", "林口": "coal", "台中": "coal", "大林": "coal", "麥寮": "coal", "coal": "coal",
    "大潭": "gas", "通霄": "gas", "興達": "gas", "南部": "gas", "gas": "gas", "lng": "gas",
    "協和": "oil", "oil": "oil",
    "水": "hydro", "hydro": "hydro",
    "solar": "solar", "太陽": "solar", "wind": "wind", "風": "wind",
}

def detect_fuel(name: str) -> str:
    for kw, fuel in FUEL_KEYWORDS.items():
        if kw.lower() in str(name).lower():
            return fuel
    return "default"

net2.poly_cost = net2.poly_cost.iloc[0:0].copy()

for idx, gen in net2.gen.iterrows():
    fuel = detect_fuel(str(gen.get("name", "")))
    pp.create_poly_cost(net2, element=idx, et="gen",
                        cp0_eur=0.0, cp1_eur_per_mw=MARGINAL_COST[fuel],
                        cp2_eur_per_mw2=0.0)

for ext_idx in net2.ext_grid.index:
    pp.create_poly_cost(net2, element=ext_idx, et="ext_grid",
                        cp0_eur=0.0, cp1_eur_per_mw=EXT_GRID_PRICE,
                        cp2_eur_per_mw2=0.0)          # <-- cp2 forced to 0

for sgen_idx in net2.sgen.index[shed_mask2]:
    pp.create_poly_cost(net2, element=sgen_idx, et="sgen",
                        cp0_eur=0.0, cp1_eur_per_mw=VOLL, cp2_eur_per_mw2=0.0)

print(f"    poly_cost rebuilt: {len(net2.poly_cost)} entries (all linear).")
print(f"    ext_grid cost now cp1={EXT_GRID_PRICE}, cp2=0.0")

# Fix C — sane Q limits so reactive dispatch has room
print("  [Fix C] Ensure generator Q limits are finite and non-degenerate ...")
for col in ["min_q_mvar", "max_q_mvar"]:
    if col not in net2.gen.columns:
        net2.gen[col] = np.nan
pmax = net2.gen["max_p_mw"].clip(lower=1.0)
qmin = net2.gen["min_q_mvar"].fillna(-0.6 * pmax)
qmax = net2.gen["max_q_mvar"].fillna( 0.6 * pmax)
degenerate = (qmax - qmin).abs() < 1.0
qmin[degenerate] = -0.6 * pmax[degenerate]
qmax[degenerate] =  0.6 * pmax[degenerate]
net2.gen["min_q_mvar"] = qmin.values
net2.gen["max_q_mvar"] = qmax.values
print(f"    {int(degenerate.sum())} degenerate Q ranges widened to +/-60% Pmax.")

# Fix D — clamp generator vm setpoints into the operating band
print("  [Fix D] Clamp generator vm_pu to [0.95, 1.05] ...")
net2.gen["vm_pu"] = net2.gen["vm_pu"].clip(lower=0.95, upper=1.05)
print("    Done.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — OPF test with CLEAN init (no stale res_bus)
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 3 : OPF TEST (clean init='flat')")

def try_opf(net_in, label, **kwargs):
    """Run AC-OPF from a clean state. Never use init='pf' here."""
    # Drop any stale result tables so the res_bus index warning cannot occur
    for tbl in ["res_bus", "res_line", "res_trafo", "res_gen",
                "res_sgen", "res_ext_grid", "res_load"]:
        if hasattr(net_in, tbl):
            setattr(net_in, tbl, getattr(net_in, tbl).iloc[0:0])
    try:
        pp.runopp(net_in, init="flat", calculate_voltage_angles=True,
                  numba=False, check_connectivity=False, verbose=False,
                  **kwargs)
        if net_in.converged:
            cost  = float(net_in.res_cost)
            slack = float(net_in.res_ext_grid.p_mw.iloc[0])
            gen_p = float(net_in.res_gen.p_mw.sum())
            shed  = float(net_in.res_sgen.loc[
                        net_in.sgen["name"].astype(str).str.startswith(SHED_PREFIX),
                        "p_mw"].sum())
            vmin  = net_in.res_bus.vm_pu.min()
            vmax  = net_in.res_bus.vm_pu.max()
            print(f"    {label:<38s} CONVERGED")
            print(f"        cost      = {cost:,.0f} NT$/hr")
            print(f"        gen P     = {gen_p:,.0f} MW")
            print(f"        slack P   = {slack:+,.0f} MW")
            print(f"        shed P    = {shed:,.1f} MW")
            print(f"        V         = [{vmin:.3f}, {vmax:.3f}] pu")
            return True
        print(f"    {label:<38s} net.converged = False")
        return False
    except Exception as exc:
        print(f"    {label:<38s} FAILED — {str(exc)[:70]}")
        return False


# Tier 1: relaxed voltage, no thermal — proves the formulation is now well-posed
print("\n  Tier 1: V=[0.85,1.15], no thermal limit")
t1 = copy.deepcopy(net2)
t1.bus["min_vm_pu"]             = 0.85
t1.bus["max_vm_pu"]             = 1.15
t1.line["max_loading_percent"]  = 1e6
t1.trafo["max_loading_percent"] = 1e6
if not t1.trafo3w.empty:
    t1.trafo3w["max_loading_percent"] = 1e6
r1 = try_opf(t1, "relaxed bounds")

# Tier 2: Taipower voltage, relaxed thermal
print("\n  Tier 2: V=[0.90,1.10], thermal 150%")
t2 = copy.deepcopy(net2)
t2.bus["min_vm_pu"]             = 0.90
t2.bus["max_vm_pu"]             = 1.10
t2.line["max_loading_percent"]  = 150.0
t2.trafo["max_loading_percent"] = 150.0
if not t2.trafo3w.empty:
    t2.trafo3w["max_loading_percent"] = 150.0
r2 = try_opf(t2, "V std + thermal 150%")

# Tier 3: full Taipower operating constraint
print("\n  Tier 3: V=[0.90,1.10], thermal 100%")
t3 = copy.deepcopy(net2)
t3.bus["min_vm_pu"]             = 0.90
t3.bus["max_vm_pu"]             = 1.10
t3.line["max_loading_percent"]  = 100.0
t3.trafo["max_loading_percent"] = 100.0
if not t3.trafo3w.empty:
    t3.trafo3w["max_loading_percent"] = 100.0
r3 = try_opf(t3, "full Taipower constraint")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Save OPF-ready network (correct API)
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 4 : SAVE OPF-READY NETWORK")

# Persist the tightest constraint set that converged
if r3:
    final = t3
elif r2:
    final = t2
elif r1:
    final = t1
else:
    final = net2  # save the patched-but-unconverged net for further inspection

pp.to_json(final, str(OUT_FILE))   # correct API: pp.to_json(net, path)
print(f"\n  Saved to: {OUT_FILE}")
print(f"  Update step09/step10:  NET_FILE = Path(r'{OUT_FILE}')")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"\n  Tier 1 (relaxed)            : {'PASS' if r1 else 'FAIL'}")
print(f"  Tier 2 (V std, thermal 150%) : {'PASS' if r2 else 'FAIL'}")
print(f"  Tier 3 (full constraint)     : {'PASS' if r3 else 'FAIL'}")
print("""
  Interpretation:
    Tier 1 PASS -> controllable + linear-cost fixes solved the OPF.
                   The earlier failure was the quadratic ext_grid cost
                   and fixed generation dispatch, not the cost magnitude.

    Tier 1 FAIL -> the slack bus itself is the bottleneck. Re-site the
                   slack onto a 345 kV backbone bus (per the Step-06 slack
                   fix discussed earlier) and consider distributed slack
                   via slack_weight, then re-run this script.

    Tier 2 PASS, Tier 3 FAIL -> thermal limits bind. Run step09 with
                   max_loading_percent = 150% as the production setting,
                   and record which lines hit 100-150% as reinforcement
                   candidates.

  For step09/step10, always run AC-OPF with init='flat' (never init='pf'
  on a net whose buses were fused), and keep all cost coefficients linear
  (cp2 = 0) unless you deliberately model a quadratic fuel curve.
""")