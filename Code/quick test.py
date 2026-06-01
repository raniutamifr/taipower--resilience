"""
Quick test - patch the NaN fields in-place and retry pp.runopp.

The verbose output from test_runopp.py showed PYPOWER's internal gen
matrix riddled with NaN in the QG column (Q dispatch initial value).
pandapower's create_gen does not initialise q_mvar (since a gen is a
PV bus from the PF standpoint), but for OPF Q becomes a decision
variable and IPOPT diverges immediately if its starting value is NaN.

This script patches the loaded JSON network in memory and confirms
whether the NaN initial values were the blocker.
"""
import traceback
import pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")

net = pp.from_json(str(NET_FILE))

# Step-9-style pre-processing (collapse switches + drop self-loops)
bb = net.switch[(net.switch.et == "b") & net.switch.closed].copy()
for _, sw in bb.iterrows():
    b1, b2 = int(sw.bus), int(sw.element)
    if b1 != b2 and b1 in net.bus.index and b2 in net.bus.index:
        try:
            pp.fuse_buses(net, b1, b2, drop=True)
        except Exception:
            pass
for df_name, c1, c2 in [
    ("line",      "from_bus", "to_bus"),
    ("impedance", "from_bus", "to_bus"),
    ("trafo",     "hv_bus",   "lv_bus"),
]:
    df = getattr(net, df_name)
    if not df.empty:
        sl = df[df[c1] == df[c2]].index
        if len(sl):
            df.drop(sl, inplace=True)

# Step-9-style load scaling
net.load["p_mw"]   *= 0.75
net.load["q_mvar"] *= 0.75
mask = net.sgen["name"].str.startswith("LoadShed_")
net.sgen.loc[mask, "max_p_mw"] *= 0.75

# OPF bounds
net.bus["min_vm_pu"] = 0.95
net.bus["max_vm_pu"] = 1.05
net.line["max_loading_percent"]  = 100.0
net.trafo["max_loading_percent"] = 100.0

# ----------------------------- THE FIX --------------------------------
print("=" * 70)
print("NaN audit BEFORE fix")
print("=" * 70)
for df_name in ["gen", "sgen"]:
    df = getattr(net, df_name)
    if df.empty:
        continue
    for col in ["p_mw", "q_mvar", "sn_mva", "vm_pu",
                "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        if col in df.columns:
            n = df[col].isna().sum()
            if n:
                print(f"  {df_name}.{col:14s} NaN count: {n}")

# Initialise the operating-point Q at zero for every gen and sgen.
# This is the actual fix.
net.gen["q_mvar"]  = 0.0
net.sgen["q_mvar"] = 0.0

# A defined rated apparent power is needed by the pu conversion. Use
# max_p_mw as a sane minimum; fall back to 1.0 MVA for very tiny units.
if "sn_mva" in net.gen.columns:
    net.gen["sn_mva"]  = net.gen["max_p_mw"].clip(lower=1.0)
if "sn_mva" in net.sgen.columns:
    net.sgen["sn_mva"] = net.sgen["max_p_mw"].clip(lower=1.0)

print("\n" + "=" * 70)
print("NaN audit AFTER fix")
print("=" * 70)
for df_name in ["gen", "sgen"]:
    df = getattr(net, df_name)
    if df.empty:
        continue
    for col in ["p_mw", "q_mvar", "sn_mva", "vm_pu",
                "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        if col in df.columns:
            n = df[col].isna().sum()
            if n:
                print(f"  {df_name}.{col:14s} STILL has NaN: {n}")
print("  (no lines printed above = all fields clean)")

# ----------------------------- AC OPF ---------------------------------
print("\n" + "=" * 70)
print("Running pp.runopp (AC OPF) with the fix in place")
print("=" * 70)
try:
    pp.runopp(net, init="flat", check_connectivity=False,
              numba=False, verbose=False)
    print(f"\n  CONVERGED  : {net.converged}")
    if net.converged:
        print(f"  Objective  : {float(net.res_cost):.1f} NT$/hr")
        print(f"  Slack P    : {float(net.res_ext_grid.p_mw.iloc[0]):+.1f} MW")
        vm = net.res_bus.vm_pu.dropna()
        print(f"  V range    : [{vm.min():.4f}, {vm.max():.4f}] pu")
        shed_mask = net.sgen.name.str.startswith("LoadShed_")
        shed = float(net.res_sgen.loc[shed_mask, "p_mw"].sum())
        print(f"  Load shed  : {shed:.2f} MW")
except Exception:
    print("  pp.runopp FAILED:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("END - if CONVERGED = True the fix works, patch Step 6 next.")
print("=" * 70)