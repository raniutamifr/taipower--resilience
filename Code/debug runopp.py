"""
Minimal test - find out exactly why pp.runopp is failing on the base
network. Step 9 currently logs the failure at debug level, so the actual
error never reaches the terminal. Here we let every exception surface.
"""
import traceback
import pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")

print("=" * 70)
print("STAGE 1 - Load network and collapse switches (mimics Step 9 main)")
print("=" * 70)
net = pp.from_json(str(NET_FILE))
print(f"Buses loaded   : {len(net.bus)}")
print(f"Gens loaded    : {len(net.gen)}")
print(f"Sgens loaded   : {len(net.sgen)}")
print(f"Switches (b)   : {((net.switch.et == 'b') & net.switch.closed).sum()}")

# Replicate the switch collapse from Step 9
bb = net.switch[(net.switch.et == "b") & net.switch.closed].copy()
for _, sw in bb.iterrows():
    b1, b2 = int(sw.bus), int(sw.element)
    if b1 == b2 or b1 not in net.bus.index or b2 not in net.bus.index:
        continue
    try:
        pp.fuse_buses(net, b1, b2, drop=True)
    except Exception:
        pass

for df_name, c1, c2 in [("line", "from_bus", "to_bus"),
                        ("impedance", "from_bus", "to_bus"),
                        ("trafo", "hv_bus", "lv_bus")]:
    df = getattr(net, df_name)
    if not df.empty:
        sl = df[df[c1] == df[c2]].index
        if len(sl):
            df.drop(sl, inplace=True)

print(f"Buses after collapse : {len(net.bus)}")

print("\n" + "=" * 70)
print("STAGE 2 - Scale load by 0.75 (mimics Step 9 LOAD_SCALE)")
print("=" * 70)
net.load["p_mw"]   *= 0.75
net.load["q_mvar"] *= 0.75
mask = net.sgen["name"].str.startswith("LoadShed_")
net.sgen.loc[mask, "max_p_mw"] *= 0.75

net.bus["min_vm_pu"] = 0.95
net.bus["max_vm_pu"] = 1.05
if "max_loading_percent" in net.line.columns:
    net.line["max_loading_percent"] = 100.0
if "max_loading_percent" in net.trafo.columns:
    net.trafo["max_loading_percent"] = 100.0
print("Load scaled, V bounds and thermal limits set.")

print("\n" + "=" * 70)
print("STAGE 3 - Sanity: run AC PF first (should converge)")
print("=" * 70)
try:
    pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
             check_connectivity=False, enforce_q_lims=False)
    print(f"AC PF converged : {net.converged}")
    if net.converged:
        print(f"  V range     : {net.res_bus.vm_pu.min():.4f} - "
              f"{net.res_bus.vm_pu.max():.4f}")
        print(f"  Slack P     : {net.res_ext_grid.p_mw.values[0]:.0f} MW")
except Exception:
    traceback.print_exc()

print("\n" + "=" * 70)
print("STAGE 4 - Run pp.runopp (THE FAILING CALL) - show full error")
print("=" * 70)
try:
    pp.runopp(net, init="flat", check_connectivity=False,
              numba=False, verbose=True)
    print(f"\nrunopp converged : {net.converged}")
    if net.converged:
        print(f"  Objective       : {float(net.res_cost):.1f}")
        print(f"  Slack P         : {float(net.res_ext_grid.p_mw.values[0]):+.1f} MW")
        print(f"  V range         : {net.res_bus.vm_pu.min():.4f} - "
              f"{net.res_bus.vm_pu.max():.4f}")
except Exception:
    print("\nRUNOPP FAILED WITH THE FOLLOWING TRACE:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("STAGE 5 - Alt: try init='pf' (warm start from AC PF results)")
print("=" * 70)
try:
    pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
             check_connectivity=False, enforce_q_lims=False)
    pp.runopp(net, init="pf", check_connectivity=False,
              numba=False, verbose=False)
    print(f"runopp(init=pf) converged : {net.converged}")
    if net.converged:
        print(f"  Objective       : {float(net.res_cost):.1f}")
        print(f"  Slack P         : {float(net.res_ext_grid.p_mw.values[0]):+.1f} MW")
except Exception:
    print("RUNOPP(init=pf) FAILED:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("STAGE 6 - Alt: try rundcopp with linearised ext_grid cost")
print("=" * 70)
try:
    eg_mask = ((net.poly_cost["et"] == "ext_grid") &
               (net.poly_cost["element"] == net.ext_grid.index[0]))
    saved_cp2 = float(net.poly_cost.loc[eg_mask, "cp2_eur_per_mw2"].iloc[0])
    net.poly_cost.loc[eg_mask, "cp2_eur_per_mw2"] = 0.0
    net.poly_cost.loc[eg_mask, "cp1_eur_per_mw"]  = 100.0
    pp.rundcopp(net, check_connectivity=False)
    print(f"rundcopp converged : {net.converged}")
    if net.converged:
        print(f"  Objective       : {float(net.res_cost):.1f}")
        print(f"  Slack P         : {float(net.res_ext_grid.p_mw.values[0]):+.1f} MW")
    net.poly_cost.loc[eg_mask, "cp2_eur_per_mw2"] = saved_cp2
    net.poly_cost.loc[eg_mask, "cp1_eur_per_mw"]  = 0.0
except Exception:
    print("RUNDCOPP FAILED:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("END")
print("=" * 70)