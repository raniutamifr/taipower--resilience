"""
AC-OPF deep diagnostic - find the EXACT reason pp.runopp fails, so we
can fix it rather than fall back to a heuristic.

Network size (2767 buses) is NOT inherently too large for AC-OPF;
IEEE test cases up to 9241 buses solve fine. When IPOPT fails fast on
a mid-size case it is almost always a data-conditioning problem, not a
size problem. This script isolates which one.

Strategy: start from the smallest possible OPF and grow until it breaks.
  Test 1 : Can a trivial OPF converge at all? (relax all limits)
  Test 2 : Does it converge with only generator cost + slack?
  Test 3 : Add voltage bounds
  Test 4 : Add line thermal limits
  Test 5 : Add the load-shed sgens
The first test that fails points straight at the culprit.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")


def fresh():
    """Load a clean copy, collapse switches, scale load to 0.75."""
    net = pp.from_json(str(NET_FILE))
    bb = net.switch[(net.switch.et == "b") & net.switch.closed].copy()
    for _, sw in bb.iterrows():
        b1, b2 = int(sw.bus), int(sw.element)
        if b1 != b2 and b1 in net.bus.index and b2 in net.bus.index:
            try:
                pp.fuse_buses(net, b1, b2, drop=True)
            except Exception:
                pass
    for dfn, c1, c2 in [("line", "from_bus", "to_bus"),
                        ("impedance", "from_bus", "to_bus"),
                        ("trafo", "hv_bus", "lv_bus")]:
        df = getattr(net, dfn)
        if not df.empty:
            sl = df[df[c1] == df[c2]].index
            if len(sl):
                df.drop(sl, inplace=True)
    net.load["p_mw"]   *= 0.75
    net.load["q_mvar"] *= 0.75
    mask = net.sgen["name"].str.startswith("LoadShed_")
    net.sgen.loc[mask, "max_p_mw"] *= 0.75
    # Make sure gen q_mvar / sn_mva are defined (OPF NaN guard)
    net.gen["q_mvar"] = 0.0
    if "sn_mva" in net.gen.columns:
        net.gen["sn_mva"] = net.gen["max_p_mw"].clip(lower=1.0)
    if "sn_mva" in net.sgen.columns and not net.sgen.empty:
        net.sgen["sn_mva"] = net.sgen["max_p_mw"].clip(lower=1.0)
    return net


def try_opf(net, label, **kwargs):
    try:
        pp.runopp(net, **kwargs)
        if net.converged:
            cost = float(net.res_cost)
            slack = float(net.res_ext_grid.p_mw.iloc[0])
            vmin = float(net.res_bus.vm_pu.min())
            vmax = float(net.res_bus.vm_pu.max())
            print(f"  [{label}] CONVERGED  cost={cost:.0f}  "
                  f"slack={slack:+.0f}MW  V=[{vmin:.3f},{vmax:.3f}]")
            return True
        print(f"  [{label}] returned but net.converged=False")
        return False
    except Exception as e:
        print(f"  [{label}] FAILED: {type(e).__name__}: {str(e)[:90]}")
        return False


print("=" * 70)
print("TEST 1 - Trivial OPF: wide voltage band, no thermal, sgens off")
print("=" * 70)
net = fresh()
net.bus["min_vm_pu"] = 0.8
net.bus["max_vm_pu"] = 1.2
net.sgen["controllable"] = False          # disable load shedding for now
net.line["max_loading_percent"]  = 1e6    # effectively no thermal limit
net.trafo["max_loading_percent"] = 1e6
if not net.trafo3w.empty:
    net.trafo3w["max_loading_percent"] = 1e6
try_opf(net, "trivial", init="flat", calculate_voltage_angles=True,
        check_connectivity=False, numba=False, verbose=False)

print("\n" + "=" * 70)
print("TEST 2 - Same but init from a converged power flow")
print("=" * 70)
net = fresh()
net.bus["min_vm_pu"] = 0.8
net.bus["max_vm_pu"] = 1.2
net.sgen["controllable"] = False
net.line["max_loading_percent"]  = 1e6
net.trafo["max_loading_percent"] = 1e6
if not net.trafo3w.empty:
    net.trafo3w["max_loading_percent"] = 1e6
try:
    pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
             check_connectivity=False, enforce_q_lims=False)
    print(f"  pre-PF converged: {net.converged}, "
          f"V=[{net.res_bus.vm_pu.min():.3f},{net.res_bus.vm_pu.max():.3f}]")
except Exception as e:
    print(f"  pre-PF failed: {e}")
try_opf(net, "init=pf", init="pf", check_connectivity=False,
        numba=False, verbose=False)

print("\n" + "=" * 70)
print("TEST 3 - Tight voltage band [0.95, 1.05], still no thermal/sgen")
print("=" * 70)
net = fresh()
net.bus["min_vm_pu"] = 0.95
net.bus["max_vm_pu"] = 1.05
net.sgen["controllable"] = False
net.line["max_loading_percent"]  = 1e6
net.trafo["max_loading_percent"] = 1e6
if not net.trafo3w.empty:
    net.trafo3w["max_loading_percent"] = 1e6
try:
    pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
             check_connectivity=False, enforce_q_lims=False)
except Exception:
    pass
try_opf(net, "Vband", init="pf", check_connectivity=False,
        numba=False, verbose=False)

print("\n" + "=" * 70)
print("TEST 4 - Add thermal limits (100%)")
print("=" * 70)
net = fresh()
net.bus["min_vm_pu"] = 0.95
net.bus["max_vm_pu"] = 1.05
net.sgen["controllable"] = False
net.line["max_loading_percent"]  = 100.0
net.trafo["max_loading_percent"] = 100.0
if not net.trafo3w.empty:
    net.trafo3w["max_loading_percent"] = 100.0
try:
    pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
             check_connectivity=False, enforce_q_lims=False)
except Exception:
    pass
try_opf(net, "thermal", init="pf", check_connectivity=False,
        numba=False, verbose=False)

print("\n" + "=" * 70)
print("TEST 5 - Full OPF: tight V + thermal + load-shed sgens controllable")
print("=" * 70)
net = fresh()
net.bus["min_vm_pu"] = 0.95
net.bus["max_vm_pu"] = 1.05
# sgens stay controllable (load shedding enabled) - this is the real config
net.line["max_loading_percent"]  = 100.0
net.trafo["max_loading_percent"] = 100.0
if not net.trafo3w.empty:
    net.trafo3w["max_loading_percent"] = 100.0
try:
    pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
             check_connectivity=False, enforce_q_lims=False)
except Exception:
    pass
try_opf(net, "full", init="pf", check_connectivity=False,
        numba=False, verbose=False)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("  First FAILED test = the culprit:")
print("    Test 1 fails -> fundamental data problem (impedance/topology)")
print("    Test 2 passes, 1 fails -> just needs warm start (init=pf)")
print("    Test 3 fails -> voltage band too tight for this network")
print("    Test 4 fails -> a line/trafo thermal rating is too small")
print("    Test 5 fails -> load-shed sgen cost or bounds break the OPF")
print("=" * 70)