"""
Find a solvable AC operating point.

The AC-OPF deep diagnostic showed even a trivial OPF and a flat-start
AC PF fail. But Step 06's self-test (init='dc', dispatched gens) DID
converge. So a stable operating point exists - the OPF just cannot find
it from a flat start. This script pins down which initialisation and
solver options reach a converged AC PF, which is the warm start the OPF
needs.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")


def fresh(load_scale=0.75):
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
    net.load["p_mw"]   *= load_scale
    net.load["q_mvar"] *= load_scale
    mask = net.sgen["name"].str.startswith("LoadShed_")
    net.sgen.loc[mask, "max_p_mw"] *= load_scale
    net.gen["q_mvar"] = 0.0
    if "sn_mva" in net.gen.columns:
        net.gen["sn_mva"] = net.gen["max_p_mw"].clip(lower=1.0)
    return net


def dispatch(net):
    """Dispatch gens proportional to load (the Step-06 self-test style)."""
    isv = net.gen.in_service
    tl = float(net.load.loc[net.load.in_service, "p_mw"].sum())
    tp = float(net.gen.loc[isv, "max_p_mw"].sum())
    if tp <= 0:
        return
    s = min(tl / tp, 1.0)
    net.gen.loc[isv, "p_mw"] = (net.gen.loc[isv, "max_p_mw"] * s).clip(
        lower=net.gen.loc[isv, "min_p_mw"])


def test_pf(label, net, **kw):
    try:
        pp.runpp(net, **kw)
        if net.converged:
            vm = net.res_bus.vm_pu
            print(f"  [{label:32s}] CONVERGED  "
                  f"V=[{vm.min():.3f},{vm.max():.3f}]  "
                  f"slack={net.res_ext_grid.p_mw.iloc[0]:+.0f}MW")
            return True
        print(f"  [{label:32s}] not converged")
        return False
    except Exception as e:
        print(f"  [{label:32s}] FAIL: {str(e)[:55]}")
        return False


print("=" * 72)
print("PROBE: which init / option reaches a converged AC power flow?")
print("=" * 72)

print("\n-- Group A: flat start variants --")
net = fresh(); dispatch(net)
test_pf("flat, no qlim", net, algorithm="nr", init="flat",
        max_iteration=100, check_connectivity=False, enforce_q_lims=False)
net = fresh(); dispatch(net)
test_pf("flat, qlim", net, algorithm="nr", init="flat",
        max_iteration=100, check_connectivity=False, enforce_q_lims=True)

print("\n-- Group B: dc init (Step-06 self-test style) --")
net = fresh(); dispatch(net)
test_pf("dc, no qlim", net, algorithm="nr", init="dc",
        max_iteration=100, check_connectivity=False, enforce_q_lims=False)
net = fresh(); dispatch(net)
test_pf("dc, qlim", net, algorithm="nr", init="dc",
        max_iteration=100, check_connectivity=False, enforce_q_lims=True)

print("\n-- Group C: dc init WITHOUT manual dispatch (gens at original pg) --")
net = fresh()
test_pf("dc, original pg, no qlim", net, algorithm="nr", init="dc",
        max_iteration=100, check_connectivity=False, enforce_q_lims=False)

print("\n-- Group D: robust algorithms --")
net = fresh(); dispatch(net)
test_pf("iwamoto_nr (robust)", net, algorithm="iwamoto_nr", init="dc",
        max_iteration=100, check_connectivity=False, enforce_q_lims=False)
net = fresh(); dispatch(net)
test_pf("gauss-seidel", net, algorithm="gs", init="dc",
        max_iteration=3000, check_connectivity=False, enforce_q_lims=False)
net = fresh(); dispatch(net)
test_pf("backward/forward sweep", net, algorithm="bfsw", init="dc",
        max_iteration=100, check_connectivity=False, enforce_q_lims=False)

print("\n-- Group E: lighter load (does it converge if load is tiny?) --")
for ls in [0.50, 0.30, 0.10]:
    net = fresh(load_scale=ls); dispatch(net)
    test_pf(f"dc, load_scale={ls}", net, algorithm="nr", init="dc",
            max_iteration=100, check_connectivity=False, enforce_q_lims=False)

print("\n-- Group F: diagnostic on the base network --")
net = fresh(); dispatch(net)
try:
    diag = pp.diagnostic(net, report_style="compact", warnings_only=True)
    print("  pp.diagnostic findings:")
    for k, v in (diag or {}).items():
        # show the headline of each issue category
        n = len(v) if hasattr(v, "__len__") else v
        print(f"    {k}: {n}")
except Exception as e:
    print(f"  diagnostic failed: {str(e)[:80]}")

print("\n" + "=" * 72)
print("READ: the first CONVERGED line tells us the init/options the OPF")
print("needs. If Group E converges only at low load, it's a Q-support /")
print("voltage-collapse problem. If nothing converges, pp.diagnostic")
print("(Group F) lists the offending elements.")
print("=" * 72)