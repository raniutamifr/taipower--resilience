import os
os.environ["JULIA_HOME"] = r"C:\Users\user\AppData\Local\Programs\Julia-1.10.9\bin"
os.environ["PATH"] = os.environ["JULIA_HOME"] + os.pathsep + os.environ["PATH"]

import pandapower as pp
import pandapower.topology as top
import copy

net = pp.from_json(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")

# ── Step 1: collapse switches (same as Step 09) ──────────────────────────────
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("opf", r"C:\reXplan-repo\Project Taipower\Code\9 OPF_Solver.py")
opf  = module_from_spec(spec)
spec.loader.exec_module(opf)

net = opf.collapse_switches_for_powermodels(net)
net = opf.full_network_sanitation(net)

print(f"Buses (total)  : {len(net.bus)}")
print(f"Buses active   : {net.bus.in_service.sum()}")
print(f"Gens  active   : {net.gen.in_service.sum()}")
print(f"Lines active   : {net.line.in_service.sum()}")

# ── Step 2: AC power flow ─────────────────────────────────────────────────────
try:
    pp.runpp(net, algorithm="nr", max_iteration=50, init="dc")
    print(f"\nAC PF converged : {net.converged}")
    if net.converged:
        print(f"Voltage range   : {net.res_bus.vm_pu.min():.3f} - {net.res_bus.vm_pu.max():.3f} pu")
        print(f"Slack P         : {net.res_ext_grid.p_mw.values[0]:.0f} MW")
except Exception as e:
    print(f"AC PF failed    : {e}")