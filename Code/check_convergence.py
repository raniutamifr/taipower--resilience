import pandapower as pp
import logging

print("=" * 55)
print("   TAIPOWER NETWORK - CONVERGENCE CHECK")
print("=" * 55)

# Load network
net = pp.from_json(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
print(f"\n Network loaded successfully!")
print(f"   Buses      : {len(net.bus)}")
print(f"   Lines      : {len(net.line)}")
print(f"   Trafos     : {len(net.trafo)}")
print(f"   Generators : {len(net.gen)} gen + {len(net.sgen)} sgen")
print(f"   Loads      : {len(net.load)}")

# ── 1. AC Power Flow ──────────────────────────────────────
print("\n" + "-" * 55)
print(" [1] AC POWER FLOW (runpp)")
print("-" * 55)
try:
    pp.runpp(net, init='flat', numba=False)
    vmin = round(net.res_bus.vm_pu.min(), 4)
    vmax = round(net.res_bus.vm_pu.max(), 4)
    print(f"   Status  : CONVERGED ✓")
    print(f"   V range : {vmin} - {vmax} pu")
except Exception as e:
    print(f"   Status  : FAILED ✗")
    print(f"   Reason  : {str(e)[:80]}")

# ── 2. AC OPF tanpa constraint (baseline) ─────────────────
print("\n" + "-" * 55)
print(" [2] AC OPF - BASELINE (no voltage constraint)")
print("-" * 55)
try:
    pp.runopp(net, numba=False)
    vmin = round(net.res_bus.vm_pu.min(), 4)
    vmax = round(net.res_bus.vm_pu.max(), 4)
    print(f"   Status  : CONVERGED ✓")
    print(f"   V range : {vmin} - {vmax} pu")
    if vmax > 1.05 or vmin < 0.95:
        print(f"   NOTE    : Voltage out of [0.95, 1.05] pu — no constraint active")
except Exception as e:
    print(f"   Status  : FAILED ✗")
    print(f"   Reason  : {str(e)[:80]}")

# ── 3. AC OPF dengan voltage constraint ───────────────────
print("\n" + "-" * 55)
print(" [3] AC OPF - WITH VOLTAGE CONSTRAINTS [0.95, 1.05]")
print("-" * 55)

# Set constraint SEBELUM runopp
net.bus['min_vm_pu'] = 0.95
net.bus['max_vm_pu'] = 1.05

# Aktifkan logging untuk lihat OPF log
pp.logger.setLevel(logging.DEBUG)

try:
    pp.runopp(net, numba=False)
    vmin = round(net.res_bus.vm_pu.min(), 4)
    vmax = round(net.res_bus.vm_pu.max(), 4)
    print(f"   Status  : CONVERGED ✓")
    print(f"   V range : {vmin} - {vmax} pu")
    # Cek apakah constraint benar-benar terpenuhi
    violations = net.res_bus[(net.res_bus.vm_pu < 0.95) | (net.res_bus.vm_pu > 1.05)]
    if len(violations) > 0:
        print(f"   WARNING : {len(violations)} bus masih melanggar batas!")
        print(violations[['vm_pu']].to_string())
    else:
        print(f"   Voltage constraints : ALL SATISFIED ✓")
except Exception as e:
    print(f"   Status  : FAILED ✗")
    print(f"   Reason  : {str(e)}")
    print(f"   --> Cek log di atas untuk detail OPF solver")

print("\n" + "=" * 55)
print(" SUMMARY")
print("=" * 55)
print("=" * 55)