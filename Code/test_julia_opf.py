"""
test_taipower_julia.py - Test Julia PowerModels dengan network Taipower asli
"""

import pandapower as pp
from pathlib import Path

RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
STEP06_DIR = RESULT_BASE / "step06"
NET_FILE = STEP06_DIR / "taipower_network.json"

print("="*60)
print("TEST JULIA POWERMODELS - TAIPOWER NETWORK")
print("="*60)

# Load network
print(f"\n[1] Loading network from {NET_FILE}")
if not NET_FILE.exists():
    print(f"    File not found! Run step06 first.")
    print(f"   Looking for: {NET_FILE}")
    exit(1)

net = pp.from_json(str(NET_FILE))
print(f"    Loaded: {len(net.bus)} buses, {len(net.gen)} gens, {len(net.load)} loads")

# Cek cost functions
print(f"\n[2] Checking cost functions...")
if hasattr(net, 'poly_cost') and len(net.poly_cost) > 0:
    print(f"    {len(net.poly_cost)} cost entries found")
    for _, row in net.poly_cost.iterrows():
        print(f"      {row['et']} {row['element']}: cp1={row.get('cp1_eur_per_mw', 0)}")
else:
    print(f"    No costs found, adding defaults...")
    for idx in net.gen.index:
        pp.create_poly_cost(net, idx, 'gen', cp1_eur_per_mw=40, cp2_eur_per_mw2=0.01)
    for idx in net.ext_grid.index:
        pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=60, cp2_eur_per_mw2=0.01)

print("\n" + "="*50)
print("TEST 1: Python runopp() - Baseline")
print("="*50)

try:
    pp.runopp(net, verbose=False, max_iteration=100, tolerance_mva=1e-6)
    if net.converged:
        print(f" Python runopp CONVERGED!")
        print(f"   Cost: {net.res_cost:.2f}")
        print(f"   Losses: {net.res_loss_pw:.2f} MW")
    else:
        print(f" Python runopp did not converge")
except Exception as e:
    print(f" Python runopp failed: {e}")

print("\n" + "="*50)
print("TEST 2: Julia runpm() - Primary Target")
print("="*50)

# Buat fresh copy untuk Julia
net_julia = pp.from_json(str(NET_FILE))

# Pastikan costs ada
if len(net_julia.poly_cost) == 0:
    for idx in net_julia.gen.index:
        pp.create_poly_cost(net_julia, idx, 'gen', cp1_eur_per_mw=40, cp2_eur_per_mw2=0.01)
    for idx in net_julia.ext_grid.index:
        pp.create_poly_cost(net_julia, idx, 'ext_grid', cp1_eur_per_mw=60, cp2_eur_per_mw2=0.01)

print("   Calling Julia PowerModels (first time ~30-60s)...")
import time
t0 = time.time()

try:
    pp.runpm(
        net_julia,
        pm_model="ACPPowerModel",
        pm_solver="ipopt",
        pm_time_limit=120.0,
        pm_log_level=0,
        delete_buffer_file=True,
        verbose=False
    )
    elapsed = time.time() - t0
    
    if net_julia.converged:
        print(f"\n Julia runpm CONVERGED in {elapsed:.1f}s!")
        print(f"   Cost: {net_julia.res_cost:.2f}")
        print(f"   Losses: {net_julia.res_loss_pw:.2f} MW")
        print(f"   Max V: {net_julia.res_bus.vm_pu.max():.4f}")
        print(f"   Min V: {net_julia.res_bus.vm_pu.min():.4f}")
    else:
        print(f"\n Julia runpm did not converge in {elapsed:.1f}s")
        
except Exception as e:
    elapsed = time.time() - t0
    print(f"\n Julia runpm FAILED after {elapsed:.1f}s: {e}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if 'net_julia' in locals() and net_julia.converged:
    print(" Julia PowerModels is WORKING for Taipower network!")
    print("   You can proceed with 1000 samples batch.")
elif 'net' in locals() and net.converged:
    print(" Julia not working, but Python runopp is WORKING.")
    print("   For 1000 samples, you can use Python runopp.")
else:
    print(" Neither Julia nor Python OPF is working.")
    print("   Need to debug network issues first.")

print("="*60)