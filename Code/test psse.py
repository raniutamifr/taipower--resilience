"""
test_psse_direct.py
====================
Load PSSE RAW directly via pandapower's built-in converter.
Skip the custom Step 1 parser (which has a bug with 3-winding transformers).

This tests ONE hypothesis: if pandapower's official PSSE converter handles
the Taipower RAW file correctly, then AC-PF/OPF should converge.


"""
import time
from pathlib import Path

import pandapower as pp
import pandapower.converter as pc

RAW_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\11507DP_base(108).raw")

print("=" * 70)
print("  Direct PSSE RAW → pandapower test")
print("=" * 70)
print(f"  File: {RAW_FILE}")
print(f"  Exists: {RAW_FILE.exists()}")
print()

if not RAW_FILE.exists():
    print(" File not found — check path")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load PSSE RAW via pandapower's official converter
# ─────────────────────────────────────────────────────────────────────────────
print("Loading PSSE RAW via pandapower converter...")
t0 = time.perf_counter()
try:
    # pandapower.converter.from_psse — battle-tested, handles 2W/3W trafos,
    # phase shifters, switched shunts, etc. correctly.
    net = pc.from_psse(str(RAW_FILE), f_hz=60)
    t_load = time.perf_counter() - t0
    print(f" Loaded in {t_load:.1f}s\n")
except Exception as e:
    t_load = time.perf_counter() - t0
    print(f" Load failed after {t_load:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Report structure
# ─────────────────────────────────────────────────────────────────────────────
print("Network structure:")
print(f"  Buses       : {len(net.bus):>5d}")
print(f"  Lines       : {len(net.line):>5d}")
print(f"  2W trafos   : {len(net.trafo):>5d}")
print(f"  3W trafos   : {len(net.trafo3w):>5d}  ← custom parser missed these!")
print(f"  Loads       : {len(net.load):>5d}")
print(f"  Generators  : {len(net.gen):>5d}")
print(f"  Static gens : {len(net.sgen):>5d}")
print(f"  Ext_grid    : {len(net.ext_grid):>5d}")
print(f"  Shunts      : {len(net.shunt):>5d}")
print()

# Slack placement
if len(net.ext_grid) > 0:
    for _, eg in net.ext_grid.iterrows():
        bus = int(eg["bus"])
        print(f"  Ext_grid at bus {bus}: vn_kv={net.bus.at[bus, 'vn_kv']}, "
              f"vm_pu={eg['vm_pu']}, name='{net.bus.at[bus, 'name']}'")
print()

# Voltage levels with gens
print("Generator distribution by voltage level:")
gen_vn = net.gen.merge(
    net.bus[["vn_kv"]], left_on="bus", right_index=True
)
vn_summary = gen_vn.groupby("vn_kv").agg(
    n_gens=("max_p_mw", "count"),
    total_pmax=("max_p_mw", "sum"),
).sort_index()
for vn, row in vn_summary.iterrows():
    print(f"  {vn:>7.1f} kV: {int(row['n_gens']):>3} gens, "
          f"total {row['total_pmax']:>7.1f} MW")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Base AC Power Flow
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Test 1: AC Power Flow (base case)")
print("=" * 70)

t0 = time.perf_counter()
try:
    pp.runpp(net, numba=False, calculate_voltage_angles=True,
             check_connectivity=True)
    t_pf = time.perf_counter() - t0
    vm_min = net.res_bus["vm_pu"].min()
    vm_max = net.res_bus["vm_pu"].max()
    max_ll = net.res_line["loading_percent"].max() if len(net.res_line) > 0 else 0
    print(f"  ✓ AC-PF CONVERGED in {t_pf:.1f}s")
    print(f"    Voltage: [{vm_min:.4f}, {vm_max:.4f}] pu")
    print(f"    Max line loading: {max_ll:.1f}%")
    print(f"    Total gen: {net.res_gen['p_mw'].sum():.1f} MW")
    print(f"    Total load: {net.load['p_mw'].sum():.1f} MW")
    print(f"    Ext_grid: {net.res_ext_grid['p_mw'].sum():+.1f} MW")
    acpf_ok = True
except Exception as e:
    t_pf = time.perf_counter() - t0
    print(f"  ✗ AC-PF FAILED in {t_pf:.1f}s: {type(e).__name__}: {e}")
    acpf_ok = False
print()

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: AC-OPF (only if AC-PF works)
# ─────────────────────────────────────────────────────────────────────────────
if acpf_ok:
    print("=" * 70)
    print("  Test 2: AC-OPF (PYPOWER solver)")
    print("=" * 70)

    # Ensure generators have cost — add dummy cost if missing
    if len(net.poly_cost) == 0:
        print("  Adding dummy linear cost function to all gens/ext_grid...")
        for idx in net.gen.index:
            pp.create_poly_cost(net, element=idx, et="gen",
                                cp1_eur_per_mw=50)  # 50 NT$/MWh linear
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, element=idx, et="ext_grid",
                                cp1_eur_per_mw=100)  # ext_grid expensive

    t0 = time.perf_counter()
    try:
        pp.runopp(net, verbose=False)
        t_opf = time.perf_counter() - t0
        vm_min = net.res_bus["vm_pu"].min()
        vm_max = net.res_bus["vm_pu"].max()
        total_cost = (net.res_gen["p_mw"].sum() * 50
                      + net.res_ext_grid["p_mw"].sum() * 100)
        print(f"   AC-OPF CONVERGED in {t_opf:.1f}s")
        print(f"    Voltage: [{vm_min:.4f}, {vm_max:.4f}] pu")
        print(f"    Total gen: {net.res_gen['p_mw'].sum():.1f} MW")
        print(f"    Ext_grid: {net.res_ext_grid['p_mw'].sum():+.1f} MW")
        print(f"    Estimated dispatch cost: ~{total_cost:.0f} NT$/hr")
    except Exception as e:
        t_opf = time.perf_counter() - t0
        print(f"   AC-OPF FAILED in {t_opf:.1f}s: {type(e).__name__}: {e}")
else:
    print("Skipping AC-OPF (base AC-PF didn't converge)")

print()
print("=" * 70)
print("  Summary")
print("=" * 70)
if acpf_ok:
    print("  Pandapower direct PSSE loader WORKS.")
    print("  Data is NOT broken — it was the custom Step 1 parser.")
    print("  Next step: export this `net` to reXplan xlsx format.")
else:
    print("  Even pandapower's official loader fails.")
    print("  This means the RAW file itself needs review at source (Taipower).")