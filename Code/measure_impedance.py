"""
Measure the actual impedance distribution in branches.csv so we can pick
the correct bus-tie threshold instead of guessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path

STEP01 = Path(r"C:\reXplan-repo\Project Taipower\Results\step01")
BASE_MVA = 100.0

br = pd.read_csv(STEP01 / "branches.csv")
bus = pd.read_csv(STEP01 / "buses.csv")

# kv per bus for zbase
kv = dict(zip(bus["bus_i"].astype(int),
              bus["base_kv"].astype(float).clip(lower=0.1)))

r = br["r_pu"].astype(float).fillna(0)
x = br["x_pu"].astype(float).fillna(0)
zmag = np.sqrt(r**2 + x**2)

print("=" * 60)
print("BRANCH SERIES IMPEDANCE |Z| (per-unit, system base)")
print("=" * 60)
for q in [0, 1, 5, 10, 25, 50]:
    print(f"  {q:>2}th percentile : {np.percentile(zmag, q):.3e} pu")
print(f"  max            : {zmag.max():.3e} pu")
print(f"  total branches : {len(br)}")

print("\n" + "=" * 60)
print("HOW MANY BRANCHES FALL BELOW EACH pu THRESHOLD")
print("=" * 60)
for thr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
    n = (zmag < thr).sum()
    print(f"  |Z| < {thr:.0e} pu : {n:>5} branches ({100*n/len(br):.1f}%)")

# Convert to ohms the same way Build_Pandamodel does, to mirror what
# pandapower's diagnostic sees (r_ohm <= 0.001).
print("\n" + "=" * 60)
print("MIRROR pandapower CHECK: branches with r_ohm<=1e-3 OR x_ohm<=1e-3")
print("=" * 60)
n_bad = 0
for _, row in br.iterrows():
    fb = int(row.get("from_bus", -1))
    rr = float(row.get("r_pu", 0) or 0)
    xx = float(row.get("x_pu", 0) or 0)
    fkv = kv.get(fb, 100.0)
    zbase = fkv**2 / BASE_MVA
    if (rr * zbase) <= 1e-3 or (xx * zbase) <= 1e-3:
        n_bad += 1
print(f"  branches pandapower would flag : {n_bad} / {len(br)}")
print(f"  -> these are the ones that must become switches")

print("\n" + "=" * 60)
print("RECOMMENDED Z_TIE_PU")
print("=" * 60)
# Pick the smallest pu threshold that captures >= the pandapower-flagged set
for thr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
    if (zmag < thr).sum() >= n_bad:
        print(f"  Use Z_TIE_PU = {thr:.0e}  "
              f"(captures {(zmag<thr).sum()} branches, >= {n_bad} flagged)")
        break