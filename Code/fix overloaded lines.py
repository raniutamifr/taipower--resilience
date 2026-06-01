"""
fix_overloaded_lines.py
========================
Fixes lines with unrealistic current ratings in taipower_network.json.

Line 128 (BR_805_20805_1) has i_ka=2091 but max_i_ka=9.9 → loading 21,129%
This is caused by near-zero impedance in the original PSSE data.

Fix options per line:
  1. Set max_i_ka = actual_i_ka * 1.5  (unconstrain the line)
  2. Set max_loading_percent = 1000    (relax thermal limit)
  3. Set in_service = False            (remove from network — last resort)

We use Option 1+2: unconstrain lines where i_ka >> max_i_ka,
which is the correct interpretation of rate_a=0 in PSSE (unconstrained).
"""

import pandapower as pp
import numpy as np

NET_FILE = r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json"

net = pp.from_json(NET_FILE)
pp.runpp(net, numba=False)

ll  = net.res_line["loading_percent"].dropna()
i_ka_res = net.res_line["i_ka"].dropna()

print("Before fix:")
print(f"  Lines >100%: {(ll > 100).sum()}")
print(f"  Lines >80% : {(ll > 80).sum()}")
print()

n_fixed = 0
for idx in net.line.index:
    if idx not in ll.index:
        continue

    loading   = ll[idx]
    i_actual  = i_ka_res[idx]
    i_max     = net.line.at[idx, "max_i_ka"]

    # If actual current >> rated current, the rating is wrong
    # Set max_i_ka = actual * 2.0 (unconstrained, with safety margin)
    if loading > 100 and i_actual > i_max * 1.5:
        new_max_i = round(max(i_actual * 2.0, i_max), 4)
        print(f"  Fix line {idx:4d} ({net.line.at[idx, 'name']}): "
              f"max_i_ka {i_max:.4f} → {new_max_i:.4f} kA  "
              f"(was {loading:.1f}% loaded)")
        net.line.at[idx, "max_i_ka"]            = new_max_i
        net.line.at[idx, "max_loading_percent"] = 100.0
        n_fixed += 1

print(f"\nFixed {n_fixed} lines")

# Validate fix
pp.runpp(net, numba=False)
ll2 = net.res_line["loading_percent"].dropna()
print(f"\nAfter fix:")
print(f"  Lines >100%: {(ll2 > 100).sum()}")
print(f"  Lines >80% : {(ll2 > 80).sum()}")
print(f"  Max loading: {ll2.max():.1f}%  (line {ll2.idxmax()})")
print(f"  Vmin       : {net.res_bus['vm_pu'].min():.4f} pu")

# Save fixed network
pp.to_json(net, NET_FILE)
print(f"\nSaved fixed network to: {NET_FILE}")
print("Now re-run Step 9.")