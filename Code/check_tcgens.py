"""
check_taichung_gens.py
======================
Audit Taichung (台中/中火) generator matching.

The cost_match_log_v3.csv shows 22 generators mapped to plant '台中'.
This script:
  1. Lists those 22 with their bus_name + capacity, so we see whether they
     came from '台中' bus prefix or '中火' bus prefix.
  2. Searches the bus inventory for any bus whose name contains '台中' or
     '中火' but whose generators were classified as fallback — i.e. real
     Taichung gens that the matcher missed.
  3. Shows summary so we know whether there are 'missing' Taichung gens
     or whether the 22 figure is already complete.

The earlier probe found 6 bus names containing '台中' and 21 containing
'中火' (printed under '== Plant-prefix hits in bus names =='). 27 bus-name
hits != 22 gen hits, which is what prompted this check — but bus count
and generator count are different things (multiple gens can share a bus,
and some bus names may match without hosting any generator).
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
LOG_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\cost_match_log_v3.csv")

def sec(t): print("\n"+"="*70+f"\n  {t}\n"+"="*70)

net = pp.from_json(str(NET_FILE))
log = pd.read_csv(LOG_FILE, encoding="utf-8-sig")

# ── A : the 22 gens already mapped to 台中 ──────────────────────────────────
sec("A : 22 GENERATORS CURRENTLY MAPPED TO 台中")
tc = log[log.plant == "台中"].copy()
print(tc[["gen","name","bus_name","cap_mw"]].to_string(index=False))

# How many came from each bus-name prefix?
print("\n  Source bus-name prefix breakdown:")
def src(s):
    s = str(s)
    if "中火" in s: return "中火"
    if "台中" in s: return "台中"
    return "other"
print(tc["bus_name"].apply(src).value_counts().to_string())

# ── B : Are there bus names with 中火 / 台中 NOT in the 22? ─────────────────
sec("B : BUSES WITH 中火 OR 台中 IN NAME, AND WHETHER THEY HOST GENERATORS")

bus_tc = net.bus[net.bus["name"].astype(str).str.contains("中火|台中",regex=True)].copy()
bus_tc["has_gen"] = bus_tc.index.isin(net.gen["bus"].values)
print(f"\n  {len(bus_tc)} buses match '中火' or '台中':")
print(bus_tc[["name","vn_kv","has_gen"]].to_string())

# Buses that DO host a gen but the gen ended up as fallback (= the gap)
fallback = log[log["how"].astype(str).str.startswith("fallback")]
bus_tc_with_gen = bus_tc[bus_tc["has_gen"]]
gen_buses_tc = net.gen[net.gen["bus"].isin(bus_tc_with_gen.index)]
gen_buses_tc = gen_buses_tc.merge(
    bus_tc[["name"]].rename(columns={"name":"bus_name_actual"}),
    left_on="bus", right_index=True
)
gen_buses_tc["in_fallback"] = gen_buses_tc.index.isin(fallback["gen"].values)

sec("C : GENERATORS ON 中火/台中 BUSES THAT FELL TO FALLBACK (the missing ones)")
missing = gen_buses_tc[gen_buses_tc["in_fallback"]]
if len(missing) == 0:
    print("  None. The 22-gen count is complete; nothing to fix.")
else:
    print(f"\n  Found {len(missing)} Taichung generators classified as fallback:")
    print(missing[["name","bus","bus_name_actual","max_p_mw"]].to_string(index=True))
    print("\n  -> Add their bus-name pattern to PLANT_PATTERNS in build_real_cost_v3.py")

# ── D : Look for anything that LOOKS like Taichung but uses a different code ─
sec("D : OTHER PLAUSIBLE TAICHUNG-RELATED BUS NAMES (any name with 中 or 火)")
candidates = net.bus[net.bus["name"].astype(str).str.contains("中|火",regex=True)]
# Restrict to plant-like buses (not D=distribution, etc.) – heuristic: vn_kv >= 11
candidates = candidates[candidates["vn_kv"] >= 11]
# Show only those that host generators
has_gen_mask = candidates.index.isin(net.gen["bus"].values)
candidates = candidates[has_gen_mask]
# And only those NOT already matched as 中火/台中
candidates = candidates[~candidates["name"].astype(str).str.contains("中火|台中",regex=True)]
print(f"\n  {len(candidates)} other gen-hosting buses with '中' or '火' (excl. 中火/台中):")
if len(candidates):
    print(candidates[["name","vn_kv"]].head(40).to_string())
else:
    print("  None — no other naming variant for Taichung in the network.")