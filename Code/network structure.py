"""
explore_network_structure.py
=============================
Diagnostic: find out what voltage levels & gen distributions ACTUALLY
exist in Taipower network. This helps debug why no 345kV+ candidates found.

Run:
    python explore_network_structure.py
"""
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

import reXplan.config as cfg
cfg.path.inputFolder = r"C:\reXplan-repo\file\input"
from reXplan.network import Network

print("Loading network...")
net_obj = Network("taipower")
net = net_obj.pp_network

# ─────────────────────────────────────────────────────────────────────────────
# Voltage level distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  VOLTAGE LEVEL DISTRIBUTION (all buses)")
print("=" * 70)

vn_counts = net.bus["vn_kv"].value_counts().sort_index()
for vn, count in vn_counts.items():
    print(f"  {vn:>8.2f} kV : {count:>5} buses")

# ─────────────────────────────────────────────────────────────────────────────
# Generator distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  GENERATOR DISTRIBUTION")
print("=" * 70)

# Merge gen with bus info
gen_detail = net.gen.merge(
    net.bus[["vn_kv", "name"]].rename(columns={"name": "bus_name"}),
    left_on="bus", right_index=True
)

print(f"\n  Total gens: {len(gen_detail)}")
print(f"  Total gen max_p_mw: {gen_detail['max_p_mw'].sum():.1f} MW")
print(f"  Gens in service: {gen_detail['in_service'].sum()}")

# Distribution by voltage level
print(f"\n  Gen count & capacity by bus voltage level:")
vn_gen = gen_detail.groupby("vn_kv").agg(
    n_gens=("max_p_mw", "count"),
    total_pmax=("max_p_mw", "sum"),
    max_single=("max_p_mw", "max"),
).reset_index()
for _, row in vn_gen.iterrows():
    print(f"  {row['vn_kv']:>8.2f} kV : {int(row['n_gens']):>4} gens, "
          f"total {row['total_pmax']:>8.1f} MW, "
          f"largest {row['max_single']:>7.1f} MW")

# ─────────────────────────────────────────────────────────────────────────────
# Top 20 largest generators
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TOP 20 LARGEST GENERATORS")
print("=" * 70)

top_gens = gen_detail.nlargest(20, "max_p_mw")[
    ["bus", "bus_name", "vn_kv", "max_p_mw", "p_mw", "in_service"]
]
print(f"\n  {'Bus':>6} {'Name':<20} {'vn_kv':>7} {'Pmax':>8} {'Pset':>8} {'InSvc':>6}")
print(f"  {'-'*6} {'-'*20} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")
for _, r in top_gens.iterrows():
    name = str(r["bus_name"])[:20]
    print(f"  {int(r['bus']):>6} {name:<20} "
          f"{r['vn_kv']:>7.2f} {r['max_p_mw']:>8.1f} {r['p_mw']:>8.1f} "
          f"{str(r['in_service']):>6}")

# ─────────────────────────────────────────────────────────────────────────────
# Buses with highest voltage level (backbone)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HIGHEST VOLTAGE BUSES")
print("=" * 70)

max_vn = net.bus["vn_kv"].max()
print(f"\n  Max vn_kv in network: {max_vn} kV")
print(f"  Buses at max voltage ({max_vn} kV): {(net.bus['vn_kv'] == max_vn).sum()}")

# Top voltage tier
top_tier_threshold = max_vn * 0.95
top_tier_buses = net.bus[net.bus["vn_kv"] >= top_tier_threshold]
print(f"  Buses at top tier (>={top_tier_threshold:.0f} kV): {len(top_tier_buses)}")

# Any gens at top tier?
gens_at_top = gen_detail[gen_detail["vn_kv"] >= top_tier_threshold]
print(f"  Gens at top tier: {len(gens_at_top)}")
if len(gens_at_top) > 0:
    print(f"  Total Pmax at top tier: {gens_at_top['max_p_mw'].sum():.1f} MW")
    print(f"\n  Top tier gens:")
    print(f"  {'Bus':>6} {'Name':<20} {'vn_kv':>7} {'Pmax':>8}")
    for _, r in gens_at_top.nlargest(10, "max_p_mw").iterrows():
        name = str(r["bus_name"])[:20]
        print(f"  {int(r['bus']):>6} {name:<20} "
              f"{r['vn_kv']:>7.2f} {r['max_p_mw']:>8.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Line loading analysis from DC-PF
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  LINE LOADING ANALYSIS (DC-PF)")
print("=" * 70)

import pandapower as pp
try:
    pp.rundcpp(net, check_connectivity=False)
    res_line = net.res_line.copy()
    res_line["from_vn_kv"] = res_line.index.map(
        lambda i: net.bus.at[int(net.line.at[i, "from_bus"]), "vn_kv"]
    )
    res_line["line_name"] = net.line["name"]
    
    # Top 20 overloaded lines
    print(f"\n  Top 20 overloaded lines:")
    print(f"  {'Line':<30} {'vn_kv':>7} {'Loading%':>10} {'max_i_ka':>10}")
    top_overload = res_line.nlargest(20, "loading_percent")
    for idx, r in top_overload.iterrows():
        line_name = str(r["line_name"])[:30]
        max_i = net.line.at[idx, "max_i_ka"]
        print(f"  {line_name:<30} {r['from_vn_kv']:>7.1f} "
              f"{r['loading_percent']:>10.1f} {max_i:>10.3f}")
    
    # Summary
    print(f"\n  Lines loading > 100%: {(res_line['loading_percent'] > 100).sum()}")
    print(f"  Lines loading > 200%: {(res_line['loading_percent'] > 200).sum()}")
    print(f"  Lines loading > 300%: {(res_line['loading_percent'] > 300).sum()}")
    
except Exception as e:
    print(f"  DC-PF failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Check source data from network_meta.json
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CHECK PSSE SLACK BUS")
print("=" * 70)

import json
from pathlib import Path

meta_path = Path(r"C:\reXplan-repo\Project Taipower\Results\step01\network_meta.json")
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"\n  network_meta.json contents:")
    for k, v in meta.items():
        if isinstance(v, list):
            print(f"    {k}: list of {len(v)} items")
        else:
            print(f"    {k}: {v}")
    
    if "slack_bus" in meta:
        slack_bus_psse = meta["slack_bus"]
        print(f"\n  → PSSE says slack bus is: {slack_bus_psse}")
        if slack_bus_psse in net.bus.index:
            print(f"  Bus exists in pp_network: ✓")
            print(f"    vn_kv: {net.bus.at[slack_bus_psse, 'vn_kv']}")
            print(f"    name: {net.bus.at[slack_bus_psse, 'name']}")
            gens_here = net.gen[net.gen["bus"] == slack_bus_psse]
            print(f"    Gens at this bus: {len(gens_here)}")
            if len(gens_here) > 0:
                print(f"    Total Pmax here: {gens_here['max_p_mw'].sum():.1f} MW")
else:
    print(f"\n  ✗ {meta_path} not found")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)