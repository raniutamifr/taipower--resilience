"""
Diagnose which buses are dropped by convert_pp_to_pm AFTER all runtime fixes.
Run this to find what needs to be added to PROBLEM_BUSES in 9_OPF_Solver.py
"""
import pandapower as pp
import pandapower.converter.pandamodels as pdm
import pandapower.topology as top

NET_FILE = r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json"

net = pp.from_json(NET_FILE)

print("=" * 65)
print("Applying ALL fixes exactly as in solve_opf()...")
print("=" * 65)

VOLL = 50_000
for gidx in net.gen.index[net.gen["bus"] == 0]:
    already = ((net.poly_cost["et"] == "gen") & (net.poly_cost["element"] == gidx))
    if not already.any():
        pp.create_poly_cost(net, gidx, "gen", cp1_eur_per_mw=0.0, cp0_eur=0.0)
        print(f"  Added zero cost to gen {gidx} on bus 0")

for egidx in net.ext_grid.index[net.ext_grid["bus"] == 0]:
    already = ((net.poly_cost["et"] == "ext_grid") & (net.poly_cost["element"] == egidx))
    if not already.any():
        pp.create_poly_cost(net, egidx, "ext_grid", cp1_eur_per_mw=VOLL, cp0_eur=0.0)
        print(f"  Added VOLL cost to ext_grid {egidx} on bus 0")

for bus_idx in [1966, 1967, 1968, 1969]:
    if bus_idx not in net.bus.index:
        continue
    for tbl in ("load", "gen", "sgen", "ext_grid"):
        df = getattr(net, tbl, None)
        if df is not None and not df.empty:
            df.loc[df["bus"] == bus_idx, "in_service"] = False
    net.line.loc[(net.line.from_bus==bus_idx)|(net.line.to_bus==bus_idx), "in_service"] = False
    net.trafo.loc[(net.trafo.hv_bus==bus_idx)|(net.trafo.lv_bus==bus_idx), "in_service"] = False
    net.bus.at[bus_idx, "in_service"] = False
    print(f"  Disabled bus {bus_idx}")

for b in [621, 622]:
    net.load.loc[net.load.bus == b, "in_service"] = False
    net.sgen.loc[net.sgen.bus == b, "in_service"] = False
    net.gen.loc[net.gen.bus == b, "in_service"] = False
    net.line.loc[(net.line.from_bus==b)|(net.line.to_bus==b), "in_service"] = False
    net.bus.at[b, "in_service"] = False
    print(f"  Disabled problem bus {b}")

mg = top.create_nxgraph(net)
islands = list(top.connected_components(mg))
slack_bus = net.ext_grid.bus.iloc[0]
main_island = next((isl for isl in islands if slack_bus in isl), max(islands, key=len))
for island in islands:
    if island != main_island:
        for bus in island:
            if bus in net.bus.index:
                net.bus.at[bus, "in_service"] = False
print(f"  Islands removed: {len(islands)-1}")

active = set(net.bus.index[net.bus["in_service"]])
for tbl in ("load", "gen", "sgen", "ext_grid"):
    df = getattr(net, tbl, None)
    if df is not None and not df.empty:
        orphans = df.index[~df["bus"].isin(active) & df["in_service"]]
        if len(orphans):
            df.loc[orphans, "in_service"] = False
for tbl, fc, tc in [("line","from_bus","to_bus"), ("trafo","hv_bus","lv_bus")]:
    df = getattr(net, tbl, None)
    if df is not None and not df.empty:
        orphans = df.index[(~df[fc].isin(active)|~df[tc].isin(active)) & df["in_service"]]
        if len(orphans):
            df.loc[orphans, "in_service"] = False

print("\n" + "=" * 65)
print("Converting to PowerModels JSON...")
pm = pdm.convert_pp_to_pm(net)

active_pp = set(net.bus.index[net.bus["in_service"]])
active_pm = set(int(k) for k in pm["bus"].keys())
missing   = active_pp - active_pm

print(f"Active buses in pandapower : {len(active_pp)}")
print(f"Active buses in PM JSON    : {len(active_pm)}")
print(f"Buses dropped by converter : {len(missing)}")

if missing:
    print(f"\nMissing buses: {sorted(missing)}")
    print("\n--- Detail per missing bus ---")
    for b in sorted(missing):
        name   = net.bus.at[b, "name"]
        loads  = net.load[(net.load.bus==b) & net.load["in_service"]]
        gens   = net.gen[(net.gen.bus==b) & net.gen["in_service"]]
        trafos = net.trafo[((net.trafo.hv_bus==b)|(net.trafo.lv_bus==b)) & net.trafo["in_service"]]
        lines  = net.line[((net.line.from_bus==b)|(net.line.to_bus==b)) & net.line["in_service"]]
        costs  = net.poly_cost[net.poly_cost["element"].isin(list(gens.index))]
        print(f"  Bus {b} ({name}): {len(loads)} loads, {len(gens)} gens, "
              f"{len(trafos)} trafos, {len(lines)} lines, {len(costs)} cost entries")

    print("\n--- PM loads pointing to missing buses ---")
    for k, v in pm["load"].items():
        if v["load_bus"] in missing:
            pp_load = net.load[net.load.bus == v["load_bus"]]
            print(f"  PM load {k} -> bus {v['load_bus']} | pp load idx: {pp_load.index.tolist()}")
else:
    print("\n✓ All buses accounted for — no converter mismatch!")