import pandapower as pp

net = pp.from_json(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")

pp.runopp(net, numba=False)

# Check NaN buses
nan_buses = net.res_bus[net.res_bus.vm_pu.isna()].copy()
nan_buses['bus_name'] = net.bus.loc[nan_buses.index, 'name']

print(f"Total NaN buses   : {len(nan_buses)}")
print(f"Total normal buses: {net.res_bus.vm_pu.notna().sum()}")

# Check if NaN buses have any connections
nan_idx = nan_buses.index.tolist()

in_line_from = net.line[net.line.from_bus.isin(nan_idx)]
in_line_to   = net.line[net.line.to_bus.isin(nan_idx)]
in_trafo_hv  = net.trafo[net.trafo.hv_bus.isin(nan_idx)]
in_trafo_lv  = net.trafo[net.trafo.lv_bus.isin(nan_idx)]

connected_via_line  = set(in_line_from.from_bus) | set(in_line_to.to_bus)
connected_via_trafo = set(in_trafo_hv.hv_bus)    | set(in_trafo_lv.lv_bus)

print(f"\nNaN buses connected via line : {len(connected_via_line)}")
print(f"NaN buses connected via trafo: {len(connected_via_trafo)}")

# Check in_service status
print(f"\nLines  with in_service=False : {(~net.line.in_service).sum()}")
print(f"Trafos with in_service=False : {(~net.trafo.in_service).sum()}")
print(f"Buses  with in_service=False : {(~net.bus.in_service).sum()}")