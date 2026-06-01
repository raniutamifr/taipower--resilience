import pandapower as pp

net = pp.from_json(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")

# Check cost functions
print("=== POLYNOMIAL COST (gen) ===")
print(net.poly_cost[['element', 'et', 'cp1_eur_per_mw']].to_string())

print("\n=== PIECEWISE COST ===")
print(net.pwl_cost.to_string())

print("\n=== COST SUMMARY ===")
print(f"Gens with cost defined : {net.poly_cost[net.poly_cost.et == 'gen'].shape[0]}")
print(f"Sgens with cost defined: {net.poly_cost[net.poly_cost.et == 'sgen'].shape[0]}")
print(f"Total poly_cost entries: {len(net.poly_cost)}")
print(f"Total pwl_cost entries : {len(net.pwl_cost)}")