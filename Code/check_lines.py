import pandapower as pp

net = pp.from_json(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
pp.runpp(net, numba=False)

ll = net.res_line["loading_percent"].dropna()
top10 = ll.nlargest(10)

print("Top 10 overloaded lines:")
print(f"{'idx':>5} {'loading%':>12} {'max_i_ka':>10} {'i_ka':>8} {'name'}")
print("-" * 70)
for idx, pct in top10.items():
    row  = net.line.loc[idx]
    i_ka = net.res_line.loc[idx, "i_ka"]
    print(f"{idx:>5} {pct:>12.1f} {row['max_i_ka']:>10.4f} {i_ka:>8.4f} {row['name']}")

print()
print(f"Lines >100%  : {(ll > 100).sum()}")
print(f"Lines >80%   : {(ll > 80).sum()}")
print()

# Ext grid info
print("Ext grid columns:", net.ext_grid.columns.tolist())
print(net.ext_grid[["name", "min_p_mw", "max_p_mw"]].to_string())