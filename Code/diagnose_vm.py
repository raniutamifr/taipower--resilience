import pandapower as pp

net = pp.from_json(r'C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json')

# ── 1. Slack bus check ────────────────────────────────────────────────
slack_bus = int(net.ext_grid.bus.values[0])
slack_vm  = float(net.ext_grid.vm_pu.values[0])
print(f"Slack bus: {slack_bus}, ext_grid vm_pu: {slack_vm:.4f}")

gens_at_slack = net.gen[net.gen.bus == slack_bus]
print(f"Gens at slack bus: {len(gens_at_slack)}")
if len(gens_at_slack):
    print(f"  Their vm_pu values: {gens_at_slack.vm_pu.unique()}")

# ── 2. Multi-gen buses with different vm_pu ───────────────────────────
gen_groups = net.gen.groupby('bus').agg(
    n_gens=('vm_pu', 'count'),
    n_unique_vm=('vm_pu', 'nunique'),
    vm_min=('vm_pu', 'min'),
    vm_max=('vm_pu', 'max'),
)
conflicts = gen_groups[gen_groups['n_unique_vm'] > 1]
print(f"\nBuses with multi-gen vm_pu conflicts: {len(conflicts)}")
print(conflicts.head(10))