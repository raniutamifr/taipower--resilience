import pandapower as pp

net = pp.from_json(r'C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json')

# ── Patch: align gen vm_pu at slack bus to ext_grid vm_pu ─────────────
slack_bus = int(net.ext_grid.bus.values[0])
slack_vm  = float(net.ext_grid.vm_pu.values[0])
mask = net.gen.bus == slack_bus
n_fixed = mask.sum()
net.gen.loc[mask, 'vm_pu'] = slack_vm
print(f"Patched {n_fixed} gen(s) at slack bus {slack_bus} -> vm_pu = {slack_vm:.4f}")

# ── Run AC power flow ────────────────────────────────────────────────
print("\n>> Running AC Newton-Raphson power flow...")
try:
    pp.runpp(net, algorithm='nr', max_iteration=50)
    print(f"\nConverged: {net.converged}")
    print(f"Bus V range            : {net.res_bus.vm_pu.min():.3f} - {net.res_bus.vm_pu.max():.3f} pu")
    print(f"Bus V < 0.95           : {(net.res_bus.vm_pu < 0.95).sum()} buses")
    print(f"Bus V > 1.05           : {(net.res_bus.vm_pu > 1.05).sum()} buses")
    print(f"Lines > 100% loading   : {(net.res_line.loading_percent > 100).sum()}")
    print(f"Max line loading       : {net.res_line.loading_percent.max():.1f}%")
    print(f"Trafo > 100% loading   : {(net.res_trafo.loading_percent > 100).sum()}")
    print(f"Trafo3W > 100% loading : {(net.res_trafo3w.loading_percent_hv > 100).sum()}")
    print(f"Total gen P            : {net.res_gen.p_mw.sum():.0f} MW")
    print(f"Total load P           : {net.res_load.p_mw.sum():.0f} MW")
    print(f"Ext grid P (slack)     : {net.res_ext_grid.p_mw.values[0]:.0f} MW")
    print(f"Total losses           : {net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum() + net.res_trafo3w.pl_mw.sum():.0f} MW")
except Exception as e:
    print(f"\nFAILED: {type(e).__name__}: {e}")