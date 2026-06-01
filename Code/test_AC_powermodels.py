import pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
net = pp.from_json(str(NET_FILE))

# Scale load ke 85% (seperti biasa)
load_scale = 0.85
net.load["p_mw"] *= load_scale
net.load["q_mvar"] *= load_scale
shed = net.sgen["name"].str.startswith("LoadShed_sgen_")
net.sgen.loc[shed, "max_p_mw"] *= load_scale

# Pastikan ext_grid controllable
if len(net.ext_grid) > 0:
    net.ext_grid["controllable"] = True

print("=" * 65)
print("TESTING POWERMODELS (Julia) + IPOPT")
print("=" * 65)
print(f"Load: {net.load['p_mw'].sum():.1f} MW")
print(f"Buses: {len(net.bus)}")
print(f"Generators: {len(net.gen)}")
print(f"Lines: {len(net.line)}")
print()

print("Running AC-OPF with PowerModels...")
try:
    pp.runpm(net, pm_model="ACPPowerModel", pm_solver="ipopt", 
             pm_log_level=0, delete_buffer_file=True)
    
    if net.converged:
        print("\n" + "=" * 40)
        print("✓✓✓ AC-OPF CONVERGED! ✓✓✓")
        print("=" * 40)
        print(f"Total generation: {net.res_gen['p_mw'].sum():.1f} MW")
        if len(net.res_ext_grid) > 0:
            print(f"External grid: {net.res_ext_grid['p_mw'].sum():.1f} MW")
        print(f"Total cost: {net.res_cost:,.2f} NT$/hr")
        print(f"Vmin: {net.res_bus['vm_pu'].min():.4f} pu")
        print(f"Vmax: {net.res_bus['vm_pu'].max():.4f} pu")
        
        # Energy Not Supplied
        ens_idx = net.sgen[net.sgen["name"].str.startswith("LoadShed_sgen_", na=False)].index
        if len(ens_idx) > 0:
            ens = net.res_sgen.loc[ens_idx, "p_mw"].clip(lower=0).sum()
            print(f"ENS (load shedding): {ens:.1f} MWh")
    else:
        print("\ AC-OPF NOT CONVERGED ")
        print("Coba dengan DC-OPF dulu...")
        try:
            pp.runpm(net, pm_model="DCPPowerModel", pm_solver="cbc", 
                     pm_log_level=0, delete_buffer_file=True)
            if net.converged:
                print(" DC-OPF converged (masalah di reactive power/voltage)")
            else:
                print(" DC-OPF also failed (masalah di active power balance)")
        except Exception as e2:
            print(f"DC-OPF error: {e2}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 65)