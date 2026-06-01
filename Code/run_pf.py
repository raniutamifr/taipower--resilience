"""
TAIPOWER NETWORK - SLACK BUS FIX & POWER FLOW
"""

import pandapower as pp
import numpy as np
import warnings
warnings.filterwarnings("ignore")

NET_PATH = r'C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json'
OUTPUT_PATH = NET_PATH.replace('.json', '_SOLVED.json')

print("=" * 65)
print("TAIPOWER NETWORK - SLACK BUS FIX")
print("=" * 65)

# Load network
net = pp.from_json(NET_PATH)

# ============================================================================
# STEP 1: Fix zero impedances
# ============================================================================
print("\n[1] Fixing Branch Parameters")

if len(net.line) > 0:
    if 'r_ohm_per_km' in net.line.columns:
        net.line.loc[net.line.r_ohm_per_km <= 0, 'r_ohm_per_km'] = 1e-6
    if 'x_ohm_per_km' in net.line.columns:
        net.line.loc[net.line.x_ohm_per_km <= 0, 'x_ohm_per_km'] = 1e-6
    if 'length_km' in net.line.columns:
        net.line.loc[net.line.length_km <= 0, 'length_km'] = 0.001

if len(net.trafo) > 0:
    if 'vk_percent' in net.trafo.columns:
        net.trafo.loc[net.trafo.vk_percent <= 0, 'vk_percent'] = 5.0
    if 'vkr_percent' in net.trafo.columns:
        net.trafo.loc[net.trafo.vkr_percent <= 0, 'vkr_percent'] = 0.5

print("    Branch parameters fixed")

# ============================================================================
# STEP 2: Identify and preserve slack bus
# ============================================================================
print("\n[2] Slack Bus Validation")

if len(net.ext_grid) > 0:
    slack_bus_original = net.ext_grid.bus.iloc[0]
    print(f"    Original slack bus: {slack_bus_original}")
else:
    print("    ERROR: No external grid defined!")
    exit()

# ============================================================================
# STEP 3: Find connected components and ensure slack is in main component
# ============================================================================
print("\n[3] Topology Analysis")

from pandapower.topology import connected_components

try:
    # Get connected components
    components, bus_to_component = connected_components(net, respect_switches=False)
    n_components = len(components)
    print(f"    Number of connected components: {n_components}")
    
    # Find which component contains the slack bus
    if slack_bus_original in bus_to_component:
        slack_component = bus_to_component[slack_bus_original]
        slack_component_size = len(components[slack_component])
        print(f"    Slack bus in component {slack_component} (size: {slack_component_size} buses)")
        
        # Identify buses NOT in slack's component
        buses_to_remove = [bus for bus, comp in bus_to_component.items() if comp != slack_component]
        print(f"    Buses not in slack component: {len(buses_to_remove)}")
    else:
        print("    WARNING: Slack bus not found in any component!")
        buses_to_remove = []
        
except Exception as e:
    print(f"    Topology analysis failed: {e}")
    buses_to_remove = []

# ============================================================================
# STEP 4: Remove only buses NOT connected to slack (keep slack bus)
# ============================================================================
print("\n[4] Removing Disconnected Components")

if len(buses_to_remove) > 0:
    # Remove isolated buses (not connected to slack)
    net.bus.drop(buses_to_remove, inplace=True, errors='ignore')
    
    # Remove associated equipment
    if len(net.gen) > 0:
        net.gen = net.gen[~net.gen.bus.isin(buses_to_remove)]
    if len(net.load) > 0:
        net.load = net.load[~net.load.bus.isin(buses_to_remove)]
    if len(net.line) > 0:
        net.line = net.line[~net.line.from_bus.isin(buses_to_remove)]
        net.line = net.line[~net.line.to_bus.isin(buses_to_remove)]
    if len(net.trafo) > 0:
        net.trafo = net.trafo[~net.trafo.hv_bus.isin(buses_to_remove)]
        net.trafo = net.trafo[~net.trafo.lv_bus.isin(buses_to_remove)]
    
    print(f"    Removed {len(buses_to_remove)} buses not connected to slack")
    
    # Re-index to avoid gaps
    net.bus.reset_index(drop=True, inplace=True)
    
print(f"    Final bus count: {len(net.bus)}")

# ============================================================================
# STEP 5: Verify slack bus still exists
# ============================================================================
print("\n[5] Slack Bus Verification")

if slack_bus_original not in net.bus.index.values:
    # Slack bus was removed - need to reassign
    print("    WARNING: Slack bus was removed! Reassigning...")
    
    # Find first bus with generator as new slack
    if len(net.gen) > 0:
        new_slack_bus = net.gen.bus.iloc[0]
        net.ext_grid.bus.iloc[0] = new_slack_bus
        print(f"    New slack bus: {new_slack_bus}")
    else:
        print("    ERROR: No generator buses available for slack!")
        exit()

# Ensure ext_grid has valid vm_pu
net.ext_grid.vm_pu = 1.02
print(f"    Slack bus: {net.ext_grid.bus.iloc[0]}, V=1.02 pu")

# ============================================================================
# STEP 6: Power balance (only on remaining network)
# ============================================================================
print("\n[6] Power Balance Correction")

total_load = net.load.p_mw.sum() if len(net.load) > 0 else 0
total_gen = net.gen.p_mw.sum() if len(net.gen) > 0 else 0

if total_gen > 0:
    scale_factor = total_load / total_gen
    net.gen.p_mw *= scale_factor
    print(f"    Load: {total_load:8.0f} MW")
    print(f"    Generation: {net.gen.p_mw.sum():8.0f} MW")
    print(f"    Scale factor: {scale_factor:.6f}")

# ============================================================================
# STEP 7: Voltage setup
# ============================================================================
print("\n[7] Voltage Initialization")

net.gen.vm_pu = 1.02
net.bus['vm_pu'] = 1.02
net.bus['va_degree'] = 0.0
print(f"    Flat start: 1.02 pu")

# ============================================================================
# STEP 8: Relax Q limits
# ============================================================================
print("\n[8] Constraint Relaxation")

if len(net.gen) > 0:
    net.gen['min_q_mvar'] = -net.gen['max_p_mw'] * 10.0
    net.gen['max_q_mvar'] = net.gen['max_p_mw'] * 10.0
    print(f"    Q limits: +/- 10.0 * Pmax")

# ============================================================================
# STEP 9: DC Power Flow
# ============================================================================
print("\n[9] DC Power Flow")

try:
    pp.rundcpp(net, check_connectivity=False)
    if net._ppc.get('success', False) and len(net.res_ext_grid) > 0:
        slack_p = net.res_ext_grid.p_mw.iloc[0]
        if not np.isnan(slack_p):
            print(f"    Status: CONVERGED")
            print(f"    Slack bus power: {slack_p:+8.0f} MW")
        else:
            print(f"    Status: CONVERGED but slack power is NaN")
            print("    Attempting to fix slack reference...")
            
            # Force slack reference
            net.ext_grid.p_mw = 0.0
            net.ext_grid.q_mvar = 0.0
            pp.rundcpp(net, check_connectivity=False)
    else:
        print(f"    Status: FAILED")
except Exception as e:
    print(f"    Status: ERROR - {str(e)[:60]}")

# ============================================================================
# STEP 10: AC Power Flow
# ============================================================================
print("\n[10] AC Power Flow")

converged = False

# Try with extremely relaxed settings
try:
    pp.runpp(net,
             algorithm='nr',
             init='flat',
             max_iteration=200,
             tolerance=1e-4,
             enforce_q_lims=False,
             check_connectivity=False)
    
    converged = net.converged
    print(f"    NR Status: {'CONVERGED' if converged else 'DIVERGED'}")
    
except Exception as e:
    print(f"    NR Error: {str(e)[:80]}")
    converged = False

if not converged:
    print("\n    Trying Gauss-Seidel with extreme settings...")
    try:
        pp.runpp(net,
                 algorithm='gs',
                 init='flat',
                 max_iteration=3000,
                 tolerance=1e-3,
                 enforce_q_lims=False,
                 check_connectivity=False)
        
        converged = net.converged
        print(f"    GS Status: {'CONVERGED' if converged else 'DIVERGED'}")
    except Exception as e:
        print(f"    GS Error: {str(e)[:80]}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"\nConverged: {converged}")

if converged:
    if len(net.res_ext_grid) > 0:
        slack_p = net.res_ext_grid.p_mw.iloc[0]
        slack_q = net.res_ext_grid.q_mvar.iloc[0]
        print(f"\nSlack Bus:")
        print(f"    Active Power:   {slack_p:+8.0f} MW")
        print(f"    Reactive Power: {slack_q:+8.0f} MVar")
    
    if len(net.res_bus) > 0:
        print(f"\nVoltage Profile:")
        print(f"    Minimum: {net.res_bus.vm_pu.min():.4f} pu")
        print(f"    Maximum: {net.res_bus.vm_pu.max():.4f} pu")
        print(f"    Mean:    {net.res_bus.vm_pu.mean():.4f} pu")
    
    if len(net.res_line) > 0:
        print(f"\nLine Loading:")
        print(f"    Maximum: {net.res_line.loading_percent.max():.1f}%")
    
    # Save
    pp.to_json(net, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    
else:
    print("\nPOWER FLOW FAILED")
    print("\nPossible root causes:")
    print("    1. Network has no path from slack to loads")
    print("    2. Branch parameters still problematic")
    print("    3. Voltage collapse condition")

print("=" * 65)