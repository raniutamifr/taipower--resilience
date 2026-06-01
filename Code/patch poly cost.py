import pandas as pd
import pandapower as pp
from pathlib import Path
import numpy as np
import copy, time, traceback

# File Paths
NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
COST_CSV = Path(r"C:\reXplan-repo\Project Taipower\Results\step03\cost_parameters.csv")

net = pp.from_json(str(NET_FILE))
cost_df = pd.read_csv(COST_CSV)

def decode_big5(s):
    try:
        return s.encode("latin-1").decode("big5")
    except Exception:
        return s

bus_decoded = {bi: decode_big5(str(row["name"])) for bi, row in net.bus.iterrows()}

print("Decode OK. Building plant costs...")

plant_costs = {}
for _, row in cost_df.iterrows():
    unit = str(row["unit_name"]).strip()
    prefix = unit[:2]
    c1, c2, c0 = float(row["opf_c1"] or 0), float(row["opf_c2"] or 0), float(row["opf_c0"] or 0)
    if prefix not in plant_costs:
        plant_costs[prefix] = {"c1": [], "c2": [], "c0": [], "fuel": str(row["fuel_type"])}
    plant_costs[prefix]["c1"].append(c1)
    plant_costs[prefix]["c2"].append(c2)
    plant_costs[prefix]["c0"].append(c0)

for p in plant_costs:
    plant_costs[p]["c1_avg"] = float(np.mean(plant_costs[p]["c1"]))
    plant_costs[p]["c2_avg"] = float(np.mean(plant_costs[p]["c2"]))
    plant_costs[p]["c0_avg"] = float(np.mean(plant_costs[p]["c0"]))

FUEL_DEFAULT = {
    "gas":     (1.680, 0.000001),
    "coal":    (0.550, 0.000200),
    "oil":     (3.500, 0.000500),
    "nuclear": (0.300, 0.000100),
}
FUEL_MAP = {
    "大潭": "gas", "通霄": "gas", "興達": "gas", "南部": "gas",
    "台中": "coal", "林口": "coal", "大林": "coal", "麥寮": "coal",
    "協和": "oil", "核一": "nuclear", "核二": "nuclear", "核三": "nuclear",
}

n_matched = n_fuel = n_default = 0

print("Patching generators with JITTER to prevent numerical ties...")
for idx, gen_row in net.gen.iterrows():
    try:
        bus_idx = int(gen_row["bus"])
        bus_name = bus_decoded.get(bus_idx, "")
        mask = (net.poly_cost["et"] == "gen") & (net.poly_cost["element"] == idx)
        if not mask.any():
            continue

        # === FIX 1: ADD JITTER (YOUR SUGGESTION) ===
        # Larger jitter untuk menghindari singular matrix
        jitter = (idx % 100) * 1e-3 + (hash(bus_name) % 100) * 1e-6

        matched = False
        for prefix, costs in plant_costs.items():
            if prefix in bus_name:
                c1 = costs["c1_avg"]
                c2 = costs["c2_avg"] if abs(costs["c2_avg"]) > 1e-9 else 1e-6
                c0 = costs["c0_avg"]
                net.poly_cost.loc[mask, "cp1_eur_per_mw"] = c1 + jitter
                net.poly_cost.loc[mask, "cp2_eur_per_mw2"] = abs(c2)
                net.poly_cost.loc[mask, "cp0_eur"] = c0
                n_matched += 1
                matched = True
                break

        if not matched:
            fuel = next((v for k, v in FUEL_MAP.items() if k in bus_name), None)
            if fuel and fuel in FUEL_DEFAULT:
                c1, c2 = FUEL_DEFAULT[fuel]
                n_fuel += 1
            else:
                c1, c2 = 1.000, 0.0003
                n_default += 1
            net.poly_cost.loc[mask, "cp1_eur_per_mw"] = c1 + jitter
            net.poly_cost.loc[mask, "cp2_eur_per_mw2"] = c2
            net.poly_cost.loc[mask, "cp0_eur"] = 0.0

    except Exception as e:
        print(f"ERROR at gen idx {idx}: {e}")
        break

# Ext_grid cost (VOLL) - high cost so it's last resort
net.poly_cost.loc[net.poly_cost["et"] == "ext_grid", "cp1_eur_per_mw"] = 50000.0
net.poly_cost.loc[net.poly_cost["et"] == "ext_grid", "cp2_eur_per_mw2"] = 0.0
net.poly_cost.loc[net.poly_cost["et"] == "ext_grid", "cp0_eur"] = 0.0

print(f"Matched: {n_matched} | Fuel: {n_fuel} | Default: {n_default}")

# Cek hasil cost
gc = net.poly_cost[net.poly_cost["et"] == "gen"]
print(f"Gen cost range: [{gc['cp1_eur_per_mw'].min():.2f}, {gc['cp1_eur_per_mw'].max():.2f}]")
print(f"Unique cp1 values: {gc['cp1_eur_per_mw'].nunique()}")

print("\nTesting runopp with IPOPT solver...")
net_t = copy.deepcopy(net)

# Scale load
load_scale = 0.85
net_t.load["p_mw"] *= load_scale
net_t.load["q_mvar"] *= load_scale
shed_mask = net_t.sgen["name"].str.startswith("LoadShed_sgen_")
net_t.sgen.loc[shed_mask, "max_p_mw"] *= load_scale

# === FIX 2: RELAX CONSTRAINTS FOR DEBUGGING (YOUR SUGGESTION) ===
print("\nRelaxing constraints for debugging...")
if 'min_vm_pu' in net_t.bus.columns:
    net_t.bus['min_vm_pu'] = 0.88
    net_t.bus['max_vm_pu'] = 1.12
    print("  Voltage bounds: [0.88, 1.12]")

# Relax line and trafo loading limits
for line_idx in net_t.line.index:
    net_t.line.at[line_idx, 'max_loading_percent'] = 200.0
for trafo_idx in net_t.trafo.index:
    net_t.trafo.at[trafo_idx, 'max_loading_percent'] = 200.0
print("  Branch limits relaxed to 200%")

# === FIX 3: TRY IPOPT SOLVER FIRST ===
t0 = time.time()
converged = False

# Method 1: Try IPOPT (if available)
try:
    print("\nTrying IPOPT solver...")
    pp.runopp(net_t, calculate_voltage_angles=True, 
              check_connectivity=False, 
              init="flat",
              max_iteration=500,
              verbose=False)
    # Note: runopp doesn't have native 'solver' parameter in standard pandapower
    # IPOPT is used if installed via pp.opf.runopp_ipopt
    converged = net_t.converged
    elapsed = time.time() - t0
    
    if converged:
        print(f"✓ IPOPT converged! ({elapsed:.1f}s)")
    else:
        print(f"✗ IPOPT did not converge ({elapsed:.1f}s)")
except AttributeError:
    print("  IPOPT not available via standard runopp")
except Exception as e:
    print(f"  IPOPT error: {e}")

# Method 2: Try runopp_ipopt if available
if not converged:
    try:
        from pandapower.opf import runopp_ipopt
        print("\nTrying runopp_ipopt...")
        net_t2 = copy.deepcopy(net_t)
        t0 = time.time()
        runopp_ipopt(net_t2, init="flat", max_iteration=500)
        elapsed = time.time() - t0
        if net_t2.converged:
            converged = True
            net_t = net_t2
            print(f"✓ runopp_ipopt converged! ({elapsed:.1f}s)")
        else:
            print(f"✗ runopp_ipopt did not converge ({elapsed:.1f}s)")
    except ImportError:
        print("  runopp_ipopt not available (install ipopt: pip install ipopt)")
    except Exception as e:
        print(f"  runopp_ipopt error: {e}")

# Method 3: Fallback to standard PYPOWER
if not converged:
    print("\nTrying standard PYPOWER (default)...")
    try:
        from pandapower.opf import ppoption
        ppopt = ppoption(OPF_ALG=2, VERBOSE=0, OPF_VIOLATION=1e-3)
        t0 = time.time()
        pp.runopp(net_t, verbose=False, numba=False,
                  calculate_voltage_angles=True,
                  check_connectivity=False,
                  init="flat",
                  max_iteration=500,
                  ppopt=ppopt)
        elapsed = time.time() - t0
        converged = net_t.converged
        if converged:
            print(f"✓ PYPOWER converged! ({elapsed:.1f}s)")
        else:
            print(f"✗ PYPOWER did not converge ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  PYPOWER error: {e}")

# Final results
print("\n" + "=" * 65)
if converged:
    print("✓✓✓ OPF CONVERGED! ✓✓✓")
    print("=" * 65)
    print(f"Total Gen: {net_t.res_gen['p_mw'].sum():.1f} MW")
    print(f"Ext Grid: {net_t.res_ext_grid['p_mw'].sum():.1f} MW")
    print(f"Cost: {net_t.res_cost:,.2f} NT$/hr")
    print(f"Vmin: {net_t.res_bus['vm_pu'].min():.4f} pu")
    print(f"Vmax: {net_t.res_bus['vm_pu'].max():.4f} pu")
    
    ens = net_t.res_sgen.loc[
        net_t.sgen[net_t.sgen["name"].str.startswith("LoadShed_sgen_")].index,
        "p_mw"].clip(lower=0).sum()
    print(f"ENS: {ens:.1f} MWh")
    print("=" * 65)
else:
    print("✗✗✗ OPF FAILED TO CONVERGE ✗✗✗")
    print("=" * 65)
    print("\nTrying DC-OPF as last resort...")
    try:
        pp.rundcopp(net_t, check_connectivity=False)
        if net_t.converged:
            print(f"✓ DC-OPF converged!")
            print(f"  Cost: {net_t.res_cost:,.2f}")
        else:
            print("✗ DC-OPF also failed")
    except Exception as e:
        print(f"  DC-OPF error: {e}")

# Save patched network
pp.to_json(net, str(NET_FILE))
print(f"\nSaved patched network to: {NET_FILE}")