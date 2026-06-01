import pandapower as pp
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================
EXCEL_PATH = r"C:\reXplan-repo\file\input\taipower\network.xlsx"
LOAD_SCALE = 1.0

print("=" * 60)
print("POWER FLOW TEST - TAIPOWER NETWORK")
print("=" * 60)

# ============================================================
# 1. CHECK FILE
# ============================================================
if not os.path.exists(EXCEL_PATH):
    print(f"\nERROR: File not found!")
    exit(1)
else:
    print(f"\nFile found: {EXCEL_PATH}")

# ============================================================
# 2. READ ALL SHEETS
# ============================================================
print("\nLoading data from Excel...")
xl = pd.ExcelFile(EXCEL_PATH)
print(f"   Sheets found: {xl.sheet_names}")

# Initialize empty network
net = pp.create_empty_network(sn_mva=100, f_hz=60)

# Create mapping dictionaries
node_id_to_bus_idx = {}  # node ID -> bus index
node_name_to_bus_idx = {}  # node name -> bus index

# ============================================================
# 3. READ NODES (as BUS)
# ============================================================
if 'nodes' in xl.sheet_names:
    nodes_df = pd.read_excel(EXCEL_PATH, sheet_name='nodes')
    print(f"   Nodes (Buses): {len(nodes_df)} rows")
    print(f"      Columns: {list(nodes_df.columns)}")
    
    for idx, row in nodes_df.iterrows():
        node_id = row.get('id', idx)
        name = row.get('name', f"Bus_{node_id}")
        vn_kv = row.get('vn_kv', 161.0)
        bus_type = 'b'
        min_vm_pu = row.get('min_vm_pu', 0.95)
        max_vm_pu = row.get('max_vm_pu', 1.05)
        
        # Store mappings
        node_id_to_bus_idx[node_id] = idx
        node_name_to_bus_idx[str(name)] = idx
        
        pp.create_bus(net, name=str(name), vn_kv=float(vn_kv), type=bus_type,
                      min_vm_pu=float(min_vm_pu), max_vm_pu=float(max_vm_pu), index=idx)
    
    print(f"      Created {len(net.bus)} buses")

# ============================================================
# 4. READ LOADS - Using node ID
# ============================================================
if 'loads' in xl.sheet_names:
    loads_df = pd.read_excel(EXCEL_PATH, sheet_name='loads')
    print(f"   Loads: {len(loads_df)} rows")
    print(f"      Columns: {list(loads_df.columns)}")
    
    load_count = 0
    for idx, row in loads_df.iterrows():
        # Try to get node_id (this is the key!)
        node_id = row.get('node', None)
        bus_id = None
        
        # Look up by node ID
        if node_id in node_id_to_bus_idx:
            bus_id = node_id_to_bus_idx[node_id]
        else:
            # Try as string
            node_str = str(node_id)
            if node_str in node_name_to_bus_idx:
                bus_id = node_name_to_bus_idx[node_str]
        
        if bus_id is None or bus_id >= len(net.bus):
            continue
        
        p_mw = row.get('p_mw', 0.0)
        q_mvar = row.get('q_mvar', 0.0)
        name = row.get('name', f"Load_{node_id}")
        
        # Skip zero loads
        if p_mw == 0 and q_mvar == 0:
            continue
            
        pp.create_load(net, bus=bus_id, p_mw=float(p_mw), q_mvar=float(q_mvar), name=str(name))
        load_count += 1
    
    print(f"      Created {load_count} loads")

# ============================================================
# 5. READ GENERATORS - Using node ID
# ============================================================
if 'generators' in xl.sheet_names:
    gen_df = pd.read_excel(EXCEL_PATH, sheet_name='generators')
    print(f"   Generators: {len(gen_df)} rows")
    print(f"      Columns: {list(gen_df.columns)}")
    
    gen_count = 0
    for idx, row in gen_df.iterrows():
        # Try to get node_id
        node_id = row.get('node', None)
        bus_id = None
        
        # Look up by node ID
        if node_id in node_id_to_bus_idx:
            bus_id = node_id_to_bus_idx[node_id]
        else:
            # Try as string
            node_str = str(node_id)
            if node_str in node_name_to_bus_idx:
                bus_id = node_name_to_bus_idx[node_str]
        
        if bus_id is None or bus_id >= len(net.bus):
            continue
        
        p_mw = row.get('p_mw', 0.0)
        vm_pu = row.get('vm_pu', 1.0)
        max_p_mw = row.get('max_p_mw', p_mw * 1.2 if p_mw > 0 else 100.0)
        min_p_mw = row.get('min_p_mw', 0.0)
        name = row.get('name', f"Gen_{node_id}")
        
        # Skip zero generators
        if p_mw == 0:
            continue
            
        # Check if this is slack bus (first non-zero generator)
        is_slack = (gen_count == 0 and p_mw > 0)
        
        pp.create_gen(net, bus=bus_id, p_mw=float(p_mw), vm_pu=float(vm_pu),
                      max_p_mw=float(max_p_mw), min_p_mw=float(min_p_mw),
                      slack=is_slack, name=str(name))
        gen_count += 1
    
    print(f"      Created {gen_count} generators")

# ============================================================
# 6. READ LINES
# ============================================================
if 'lines' in xl.sheet_names:
    lines_df = pd.read_excel(EXCEL_PATH, sheet_name='lines')
    print(f"   Lines: {len(lines_df)} rows")
    
    # Read ln_type for line parameters
    ln_type_df = None
    if 'ln_type' in xl.sheet_names:
        ln_type_df = pd.read_excel(EXCEL_PATH, sheet_name='ln_type')
        print(f"      Line types: {len(ln_type_df)} types")
    
    line_count = 0
    for idx, row in lines_df.iterrows():
        from_bus = row.get('from_bus', None)
        to_bus = row.get('to_bus', None)
        
        if from_bus is None or to_bus is None or pd.isna(from_bus) or pd.isna(to_bus):
            continue
        
        try:
            from_bus = int(float(from_bus))
            to_bus = int(float(to_bus))
        except (ValueError, TypeError):
            continue
        
        if from_bus >= len(net.bus) or to_bus >= len(net.bus):
            continue
        
        length_km = row.get('length_km', 1.0)
        if pd.isna(length_km) or length_km is None:
            length_km = 1.0
        
        name = row.get('name', f"Line_{idx}")
        
        # Get line parameters from type if available
        r_ohm_per_km = 0.1
        x_ohm_per_km = 0.3
        c_nf_per_km = 10.0
        
        line_type = row.get('type', None)
        if line_type and ln_type_df is not None:
            type_row = ln_type_df[ln_type_df['name'] == line_type]
            if len(type_row) > 0:
                r_ohm_per_km = type_row.iloc[0].get('r_ohm_per_km', r_ohm_per_km)
                x_ohm_per_km = type_row.iloc[0].get('x_ohm_per_km', x_ohm_per_km)
                c_nf_per_km = type_row.iloc[0].get('c_nf_per_km', c_nf_per_km)
        
        try:
            pp.create_line_from_parameters(
                net, 
                from_bus=from_bus, 
                to_bus=to_bus,
                length_km=float(length_km), 
                name=str(name),
                r_ohm_per_km=float(r_ohm_per_km), 
                x_ohm_per_km=float(x_ohm_per_km),
                c_nf_per_km=float(c_nf_per_km),
                max_i_ka=1.0
            )
            line_count += 1
        except Exception as e:
            continue
    
    print(f"      Created {line_count} lines")

# ============================================================
# 7. READ TRANSFORMERS
# ============================================================
if 'transformers' in xl.sheet_names:
    trafo_df = pd.read_excel(EXCEL_PATH, sheet_name='transformers')
    print(f"   Transformers: {len(trafo_df)} rows")
    
    trafo_count = 0
    for idx, row in trafo_df.iterrows():
        hv_bus = row.get('hv_bus', None)
        lv_bus = row.get('lv_bus', None)
        
        if hv_bus is None or lv_bus is None or pd.isna(hv_bus) or pd.isna(lv_bus):
            continue
        
        try:
            hv_bus = int(float(hv_bus))
            lv_bus = int(float(lv_bus))
        except (ValueError, TypeError):
            continue
        
        if hv_bus >= len(net.bus) or lv_bus >= len(net.bus):
            continue
        
        # Get transformer parameters
        sn_mva = row.get('sn_mva', 100.0)
        vn_hv_kv = row.get('vn_hv_kv', 161.0)
        vn_lv_kv = row.get('vn_lv_kv', 69.0)
        vk_percent = row.get('vk_percent', 8.0)
        vkr_percent = row.get('vkr_percent', 0.2)
        name = row.get('name', f"Trafo_{idx}")
        
        try:
            pp.create_transformer(net, hv_bus=hv_bus, lv_bus=lv_bus,
                                  sn_mva=float(sn_mva), vn_hv_kv=float(vn_hv_kv), vn_lv_kv=float(vn_lv_kv),
                                  vk_percent=float(vk_percent), vkr_percent=float(vkr_percent),
                                  name=str(name))
            trafo_count += 1
        except Exception as e:
            continue
    
    print(f"      Created {trafo_count} transformers")

# ============================================================
# 8. DISPLAY STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("NETWORK STATISTICS")
print("=" * 60)
print(f"   Buses: {len(net.bus)}")
print(f"   Loads: {len(net.load)}")
print(f"   Generators: {len(net.gen)}")
print(f"   Lines: {len(net.line)}")
print(f"   Transformers: {len(net.trafo)}")

# Calculate total load
if len(net.load) > 0:
    total_load = net.load['p_mw'].sum()
    print(f"\n   Total Load: {total_load:.2f} MW")

# ============================================================
# 9. ADD MISSING SLACK IF NEEDED
# ============================================================
slack_exists = False
for idx in net.gen.index:
    if net.gen.at[idx, 'slack']:
        slack_exists = True
        break

if not slack_exists and len(net.gen) > 0:
    print("\nWARNING: No slack generator found. Setting first generator as slack...")
    net.gen.at[net.gen.index[0], 'slack'] = True

# ============================================================
# 10. RUN POWER FLOW
# ============================================================
print("\n" + "=" * 60)
print("RUNNING POWER FLOW")
print("=" * 60)

try:
    pp.runpp(net, verbose=False)
    print("\nPOWER FLOW CONVERGED!")
    
    # Display results
    print("\nResults:")
    if len(net.res_gen) > 0:
        total_gen = net.res_gen['p_mw'].sum()
        print(f"   Total Generation: {total_gen:.2f} MW")
    if len(net.load) > 0:
        total_load = net.load['p_mw'].sum()
        print(f"   Total Load: {total_load:.2f} MW")
    
    # Display buses with voltage
    print("\n   Buses with valid voltage:")
    valid_buses = 0
    for i in range(min(20, len(net.res_bus))):
        vm = net.res_bus.loc[i, 'vm_pu']
        if not pd.isna(vm):
            valid_buses += 1
            bus_name = net.bus.loc[i, 'name']
            print(f"      Bus {i} ({bus_name}): V={vm:.4f} pu")
    
    if valid_buses == 0:
        print("      No buses have valid voltage!")
        print("      Check if lines are properly connected.")
    
    # Display generators with output
    print("\n   Generators with output:")
    gen_output = 0
    for i in range(min(20, len(net.res_gen))):
        p = net.res_gen.iloc[i]['p_mw']
        if p > 0:
            gen_output += 1
            gen_name = net.gen.loc[net.res_gen.index[i], 'name']
            print(f"      {gen_name}: {p:.2f} MW")
    
    if gen_output == 0:
        print("      No generators are producing power!")
    
except Exception as e:
    print(f"\nPOWER FLOW FAILED: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)