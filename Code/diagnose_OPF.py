"""
Taipower Network OPF Feasibility Diagnostic
============================================

Combines the data-sanity, load-scale, and native-OPF tests we need to
isolate the OPF non-convergence:

    Section 1 : Network inventory and data sanity
                (NaN cells, Pmin>Pmax, Qmin>Qmax, cost-data shape)
    Section 2 : Capacity vs demand check
    Section 3 : DC power flow sweep over load scales (topology test)
    Section 4 : AC power flow sweep over load scales (nonlinearity test)
    Section 5 : Binary search - maximum feasible load scale
    Section 6 : Bottleneck inspection at the max feasible scale
    Section 7 : Native pandapower DC-OPF test (no Julia)
    Section 8 : Native pandapower AC-OPF test (no Julia)
    Section 9 : Summary and recommendation

Native pandapower solvers are used everywhere - no PowerModels.jl - so a
failure here proves the issue is in the network data, not in the Julia
bridge.
"""

import pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
LOAD_SCALES = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

# Step 6 contract: LoadShed virtual generators use this exact prefix
LOADSHED_PREFIX = "LoadShed_"


def scale_load(net, scale):
    """Apply load scaling consistent with Step 9 contract."""
    net.load["p_mw"]   *= scale
    net.load["q_mvar"] *= scale
    mask = net.sgen["name"].str.startswith(LOADSHED_PREFIX)
    net.sgen.loc[mask, "max_p_mw"] *= scale


# =============================================================================
print("=" * 70)
print("SECTION 1 - Network inventory and data sanity")
print("=" * 70)
# =============================================================================
net = pp.from_json(str(NET_FILE))

print(f"Buses        : {len(net.bus)} total | {net.bus.in_service.sum()} in service")
print(f"Gens         : {len(net.gen)} total | {net.gen.in_service.sum()} in service")
print(f"Sgens        : {len(net.sgen)} total | "
      f"LoadShed_ = {net.sgen.name.str.startswith(LOADSHED_PREFIX).sum()}")
print(f"Loads        : {len(net.load)}")
print(f"Lines        : {len(net.line)} | in service = {net.line.in_service.sum()}")
print(f"Trafo 2W     : {len(net.trafo)}")
print(f"Trafo 3W     : {len(net.trafo3w)}")
print(f"Impedances   : {len(net.impedance)}")
print(f"Switches     : {len(net.switch)} | "
      f"closed bus-bus = {((net.switch.et == 'b') & net.switch.closed).sum()}")

g = net.gen
print(f"\nGenerator data:")
print(f"  Pmin range : {g.min_p_mw.min():.1f} to {g.min_p_mw.max():.1f}")
print(f"  Pmax range : {g.max_p_mw.min():.1f} to {g.max_p_mw.max():.1f}")
print(f"  Qmin range : {g.min_q_mvar.min():.1f} to {g.min_q_mvar.max():.1f}")
print(f"  Qmax range : {g.max_q_mvar.min():.1f} to {g.max_q_mvar.max():.1f}")
print(f"  Pmin > Pmax violations : {(g.min_p_mw > g.max_p_mw).sum()}")
print(f"  Pmax <= 0              : {(g.max_p_mw <= 0).sum()}")
print(f"  Qmin > Qmax violations : {(g.min_q_mvar > g.max_q_mvar).sum()}")
print(f"  NaN in Pmin / Pmax     : {g.min_p_mw.isna().sum()} / {g.max_p_mw.isna().sum()}")
print(f"  NaN in Qmin / Qmax     : {g.min_q_mvar.isna().sum()} / {g.max_q_mvar.isna().sum()}")

print(f"\nCost data (poly_cost):")
pc = net.poly_cost
print(f"  Rows           : {len(pc)}")
print(f"  NaN cells      : {pc.isna().sum().sum()}")
print(f"  cp1 range      : {pc.cp1_eur_per_mw.min():.1f} to {pc.cp1_eur_per_mw.max():.1f}")
print(f"  cp2 range      : {pc.cp2_eur_per_mw2.min():.4f} to {pc.cp2_eur_per_mw2.max():.4f}")
print(f"\n  Ext_grid cost rows:")
print(pc[pc.et == "ext_grid"][["et", "element", "cp0_eur",
                                "cp1_eur_per_mw", "cp2_eur_per_mw2"]].to_string(index=False))

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2 - Capacity vs demand")
print("=" * 70)
# =============================================================================
total_load_nom = float(net.load[net.load.in_service].p_mw.sum())
total_pmax     = float(net.gen[net.gen.in_service].max_p_mw.sum())
total_pmin     = float(net.gen[net.gen.in_service].min_p_mw.sum())
total_qmax     = float(net.gen[net.gen.in_service].max_q_mvar.sum())
print(f"Total load (nominal)        : {total_load_nom:8.0f} MW")
print(f"Total gen Pmin              : {total_pmin:8.0f} MW")
print(f"Total gen Pmax              : {total_pmax:8.0f} MW")
print(f"Total gen Qmax              : {total_qmax:8.0f} Mvar")
print(f"P margin (Pmax - load_nom)  : {total_pmax - total_load_nom:8.0f} MW")
if total_pmin > total_load_nom * 0.5:
    print(f"  WARNING: total Pmin ({total_pmin:.0f}) is large; gens cannot")
    print(f"  back off below {total_pmin:.0f} MW. Surplus must go to slack")
    print(f"  or be absorbed by negative loads.")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3 - DC power flow sweep (topology test)")
print("=" * 70)
# =============================================================================
for scale in LOAD_SCALES:
    net = pp.from_json(str(NET_FILE))
    scale_load(net, scale)
    total_load = float(net.load[net.load.in_service].p_mw.sum())
    try:
        pp.rundcpp(net, check_connectivity=False)
        if net.converged:
            slack_p = float(net.res_ext_grid.p_mw.values[0])
            print(f"  scale {scale:.2f} ({total_load:6.0f} MW) : "
                  f"DC PF OK | slack P = {slack_p:+7.0f} MW")
        else:
            print(f"  scale {scale:.2f} ({total_load:6.0f} MW) : "
                  f"DC PF NOT CONVERGED")
    except Exception as e:
        print(f"  scale {scale:.2f} : DC PF ERROR - {type(e).__name__}: {str(e)[:60]}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4 - AC power flow sweep (nonlinearity test)")
print("=" * 70)
# =============================================================================
for scale in LOAD_SCALES:
    net = pp.from_json(str(NET_FILE))
    scale_load(net, scale)
    total_load = float(net.load[net.load.in_service].p_mw.sum())
    try:
        pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
                 check_connectivity=False, enforce_q_lims=False)
        if net.converged:
            vmin = float(net.res_bus.vm_pu.min())
            vmax = float(net.res_bus.vm_pu.max())
            n_uv = int((net.res_bus.vm_pu < 0.95).sum())
            n_ov = int((net.res_bus.vm_pu > 1.05).sum())
            n_oload = int((net.res_line.loading_percent > 100).sum())
            print(f"  scale {scale:.2f} ({total_load:6.0f} MW) : AC PF OK | "
                  f"V=[{vmin:.3f},{vmax:.3f}] | UV={n_uv} OV={n_ov} "
                  f"OvLoad={n_oload}")
        else:
            print(f"  scale {scale:.2f} ({total_load:6.0f} MW) : "
                  f"AC PF NOT CONVERGED")
    except Exception as e:
        print(f"  scale {scale:.2f} : AC PF ERROR - {type(e).__name__}: {str(e)[:60]}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5 - Binary search for maximum feasible AC PF load scale")
print("=" * 70)
# =============================================================================
lo, hi = 0.30, 0.85
feasible = 0.30
for _ in range(8):
    mid = (lo + hi) / 2
    net = pp.from_json(str(NET_FILE))
    scale_load(net, mid)
    try:
        pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
                 check_connectivity=False, enforce_q_lims=False)
        ok = net.converged
    except Exception:
        ok = False
    if ok:
        feasible = mid
        lo = mid
        print(f"  scale {mid:.3f} : feasible")
    else:
        hi = mid
        print(f"  scale {mid:.3f} : not feasible")
print(f"\n>>> Maximum feasible AC PF load scale : {feasible:.3f}")

# =============================================================================
print("\n" + "=" * 70)
print(f"SECTION 6 - Bottleneck inspection at scale = {feasible:.3f}")
print("=" * 70)
# =============================================================================
net = pp.from_json(str(NET_FILE))
scale_load(net, feasible)
try:
    pp.runpp(net, algorithm="nr", init="dc", max_iteration=80,
             check_connectivity=False, enforce_q_lims=False)
    if net.converged:
        vm = net.res_bus.vm_pu.dropna()
        print(f"V range : {vm.min():.4f} - {vm.max():.4f}")
        ov  = vm[vm > 1.05].sort_values(ascending=False)
        uv  = vm[vm < 0.95].sort_values()
        print(f"Over-voltage  buses (>1.05) : {len(ov)}")
        print(f"Under-voltage buses (<0.95) : {len(uv)}")
        for bi in ov.head(5).index:
            print(f"  bus {bi:5d} : V = {vm[bi]:.4f} pu")
        for bi in uv.head(5).index:
            print(f"  bus {bi:5d} : V = {vm[bi]:.4f} pu")

        ll = net.res_line.loading_percent.dropna()
        over = ll[ll > 100].sort_values(ascending=False)
        print(f"\nOverloaded lines (>100%) : {len(over)}")
        for li in over.head(10).index:
            nm = net.line.at[li, "name"]
            print(f"  line {li:5d} {nm} : loading = {ll[li]:6.1f} %")
    else:
        print("AC PF did not converge at the reported feasible scale - rerun.")
except Exception as e:
    print(f"AC PF error: {type(e).__name__}: {e}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7 - Native pandapower DC-OPF (no PowerModels.jl)")
print("=" * 70)
# =============================================================================
net = pp.from_json(str(NET_FILE))
scale_load(net, feasible)
try:
    pp.rundcopp(net, check_connectivity=False)
    print(f"DC-OPF converged : {net.converged}")
    if net.converged:
        print(f"  Objective          : {float(net.res_cost):.1f}")
        print(f"  Slack P            : {float(net.res_ext_grid.p_mw.values[0]):+.1f} MW")
        mask = net.sgen.name.str.startswith(LOADSHED_PREFIX)
        shed = float(net.res_sgen.loc[mask, "p_mw"].sum())
        print(f"  Total load shed    : {shed:.2f} MW")
except Exception as e:
    print(f"DC-OPF ERROR : {type(e).__name__}: {e}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8 - Native pandapower AC-OPF (no PowerModels.jl)")
print("=" * 70)
# =============================================================================
net = pp.from_json(str(NET_FILE))
scale_load(net, feasible)
try:
    pp.runopp(net, init="results", check_connectivity=False,
              numba=False, verbose=False)
    print(f"AC-OPF converged : {net.converged}")
    if net.converged:
        print(f"  Objective          : {float(net.res_cost):.1f}")
        print(f"  Slack P            : {float(net.res_ext_grid.p_mw.values[0]):+.1f} MW")
        vm = net.res_bus.vm_pu.dropna()
        print(f"  V range            : {vm.min():.4f} - {vm.max():.4f}")
        mask = net.sgen.name.str.startswith(LOADSHED_PREFIX)
        shed = float(net.res_sgen.loc[mask, "p_mw"].sum())
        print(f"  Total load shed    : {shed:.2f} MW")
except Exception as e:
    print(f"AC-OPF ERROR : {type(e).__name__}: {str(e)[:200]}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9 - Summary and recommendation")
print("=" * 70)
# =============================================================================
print(f"  Max feasible AC PF load scale : {feasible:.3f}")
print()
print("  Interpretation guide:")
print("    * If AC PF works but native AC-OPF fails: cost/bounds issue")
print("    * If both AC PF and AC-OPF work: PowerModels.jl bridge issue")
print("    * If even AC PF fails at every scale: network data issue")
print("    * If feasible scale << 0.75: load too heavy for network -")
print("      revise LOAD_SCALE in Step 9 to match feasible_scale")
print("=" * 70)