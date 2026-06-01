"""
Step 08a — Pre-OPF Forensic Diagnostic
=======================================
Loads the Step 06 network JSON and runs a battery of structural checks that
explain *why* AC-OPF fails, before the solver is ever called.

Run this FIRST. Read the report. THEN run Step 09.

Diagnostics performed
---------------------
  1.  Bus inventory by voltage tier   (where are 345/161/69/22-25 kV buses?)
  2.  Generation inventory by voltage (THE classic Taipower issue: 0 gens @ 345 kV)
  3.  Slack bus sanity                (is ext_grid at backbone or distribution?)
  4.  Power balance                   (total gen capacity vs total load)
  5.  Connectivity / islanding        (count of connected components, slack islands)
  6.  Reactive support inventory      (Q capability vs load Q demand)
  7.  Branch impedance histogram      (x_pu near zero → numerical conditioning)
  8.  Rate_A coverage                 (how many lines have meaningful thermal limits)
  9.  DC power flow                   (does the network even balance linearly?)
 10.  Pre-OPF overload screen         (which lines are >150% loaded in DC-PF?)
 11.  Cost coverage                   (how many gens have real cost vs default?)
 12.  Hazard factor coverage          (how many components mapped to typhoon data?)

Outputs
-------
  Results/step08_diag/diagnostic_report.txt   — human-readable report
  Results/step08_diag/diagnostic_findings.json — machine-readable findings
  Results/step08_diag/overloaded_branches.csv  — lines that need attention
  Results/step08_diag/voltage_tier_summary.csv — buses per kV tier
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as pt
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
STEP06_DIR  = RESULT_BASE / "step06"
OUT_DIR     = RESULT_BASE / "step08_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NET_FILE     = STEP06_DIR / "taipower_network.json"
REPORT_TXT   = OUT_DIR / "diagnostic_report.txt"
FINDINGS_JSON= OUT_DIR / "diagnostic_findings.json"
OVERLOAD_CSV = OUT_DIR / "overloaded_branches.csv"
KV_SUMMARY   = OUT_DIR / "voltage_tier_summary.csv"

# Taipower voltage tier classification (engineering judgement, not arbitrary)
TIER_LIMITS = {
    "EHV (345 kV)":      (300.0, 400.0),
    "HV  (161 kV)":      (140.0, 200.0),
    "Sub (69 kV)":       (60.0,  80.0),
    "MV  (22-33 kV)":    (15.0,  40.0),
    "LV  (<15 kV)":      (0.0,   15.0),
}

# Text format helpers
def _line(c="="): return c * 78
def _box(title): return f"\n{_line()}\n  {title}\n{_line()}"


# ─────────────────────────────────────────────────────────────────────────────
# Section 1-2: Buses and generators by voltage tier
# ─────────────────────────────────────────────────────────────────────────────
def classify_tier(kv: float) -> str:
    for tier, (lo, hi) in TIER_LIMITS.items():
        if lo <= kv < hi:
            return tier
    return f"Other ({kv:.0f} kV)"


def diagnose_voltage_tiers(net: pp.pandapowerNet) -> dict:
    """Buses, gens, loads, lines per voltage tier."""
    out: dict = {"by_tier": {}}

    bus_kv = net.bus["vn_kv"].copy()
    bus_active = net.bus["in_service"].astype(bool)

    # Map each gen/load/line to its bus's kV
    gen_bus_kv  = net.gen["bus"].map(bus_kv) if not net.gen.empty else pd.Series(dtype=float)
    load_bus_kv = net.load["bus"].map(bus_kv) if not net.load.empty else pd.Series(dtype=float)
    line_from_kv= net.line["from_bus"].map(bus_kv) if not net.line.empty else pd.Series(dtype=float)

    tier_rows = []
    for tier in list(TIER_LIMITS.keys()) + ["Other"]:
        lo, hi = TIER_LIMITS.get(tier, (None, None))
        if lo is None:
            # "Other" catches buses outside known tiers
            mask_bus  = ~bus_kv.apply(lambda v: any(l <= v < h for l,h in TIER_LIMITS.values()))
        else:
            mask_bus  = (bus_kv >= lo) & (bus_kv < hi)
        mask_active = mask_bus & bus_active

        n_bus = int(mask_active.sum())

        if not gen_bus_kv.empty:
            if lo is None:
                m = ~gen_bus_kv.apply(lambda v: any(l <= v < h for l,h in TIER_LIMITS.values()))
            else:
                m = (gen_bus_kv >= lo) & (gen_bus_kv < hi)
            gens_in_tier = net.gen.loc[m & net.gen["in_service"].astype(bool)]
            n_gen = len(gens_in_tier)
            pmax_tier = float(gens_in_tier["max_p_mw"].sum()) if n_gen else 0.0
        else:
            n_gen, pmax_tier = 0, 0.0

        if not load_bus_kv.empty:
            if lo is None:
                m = ~load_bus_kv.apply(lambda v: any(l <= v < h for l,h in TIER_LIMITS.values()))
            else:
                m = (load_bus_kv >= lo) & (load_bus_kv < hi)
            n_load = int((m & net.load["in_service"].astype(bool)).sum())
            p_load = float(net.load.loc[m & net.load["in_service"].astype(bool), "p_mw"].sum())
        else:
            n_load, p_load = 0, 0.0

        if not line_from_kv.empty:
            if lo is None:
                m = ~line_from_kv.apply(lambda v: any(l <= v < h for l,h in TIER_LIMITS.values()))
            else:
                m = (line_from_kv >= lo) & (line_from_kv < hi)
            n_line = int((m & net.line["in_service"].astype(bool)).sum())
        else:
            n_line = 0

        tier_rows.append({
            "tier":          tier,
            "n_bus_active":  n_bus,
            "n_gen":         n_gen,
            "gen_pmax_mw":   round(pmax_tier, 1),
            "n_load":        n_load,
            "load_p_mw":     round(p_load, 1),
            "n_line":        n_line,
        })
        out["by_tier"][tier] = tier_rows[-1]

    df = pd.DataFrame(tier_rows)
    df.to_csv(KV_SUMMARY, index=False, encoding="utf-8-sig")

    # The classic red flag: zero gens at EHV
    ehv = out["by_tier"].get("EHV (345 kV)", {})
    out["red_flag_zero_ehv_gen"] = (ehv.get("n_bus_active", 0) > 10
                                     and ehv.get("n_gen", 0) == 0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Slack bus sanity
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_slack(net: pp.pandapowerNet) -> dict:
    if net.ext_grid.empty:
        return {"ok": False, "reason": "No ext_grid defined"}

    slack_idx  = int(net.ext_grid["bus"].iloc[0])
    slack_kv   = float(net.bus.at[slack_idx, "vn_kv"])
    slack_name = str(net.bus.at[slack_idx, "name"])
    slack_active = bool(net.bus.at[slack_idx, "in_service"])

    # Count gens connected at the same voltage level as slack
    gens_same_kv = 0
    if not net.gen.empty:
        gens_same_kv = int((net.gen["bus"].map(net.bus["vn_kv"])
                            .between(slack_kv * 0.95, slack_kv * 1.05)).sum())

    out = {
        "slack_bus_idx": slack_idx,
        "slack_bus_name": slack_name,
        "slack_kv":     slack_kv,
        "slack_active": slack_active,
        "gens_at_same_kv": gens_same_kv,
        "ok": True,
    }

    # Red flag: slack at <100 kV with many loads downstream
    if slack_kv < 100.0:
        out["red_flag_slack_at_distribution"] = True
        out["recommendation"] = (
            f"Slack at {slack_kv:.1f} kV is on distribution/MV. "
            f"Relocate to a 345 kV bus with attached generation."
        )
    else:
        out["red_flag_slack_at_distribution"] = False

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Power balance
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_power_balance(net: pp.pandapowerNet) -> dict:
    pmax_total = float(net.gen.loc[net.gen["in_service"], "max_p_mw"].sum()) \
                 if not net.gen.empty else 0.0
    pmin_total = float(net.gen.loc[net.gen["in_service"], "min_p_mw"].sum()) \
                 if not net.gen.empty else 0.0
    pload_total = float(net.load.loc[net.load["in_service"], "p_mw"].sum())
    qload_total = float(net.load.loc[net.load["in_service"], "q_mvar"].sum())

    qmax_total = float(net.gen.loc[net.gen["in_service"], "max_q_mvar"].sum()) \
                 if not net.gen.empty else 0.0
    qmin_total = float(net.gen.loc[net.gen["in_service"], "min_q_mvar"].sum()) \
                 if not net.gen.empty else 0.0

    headroom_p = pmax_total - pload_total
    out = {
        "pmax_gen_mw":   round(pmax_total, 1),
        "pmin_gen_mw":   round(pmin_total, 1),
        "pload_mw":      round(pload_total, 1),
        "qload_mvar":    round(qload_total, 1),
        "qmax_gen_mvar": round(qmax_total, 1),
        "qmin_gen_mvar": round(qmin_total, 1),
        "p_headroom_mw": round(headroom_p, 1),
        "p_reserve_pct": round(100 * headroom_p / max(pload_total, 1), 1),
    }
    out["red_flag_low_reserve"] = (out["p_reserve_pct"] < 10.0)
    out["red_flag_q_deficit"]   = (qmax_total < qload_total * 0.4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Connectivity / islanding
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_connectivity(net: pp.pandapowerNet) -> dict:
    try:
        mg = pt.create_nxgraph(net, respect_switches=False)
        components = sorted(nx.connected_components(mg), key=len, reverse=True)
        sizes = [len(c) for c in components]
        slack_bus = int(net.ext_grid["bus"].iloc[0]) if not net.ext_grid.empty else -1
        slack_island = -1
        for i, comp in enumerate(components):
            if slack_bus in comp:
                slack_island = i
                break
        out = {
            "n_components":    len(components),
            "largest_size":    sizes[0] if sizes else 0,
            "second_size":     sizes[1] if len(sizes) > 1 else 0,
            "slack_in_island": slack_island,
            "slack_island_size": sizes[slack_island] if slack_island >= 0 else 0,
            "isolated_buses_total": sum(sizes[1:]) if len(sizes) > 1 else 0,
        }
        # Isolated buses that still have loads → ENS-unsupportable
        if slack_island >= 0:
            main = components[slack_island]
            isolated_load_mw = 0.0
            for _, row in net.load.iterrows():
                if int(row["bus"]) not in main and row["in_service"]:
                    isolated_load_mw += row["p_mw"]
            out["isolated_load_mw"] = round(isolated_load_mw, 1)
            out["red_flag_orphan_load"] = isolated_load_mw > 10.0
        return out
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Branch impedance histogram
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_branch_impedance(net: pp.pandapowerNet) -> dict:
    out = {}
    if not net.line.empty:
        x_pu = (net.line["x_ohm_per_km"] * net.line["length_km"]) / \
               (net.line["from_bus"].map(net.bus["vn_kv"]) ** 2 / 100.0)
        out["line_x_pu_min"]    = float(x_pu.min())
        out["line_x_pu_median"] = float(x_pu.median())
        out["line_x_pu_max"]    = float(x_pu.max())
        out["lines_x_below_1e_5"] = int((x_pu.abs() < 1e-5).sum())
        out["red_flag_near_zero_x"] = out["lines_x_below_1e_5"] > 0
    if not net.trafo.empty:
        vk = net.trafo["vk_percent"]
        out["trafo_vk_pct_min"]    = float(vk.min())
        out["trafo_vk_pct_median"] = float(vk.median())
        out["trafo_vk_pct_max"]    = float(vk.max())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 9-10: DC-PF and pre-OPF overload screen
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_dcpf_overloads(net: pp.pandapowerNet) -> dict:
    out = {}
    try:
        pp.rundcpp(net, check_connectivity=False, verbose=False)
        out["dc_pf_converged"] = True
        out["dc_gen_mw"]    = float(net.res_gen["p_mw"].sum()) if not net.res_gen.empty else 0.0
        out["dc_ext_mw"]    = float(net.res_ext_grid["p_mw"].sum()) if not net.res_ext_grid.empty else 0.0
        out["dc_load_mw"]   = float(net.res_load["p_mw"].sum()) if not net.res_load.empty else 0.0

        # Overloaded branches
        bad_lines = pd.DataFrame()
        if not net.res_line.empty:
            ll = net.res_line["loading_percent"].abs()
            mask = ll > 150.0
            if mask.any():
                bad_lines = net.line.loc[mask].copy()
                bad_lines["loading_pct"] = ll[mask].round(1)
                bad_lines["from_kv"] = bad_lines["from_bus"].map(net.bus["vn_kv"])
                bad_lines["to_kv"]   = bad_lines["to_bus"].map(net.bus["vn_kv"])

        if not net.res_trafo.empty:
            ll = net.res_trafo["loading_percent"].abs()
            mask = ll > 150.0
            if mask.any():
                bad_trf = net.trafo.loc[mask].copy()
                bad_trf["loading_pct"] = ll[mask].round(1)
                bad_trf["from_kv"] = bad_trf["hv_bus"].map(net.bus["vn_kv"])
                bad_trf["to_kv"]   = bad_trf["lv_bus"].map(net.bus["vn_kv"])
                bad_trf["element_type"] = "trafo"
                bad_lines = pd.concat([
                    bad_lines.assign(element_type="line"),
                    bad_trf], ignore_index=True, sort=False) if not bad_lines.empty else bad_trf

        if not bad_lines.empty:
            bad_lines.sort_values("loading_pct", ascending=False).to_csv(
                OVERLOAD_CSV, index=False, encoding="utf-8-sig")
            out["overloaded_count"]   = int(len(bad_lines))
            out["max_loading_pct"]    = float(bad_lines["loading_pct"].max())
        else:
            out["overloaded_count"] = 0
            out["max_loading_pct"]  = 0.0

        out["red_flag_overloads"] = out["overloaded_count"] > 0
    except Exception as e:
        out["dc_pf_converged"] = False
        out["dc_pf_error"] = str(e)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: Cost coverage
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_cost_coverage(net: pp.pandapowerNet) -> dict:
    """How many generators have a real cost vs the default ~1000 fallback?"""
    if net.poly_cost.empty or net.gen.empty:
        return {"n_gen_cost": 0, "n_default": 0}
    gen_costs = net.poly_cost[net.poly_cost["et"] == "gen"]
    # Default fallback was c1 = 1000 in your Step 06
    n_default = int((gen_costs["cp1_eur_per_mw"].between(999, 1001) &
                     gen_costs["cp2_eur_per_mw2"].between(-0.001, 0.001)).sum())
    return {
        "n_gen_cost_records": int(len(gen_costs)),
        "n_default_fallback": n_default,
        "default_pct": round(100 * n_default / max(len(gen_costs), 1), 1),
        "red_flag_no_merit_order": n_default / max(len(gen_costs), 1) > 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 12: Hazard factor coverage
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_hazard_coverage(net: pp.pandapowerNet) -> dict:
    hf = getattr(net, "hazard_factors", None)
    if hf is None or (isinstance(hf, list) and len(hf) == 0):
        return {"hazard_factors_attached": False}
    df = pd.DataFrame(hf)
    return {
        "hazard_factors_attached": True,
        "n_records": len(df),
        "n_lines":   int((df["comp_type"] == "line").sum()) if "comp_type" in df else 0,
        "n_trafos":  int((df["comp_type"] == "transformer").sum()) if "comp_type" in df else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_report(findings: dict) -> str:
    L = []
    L.append(_line())
    L.append("  TAIPOWER NETWORK — PRE-OPF FORENSIC DIAGNOSTIC")
    L.append(_line())
    L.append(f"  Source: {NET_FILE}")
    L.append("")

    # Voltage tiers
    L.append(_box("1. BUS / GEN / LOAD INVENTORY BY VOLTAGE TIER"))
    L.append(f"  {'Tier':<20s} {'Buses':>8s} {'Gens':>6s} "
             f"{'Pmax (MW)':>12s} {'Loads':>7s} {'P_load (MW)':>14s} {'Lines':>7s}")
    L.append(f"  {'-'*78}")
    for tier, row in findings["voltage_tiers"]["by_tier"].items():
        L.append(f"  {tier:<20s} {row['n_bus_active']:>8d} {row['n_gen']:>6d} "
                 f"{row['gen_pmax_mw']:>12.1f} {row['n_load']:>7d} "
                 f"{row['load_p_mw']:>14.1f} {row['n_line']:>7d}")
    if findings["voltage_tiers"].get("red_flag_zero_ehv_gen"):
        L.append("")
        L.append("  >>> RED FLAG: zero generators at 345 kV with many EHV buses.")
        L.append("       Backbone has no local voltage support. AC-OPF will struggle.")
        L.append("       MITIGATION: Step 09 must add virtual gens at large EHV buses,")
        L.append("       or PSS/E parser must extract generator step-up transformers.")

    # Slack
    L.append(_box("2. SLACK BUS PLACEMENT"))
    s = findings["slack"]
    L.append(f"  Ext_grid bus  : {s['slack_bus_idx']}  ({s['slack_bus_name']})")
    L.append(f"  Voltage level : {s['slack_kv']:.1f} kV")
    L.append(f"  Active        : {s['slack_active']}")
    L.append(f"  Gens at same kV: {s['gens_at_same_kv']}")
    if s.get("red_flag_slack_at_distribution"):
        L.append("")
        L.append(f"  >>> RED FLAG: {s['recommendation']}")

    # Power balance
    L.append(_box("3. POWER BALANCE"))
    pb = findings["power_balance"]
    L.append(f"  Total P generation capacity : {pb['pmax_gen_mw']:>12.1f} MW")
    L.append(f"  Total P load (in-service)   : {pb['pload_mw']:>12.1f} MW")
    L.append(f"  Active reserve              : {pb['p_headroom_mw']:>12.1f} MW  ({pb['p_reserve_pct']:.1f}%)")
    L.append(f"  Total Q capability (max)    : {pb['qmax_gen_mvar']:>12.1f} Mvar")
    L.append(f"  Total Q load                : {pb['qload_mvar']:>12.1f} Mvar")
    if pb.get("red_flag_low_reserve"):
        L.append("  >>> RED FLAG: P reserve below 10% — system is generation-deficient.")
    if pb.get("red_flag_q_deficit"):
        L.append("  >>> RED FLAG: Q capability looks low vs Q load — check inverter limits.")

    # Connectivity
    L.append(_box("4. CONNECTIVITY"))
    c = findings["connectivity"]
    if "error" in c:
        L.append(f"  ERROR: {c['error']}")
    else:
        L.append(f"  Connected components : {c['n_components']}")
        L.append(f"  Largest island       : {c['largest_size']} buses")
        L.append(f"  Slack in island #    : {c['slack_in_island']} "
                 f"(size {c['slack_island_size']})")
        if c.get("red_flag_orphan_load"):
            L.append(f"  >>> RED FLAG: {c['isolated_load_mw']:.1f} MW of load is in islands")
            L.append("       that contain NO slack — those loads will always shed.")

    # Branch impedance
    L.append(_box("5. BRANCH IMPEDANCE"))
    bi = findings["branch_impedance"]
    if "line_x_pu_min" in bi:
        L.append(f"  Line x_pu  : min={bi['line_x_pu_min']:.6f}  "
                 f"median={bi['line_x_pu_median']:.4f}  max={bi['line_x_pu_max']:.4f}")
        L.append(f"  Lines with |x_pu| < 1e-5 : {bi['lines_x_below_1e_5']}")
    if "trafo_vk_pct_min" in bi:
        L.append(f"  Trafo vk_%  : min={bi['trafo_vk_pct_min']:.2f}  "
                 f"median={bi['trafo_vk_pct_median']:.2f}  max={bi['trafo_vk_pct_max']:.2f}")
    if bi.get("red_flag_near_zero_x"):
        L.append("  >>> RED FLAG: lines with near-zero reactance cause singular Y-bus.")

    # DC-PF and overloads
    L.append(_box("6. DC POWER FLOW + OVERLOAD SCREEN"))
    dc = findings["dcpf"]
    if dc.get("dc_pf_converged"):
        L.append(f"  DC-PF: CONVERGED")
        L.append(f"  Generation : {dc['dc_gen_mw']:.0f} MW")
        L.append(f"  Ext grid   : {dc['dc_ext_mw']:>+.0f} MW  "
                 f"({'absorbing' if dc['dc_ext_mw'] < 0 else 'injecting'})")
        L.append(f"  Load       : {dc['dc_load_mw']:.0f} MW")
        L.append(f"  Overloaded (>150%): {dc['overloaded_count']}  "
                 f"(max {dc['max_loading_pct']:.0f}%)")
        if dc.get("red_flag_overloads"):
            L.append("  >>> RED FLAG: DC-PF shows overloads. AC-OPF will be infeasible")
            L.append(f"       unless mitigated. See {OVERLOAD_CSV.name} for the list.")
            L.append("       MITIGATION: Step 09 must scale rate_A on those branches,")
            L.append("       OR relocate slack to balance flows, OR add parallel rating.")
        if abs(dc["dc_ext_mw"]) > 5000:
            L.append(f"  >>> RED FLAG: ext_grid is moving {abs(dc['dc_ext_mw']):.0f} MW.")
            L.append("       Slack location is forcing huge cross-grid flows.")
    else:
        L.append(f"  DC-PF: FAILED — {dc.get('dc_pf_error', 'unknown')}")
        L.append("       Cannot proceed to AC-OPF without DC-PF convergence.")

    # Cost coverage
    L.append(_box("7. GENERATOR COST COVERAGE"))
    cc = findings["cost_coverage"]
    L.append(f"  Total cost records  : {cc.get('n_gen_cost_records', 0)}")
    L.append(f"  Default fallback (c1=1000) : {cc.get('n_default_fallback', 0)} "
             f"({cc.get('default_pct', 0):.0f}%)")
    if cc.get("red_flag_no_merit_order"):
        L.append("  >>> RED FLAG: Most gens use default cost. OPF cannot see real")
        L.append("       merit order — dispatch will be cost-neutral. Convergence is")
        L.append("       still possible, but the dispatch is economically meaningless.")

    # Hazard coverage
    L.append(_box("8. HAZARD FACTOR COVERAGE"))
    hc = findings["hazard"]
    if hc.get("hazard_factors_attached"):
        L.append(f"  Records attached : {hc['n_records']}")
        L.append(f"    Lines    : {hc['n_lines']}")
        L.append(f"    Trafos   : {hc['n_trafos']}")
    else:
        L.append("  Not attached. SMC will use baseline rates only.")

    # Final verdict
    L.append(_box("VERDICT"))
    red_flags = [
        ("Zero EHV generation",     findings["voltage_tiers"].get("red_flag_zero_ehv_gen")),
        ("Slack at distribution",   findings["slack"].get("red_flag_slack_at_distribution")),
        ("Low active reserve",      findings["power_balance"].get("red_flag_low_reserve")),
        ("Q deficit",               findings["power_balance"].get("red_flag_q_deficit")),
        ("Orphan loads",            findings["connectivity"].get("red_flag_orphan_load")),
        ("Near-zero reactance",     findings["branch_impedance"].get("red_flag_near_zero_x")),
        ("DC-PF overloads",         findings["dcpf"].get("red_flag_overloads")),
        ("No merit order",          findings["cost_coverage"].get("red_flag_no_merit_order")),
    ]
    fired = [name for name, val in red_flags if val]
    L.append(f"  Red flags fired : {len(fired)} / {len(red_flags)}")
    for name in fired:
        L.append(f"    - {name}")
    if not fired:
        L.append("  No red flags. Proceed directly to Step 09.")
    else:
        L.append("")
        L.append("  RECOMMENDED NEXT STEP:")
        L.append("    Run  8b_robust_acopf.py")
        L.append("    It applies repairs that target each red flag listed above.")
        L.append("    All repairs are logged. Document them in your paper as a")
        L.append("    'network sanitisation' preprocessing step — this is normal")
        L.append("    practice when consuming raw utility PSS/E files.")

    L.append(_line())
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not NET_FILE.exists():
        log.error(f"Step 06 network not found: {NET_FILE}")
        log.error("Run Step 06 first.")
        return

    log.info(f"Loading network from {NET_FILE} ...")
    net = pp.from_json(str(NET_FILE))
    log.info(f"  {len(net.bus)} buses, {len(net.gen)} gens, {len(net.line)} lines, "
             f"{len(net.trafo)} 2W trafos, {len(net.trafo3w)} 3W trafos, "
             f"{len(net.load)} loads")

    findings = {
        "voltage_tiers":     diagnose_voltage_tiers(net),
        "slack":             diagnose_slack(net),
        "power_balance":     diagnose_power_balance(net),
        "connectivity":      diagnose_connectivity(net),
        "branch_impedance":  diagnose_branch_impedance(net),
        "dcpf":              diagnose_dcpf_overloads(net),
        "cost_coverage":     diagnose_cost_coverage(net),
        "hazard":            diagnose_hazard_coverage(net),
    }

    report = render_report(findings)
    print(report)

    REPORT_TXT.write_text(report, encoding="utf-8")
    FINDINGS_JSON.write_text(
        json.dumps(findings, indent=2, default=str), encoding="utf-8"
    )

    log.info(f"\nReport saved to: {REPORT_TXT}")
    log.info(f"Findings JSON  : {FINDINGS_JSON}")
    log.info(f"Voltage summary: {KV_SUMMARY}")
    if OVERLOAD_CSV.exists():
        log.info(f"Overloads CSV  : {OVERLOAD_CSV}")


if __name__ == "__main__":
    main()