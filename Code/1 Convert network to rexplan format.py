"""
Step 01 - PSS/E RAW Parser (Complete Rebuild)
=============================================
Parses Taipower 11507DP_base_108_.raw (PSS/E v32, BIG5 encoding)
and exports all network data to CSV files for downstream steps.

Key fix vs previous version:
  - Exports ALL 2761 active buses (ide != 4), not just 1970
  - Includes LV terminal buses of 3-winding transformers
  - Correct PSS/E v32 field indexing for branches and transformers
  - Chinese characters preserved as-is (BIG5 encoding)

Output files (all in RESULT_BASE/step01/):
  buses.csv, generators.csv, loads.csv, branches.csv,
  transformers_2w.csv, transformers_3w.csv
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

RAW_FILE    = Path(r"C:\reXplan-repo\Project Taipower\Data\11507DP_base(108).raw")
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
OUT_DIR     = RESULT_BASE / "step01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MVA = 100.0


# ==============================================================================
# File reader
# ==============================================================================
def read_raw(path: Path) -> list:
    """Read PSS/E RAW file with BIG5 encoding (Taiwanese Chinese)."""
    with open(path, "rb") as f:
        text = f.read().decode("big5")
    return text.splitlines()


def get_section(lines: list, start_line: int, end_line: int) -> list:
    """Return non-empty, non-comment lines between two line indices."""
    result = []
    for line in lines[start_line:end_line]:
        s = line.strip()
        if s and not s.startswith("/"):
            result.append(s)
    return result


def find_section_bounds(lines: list) -> dict:
    """Find line indices for all PSS/E data sections."""
    markers = {
        "bus_end":    "End of Bus data",
        "load_end":   "End of Load data",
        "shunt_end":  "End of Fixed shunt data",
        "gen_end":    "End of Generator data",
        "branch_end": "End of Branch data",
        "trafo_end":  "End of Transformer data",
    }
    bounds = {}
    for key, marker in markers.items():
        for i, line in enumerate(lines):
            if marker in line:
                bounds[key] = i
                break
    return bounds


# ==============================================================================
# Section parsers
# ==============================================================================
def parse_buses(lines: list) -> pd.DataFrame:
    """
    PSS/E v32 Bus record:
    I, NAME, BASKV, IDE, AREA, ZONE, OWNER, VM, VA
    IDE: 1=PQ load, 2=PV gen, 3=swing, 4=isolated (skip)
    """
    records = []
    for line in lines:
        try:
            parts   = line.split(",")
            bus_i   = int(parts[0].strip())
            name    = parts[1].strip().strip("'").strip()
            base_kv = float(parts[2].strip())
            ide     = int(parts[3].strip())
            area    = int(parts[4].strip()) if len(parts) > 4 else 1
            zone    = int(parts[5].strip()) if len(parts) > 5 else 1
            vm      = float(parts[7].strip()) if len(parts) > 7 else 1.0
            va      = float(parts[8].strip()) if len(parts) > 8 else 0.0
            records.append({
                "bus_i":   bus_i,
                "name":    name,
                "base_kv": base_kv,
                "ide":     ide,
                "area":    area,
                "zone":    zone,
                "vm":      vm,
                "va":      va,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    # Keep all buses including ide=4 for reference, filter downstream
    return df


def parse_loads(lines: list) -> pd.DataFrame:
    """
    PSS/E v32 Load record:
    I, ID, STATUS, AREA, ZONE, PL, QL, IP, IQ, YP, YQ, OWNER, SCALE, INTRPT
    """
    records = []
    for line in lines:
        try:
            parts  = line.split(",")
            bus_i  = int(parts[0].strip())
            load_id= parts[1].strip().strip("'").strip()
            status = int(parts[2].strip()) if len(parts) > 2 else 1
            pl     = float(parts[5].strip()) if len(parts) > 5 else 0.0
            ql     = float(parts[6].strip()) if len(parts) > 6 else 0.0
            records.append({
                "bus_i":   bus_i,
                "id":      load_id,
                "status":  status,
                "pl_mw":   pl,
                "ql_mvar": ql,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def parse_generators(lines: list) -> pd.DataFrame:
    """
    PSS/E v32 Generator record:
    I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX, RT, XT, GTAP, STAT,
    RMPCT, PT, PB, O1, F1, ...
    """
    records = []
    for line in lines:
        try:
            parts  = line.split(",")
            bus_i  = int(parts[0].strip())
            gen_id = parts[1].strip().strip("'").strip()
            pg     = float(parts[2].strip())
            qg     = float(parts[3].strip())
            qt     = float(parts[4].strip())   # Qmax
            qb     = float(parts[5].strip())   # Qmin
            vs     = float(parts[6].strip())   # voltage setpoint
            mbase  = float(parts[8].strip()) if len(parts) > 8 else BASE_MVA
            stat   = int(parts[14].strip())  if len(parts) > 14 else 1
            pmax   = float(parts[16].strip()) if len(parts) > 16 else 9999.0
            pmin   = float(parts[17].strip()) if len(parts) > 17 else 0.0
            records.append({
                "bus_i":     bus_i,
                "gen_id":    gen_id,
                "pg_mw":     pg,
                "qg_mvar":   qg,
                "qt_mvar":   qt,
                "qb_mvar":   qb,
                "vs_pu":     vs,
                "mbase_mva": mbase,
                "status":    stat,
                "pmax_mw":   pmax,
                "pmin_mw":   pmin,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def parse_branches(lines: list) -> pd.DataFrame:
    """
    PSS/E v32 Branch record:
    I, J, CKT, R, X, B, RATEA, RATEB, RATEC, GI, BI, GJ, BJ, ST, MET, LEN, O1, F1...
    STATUS is field index 13.
    """
    records = []
    for line in lines:
        try:
            parts   = line.split(",")
            fb      = int(parts[0].strip())
            tb      = int(parts[1].strip())
            ckt     = parts[2].strip().strip("'").strip()
            r       = float(parts[3].strip())
            x       = float(parts[4].strip())
            b       = float(parts[5].strip())
            rate_a  = float(parts[6].strip()) if len(parts) > 6 else 9999.0
            rate_b  = float(parts[7].strip()) if len(parts) > 7 else 9999.0
            rate_c  = float(parts[8].strip()) if len(parts) > 8 else 9999.0
            status  = int(parts[13].strip())  if len(parts) > 13 else 1
            records.append({
                "from_bus":   fb,
                "to_bus":     tb,
                "ckt":        ckt,
                "r_pu":       r,
                "x_pu":       x,
                "b_pu":       b,
                "rate_a_mva": rate_a,
                "rate_b_mva": rate_b,
                "rate_c_mva": rate_c,
                "status":     status,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def parse_transformers(raw_lines: list) -> tuple:
    """
    PSS/E v32 Transformer records.
    2-winding: 4 lines each (K=0)
    3-winding: 5 lines each (K!=0)

    Line 1: I, J, K, CKT, CW, CZ, CM, MAG1, MAG2, NMETR, NAME, STAT, ...
    Line 2: R1-2, X1-2, SBASE1-2 [, R2-3, X2-3, SBASE2-3, R3-1, X3-1, SBASE3-1] (3W only)
    Line 3 (winding 1): WINDV1, NOMV1, ANG1, RATA1, RATB1, RATC1, COD1, CONT1, ...
    Line 4 (winding 2): WINDV2, NOMV2 [, ANG2, RATA2, ...] (3W only has full winding data)
    Line 5 (winding 3, 3W only): WINDV3, NOMV3, ANG3, RATA3, ...
    """
    trafos_2w = []
    trafos_3w = []
    i = 0

    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line or line.startswith("/"):
            i += 1
            continue

        try:
            parts = line.split(",")
            I     = int(parts[0].strip())
            J     = int(parts[1].strip())
            K     = int(parts[2].strip())
            stat  = int(parts[11].strip()) if len(parts) > 11 else 1

            if K == 0:
                # 2-winding transformer
                l2 = raw_lines[i+1].strip().split(",") if i+1 < len(raw_lines) else []
                l3 = raw_lines[i+2].strip().split(",") if i+2 < len(raw_lines) else []
                l4 = raw_lines[i+3].strip().split(",") if i+3 < len(raw_lines) else []

                r12     = float(l2[0]) if l2 else 0.0
                x12     = float(l2[1]) if len(l2) > 1 else 0.01
                sbase12 = float(l2[2]) if len(l2) > 2 else BASE_MVA
                windv1  = float(l3[0]) if l3 else 1.0
                nomv1   = float(l3[1]) if len(l3) > 1 else 0.0
                ang1    = float(l3[2]) if len(l3) > 2 else 0.0
                rata1   = float(l3[3]) if len(l3) > 3 else 0.0
                windv2  = float(l4[0]) if l4 else 1.0

                trafos_2w.append({
                    "from_bus":    I,
                    "to_bus":      J,
                    "k_bus":       0,
                    "status":      stat,
                    "r12_pu":      r12,
                    "x12_pu":      x12,
                    "sbase12_mva": sbase12,
                    "windv1":      windv1,
                    "nomv1":       nomv1,
                    "ang1_deg":    ang1,
                    "rata1_mva":   rata1,
                    "windv2":      windv2,
                })
                i += 4

            else:
                # 3-winding transformer
                l2 = raw_lines[i+1].strip().split(",") if i+1 < len(raw_lines) else []
                l3 = raw_lines[i+2].strip().split(",") if i+2 < len(raw_lines) else []
                l4 = raw_lines[i+3].strip().split(",") if i+3 < len(raw_lines) else []
                l5 = raw_lines[i+4].strip().split(",") if i+4 < len(raw_lines) else []

                # Line 2: R12, X12, SBASE12, R23, X23, SBASE23, R31, X31, SBASE31
                r12     = float(l2[0]) if l2 else 0.0
                x12     = float(l2[1]) if len(l2) > 1 else 0.1
                sbase12 = float(l2[2]) if len(l2) > 2 else BASE_MVA
                r23     = float(l2[3]) if len(l2) > 3 else 0.0
                x23     = float(l2[4]) if len(l2) > 4 else 0.1
                sbase23 = float(l2[5]) if len(l2) > 5 else BASE_MVA
                r31     = float(l2[6]) if len(l2) > 6 else 0.0
                x31     = float(l2[7]) if len(l2) > 7 else 0.1
                sbase31 = float(l2[8]) if len(l2) > 8 else BASE_MVA

                # Winding voltages from buses
                windv1 = float(l3[0]) if l3 else 1.0
                nomv1  = float(l3[1]) if len(l3) > 1 else 0.0
                ang1   = float(l3[2]) if len(l3) > 2 else 0.0
                rata1  = float(l3[3]) if len(l3) > 3 else 0.0
                windv2 = float(l4[0]) if l4 else 1.0
                nomv2  = float(l4[1]) if len(l4) > 1 else 0.0
                ang2   = float(l4[2]) if len(l4) > 2 else 0.0
                rata2  = float(l4[3]) if len(l4) > 3 else 0.0
                windv3 = float(l5[0]) if l5 else 1.0
                nomv3  = float(l5[1]) if len(l5) > 1 else 0.0
                ang3   = float(l5[2]) if len(l5) > 2 else 0.0
                rata3  = float(l5[3]) if len(l5) > 3 else 0.0

                trafos_3w.append({
                    "hv_bus":      I,
                    "mv_bus":      J,
                    "lv_bus":      K,
                    "status":      stat,
                    "r12_pu":      r12,
                    "x12_pu":      x12,
                    "sbase12_mva": sbase12,
                    "r23_pu":      r23,
                    "x23_pu":      x23,
                    "sbase23_mva": sbase23,
                    "r31_pu":      r31,
                    "x31_pu":      x31,
                    "sbase31_mva": sbase31,
                    "windv1":      windv1,
                    "nomv1":       nomv1,
                    "ang1_deg":    ang1,
                    "rata1_mva":   rata1,
                    "windv2":      windv2,
                    "nomv2":       nomv2,
                    "ang2_deg":    ang2,
                    "rata2_mva":   rata2,
                    "windv3":      windv3,
                    "nomv3":       nomv3,
                    "ang3_deg":    ang3,
                    "rata3_mva":   rata3,
                })
                i += 5

        except Exception:
            i += 1

    return pd.DataFrame(trafos_2w), pd.DataFrame(trafos_3w)


# ==============================================================================
# Derive kV for 3W windings from bus data
# ==============================================================================
def enrich_transformers_3w(tr3_df: pd.DataFrame, buses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vn_hv_kv, vn_mv_kv, vn_lv_kv from buses base_kv.
    Also compute per-unit reactances for each winding
    using the star-equivalent circuit.
    """
    kv_map = dict(zip(buses_df["bus_i"], buses_df["base_kv"]))

    tr3_df = tr3_df.copy()
    tr3_df["vn_hv_kv"] = tr3_df["hv_bus"].map(kv_map).fillna(345.0)
    tr3_df["vn_mv_kv"] = tr3_df["mv_bus"].map(kv_map).fillna(161.0)
    tr3_df["vn_lv_kv"] = tr3_df["lv_bus"].map(kv_map).fillna(13.8)
    tr3_df["sn_hv_mva"] = tr3_df["sbase12_mva"]
    tr3_df["sn_mv_mva"] = tr3_df["sbase23_mva"]
    tr3_df["sn_lv_mva"] = tr3_df["sbase31_mva"]

    z12 = tr3_df["x12_pu"]
    z23 = tr3_df["x23_pu"]
    z31 = tr3_df["x31_pu"]
    tr3_df["x_hv_pu"] = 0.5 * (z12 + z31 - z23)
    tr3_df["x_mv_pu"] = 0.5 * (z12 + z23 - z31)
    tr3_df["x_lv_pu"] = 0.5 * (z23 + z31 - z12)

    r12 = tr3_df["r12_pu"]
    r23 = tr3_df["r23_pu"]
    r31 = tr3_df["r31_pu"]
    tr3_df["r_hv_pu"] = 0.5 * (r12 + r31 - r23)
    tr3_df["r_mv_pu"] = 0.5 * (r12 + r23 - r31)
    tr3_df["r_lv_pu"] = 0.5 * (r23 + r31 - r12)

    return tr3_df


# ==============================================================================
# Switched shunt parser
# ==============================================================================
def parse_switched_shunts(lines: list) -> pd.DataFrame:
    """
    PSS/E v32 Switched Shunt record:
    I, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, RMIDNT, BINIT, N1, B1, ...

    Switched shunts represent capacitor banks and reactors used for reactive
    power compensation and voltage control. They are critical for AC-OPF
    convergence as they provide reactive power support at load buses.

    BINIT : initial reactive power injection (Mvar, positive = capacitive)
    B1    : susceptance of one shunt block (Mvar at nominal voltage)
    N1    : number of steps available in block 1
    """
    records = []
    for line in lines:
        try:
            parts  = line.split(",")
            bus_i  = int(parts[0].strip())
            modsw  = int(parts[1].strip()) if len(parts) > 1 else 0
            stat   = int(parts[3].strip()) if len(parts) > 3 else 1
            vswhi  = float(parts[4].strip()) if len(parts) > 4 else 1.05
            vswlo  = float(parts[5].strip()) if len(parts) > 5 else 0.95
            swrem  = int(float(parts[6].strip())) if len(parts) > 6 else 0
            binit  = float(parts[9].strip()) if len(parts) > 9 else 0.0

            # Parse all N-B pairs (up to 8 blocks)
            q_total = 0.0
            for k in range(8):
                n_idx = 10 + k * 2
                b_idx = 11 + k * 2
                if n_idx >= len(parts):
                    break
                n_steps = int(float(parts[n_idx].strip())) if parts[n_idx].strip() else 0
                b_step  = float(parts[b_idx].strip()) if b_idx < len(parts) and parts[b_idx].strip() else 0.0
                q_total += abs(n_steps * b_step)

            records.append({
                "bus_i":     bus_i,
                "modsw":     modsw,
                "status":    stat,
                "vswhi_pu":  vswhi,
                "vswlo_pu":  vswlo,
                "swrem_bus": swrem,
                "binit_mvar": binit,
                "q_max_mvar": abs(q_total),
            })
        except Exception:
            continue

    return pd.DataFrame(records)


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 65)
    print("  STEP 01 - PSS/E RAW PARSER (Complete Rebuild)")
    print("=" * 65)
    print(f"  Input : {RAW_FILE}")
    print(f"  Output: {OUT_DIR}")

    lines = read_raw(RAW_FILE)
    print(f"  File  : {len(lines)} lines, BIG5 encoded")

    bounds = find_section_bounds(lines)
    print(f"\n  Section bounds:")
    for k, v in bounds.items():
        print(f"    {k:15s} : line {v}")

    # Section line ranges
    bus_lines   = get_section(lines, 3,                    bounds["bus_end"])
    load_lines  = get_section(lines, bounds["bus_end"]+1,  bounds["load_end"])
    gen_lines   = get_section(lines, bounds["shunt_end"]+1,bounds["gen_end"])
    br_lines    = get_section(lines, bounds["gen_end"]+1,  bounds["branch_end"])
    tr_raw      = [l.strip() for l in lines[bounds["branch_end"]+1:bounds["trafo_end"]]
                   if l.strip() and not l.strip().startswith("/")]

    # Switched shunts (capacitor banks and reactors)
    facts_end_marker   = "End of FACTS device data"
    sw_shunt_end_marker= "End of Switched shunt data"
    facts_end_line = next((i for i,l in enumerate(lines) if facts_end_marker in l), None)
    sw_shunt_end   = next((i for i,l in enumerate(lines) if sw_shunt_end_marker in l), None)
    if facts_end_line and sw_shunt_end:
        sw_shunt_lines = get_section(lines, facts_end_line+1, sw_shunt_end)
    else:
        sw_shunt_lines = []

    # Parse
    buses_df = parse_buses(bus_lines)
    loads_df = parse_loads(load_lines)
    gens_df  = parse_generators(gen_lines)
    br_df    = parse_branches(br_lines)
    tr2_df, tr3_df = parse_transformers(tr_raw)
    sw_df    = parse_switched_shunts(sw_shunt_lines)

    # Enrich 3W transformers with kV and star-circuit reactances
    tr3_df = enrich_transformers_3w(tr3_df, buses_df)

    # Summary before filtering
    print(f"\n  RAW data extracted:")
    print(f"    Buses (total)      : {len(buses_df)}")
    print(f"    Buses (active)     : {(buses_df['ide']!=4).sum()}")
    print(f"    Generators         : {len(gens_df)}")
    print(f"    Loads              : {len(loads_df)}")
    print(f"    Branches           : {len(br_df)}")
    print(f"    Transformers 2W    : {len(tr2_df)}")
    print(f"    Transformers 3W    : {len(tr3_df)}")
    print(f"    Switched shunts    : {len(sw_df)}  "
          f"(active: {(sw_df['status']==1).sum() if len(sw_df) else 0})")

    # Identify swing bus
    swing = buses_df[buses_df["ide"] == 3]
    print(f"\n  Swing bus (ide=3):")
    for _, row in swing.iterrows():
        print(f"    Bus {row['bus_i']} ({row['name']}) @ {row['base_kv']} kV")

    # Active bus set = all buses except ide=4
    active_set = set(buses_df[buses_df["ide"] != 4]["bus_i"])
    all_set    = set(buses_df["bus_i"])

    # Verify 3W LV buses
    lv_in_active = set(tr3_df["lv_bus"]) & active_set
    lv_missing   = set(tr3_df["lv_bus"]) - all_set
    print(f"\n  3W trafo LV bus check:")
    print(f"    LV buses in active set : {len(lv_in_active)} / {len(set(tr3_df['lv_bus']))}")
    print(f"    LV buses NOT in raw    : {len(lv_missing)}  (should be 0)")

    active_tr3 = tr3_df[
        tr3_df["hv_bus"].isin(active_set) &
        tr3_df["mv_bus"].isin(active_set) &
        tr3_df["lv_bus"].isin(active_set) &
        (tr3_df["status"] == 1)
    ]
    print(f"    Active 3W trafos       : {len(active_tr3)} / {len(tr3_df)}")

    # Power balance check
    active_gen = gens_df[gens_df["bus_i"].isin(active_set) & (gens_df["status"] == 1)]
    active_load= loads_df[loads_df["bus_i"].isin(active_set) & (loads_df["status"] == 1)]
    pmax_total = active_gen["pmax_mw"].clip(upper=9000).sum()
    pg_total   = active_gen["pg_mw"].sum()
    pl_total   = active_load["pl_mw"].sum()
    print(f"\n  Power balance:")
    print(f"    Active gen Pmax  : {pmax_total:.0f} MW")
    print(f"    Active gen Pg    : {pg_total:.0f} MW")
    print(f"    Active load PL   : {pl_total:.0f} MW")
    print(f"    Surplus Pmax-PL  : {pmax_total - pl_total:.0f} MW")

    # Build network_meta.json
    isolated_list = sorted(list(set(buses_df[buses_df["ide"] == 4]["bus_i"])))
    meta = {
        "system_mva":        BASE_MVA,
        "raw_file":          str(RAW_FILE.name),
        "raw_version":       32,
        "encoding":          "big5",
        "slack_bus":         int(swing["bus_i"].iloc[0]) if len(swing) else 1,
        "n_buses_total":     len(buses_df),
        "n_buses_active":    len(active_set),
        "n_isolated":        len(isolated_list),
        "isolated_bus_list": isolated_list,
        "n_generators":      len(gens_df),
        "n_loads":           len(loads_df),
        "n_branches":        len(br_df),
        "n_trafo_2w":        len(tr2_df),
        "n_trafo_3w":        len(tr3_df),
        "n_switched_shunts": len(sw_df),
        "total_pmax_mw":     round(pmax_total, 1),
        "total_pg_mw":       round(pg_total, 1),
        "total_pl_mw":       round(pl_total, 1),
    }

    # Save all files
    buses_df.to_csv(OUT_DIR / "buses.csv",              index=False, encoding="utf-8-sig")
    loads_df.to_csv(OUT_DIR / "loads.csv",              index=False, encoding="utf-8-sig")
    gens_df.to_csv( OUT_DIR / "generators.csv",         index=False, encoding="utf-8-sig")
    br_df.to_csv(   OUT_DIR / "branches.csv",           index=False, encoding="utf-8-sig")
    tr2_df.to_csv(  OUT_DIR / "transformers_2w.csv",    index=False, encoding="utf-8-sig")
    tr3_df.to_csv(  OUT_DIR / "transformers_3w.csv",    index=False, encoding="utf-8-sig")
    sw_df.to_csv(   OUT_DIR / "switched_shunts.csv",    index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "network_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Files saved to: {OUT_DIR}")
    print(f"    buses.csv              : {len(buses_df)} rows")
    print(f"    generators.csv         : {len(gens_df)} rows")
    print(f"    loads.csv              : {len(loads_df)} rows")
    print(f"    branches.csv           : {len(br_df)} rows")
    print(f"    transformers_2w.csv    : {len(tr2_df)} rows")
    print(f"    transformers_3w.csv    : {len(tr3_df)} rows")
    print(f"    switched_shunts.csv    : {len(sw_df)} rows")
    print(f"    network_meta.json      : OK")
    print("=" * 65)
    print("  STEP 01 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()