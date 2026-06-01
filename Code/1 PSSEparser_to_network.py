"""
Step 01 — Parse PSSE RAW Network File (PSS/E v32 format)
=========================================================
Taipower 2025 (Minguo Year 114) base case: 11507DP_base(108).raw

Key engineering decisions (per Taipower feedback):
  1. Negative X values are -0.000000E+0 → treated as 0.0 (not negative)
  2. Isolated buses must be identified and removed
  3. If no explicit slack bus: choose generator bus with largest Pmax
  4. Encoding: latin-1 (RAW) with Big5 Chinese bus names → decoded to UTF-8 labels
"""

import re
import sys
import logging
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
RAW_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\11507DP_base(108).raw")
OUT_DIR   = Path(r"C:\reXplan-repo\Project Taipower\Results\step01")
ENCODING  = "latin-1"          # PSSE RAW standard; Big5 names embedded


def decode_big5(s: str) -> str:
    """Decode a latin-1 string that contains Big5-encoded Chinese characters."""
    try:
        return s.encode("latin-1").decode("big5")
    except Exception:
        return s
BASE_MVA  = 100.0              # system MVA base (from RAW header)
X_ZERO_THRESHOLD = 1e-9        # |X| below this → treat as 0


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BusRecord:
    bus_i:    int
    name:     str
    base_kv:  float
    ide:      int    # 1=load, 2=gen, 3=slack, 4=isolated
    area:     int
    zone:     int
    owner:    int
    vm:       float  # pu voltage magnitude
    va:       float  # voltage angle deg

@dataclass
class GeneratorRecord:
    bus:      int
    id:       str
    pg:       float  # MW dispatch
    qg:       float  # Mvar dispatch
    qt:       float  # Mvar max
    qb:       float  # Mvar min
    vs:       float  # voltage setpoint pu
    ireg:     int
    mbase:    float  # MVA rating
    zr:       float
    zx:       float
    rt:       float
    xt:       float
    gtap:     float
    stat:     int    # 1=in-service
    rmpct:    float
    pt:       float  # MW max
    pb:       float  # MW min

@dataclass
class BranchRecord:
    from_bus: int
    to_bus:   int
    ckt:      str
    r:        float  # pu resistance
    x:        float  # pu reactance
    b:        float  # pu total charging susceptance
    rate_a:   float  # MVA normal rating
    rate_b:   float  # MVA emergency rating
    rate_c:   float  # MVA short-term emergency rating
    st:       int    # 1=in-service

@dataclass
class TransformerRecord:
    from_bus: int
    to_bus:   int
    k_bus:    int    # 0 for 2-winding
    ckt:      str
    cw:       int
    cz:       int
    cm:       int
    mag1:     float
    mag2:     float
    nmetr:    int
    name:     str
    stat:     int
    o1:       int
    # winding 1 data
    r12:      float  # pu resistance
    x12:      float  # pu reactance
    sbase12:  float
    windv1:   float
    nomv1:    float
    ang1:     float
    rata1:    float
    ratb1:    float
    ratc1:    float
    cod1:     int
    cont1:    int
    rma1:     float
    rmi1:     float
    vma1:     float
    vmi1:     float
    ntp1:     int
    tab1:     int
    cr1:      float
    cx1:      float
    cnxa1:    float
    windv2:   float
    nomv2:    float

@dataclass
class NetworkData:
    """Container for all parsed network data"""
    system_mva:   float
    version:      int
    buses:        pd.DataFrame = field(default_factory=pd.DataFrame)
    generators:   pd.DataFrame = field(default_factory=pd.DataFrame)
    branches:     pd.DataFrame = field(default_factory=pd.DataFrame)
    transformers: pd.DataFrame = field(default_factory=pd.DataFrame)
    loads:        pd.DataFrame = field(default_factory=pd.DataFrame)
    shunts:       pd.DataFrame = field(default_factory=pd.DataFrame)
    slack_bus:    Optional[int] = None
    isolated_buses: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Section splitter
# ──────────────────────────────────────────────────────────────────────────────
def split_sections(raw_text: str) -> dict:
    """Split RAW file into named sections by '0 /End of X data...' markers."""
    section_map = {
        "bus":         "End of Bus data",
        "load":        "End of Load data",
        "fixed_shunt": "End of Fixed shunt data",
        "generator":   "End of Generator data",
        "branch":      "End of Branch data",
        "transformer": "End of Transformer data",
        "area":        "End of Area interchange data",
    }
    # Build ordered list of (marker_text, line_number)
    lines = raw_text.splitlines()
    markers = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("0 /End of"):
            for key, text in section_map.items():
                if text in stripped:
                    markers[key] = i
                    break

    # Extract sections between consecutive markers
    # Header = lines 0..(bus_marker-1), Bus = bus_marker+1..load_marker-1, etc.
    ordered_keys = list(section_map.keys())
    sections = {}
    prev_end = 0
    sections["header"] = lines[0:3]

    for idx, key in enumerate(ordered_keys):
        if key not in markers:
            continue
        end_line = markers[key]
        # start is after previous marker
        if idx == 0:
            start = 3   # after 3-line header
        else:
            prev_key = ordered_keys[idx - 1]
            start = markers.get(prev_key, 0) + 1
        sections[key] = lines[start:end_line]

    # Transformer section needs everything after branch marker
    if "transformer" in markers and "branch" in markers:
        sections["transformer"] = lines[markers["branch"] + 1 : markers["transformer"]]

    return sections


# ──────────────────────────────────────────────────────────────────────────────
# Bus parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_buses(lines: list) -> pd.DataFrame:
    """
    PSSE v32 Bus record format:
    I, 'NAME', BASKV, IDE, AREA, ZONE, OWNER, VM, VA
    """
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        # Extract quoted name first
        m = re.match(r"\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line)
        if not m:
            continue
        bus_i = int(m.group(1))
        name  = decode_big5(m.group(2).strip())
        rest  = m.group(3).split(",")
        try:
            base_kv = float(rest[0])
            ide     = int(rest[1])
            area    = int(rest[2])
            zone    = int(rest[3])
            owner   = int(rest[4])
            vm      = float(rest[5])
            va      = float(rest[6])
        except (IndexError, ValueError):
            continue
        records.append(BusRecord(bus_i, name, base_kv, ide, area, zone, owner, vm, va))

    df = pd.DataFrame([vars(r) for r in records])
    log.info(f"Parsed {len(df)} buses")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Load parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_loads(lines: list) -> pd.DataFrame:
    """
    PSSE v32 Load record:
    I, ID, STATUS, AREA, ZONE, PL, QL, IP, IQ, YP, YQ, OWNER, SCALE, INTRPT
    """
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        m = re.match(r"\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line)
        if not m:
            continue
        bus_i = int(m.group(1))
        cid   = m.group(2).strip()
        parts = m.group(3).split(",")
        try:
            status = int(parts[0])
            area   = int(parts[1])
            zone   = int(parts[2])
            pl     = float(parts[3])   # MW constant power
            ql     = float(parts[4])   # Mvar constant power
        except (IndexError, ValueError):
            continue
        records.append({"bus_i": bus_i, "id": cid, "status": status,
                         "area": area, "zone": zone, "pl_mw": pl, "ql_mvar": ql})

    df = pd.DataFrame(records)
    log.info(f"Parsed {len(df)} load records")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Generator parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_generators(lines: list) -> pd.DataFrame:
    """
    PSSE v32 Generator record:
    I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX, RT, XT, GTAP, STAT, RMPCT, PT, PB, ...
    """
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        m = re.match(r"\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line)
        if not m:
            continue
        bus_i = int(m.group(1))
        gid   = m.group(2).strip()
        parts = m.group(3).split(",")
        try:
            pg    = float(parts[0])
            qg    = float(parts[1])
            qt    = float(parts[2])
            qb    = float(parts[3])
            vs    = float(parts[4])
            ireg  = int(parts[5])
            mbase = float(parts[6])
            zr    = float(parts[7])
            zx    = float(parts[8])
            rt    = float(parts[9])
            xt    = float(parts[10])
            gtap  = float(parts[11])
            stat  = int(parts[12])
            rmpct = float(parts[13])
            pt    = float(parts[14])  # MW max
            pb    = float(parts[15]) if len(parts) > 15 else 0.0
        except (IndexError, ValueError):
            continue
        records.append({
            "bus_i": bus_i, "gen_id": gid, "pg_mw": pg, "qg_mvar": qg,
            "qt_mvar": qt, "qb_mvar": qb, "vs_pu": vs, "ireg": ireg,
            "mbase_mva": mbase, "zr_pu": zr, "zx_pu": zx,
            "rt_pu": rt, "xt_pu": xt, "gtap": gtap, "status": stat,
            "rmpct": rmpct, "pmax_mw": pt, "pmin_mw": pb
        })

    df = pd.DataFrame(records)
    log.info(f"Parsed {len(df)} generator records")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Branch parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_branches(lines: list) -> pd.DataFrame:
    """
    PSSE v32 Branch record:
    I, J, CKT, R, X, B, RATEA, RATEB, RATEC, GI, BI, GJ, BJ, ST, MET, LEN, O1, F1, ...
    """
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        m = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line)
        if not m:
            continue
        from_bus = int(m.group(1))
        to_bus   = int(m.group(2))
        ckt      = m.group(3).strip()
        parts    = m.group(4).split(",")
        try:
            r      = float(parts[0])
            x      = float(parts[1])
            b      = float(parts[2])
            rate_a = float(parts[3])
            rate_b = float(parts[4])
            rate_c = float(parts[5])
            # Skip GI, BI, GJ, BJ (indices 6-9)
            st = int(parts[10]) if len(parts) > 10 else 1
        except (IndexError, ValueError):
            continue

        # Fix -0.000000E+0 → 0.0 (Taipower feedback)
        if abs(x) < X_ZERO_THRESHOLD:
            x = 1e-6  # assign tiny value to avoid singularity

        records.append({
            "from_bus": from_bus, "to_bus": to_bus, "ckt": ckt,
            "r_pu": r, "x_pu": x, "b_pu": b,
            "rate_a_mva": rate_a, "rate_b_mva": rate_b, "rate_c_mva": rate_c,
            "status": st
        })

    df = pd.DataFrame(records)
    log.info(f"Parsed {len(df)} branch records")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Transformer parser (2-winding, 4-line records)
# ──────────────────────────────────────────────────────────────────────────────
def parse_transformers(lines: list) -> pd.DataFrame:
    """
    PSSE v32 Two-winding transformer: 4 lines per record
    Line 1: I, J, K, CKT, CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, O1,...
    Line 2: R1-2, X1-2, SBASE1-2
    Line 3: WINDV1, NOMV1, ANG1, RATA1, RATB1, RATC1, COD1, CONT1, RMA1, RMI1, VMA1, VMI1,...
    Line 4: WINDV2, NOMV2
    """
    records = []
    i = 0
    while i < len(lines) - 3:
        line1 = lines[i].strip()
        if not line1 or line1.startswith("@"):
            i += 1
            continue

        # Check if this is a transformer record (has quoted name in line1)
        m1 = re.match(
            r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line1
        )
        if not m1:
            i += 1
            continue

        from_bus = int(m1.group(1))
        to_bus   = int(m1.group(2))
        k_bus    = int(m1.group(3))
        ckt      = m1.group(4).strip()
        rest1    = m1.group(5).split(",")
        try:
            cw   = int(rest1[0])
            cz   = int(rest1[1])
            cm   = int(rest1[2])
            mag1 = float(rest1[3])
            mag2 = float(rest1[4])
            nmetr= int(rest1[5])
            # rest1[6] = quoted name (already captured)
            stat = int(rest1[8]) if len(rest1) > 8 else 1
        except (IndexError, ValueError):
            i += 4
            continue

        # Line 2: winding impedance
        line2 = lines[i + 1].strip()
        parts2 = line2.split(",")
        try:
            r12    = float(parts2[0])
            x12    = float(parts2[1])
            sbase12= float(parts2[2])
        except (IndexError, ValueError):
            r12 = x12 = sbase12 = 0.0

        # Fix near-zero X (same Taipower feedback)
        if abs(x12) < X_ZERO_THRESHOLD:
            x12 = 1e-6

        # Line 3: winding 1 tap / rating
        line3 = lines[i + 2].strip()
        parts3 = line3.split(",")
        try:
            windv1 = float(parts3[0])
            nomv1  = float(parts3[1])
            ang1   = float(parts3[2])
            rata1  = float(parts3[3])
            ratb1  = float(parts3[4])
            ratc1  = float(parts3[5])
        except (IndexError, ValueError):
            windv1 = 1.0; nomv1 = 0.0; ang1 = 0.0
            rata1  = 0.0; ratb1 = 0.0; ratc1= 0.0

        # Line 4: winding 2 tap
        line4 = lines[i + 3].strip()
        parts4 = line4.split(",")
        try:
            windv2 = float(parts4[0])
            nomv2  = float(parts4[1])
        except (IndexError, ValueError):
            windv2 = 1.0; nomv2 = 0.0

        records.append({
            "from_bus": from_bus, "to_bus": to_bus, "k_bus": k_bus, "ckt": ckt,
            "cw": cw, "cz": cz, "cm": cm, "mag1": mag1, "mag2": mag2,
            "nmetr": nmetr, "status": stat,
            "r12_pu": r12, "x12_pu": x12, "sbase12_mva": sbase12,
            "windv1": windv1, "nomv1": nomv1, "ang1_deg": ang1,
            "rata1_mva": rata1, "ratb1_mva": ratb1, "ratc1_mva": ratc1,
            "windv2": windv2, "nomv2": nomv2
        })
        i += 4

    df = pd.DataFrame(records)
    log.info(f"Parsed {len(df)} transformer records")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Isolated bus detection
# ──────────────────────────────────────────────────────────────────────────────
def find_isolated_buses(buses_df: pd.DataFrame,
                        branches_df: pd.DataFrame,
                        transformers_df: pd.DataFrame) -> list:
    """
    Build undirected connectivity graph and identify buses not reachable
    from the main connected component. Also flag IDE=4 (explicitly isolated).
    """
    G = nx.Graph()
    all_bus_ids = set(buses_df["bus_i"].tolist())
    G.add_nodes_from(all_bus_ids)

    # Add branch edges (in-service only)
    if not branches_df.empty:
        active_br = branches_df[branches_df["status"] == 1]
        for _, row in active_br.iterrows():
            G.add_edge(int(row["from_bus"]), int(row["to_bus"]))

    # Add transformer edges (in-service only)
    if not transformers_df.empty:
        active_tr = transformers_df[transformers_df["status"] == 1]
        for _, row in active_tr.iterrows():
            G.add_edge(int(row["from_bus"]), int(row["to_bus"]))

    # Find largest connected component
    components = list(nx.connected_components(G))
    if not components:
        return list(all_bus_ids)

    main_component = max(components, key=len)
    topologically_isolated = all_bus_ids - main_component

    # Also include IDE=4 buses
    ide4_buses = set(buses_df[buses_df["ide"] == 4]["bus_i"].tolist())

    isolated = sorted(topologically_isolated | ide4_buses)
    log.info(f"Found {len(isolated)} isolated buses "
             f"({len(topologically_isolated)} topological, {len(ide4_buses)} IDE=4)")
    if isolated:
        log.info(f"  Isolated bus IDs (first 20): {isolated[:20]}")
    return isolated


# ──────────────────────────────────────────────────────────────────────────────
# Slack bus identification
# ──────────────────────────────────────────────────────────────────────────────
def identify_slack_bus(buses_df: pd.DataFrame,
                       generators_df: pd.DataFrame,
                       isolated: list) -> int:
    """
    1. Check if any bus has IDE=3 (swing bus) and is not isolated → use it
    2. Otherwise: choose generator bus with largest Pmax, not isolated
    """
    valid_buses = buses_df[~buses_df["bus_i"].isin(isolated)]

    # Check IDE=3
    swing_buses = valid_buses[valid_buses["ide"] == 3]["bus_i"].tolist()
    if swing_buses:
        slack = swing_buses[0]
        log.info(f"Slack bus: {slack} (IDE=3, explicit swing bus)")
        return slack

    # No explicit swing → largest Pmax generator
    valid_gen_buses = set(valid_buses["bus_i"].tolist())
    gen_valid = generators_df[
        (generators_df["bus_i"].isin(valid_gen_buses)) &
        (generators_df["status"] == 1)
    ]
    if gen_valid.empty:
        log.warning("No valid in-service generators found! Using bus 1 as slack.")
        return int(valid_buses["bus_i"].iloc[0])

    # Sum Pmax by bus (multiple generators per bus)
    bus_pmax = gen_valid.groupby("bus_i")["pmax_mw"].sum()
    slack = int(bus_pmax.idxmax())
    log.info(f"Slack bus: {slack} (largest Pmax = {bus_pmax.max():.1f} MW, "
             f"no explicit swing bus)")
    return slack


# ──────────────────────────────────────────────────────────────────────────────
# Main parse function
# ──────────────────────────────────────────────────────────────────────────────
def parse_raw(filepath: Path) -> NetworkData:
    log.info(f"Reading RAW file: {filepath}")
    raw_text = filepath.read_text(encoding=ENCODING)

    # Parse header: first line  → system MVA, version
    first_line = raw_text.splitlines()[0]
    parts0 = first_line.split(",")
    system_mva = float(parts0[1].strip()) if len(parts0) > 1 else 100.0
    version    = int(parts0[2].strip())   if len(parts0) > 2 else 32
    log.info(f"System MVA base: {system_mva}, PSS/E version: {version}")

    sections = split_sections(raw_text)

    buses_df    = parse_buses(sections.get("bus", []))
    loads_df    = parse_loads(sections.get("load", []))
    gens_df     = parse_generators(sections.get("generator", []))
    branches_df = parse_branches(sections.get("branch", []))
    trafo_df    = parse_transformers(sections.get("transformer", []))

    # ── Isolated bus detection
    isolated = find_isolated_buses(buses_df, branches_df, trafo_df)

    # ── Slack bus
    slack = identify_slack_bus(buses_df, gens_df, isolated)

    # ── Mark slack in buses_df
    buses_df.loc[buses_df["bus_i"] == slack, "ide"] = 3

    net = NetworkData(
        system_mva   = system_mva,
        version      = version,
        buses        = buses_df,
        generators   = gens_df,
        branches     = branches_df,
        transformers = trafo_df,
        loads        = loads_df,
        slack_bus    = slack,
        isolated_buses = isolated,
    )
    return net


# ──────────────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(net: NetworkData):
    print("\n" + "=" * 65)
    print("  TAIPOWER NETWORK SUMMARY — Step 01 Parse Results")
    print("=" * 65)
    print(f"  System MVA base       : {net.system_mva:.0f} MVA")
    print(f"  PSS/E version         : {net.version}")
    print(f"  Total buses           : {len(net.buses):>6d}")
    print(f"  Isolated buses removed: {len(net.isolated_buses):>6d}")
    print(f"  Active buses          : {len(net.buses) - len(net.isolated_buses):>6d}")
    print(f"  Generators            : {len(net.generators):>6d}")
    print(f"  In-service generators : {(net.generators['status']==1).sum():>6d}")
    print(f"  Branches              : {len(net.branches):>6d}")
    print(f"  Transformers (2W)     : {len(net.transformers):>6d}")
    print(f"  Load records          : {len(net.loads):>6d}")
    print(f"  Slack bus             : {net.slack_bus}")

    total_gen_mw  = net.generators[net.generators["status"]==1]["pmax_mw"].sum()
    total_load_mw = net.loads["pl_mw"].sum()
    print(f"  Total generation cap. : {total_gen_mw:>10.1f} MW")
    print(f"  Total load (constant P): {total_load_mw:>10.1f} MW")

    # Voltage level distribution
    kv_groups = net.buses.groupby(
        pd.cut(net.buses["base_kv"],
               bins=[0, 1, 35, 70, 115, 162, 250, 350, 500, 800],
               labels=["<1kV","34.5kV","69kV","110kV","161kV","230kV","345kV","500kV",">500kV"])
    ).size()
    print("\n  Voltage level distribution:")
    for level, count in kv_groups.items():
        if count > 0:
            print(f"    {str(level):>10s}: {count:>5d} buses")
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# Save to CSV for downstream steps
# ──────────────────────────────────────────────────────────────────────────────
def save_network_csv(net: NetworkData, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter out isolated buses
    active_buses = net.buses[~net.buses["bus_i"].isin(net.isolated_buses)]
    active_buses.to_csv(out_dir / "buses.csv", index=False, encoding="utf-8")

    # Filter generators to active buses only
    active_gen = net.generators[~net.generators["bus_i"].isin(net.isolated_buses)]
    active_gen.to_csv(out_dir / "generators.csv", index=False)

    # Filter branches to active buses
    iso_set = set(net.isolated_buses)
    active_br = net.branches[
        (~net.branches["from_bus"].isin(iso_set)) &
        (~net.branches["to_bus"].isin(iso_set))
    ]
    active_br.to_csv(out_dir / "branches.csv", index=False)

    # Filter transformers to active buses
    active_tr = net.transformers[
        (~net.transformers["from_bus"].isin(iso_set)) &
        (~net.transformers["to_bus"].isin(iso_set))
    ]
    active_tr.to_csv(out_dir / "transformers.csv", index=False)
    net.loads.to_csv(out_dir / "loads.csv", index=False)

    # Save metadata
    meta = {
        "system_mva": net.system_mva,
        "version": net.version,
        "slack_bus": net.slack_bus,
        "n_buses_total": len(net.buses),
        "n_buses_active": len(active_buses),
        "n_isolated": len(net.isolated_buses),
        "isolated_bus_list": net.isolated_buses,
    }
    import json
    (out_dir / "network_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info(f"Network data saved to: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    net = parse_raw(RAW_FILE)
    print_summary(net)
    save_network_csv(net, OUT_DIR)
    return net


if __name__ == "__main__":
    main()