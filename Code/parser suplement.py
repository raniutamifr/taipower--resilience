"""
Step 01b — PSSE Parser Supplement
==================================
Re-parses the raw PSS/E v32 file to extract three element classes that
Step 01 dropped entirely:

  1.  Three-winding transformers   →  transformers_3w.csv
  2.  Switched shunts (cap/reactor) →  switched_shunts.csv
  3.  Fixed shunts                  →  fixed_shunts.csv

It also splits the original `transformers.csv` into a clean two-winding-only
file (`transformers_2w.csv`), which is what Step 06 expects.

WHY THIS MATTERS — the convergence diagnosis
---------------------------------------------
Step 01's transformer parser advances 4 lines per record. PSS/E v32 stores
two-winding transformers in 4 lines, but three-winding transformers in 5
lines. The unconditional `i += 4` corrupts every record after the first
3W transformer in the file.

Worse: Step 06 reads `transformers_2w.csv` and `transformers_3w.csv`
separately. Step 01 never produced `transformers_3w.csv` at all, so Step 06
loaded an empty DataFrame and dropped every 3W trafo.

In Taipower's network these 3W transformers connect:
    345 kV  ──HV──┐
    161 kV  ──MV──┤  3W trafo at major substations
     22 kV  ──LV──┘  (generator step-up at LV)

Dropping them severs every 345 kV bus from generation and load — exactly
the "0 generators at 345 kV" symptom we've been seeing for months.

PSS/E v32 record formats
------------------------
2W transformer (k_bus == 0):
    Line 1 : I, J, K=0, CKT, CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, O1, F1, ...
    Line 2 : R1-2, X1-2, SBASE1-2
    Line 3 : WINDV1, NOMV1, ANG1, RATA1, RATB1, RATC1, COD1, ...
    Line 4 : WINDV2, NOMV2

3W transformer (k_bus > 0):
    Line 1 : I, J, K>0, CKT, CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, ...
    Line 2 : R1-2, X1-2, SBASE1-2, R2-3, X2-3, SBASE2-3, R3-1, X3-1, SBASE3-1, VMSTAR, ANSTAR
    Line 3 : WINDV1, NOMV1, ANG1, RATA1, RATB1, RATC1, COD1, ...
    Line 4 : WINDV2, NOMV2, ANG2, RATA2, RATB2, RATC2, ...
    Line 5 : WINDV3, NOMV3, ANG3, RATA3, RATB3, RATC3, ...

Switched shunt:
    I, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, 'RMIDNT', BINIT, N1, B1, N2, B2, ...
    BINIT = base-case Mvar (+ capacitive, − inductive)

Fixed shunt:
    I, ID, STATUS, GL, BL
    BL = Mvar (+ capacitive, − inductive)

This script writes ONLY the missing files. It does not touch any of the
files Step 01 already wrote, so the rest of the pipeline is unaffected.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\11507DP_base(108).raw")
OUT_DIR  = Path(r"C:\reXplan-repo\Project Taipower\Results\step01")
ENCODING = "latin-1"

# Star-circuit conversion uses the system base
BASE_MVA = 100.0
X_ZERO_THRESHOLD = 1e-9


def decode_big5(s: str) -> str:
    try:
        return s.encode("latin-1").decode("big5")
    except Exception:
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Section locator — finds the byte range of each PSS/E section
# ─────────────────────────────────────────────────────────────────────────────
SECTION_MARKERS = [
    ("bus",            "End of Bus data"),
    ("load",           "End of Load data"),
    ("fixed_shunt",    "End of Fixed shunt data"),
    ("generator",      "End of Generator data"),
    ("branch",         "End of Branch data"),
    ("transformer",    "End of Transformer data"),
    ("area",           "End of Area interchange data"),
    ("two_terminal",   "End of Two-terminal dc data"),
    ("vsc",            "End of VSC dc line data"),
    ("impedance_corr", "End of Impedance correction data"),
    ("multi_terminal", "End of Multi-terminal dc data"),
    ("multi_section",  "End of Multi-section line data"),
    ("zone",           "End of Zone data"),
    ("interarea",      "End of Inter-area transfer data"),
    ("owner",          "End of Owner data"),
    ("facts",          "End of FACTS device data"),
    ("switched_shunt", "End of Switched shunt data"),
]


def locate_sections(raw_text: str) -> dict:
    """Return {section_name: (start_line_idx, end_line_idx)} for each section."""
    lines = raw_text.splitlines()
    end_markers: dict = {}
    for i, line in enumerate(lines):
        if "0 /" in line and "End of" in line:
            for key, text in SECTION_MARKERS:
                if text in line:
                    end_markers[key] = i
                    break

    # Build start indices (3-line header before bus section)
    sections = {}
    prev_end = 3  # after 3-line header
    for key, _ in SECTION_MARKERS:
        if key in end_markers:
            sections[key] = (prev_end, end_markers[key])
            prev_end = end_markers[key] + 1
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Transformer split (2W vs 3W) — the main fix
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trafo3WRecord:
    from_bus: int
    to_bus:   int
    mid_bus:  int
    ckt:      str
    name:     str
    status:   int
    # Winding 1-2-3 mesh impedance (PSS/E pairwise)
    r12_pu:   float
    x12_pu:   float
    sbase12_mva: float
    r23_pu:   float
    x23_pu:   float
    sbase23_mva: float
    r31_pu:   float
    x31_pu:   float
    sbase31_mva: float
    vmstar:   float
    anstar:   float
    # Per-winding ratings
    rata1_mva: float
    rata2_mva: float
    rata3_mva: float
    # Per-winding nominal voltages
    nomv1: float
    nomv2: float
    nomv3: float
    windv1: float
    windv2: float
    windv3: float
    ang1: float
    ang2: float
    ang3: float


def _split_csv(line: str) -> list:
    """Split a PSS/E CSV-ish line, handling quoted strings."""
    out = []
    cur = ""
    in_q = False
    for ch in line:
        if ch == "'":
            in_q = not in_q
            cur += ch
        elif ch == "," and not in_q:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur.strip())
    return out


def _f(parts: list, i: int, default: float = 0.0) -> float:
    try:
        return float(parts[i])
    except (IndexError, ValueError, TypeError):
        return default


def _i(parts: list, idx: int, default: int = 0) -> int:
    try:
        return int(float(parts[idx]))
    except (IndexError, ValueError, TypeError):
        return default


def parse_transformers_split(lines: list, bus_kv: dict = None) -> tuple:
    """
    Re-parse the transformer section, correctly handling 2W (4-line) vs
    3W (5-line) records. Returns (df_2w, df_3w).

    bus_kv : optional dict {bus_i: base_kv} used to fill nomv when PSS/E
             RAW has 0 (which means "use bus base voltage").
    """
    if bus_kv is None:
        bus_kv = {}
    rec_2w = []
    rec_3w = []

    n = len(lines)
    i = 0
    n_skipped_blank = 0
    n_2w = 0
    n_3w = 0

    while i < n:
        line1 = lines[i].strip()
        if not line1 or line1.startswith("@"):
            n_skipped_blank += 1
            i += 1
            continue
        # Line 1 must start with at least 3 comma-separated integers
        # (I, J, K). K decides 2W vs 3W.
        m = re.match(
            r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line1)
        if not m:
            # Not a header line — skip
            i += 1
            continue

        from_bus = int(m.group(1))
        to_bus   = int(m.group(2))
        k_bus    = int(m.group(3))
        ckt      = m.group(4).strip()
        rest1    = _split_csv(m.group(5))
        # rest1 fields: CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, O1, F1, ...
        # name index = 6, status index = 7
        try:
            name = rest1[6].strip().strip("'") if len(rest1) > 6 else ""
            name = decode_big5(name)
        except Exception:
            name = ""
        status = _i(rest1, 7, 1)

        # Boundary check — need at least 4 more lines for 2W, 5 for 3W
        is_3w = (k_bus != 0)
        record_len = 5 if is_3w else 4
        if i + record_len > n:
            log.warning(f"Truncated record at line {i} (need {record_len} more)")
            break

        line2_parts = _split_csv(lines[i + 1].strip())
        line3_parts = _split_csv(lines[i + 2].strip())
        line4_parts = _split_csv(lines[i + 3].strip())

        if is_3w:
            line5_parts = _split_csv(lines[i + 4].strip())

            # Line 2 (3W): R12, X12, SBASE12, R23, X23, SBASE23, R31, X31, SBASE31, VMSTAR, ANSTAR
            r12     = _f(line2_parts, 0); x12     = _f(line2_parts, 1)
            sbase12 = _f(line2_parts, 2, BASE_MVA)
            r23     = _f(line2_parts, 3); x23     = _f(line2_parts, 4)
            sbase23 = _f(line2_parts, 5, BASE_MVA)
            r31     = _f(line2_parts, 6); x31     = _f(line2_parts, 7)
            sbase31 = _f(line2_parts, 8, BASE_MVA)
            vmstar  = _f(line2_parts, 9, 1.0)
            anstar  = _f(line2_parts, 10, 0.0)

            # Fix near-zero X
            for var_name, val in [("x12", x12), ("x23", x23), ("x31", x31)]:
                if abs(val) < X_ZERO_THRESHOLD:
                    if var_name == "x12": x12 = 1e-6
                    if var_name == "x23": x23 = 1e-6
                    if var_name == "x31": x31 = 1e-6

            # Line 3 (winding 1)
            windv1 = _f(line3_parts, 0, 1.0)
            nomv1  = _f(line3_parts, 1, 0.0)
            ang1   = _f(line3_parts, 2, 0.0)
            rata1  = _f(line3_parts, 3, 0.0)

            # Line 4 (winding 2)
            windv2 = _f(line4_parts, 0, 1.0)
            nomv2  = _f(line4_parts, 1, 0.0)
            ang2   = _f(line4_parts, 2, 0.0)
            rata2  = _f(line4_parts, 3, 0.0)

            # Line 5 (winding 3)
            windv3 = _f(line5_parts, 0, 1.0)
            nomv3  = _f(line5_parts, 1, 0.0)
            ang3   = _f(line5_parts, 2, 0.0)
            rata3  = _f(line5_parts, 3, 0.0)

            # Convert pairwise (mesh) impedances to per-winding star (Y) equivalent.
            # Standard Y-Δ → star conversion for three windings:
            #   z1 = (z12 + z31 − z23) / 2
            #   z2 = (z12 + z23 − z31) / 2
            #   z3 = (z23 + z31 − z12) / 2
            # This is what pandapower's create_transformer3w_from_parameters wants
            # via vk_hv/mv/lv_percent. We compute the per-winding x_pu and r_pu,
            # then Step 06 multiplies by 100 to get vk%.
            x_hv = (x12 + x31 - x23) / 2.0
            x_mv = (x12 + x23 - x31) / 2.0
            x_lv = (x23 + x31 - x12) / 2.0
            r_hv = (r12 + r31 - r23) / 2.0
            r_mv = (r12 + r23 - r31) / 2.0
            r_lv = (r23 + r31 - r12) / 2.0

            # Sort windings into HV/MV/LV roles by nominal voltage.
            # PSS/E winding 1/2/3 doesn't always correspond to HV/MV/LV.
            # pandapower's create_transformer3w_from_parameters requires
            # vn_hv_kv > vn_mv_kv > vn_lv_kv strictly.
            # If nomv is 0 in the RAW, PSS/E convention is "use the bus
            # base voltage" — so fall back to bus_kv lookup.
            def _eff_nomv(bus, nomv):
                return nomv if nomv > 0 else float(bus_kv.get(bus, 0.0))

            windings = [
                {"bus": from_bus, "nomv": _eff_nomv(from_bus, nomv1),
                 "windv": windv1, "ang": ang1,
                 "rata": rata1, "r": r_hv, "x": x_hv},
                {"bus": to_bus,   "nomv": _eff_nomv(to_bus, nomv2),
                 "windv": windv2, "ang": ang2,
                 "rata": rata2, "r": r_mv, "x": x_mv},
                {"bus": k_bus,    "nomv": _eff_nomv(k_bus, nomv3),
                 "windv": windv3, "ang": ang3,
                 "rata": rata3, "r": r_lv, "x": x_lv},
            ]
            # Sort by nominal voltage descending. Tie-break by rating.
            windings.sort(key=lambda w: (w["nomv"], w["rata"]), reverse=True)
            w_hv, w_mv, w_lv = windings[0], windings[1], windings[2]

            rec_3w.append({
                "from_bus":    from_bus,
                "to_bus":      to_bus,
                "mid_bus":     k_bus,
                "ckt":         ckt,
                "name":        name,
                "status":      status,
                # Mesh impedances (kept for fallback / audit)
                "r12_pu":      r12, "x12_pu": x12, "sbase12_mva": sbase12,
                "r23_pu":      r23, "x23_pu": x23, "sbase23_mva": sbase23,
                "r31_pu":      r31, "x31_pu": x31, "sbase31_mva": sbase31,
                "vmstar":      vmstar, "anstar": anstar,
                # Per-winding ratings/voltages in PSS/E order (for audit)
                "rata1_mva":   rata1, "rata2_mva": rata2, "rata3_mva": rata3,
                "nomv1":       nomv1, "nomv2": nomv2, "nomv3": nomv3,
                "windv1":      windv1, "windv2": windv2, "windv3": windv3,
                "ang1_deg":    ang1, "ang2_deg": ang2, "ang3_deg": ang3,
                # ---- Role-sorted fields (what Step 06 actually reads) ----
                "hv_bus":      w_hv["bus"],
                "mv_bus":      w_mv["bus"],
                "lv_bus":      w_lv["bus"],
                "vn_hv_kv":    w_hv["nomv"],
                "vn_mv_kv":    w_mv["nomv"],
                "vn_lv_kv":    w_lv["nomv"],
                "sn_hv_mva":   max(w_hv["rata"], 1.0),
                "sn_mv_mva":   max(w_mv["rata"], 1.0),
                "sn_lv_mva":   max(w_lv["rata"], 1.0),
                "r_hv_pu":     w_hv["r"], "x_hv_pu": w_hv["x"],
                "r_mv_pu":     w_mv["r"], "x_mv_pu": w_mv["x"],
                "r_lv_pu":     w_lv["r"], "x_lv_pu": w_lv["x"],
            })
            n_3w += 1
            i += 5
        else:
            # 2W transformer — same logic as Step 01 but cleaner
            r12     = _f(line2_parts, 0)
            x12     = _f(line2_parts, 1)
            if abs(x12) < X_ZERO_THRESHOLD:
                x12 = 1e-6
            sbase12 = _f(line2_parts, 2, BASE_MVA)

            windv1 = _f(line3_parts, 0, 1.0)
            nomv1  = _f(line3_parts, 1, 0.0)
            ang1   = _f(line3_parts, 2, 0.0)
            rata1  = _f(line3_parts, 3, 0.0)
            ratb1  = _f(line3_parts, 4, 0.0)
            ratc1  = _f(line3_parts, 5, 0.0)

            windv2 = _f(line4_parts, 0, 1.0)
            nomv2  = _f(line4_parts, 1, 0.0)

            rec_2w.append({
                "from_bus":    from_bus,
                "to_bus":      to_bus,
                "k_bus":       0,
                "ckt":         ckt,
                "name":        name,
                "status":      status,
                "r12_pu":      r12,
                "x12_pu":      x12,
                "sbase12_mva": sbase12,
                "windv1":      windv1,
                "nomv1":       nomv1,
                "ang1_deg":    ang1,
                "rata1_mva":   rata1,
                "ratb1_mva":   ratb1,
                "ratc1_mva":   ratc1,
                "windv2":      windv2,
                "nomv2":       nomv2,
            })
            n_2w += 1
            i += 4

    log.info(f"  Parsed {n_2w} two-winding, {n_3w} three-winding transformers "
             f"({n_skipped_blank} blank/comment lines skipped)")
    return pd.DataFrame(rec_2w), pd.DataFrame(rec_3w)


# ─────────────────────────────────────────────────────────────────────────────
# Switched shunts — entirely missing in Step 01
# ─────────────────────────────────────────────────────────────────────────────
def parse_switched_shunts(lines: list) -> pd.DataFrame:
    """
    PSS/E v32 switched shunt record:
      I, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, 'RMIDNT', BINIT,
      N1, B1, N2, B2, ..., N8, B8

    BINIT is the base-case shunt admittance in Mvar at 1.0 pu voltage.
    Positive = capacitive (injects Q), negative = inductive (absorbs Q).
    Pandapower's shunt convention: q_mvar > 0 = absorbing (inductive),
    so we flip sign in Step 06.
    """
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        parts = _split_csv(line)
        if len(parts) < 10:
            continue
        try:
            bus_i  = int(parts[0])
            modsw  = _i(parts, 1, 0)
            adjm   = _i(parts, 2, 0)
            stat   = _i(parts, 3, 1)
            vswhi  = _f(parts, 4, 1.0)
            vswlo  = _f(parts, 5, 1.0)
            swrem  = _i(parts, 6, 0)
            rmpct  = _f(parts, 7, 100.0)
            # parts[8] is 'RMIDNT' (quoted string identifier)
            binit  = _f(parts, 9, 0.0)
        except (IndexError, ValueError):
            continue

        records.append({
            "bus_i":      bus_i,
            "modsw":      modsw,
            "adjm":       adjm,
            "status":     stat,
            "vswhi":      vswhi,
            "vswlo":      vswlo,
            "swrem":      swrem,
            "rmpct":      rmpct,
            "binit_mvar": binit,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        log.info(f"  Parsed {len(df)} switched shunts "
                 f"(active: {int((df['status']==1).sum())}, "
                 f"total binit: {df.loc[df['status']==1, 'binit_mvar'].sum():.0f} Mvar)")
    else:
        log.info("  Parsed 0 switched shunts")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fixed shunts — also missing
# ─────────────────────────────────────────────────────────────────────────────
def parse_fixed_shunts(lines: list) -> pd.DataFrame:
    """
    PSS/E v32 fixed shunt record:
      I, 'ID', STATUS, GL, BL

    GL = MW absorbed at 1.0 pu (positive = resistive load)
    BL = Mvar injected at 1.0 pu (positive = capacitive)
    """
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        parts = _split_csv(line)
        if len(parts) < 5:
            continue
        try:
            bus_i  = int(parts[0])
            sh_id  = parts[1].strip().strip("'")
            status = _i(parts, 2, 1)
            gl     = _f(parts, 3, 0.0)
            bl     = _f(parts, 4, 0.0)
        except (IndexError, ValueError):
            continue
        records.append({
            "bus_i":  bus_i,
            "id":     sh_id,
            "status": status,
            "gl_mw":  gl,
            "bl_mvar": bl,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        log.info(f"  Parsed {len(df)} fixed shunts "
                 f"(active: {int((df['status']==1).sum())}, "
                 f"total bl: {df.loc[df['status']==1, 'bl_mvar'].sum():.0f} Mvar)")
    else:
        log.info("  Parsed 0 fixed shunts")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sanity report — what we added vs what Step 01 had
# ─────────────────────────────────────────────────────────────────────────────
def report_findings(df_2w: pd.DataFrame, df_3w: pd.DataFrame,
                    df_sw: pd.DataFrame, df_fs: pd.DataFrame,
                    buses_path: Path) -> dict:
    """Cross-reference findings with the existing buses.csv."""
    findings = {}

    # How many 3W trafos reach EHV?
    if not df_3w.empty and buses_path.exists():
        buses = pd.read_csv(buses_path, encoding="utf-8-sig")
        bus_kv = dict(zip(buses["bus_i"], buses["base_kv"]))

        ehv_3w = 0
        for _, row in df_3w.iterrows():
            kvs = [bus_kv.get(int(row["from_bus"]), 0),
                   bus_kv.get(int(row["to_bus"]), 0),
                   bus_kv.get(int(row["mid_bus"]), 0)]
            if max(kvs) >= 300.0:
                ehv_3w += 1
        findings["n_3w_total"]      = len(df_3w)
        findings["n_3w_reaching_ehv"] = ehv_3w
        findings["n_3w_active"]     = int((df_3w["status"] == 1).sum())

        # Capacity sum at EHV
        ehv_buses = {int(b): float(kv) for b, kv in bus_kv.items() if kv >= 300.0}
        sw_at_ehv = df_sw[df_sw["bus_i"].isin(ehv_buses.keys())] if not df_sw.empty else pd.DataFrame()
        findings["switched_shunt_at_ehv_mvar"] = (
            float(sw_at_ehv.loc[sw_at_ehv["status"]==1, "binit_mvar"].sum())
            if not sw_at_ehv.empty else 0.0
        )

    findings["n_switched_shunts"] = len(df_sw)
    findings["n_switched_active"] = int((df_sw["status"] == 1).sum()) if not df_sw.empty else 0
    findings["total_switched_mvar_active"] = (
        float(df_sw.loc[df_sw["status"]==1, "binit_mvar"].sum())
        if not df_sw.empty else 0.0
    )

    findings["n_fixed_shunts"] = len(df_fs)
    findings["n_fixed_active"] = int((df_fs["status"] == 1).sum()) if not df_fs.empty else 0
    findings["total_fixed_mvar_active"] = (
        float(df_fs.loc[df_fs["status"]==1, "bl_mvar"].sum())
        if not df_fs.empty else 0.0
    )

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 78)
    log.info("  STEP 01b — PSSE PARSER SUPPLEMENT")
    log.info("=" * 78)

    if not RAW_FILE.exists():
        log.error(f"Raw file not found: {RAW_FILE}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading: {RAW_FILE}")
    raw_text = RAW_FILE.read_text(encoding=ENCODING)
    log.info(f"  {len(raw_text):,} chars, {raw_text.count(chr(10)):,} lines")

    log.info("\n>> Locating sections")
    sections = locate_sections(raw_text)
    lines = raw_text.splitlines()
    for name, (s, e) in sections.items():
        log.info(f"  {name:<16s}: lines {s:>5d}–{e:<5d}  ({e-s} records)")

    # ── Load bus voltages for transformer winding role assignment ─────────
    bus_kv = {}
    buses_csv = OUT_DIR / "buses.csv"
    if buses_csv.exists():
        try:
            _buses = pd.read_csv(buses_csv, encoding="utf-8-sig")
            bus_kv = dict(zip(_buses["bus_i"].astype(int),
                              _buses["base_kv"].astype(float)))
            log.info(f"  Loaded {len(bus_kv)} bus voltages from {buses_csv.name}")
        except Exception as e:
            log.warning(f"  Failed to read buses.csv for kV lookup: {e}")

    # ── Transformers (split 2W vs 3W) ───────────────────────────────────────
    log.info("\n>> Parsing transformers (2W + 3W)")
    if "transformer" in sections:
        s, e = sections["transformer"]
        df_2w, df_3w = parse_transformers_split(lines[s:e], bus_kv)
    else:
        log.warning("  Transformer section not found")
        df_2w, df_3w = pd.DataFrame(), pd.DataFrame()

    # ── Switched shunts ─────────────────────────────────────────────────────
    log.info("\n>> Parsing switched shunts")
    if "switched_shunt" in sections:
        s, e = sections["switched_shunt"]
        df_sw = parse_switched_shunts(lines[s:e])
    else:
        log.warning("  Switched shunt section not found")
        df_sw = pd.DataFrame()

    # ── Fixed shunts ────────────────────────────────────────────────────────
    log.info("\n>> Parsing fixed shunts")
    if "fixed_shunt" in sections:
        s, e = sections["fixed_shunt"]
        df_fs = parse_fixed_shunts(lines[s:e])
    else:
        log.warning("  Fixed shunt section not found")
        df_fs = pd.DataFrame()

    # ── Write outputs ───────────────────────────────────────────────────────
    log.info("\n>> Writing CSV outputs")
    if not df_2w.empty:
        path = OUT_DIR / "transformers_2w.csv"
        df_2w.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s}: {len(df_2w)} records")

    if not df_3w.empty:
        path = OUT_DIR / "transformers_3w.csv"
        df_3w.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s}: {len(df_3w)} records  *** NEW ***")
    else:
        log.warning("  transformers_3w.csv: 0 records — RAW has no 3W trafos?")

    if not df_sw.empty:
        path = OUT_DIR / "switched_shunts.csv"
        df_sw.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s}: {len(df_sw)} records  *** NEW ***")

    if not df_fs.empty:
        path = OUT_DIR / "fixed_shunts.csv"
        df_fs.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s}: {len(df_fs)} records  *** NEW ***")

    # ── Sanity findings ─────────────────────────────────────────────────────
    log.info("\n>> Sanity findings")
    findings = report_findings(df_2w, df_3w, df_sw, df_fs,
                                OUT_DIR / "buses.csv")
    for k, v in findings.items():
        log.info(f"  {k:<35s}: {v}")

    findings_path = OUT_DIR / "step01b_findings.json"
    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    # ── Verdict ─────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 78)
    log.info("  VERDICT")
    log.info("=" * 78)
    if findings.get("n_3w_total", 0) > 0:
        log.info(f"  3-winding transformers recovered  : {findings['n_3w_total']}")
        log.info(f"    of which reach 345 kV           : {findings.get('n_3w_reaching_ehv', 0)}")
        log.info(f"    active in base case             : {findings.get('n_3w_active', 0)}")
        log.info("")
        log.info("  These records are the 345-kV step-down transformers Step 06")
        log.info("  was missing. Re-run Step 06 — the 'zero gens at EHV' red flag")
        log.info("  should now be gone or much reduced.")
    else:
        log.info("  No 3W transformers in the raw file. This means the convergence")
        log.info("  issue is elsewhere — keep using 8b_robust_acopf.py's R2 backfill.")

    if findings.get("total_switched_mvar_active", 0) != 0:
        log.info(f"  Switched shunt capacity (active)  : "
                 f"{findings['total_switched_mvar_active']:+.0f} Mvar")
        if findings.get("switched_shunt_at_ehv_mvar"):
            log.info(f"    at 345 kV buses                : "
                     f"{findings['switched_shunt_at_ehv_mvar']:+.0f} Mvar")
    if findings.get("total_fixed_mvar_active", 0) != 0:
        log.info(f"  Fixed shunt capacity (active)     : "
                 f"{findings['total_fixed_mvar_active']:+.0f} Mvar")

    log.info("")
    log.info("  NEXT STEPS:")
    log.info("    1. Re-run Step 06 (it now reads the new CSVs automatically).")
    log.info("    2. Run 8a_pre_opf_diagnostic.py — see how many red flags drop.")
    log.info("    3. Run 8b_robust_acopf.py — should converge with fewer repairs.")
    log.info("=" * 78)


if __name__ == "__main__":
    main()