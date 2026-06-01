"""
Step 01b - PSS/E Parser Supplement
====================================
Re-parses the raw PSS/E v32 file to extract three element classes that
Step 01 omits:

  1. Three-winding transformers   ->  transformers_3w.csv
  2. Switched shunts (cap/reactor) ->  switched_shunts.csv
  3. Fixed shunts                 ->  fixed_shunts.csv

It also splits the original `transformers.csv` into a clean two-winding-only
file (`transformers_2w.csv`) which is what Step 06 expects.

CORRECTNESS NOTE - PSS/E v32 mesh-to-star impedance conversion
---------------------------------------------------------------
PSS/E v32 stores three-winding transformers using PAIRWISE (mesh) impedances:
  R12 + jX12 on its own MVA base SBASE12
  R23 + jX23 on its own MVA base SBASE23
  R31 + jX31 on its own MVA base SBASE31

These bases are NOT identical. The standard delta-to-star conversion

    Z1 = (Z12 + Z31 - Z23) / 2
    Z2 = (Z12 + Z23 - Z31) / 2
    Z3 = (Z23 + Z31 - Z12) / 2

is only valid when Z12, Z23, Z31 are expressed on the SAME base. Applying it
directly to PSS/E values produces star impedances on an undefined mixed base,
which causes singular Y-bus matrices when pandapower constructs the admittance
representation.

CORRECT PROCEDURE:

  Step A. Convert each pairwise impedance to the system base (100 MVA):
              Z_ij_sys = Z_ij * (S_sys / SBASE_ij)
          This uses the standard per-unit base-change rule
              Z_pu_new = Z_pu_old * (S_new / S_old)

  Step B. Compute star equivalents on the system base:
              Z_hv_sys = (Z12_sys + Z31_sys - Z23_sys) / 2
              Z_mv_sys = (Z12_sys + Z23_sys - Z31_sys) / 2
              Z_lv_sys = (Z23_sys + Z31_sys - Z12_sys) / 2

  Step C. Convert each star impedance to the winding's OWN rated MVA base,
          which is what pandapower requires for vk_*_percent:
              Z_hv_on_SN_hv = Z_hv_sys * (SN_hv / S_sys)
              vk_hv_percent = |Z_hv_on_SN_hv| * 100
                           = |Z_hv_sys| * SN_hv         (since S_sys = 100 MVA)

This file writes vk_hv_percent, vk_mv_percent, vk_lv_percent and the
real-part counterparts vkr_hv_percent, vkr_mv_percent, vkr_lv_percent
directly into transformers_3w.csv, so Step 06 can use them without further
base conversion.

References
----------
  - PSS/E Program Operation Manual v32, Volume 2, Section 5.2.1
  - Bergen and Vittal, "Power Systems Analysis", 2nd ed., Ch. 5
  - Coffrin et al., PowerModels.jl correct_network_data() routine
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================
RAW_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\11507DP_base(108).raw")
OUT_DIR  = Path(r"C:\reXplan-repo\Project Taipower\Results\step01")
ENCODING = "latin-1"

SYSTEM_MVA       = 100.0    # pandapower / PSS/E system base
X_ZERO_THRESHOLD = 1e-9     # below this, reactance is treated as zero

# pandapower vk_percent bounds (typical transformer short-circuit voltage)
VK_PCT_MIN       = 1.0      # below 1% is numerically unstable
VK_PCT_MAX       = 30.0     # above 30% is physically implausible
VKR_PCT_MAX      = 5.0      # resistive part typically << reactive part

# PSS/E v32 section end markers (used to locate byte ranges)
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


# ==============================================================================
# Utility functions
# ==============================================================================
def decode_big5(text: str) -> str:
    """Decode Big-5 encoded substring that arrived as Latin-1 bytes."""
    try:
        return text.encode("latin-1").decode("big5")
    except Exception:
        return text


def split_psse_csv(line: str) -> list:
    """Split a PSS/E CSV-ish line, respecting single-quoted string fields."""
    out, current, in_quote = [], "", False
    for char in line:
        if char == "'":
            in_quote = not in_quote
            current += char
        elif char == "," and not in_quote:
            out.append(current.strip())
            current = ""
        else:
            current += char
    if current:
        out.append(current.strip())
    return out


def get_float(parts: list, idx: int, default: float = 0.0) -> float:
    """Safely extract a float from a list of strings."""
    try:
        return float(parts[idx])
    except (IndexError, ValueError, TypeError):
        return default


def get_int(parts: list, idx: int, default: int = 0) -> int:
    """Safely extract an int from a list of strings."""
    try:
        return int(float(parts[idx]))
    except (IndexError, ValueError, TypeError):
        return default


def locate_sections(raw_text: str) -> dict:
    """Find the line ranges of each PSS/E v32 section."""
    lines       = raw_text.splitlines()
    end_markers = {}
    for i, line in enumerate(lines):
        if "0 /" in line and "End of" in line:
            for key, text in SECTION_MARKERS:
                if text in line:
                    end_markers[key] = i
                    break

    sections = {}
    prev_end = 3                              # 3-line header before bus section
    for key, _ in SECTION_MARKERS:
        if key in end_markers:
            sections[key] = (prev_end, end_markers[key])
            prev_end      = end_markers[key] + 1
    return sections


# ==============================================================================
# Mesh-to-star base conversion (the key correctness fix)
# ==============================================================================
def mesh_to_star_per_winding(
    r12_pu: float, x12_pu: float, sbase12_mva: float,
    r23_pu: float, x23_pu: float, sbase23_mva: float,
    r31_pu: float, x31_pu: float, sbase31_mva: float,
    sn_hv_mva:   float, sn_mv_mva:   float, sn_lv_mva:   float,
) -> dict:
    """
    Convert PSS/E pairwise (mesh) impedances to pandapower vk_*_percent.

    Implements the three-step procedure from this module's docstring:
        A. Convert each Z_ij to the system base (100 MVA).
        B. Compute star equivalents on the system base.
        C. Re-express each star impedance on its own winding's rated MVA base
           and multiply by 100 to obtain vk_*_percent.
    """
    # Step A: convert pairwise impedances to system base
    sb12 = max(sbase12_mva, 1.0)
    sb23 = max(sbase23_mva, 1.0)
    sb31 = max(sbase31_mva, 1.0)

    r12_sys = r12_pu * (SYSTEM_MVA / sb12)
    x12_sys = x12_pu * (SYSTEM_MVA / sb12)
    r23_sys = r23_pu * (SYSTEM_MVA / sb23)
    x23_sys = x23_pu * (SYSTEM_MVA / sb23)
    r31_sys = r31_pu * (SYSTEM_MVA / sb31)
    x31_sys = x31_pu * (SYSTEM_MVA / sb31)

    # Step B: star equivalents on system base
    r_hv_sys = (r12_sys + r31_sys - r23_sys) / 2.0
    x_hv_sys = (x12_sys + x31_sys - x23_sys) / 2.0
    r_mv_sys = (r12_sys + r23_sys - r31_sys) / 2.0
    x_mv_sys = (x12_sys + x23_sys - x31_sys) / 2.0
    r_lv_sys = (r23_sys + r31_sys - r12_sys) / 2.0
    x_lv_sys = (x23_sys + x31_sys - x12_sys) / 2.0

    # Step C: per-winding base, then to percent
    sn_hv = max(sn_hv_mva, 1.0)
    sn_mv = max(sn_mv_mva, 1.0)
    sn_lv = max(sn_lv_mva, 1.0)

    vk_hv  = abs(x_hv_sys) * sn_hv      # |Z_pu_winding| * 100 = |Z_pu_sys| * sn
    vk_mv  = abs(x_mv_sys) * sn_mv
    vk_lv  = abs(x_lv_sys) * sn_lv
    vkr_hv = abs(r_hv_sys) * sn_hv
    vkr_mv = abs(r_mv_sys) * sn_mv
    vkr_lv = abs(r_lv_sys) * sn_lv

    # Clamp into physically reasonable ranges
    vk_hv  = max(VK_PCT_MIN, min(VK_PCT_MAX, vk_hv))
    vk_mv  = max(VK_PCT_MIN, min(VK_PCT_MAX, vk_mv))
    vk_lv  = max(VK_PCT_MIN, min(VK_PCT_MAX, vk_lv))
    vkr_hv = max(0.0, min(VKR_PCT_MAX, min(vkr_hv, vk_hv * 0.99)))
    vkr_mv = max(0.0, min(VKR_PCT_MAX, min(vkr_mv, vk_mv * 0.99)))
    vkr_lv = max(0.0, min(VKR_PCT_MAX, min(vkr_lv, vk_lv * 0.99)))

    return {
        # System-base star impedances (for audit / fallback)
        "r_hv_pu": r_hv_sys, "x_hv_pu": x_hv_sys,
        "r_mv_pu": r_mv_sys, "x_mv_pu": x_mv_sys,
        "r_lv_pu": r_lv_sys, "x_lv_pu": x_lv_sys,
        # Per-winding-base vk percentages (what pandapower needs)
        "vk_hv_pct":  vk_hv,  "vkr_hv_pct": vkr_hv,
        "vk_mv_pct":  vk_mv,  "vkr_mv_pct": vkr_mv,
        "vk_lv_pct":  vk_lv,  "vkr_lv_pct": vkr_lv,
    }


# ==============================================================================
# Transformer parser (2W vs 3W aware)
# ==============================================================================
def parse_transformers_split(lines: list, bus_kv: dict) -> tuple:
    """
    Re-parse the transformer section, correctly handling the 4-line (2W) vs
    5-line (3W) record-length difference that the original Step 01 missed.
    """
    rec_2w, rec_3w = [], []
    n_2w = n_3w    = 0
    n              = len(lines)
    i              = 0

    while i < n:
        line1 = lines[i].strip()
        if not line1 or line1.startswith("@"):
            i += 1
            continue

        # PSS/E header: "I, J, K, 'CKT', CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, ..."
        m = re.match(
            r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line1)
        if not m:
            i += 1
            continue

        from_bus = int(m.group(1))
        to_bus   = int(m.group(2))
        k_bus    = int(m.group(3))
        ckt      = m.group(4).strip()
        rest1    = split_psse_csv(m.group(5))

        name = decode_big5(rest1[6].strip().strip("'")) if len(rest1) > 6 else ""
        status = get_int(rest1, 7, 1)

        is_3w      = (k_bus != 0)
        record_len = 5 if is_3w else 4
        if i + record_len > n:
            log.warning(f"  Truncated transformer record at line {i}")
            break

        line2 = split_psse_csv(lines[i + 1].strip())
        line3 = split_psse_csv(lines[i + 2].strip())
        line4 = split_psse_csv(lines[i + 3].strip())

        if is_3w:
            line5 = split_psse_csv(lines[i + 4].strip())

            # Line 2: pairwise impedances on their own bases
            r12     = get_float(line2, 0)
            x12     = get_float(line2, 1)
            sbase12 = get_float(line2, 2, SYSTEM_MVA)
            r23     = get_float(line2, 3)
            x23     = get_float(line2, 4)
            sbase23 = get_float(line2, 5, SYSTEM_MVA)
            r31     = get_float(line2, 6)
            x31     = get_float(line2, 7)
            sbase31 = get_float(line2, 8, SYSTEM_MVA)
            vmstar  = get_float(line2, 9, 1.0)
            anstar  = get_float(line2, 10, 0.0)

            # Minimum reactance to prevent singular Y-bus contribution
            x12 = x12 if abs(x12) >= X_ZERO_THRESHOLD else 1e-6
            x23 = x23 if abs(x23) >= X_ZERO_THRESHOLD else 1e-6
            x31 = x31 if abs(x31) >= X_ZERO_THRESHOLD else 1e-6

            # Lines 3-5: per-winding data
            windv1, nomv1, ang1, rata1 = (get_float(line3, 0, 1.0), get_float(line3, 1, 0.0),
                                           get_float(line3, 2, 0.0), get_float(line3, 3, 0.0))
            windv2, nomv2, ang2, rata2 = (get_float(line4, 0, 1.0), get_float(line4, 1, 0.0),
                                           get_float(line4, 2, 0.0), get_float(line4, 3, 0.0))
            windv3, nomv3, ang3, rata3 = (get_float(line5, 0, 1.0), get_float(line5, 1, 0.0),
                                           get_float(line5, 2, 0.0), get_float(line5, 3, 0.0))

            # PSS/E nominal voltage of 0 means "use the bus base voltage"
            def resolve_kv(bus_idx: int, nominal: float) -> float:
                return nominal if nominal > 0 else float(bus_kv.get(bus_idx, 0.0))

            windings = [
                {"bus": from_bus, "nomv": resolve_kv(from_bus, nomv1),
                 "rata": rata1, "windv": windv1, "ang": ang1},
                {"bus": to_bus,   "nomv": resolve_kv(to_bus,   nomv2),
                 "rata": rata2, "windv": windv2, "ang": ang2},
                {"bus": k_bus,    "nomv": resolve_kv(k_bus,    nomv3),
                 "rata": rata3, "windv": windv3, "ang": ang3},
            ]
            # pandapower requires vn_hv_kv > vn_mv_kv > vn_lv_kv strictly
            windings.sort(key=lambda w: (w["nomv"], w["rata"]), reverse=True)
            w_hv, w_mv, w_lv = windings

            # Correct mesh-to-star-to-per-winding conversion
            z = mesh_to_star_per_winding(
                r12, x12, sbase12,
                r23, x23, sbase23,
                r31, x31, sbase31,
                sn_hv_mva = max(w_hv["rata"], 1.0),
                sn_mv_mva = max(w_mv["rata"], 1.0),
                sn_lv_mva = max(w_lv["rata"], 1.0),
            )

            rec_3w.append({
                # Identifiers
                "from_bus":     from_bus, "to_bus":   to_bus,    "mid_bus":  k_bus,
                "ckt":          ckt,      "name":     name,      "status":   status,
                # Role-sorted per-winding data
                "hv_bus":       w_hv["bus"], "mv_bus":   w_mv["bus"], "lv_bus":   w_lv["bus"],
                "vn_hv_kv":     w_hv["nomv"], "vn_mv_kv": w_mv["nomv"], "vn_lv_kv": w_lv["nomv"],
                "sn_hv_mva":    max(w_hv["rata"], 1.0),
                "sn_mv_mva":    max(w_mv["rata"], 1.0),
                "sn_lv_mva":    max(w_lv["rata"], 1.0),
                "shift_hv_deg": w_hv["ang"],
                "shift_mv_deg": w_mv["ang"],
                "shift_lv_deg": w_lv["ang"],
                # Star-equivalent impedances on system base (audit)
                "r_hv_pu": z["r_hv_pu"], "x_hv_pu": z["x_hv_pu"],
                "r_mv_pu": z["r_mv_pu"], "x_mv_pu": z["x_mv_pu"],
                "r_lv_pu": z["r_lv_pu"], "x_lv_pu": z["x_lv_pu"],
                # Pre-computed pandapower parameters (correct base)
                "vk_hv_pct":  z["vk_hv_pct"],  "vkr_hv_pct": z["vkr_hv_pct"],
                "vk_mv_pct":  z["vk_mv_pct"],  "vkr_mv_pct": z["vkr_mv_pct"],
                "vk_lv_pct":  z["vk_lv_pct"],  "vkr_lv_pct": z["vkr_lv_pct"],
                # Original PSS/E mesh values (audit / debug)
                "r12_pu": r12, "x12_pu": x12, "sbase12_mva": sbase12,
                "r23_pu": r23, "x23_pu": x23, "sbase23_mva": sbase23,
                "r31_pu": r31, "x31_pu": x31, "sbase31_mva": sbase31,
                "vmstar": vmstar, "anstar": anstar,
            })
            n_3w += 1
            i    += 5
        else:
            r12     = get_float(line2, 0)
            x12     = get_float(line2, 1)
            if abs(x12) < X_ZERO_THRESHOLD:
                x12 = 1e-6
            sbase12 = get_float(line2, 2, SYSTEM_MVA)

            windv1, nomv1, ang1 = (get_float(line3, 0, 1.0),
                                    get_float(line3, 1, 0.0),
                                    get_float(line3, 2, 0.0))
            rata1, ratb1, ratc1 = (get_float(line3, 3, 0.0),
                                    get_float(line3, 4, 0.0),
                                    get_float(line3, 5, 0.0))
            windv2, nomv2 = (get_float(line4, 0, 1.0),
                              get_float(line4, 1, 0.0))

            rec_2w.append({
                "from_bus":  from_bus, "to_bus":  to_bus, "k_bus": 0,
                "ckt":       ckt,      "name":    name,   "status": status,
                "r12_pu":    r12,      "x12_pu":  x12,    "sbase12_mva": sbase12,
                "windv1":    windv1,   "nomv1":   nomv1,  "ang1_deg":    ang1,
                "rata1_mva": rata1,    "ratb1_mva": ratb1, "ratc1_mva":  ratc1,
                "windv2":    windv2,   "nomv2":   nomv2,
            })
            n_2w += 1
            i    += 4

    log.info(f"  Parsed {n_2w} two-winding and {n_3w} three-winding transformers")
    return pd.DataFrame(rec_2w), pd.DataFrame(rec_3w)


# ==============================================================================
# Switched and fixed shunts
# ==============================================================================
def parse_switched_shunts(lines: list) -> pd.DataFrame:
    """
    Parse the switched shunt section. Sign convention:
        BINIT > 0   capacitive  (injects Mvar)
        BINIT < 0   inductive   (absorbs Mvar)
    """
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        parts = split_psse_csv(line)
        if len(parts) < 10:
            continue
        try:
            records.append({
                "bus_i":      int(parts[0]),
                "modsw":      get_int(parts, 1, 0),
                "adjm":       get_int(parts, 2, 0),
                "status":     get_int(parts, 3, 1),
                "vswhi":      get_float(parts, 4, 1.0),
                "vswlo":      get_float(parts, 5, 1.0),
                "swrem":      get_int(parts, 6, 0),
                "rmpct":      get_float(parts, 7, 100.0),
                "binit_mvar": get_float(parts, 9, 0.0),
            })
        except (IndexError, ValueError):
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        active = df[df["status"] == 1]
        log.info(f"  Parsed {len(df)} switched shunts "
                 f"(active: {len(active)}, total: {active['binit_mvar'].sum():+.0f} Mvar)")
    return df


def parse_fixed_shunts(lines: list) -> pd.DataFrame:
    """Parse the fixed shunt section."""
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        parts = split_psse_csv(line)
        if len(parts) < 5:
            continue
        try:
            records.append({
                "bus_i":   int(parts[0]),
                "id":      parts[1].strip().strip("'"),
                "status":  get_int(parts, 2, 1),
                "gl_mw":   get_float(parts, 3, 0.0),
                "bl_mvar": get_float(parts, 4, 0.0),
            })
        except (IndexError, ValueError):
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        active = df[df["status"] == 1]
        log.info(f"  Parsed {len(df)} fixed shunts "
                 f"(active: {len(active)}, total: {active['bl_mvar'].sum():+.0f} Mvar)")
    return df


# ==============================================================================
# Audit findings
# ==============================================================================
def audit_findings(df_3w: pd.DataFrame, df_sw: pd.DataFrame,
                   df_fs: pd.DataFrame, bus_kv: dict) -> dict:
    """Generate summary statistics for the verdict section."""
    findings = {
        "n_3w_total":  len(df_3w),
        "n_3w_active": int((df_3w["status"] == 1).sum()) if not df_3w.empty else 0,
    }

    if not df_3w.empty:
        ehv_count = 0
        for _, row in df_3w.iterrows():
            kvs = [bus_kv.get(int(row["hv_bus"]), 0),
                   bus_kv.get(int(row["mv_bus"]), 0),
                   bus_kv.get(int(row["lv_bus"]), 0)]
            if max(kvs) >= 300.0:
                ehv_count += 1
        findings["n_3w_reaching_ehv"] = ehv_count
        findings["vk_hv_min_pct"]     = float(df_3w["vk_hv_pct"].min())
        findings["vk_hv_median_pct"]  = float(df_3w["vk_hv_pct"].median())
        findings["vk_hv_max_pct"]     = float(df_3w["vk_hv_pct"].max())

    findings["n_switched_shunts"]         = len(df_sw)
    findings["n_switched_active"]         = int((df_sw["status"] == 1).sum()) if not df_sw.empty else 0
    findings["total_switched_mvar_active"] = (
        float(df_sw.loc[df_sw["status"] == 1, "binit_mvar"].sum())
        if not df_sw.empty else 0.0)

    findings["n_fixed_shunts"] = len(df_fs)
    findings["n_fixed_active"] = int((df_fs["status"] == 1).sum()) if not df_fs.empty else 0
    findings["total_fixed_mvar_active"] = (
        float(df_fs.loc[df_fs["status"] == 1, "bl_mvar"].sum())
        if not df_fs.empty else 0.0)

    return findings


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    log.info("=" * 78)
    log.info("  STEP 01b - PSS/E PARSER SUPPLEMENT")
    log.info("=" * 78)

    if not RAW_FILE.exists():
        log.error(f"Raw file not found: {RAW_FILE}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading: {RAW_FILE}")
    raw_text = RAW_FILE.read_text(encoding=ENCODING)
    log.info(f"  {len(raw_text):,} characters | {raw_text.count(chr(10)):,} lines")

    log.info("\n>> Locating PSS/E sections")
    sections = locate_sections(raw_text)
    lines    = raw_text.splitlines()
    for name, (start, end) in sections.items():
        log.info(f"  {name:<16s} : lines {start:>5d}-{end:<5d} ({end-start} records)")

    # Load bus voltages from Step 01's buses.csv for nomv fallback
    bus_kv    = {}
    buses_csv = OUT_DIR / "buses.csv"
    if buses_csv.exists():
        try:
            buses_df = pd.read_csv(buses_csv, encoding="utf-8-sig")
            bus_kv   = dict(zip(buses_df["bus_i"].astype(int),
                                buses_df["base_kv"].astype(float)))
            log.info(f"\n  Loaded {len(bus_kv)} bus voltages from {buses_csv.name}")
        except Exception as e:
            log.warning(f"  Failed to read buses.csv for kV lookup: {e}")

    log.info("\n>> Parsing transformers (2W and 3W, correct base conversion)")
    if "transformer" in sections:
        start, end   = sections["transformer"]
        df_2w, df_3w = parse_transformers_split(lines[start:end], bus_kv)
    else:
        log.warning("  Transformer section not found")
        df_2w, df_3w = pd.DataFrame(), pd.DataFrame()

    log.info("\n>> Parsing switched shunts")
    if "switched_shunt" in sections:
        start, end = sections["switched_shunt"]
        df_sw      = parse_switched_shunts(lines[start:end])
    else:
        df_sw = pd.DataFrame()

    log.info("\n>> Parsing fixed shunts")
    if "fixed_shunt" in sections:
        start, end = sections["fixed_shunt"]
        df_fs      = parse_fixed_shunts(lines[start:end])
    else:
        df_fs = pd.DataFrame()

    log.info("\n>> Writing CSV outputs")
    if not df_2w.empty:
        path = OUT_DIR / "transformers_2w.csv"
        df_2w.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s} : {len(df_2w)} records")

    if not df_3w.empty:
        path = OUT_DIR / "transformers_3w.csv"
        df_3w.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s} : {len(df_3w)} records  (vk_pct columns included)")

    if not df_sw.empty:
        path = OUT_DIR / "switched_shunts.csv"
        df_sw.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s} : {len(df_sw)} records")

    if not df_fs.empty:
        path = OUT_DIR / "fixed_shunts.csv"
        df_fs.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  {path.name:<25s} : {len(df_fs)} records")

    log.info("\n>> Audit findings")
    findings = audit_findings(df_3w, df_sw, df_fs, bus_kv)
    for key, value in findings.items():
        if isinstance(value, float):
            log.info(f"  {key:<35s} : {value:.3f}")
        else:
            log.info(f"  {key:<35s} : {value}")

    (OUT_DIR / "step01b_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8")

    log.info("\n" + "=" * 78)
    log.info("  VERDICT")
    log.info("=" * 78)
    if findings.get("n_3w_total", 0) > 0:
        log.info(f"  3-winding transformers parsed     : {findings['n_3w_total']}")
        log.info(f"    reaching 345 kV                 : {findings.get('n_3w_reaching_ehv', 0)}")
        log.info(f"    active in base case             : {findings.get('n_3w_active', 0)}")
        log.info("")
        log.info("  vk_hv_percent statistics (per-winding base, correct conversion):")
        log.info(f"    min    : {findings.get('vk_hv_min_pct',    0):.2f} %")
        log.info(f"    median : {findings.get('vk_hv_median_pct', 0):.2f} %")
        log.info(f"    max    : {findings.get('vk_hv_max_pct',    0):.2f} %")
        log.info("")
        log.info("  Healthy range for transformer vk_hv_percent is typically [5, 20] %.")
        log.info("  These pre-computed values are ready for Step 06 -- no further base")
        log.info("  conversion is needed downstream.")

    if findings.get("total_switched_mvar_active", 0) != 0:
        log.info(f"  Switched shunt capacity (active)  : "
                 f"{findings['total_switched_mvar_active']:+.0f} Mvar")
    if findings.get("total_fixed_mvar_active", 0) != 0:
        log.info(f"  Fixed shunt capacity (active)     : "
                 f"{findings['total_fixed_mvar_active']:+.0f} Mvar")
    log.info("=" * 78)


if __name__ == "__main__":
    main()