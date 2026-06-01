"""
Step 00 — Convert Taipower parsed data → reXplan network.xlsx
=============================================================
Reads outputs dari step01 (PSSE parser), step03 (cost), step04 (hazard)
dan menghasilkan network.xlsx yang bisa langsung dibaca oleh:

    network = rx.network.Network('taipower')

Format network.xlsx mengikuti fields_map.csv dari reXplan package.

Sheet yang dihasilkan:
  bus       → dari step01/buses.csv
  line      → dari step01/branches.csv
  trafo     → dari step01/transformers.csv
  load      → dari step01/loads.csv
  gen       → dari step01/generators.csv + step03/cost_parameters.csv
  ext_grid  → slack bus (dari network_meta.json)

reXplan-specific fields yang ditambahkan:
  fragility_curve → mapped dari step04/hazard_factors.csv (composite_hf)
  kf              → vulnerability factor (default 1.0, scaled by HF)
  resilienceFull  → full restoration time (jam)
  weatherTTR      → weather-dependent TTR (jam)
  normalTTR       → normal TTR (jam)

Output:
  file/input/taipower/network.xlsx   ← langsung bisa dibaca reXplan
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"C:\reXplan-repo\Project Taipower")
RESULT_BASE = BASE_DIR / "Results"
STEP01_DIR  = RESULT_BASE / "step01"
STEP03_DIR  = RESULT_BASE / "step03"
STEP04_DIR  = RESULT_BASE / "step04"

# Output: langsung ke folder yang reXplan baca
REXPLAN_DIR = Path(r"C:\reXplan-repo\file\input\taipower")
OUT_FILE    = REXPLAN_DIR / "network.xlsx"

# Constants
BASE_MVA        = 100.0
FREQ_HZ         = 60.0
PSSE_SENTINEL   = 9000.0

# Default TTR (Time To Repair) dalam jam — berdasarkan standar Taipower
DEFAULT_TTR = {
    "bus":   {"normalTTR": 4.0,  "weatherTTR": 12.0, "resilienceFull": 24.0},
    "line":  {"normalTTR": 8.0,  "weatherTTR": 24.0, "resilienceFull": 72.0},
    "trafo": {"normalTTR": 24.0, "weatherTTR": 72.0, "resilienceFull": 168.0},
    "gen":   {"normalTTR": 48.0, "weatherTTR": 48.0, "resilienceFull": 720.0},
    "load":  {"normalTTR": 2.0,  "weatherTTR": 8.0,  "resilienceFull": 24.0},
}

# Default fragility curve names — sesuai dengan yang reXplan generate otomatis
DEFAULT_FC = {
    "line":       "towers_1",    # transmission tower
    "trafo":      "substation_1",
    "gen":        "generator_1",
    "bus":        "substation_1",
    "load":       "distribution_1",
}

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def sf(val, default=0.0) -> float:
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def load_hazard_lookup(hf_csv: Path) -> dict:
    """
    Build lookup: comp_name → {composite_hf, kf}
    kf = vulnerability factor, scaled dari composite_hf.
    HF=1.0 → kf=1.0 (normal), HF>1 → kf>1 (lebih rentan).
    """
    if not hf_csv.exists():
        return {}
    hf_df = pd.read_csv(hf_csv)
    lookup = {}
    for _, row in hf_df.iterrows():
        name = str(row.get("comp_name", "")).strip()
        hf   = float(row.get("composite_hf", 1.0) or 1.0)
        kf   = min(hf, 5.0)   # cap kf at 5.0 per reXplan convention
        lookup[name] = {"composite_hf": hf, "kf": kf}
    log.info(f"Hazard lookup built: {len(lookup)} components")
    return lookup


def load_cost_lookup(cost_csv: Path) -> dict:
    """Build lookup: unit_name_prefix → {cp2, cp1, cp0}"""
    if not cost_csv.exists():
        return {}
    df = pd.read_csv(cost_csv)
    lookup = {}
    for _, row in df.iterrows():
        name = str(row.get("unit_name", "")).strip()
        lookup[name] = {
            "cp2_eur_per_mw2": float(row.get("opf_c2", 0.0) or 0.0),
            "cp1_eur_per_mw":  float(row.get("opf_c1", 1000.0) or 1000.0),
            "cp0_eur":         float(row.get("opf_c0", 0.0) or 0.0),
        }
    return lookup


def match_cost(bus_name: str, cost_lookup: dict) -> dict:
    """Fuzzy match bus name ke cost data."""
    FUEL_DEFAULTS = {
        "大潭": (0.0, 1440.0, 0.0), "通霄": (0.0, 1440.0, 0.0),
        "興達": (0.0, 1440.0, 0.0), "南部": (0.0, 1440.0, 0.0),
        "台中": (0.0,  550.0, 0.0), "林口": (0.0,  550.0, 0.0),
        "大林": (0.0,  550.0, 0.0), "協和": (0.0, 1800.0, 0.0),
        "核一": (0.0,  300.0, 0.0), "核二": (0.0,  300.0, 0.0),
        "核三": (0.0,  300.0, 0.0), "麥寮": (0.0,  550.0, 0.0),
    }
    for unit, costs in cost_lookup.items():
        if len(unit) >= 2 and unit[:2] in str(bus_name):
            return costs
    for prefix, (c2, c1, c0) in FUEL_DEFAULTS.items():
        if prefix in str(bus_name):
            return {"cp2_eur_per_mw2": c2, "cp1_eur_per_mw": c1, "cp0_eur": c0}
    return {"cp2_eur_per_mw2": 0.0, "cp1_eur_per_mw": 1000.0, "cp0_eur": 0.0}


def match_hf(name: str, hf_lookup: dict) -> dict:
    """Fuzzy match component name ke hazard factor."""
    clean = str(name).strip()
    if clean in hf_lookup:
        return hf_lookup[clean]
    # Partial match
    for hf_name, vals in hf_lookup.items():
        if len(hf_name) >= 3 and (hf_name in clean or clean in hf_name):
            return vals
    return {"composite_hf": 1.0, "kf": 1.0}


# ──────────────────────────────────────────────────────────────────────────────
# Sheet builders
# ──────────────────────────────────────────────────────────────────────────────
def build_bus_sheet(buses_df: pd.DataFrame, isolated: set, hf_lookup: dict) -> pd.DataFrame:
    """
    reXplan bus sheet columns:
    name, index, vn_kv, longitude, latitude, zone,
    max_vm_pu, min_vm_pu, in_service,
    fragility_curve, kf, resilienceFull, weatherTTR, normalTTR
    """
    rows = []
    for _, row in buses_df.iterrows():
        bus_i = int(sf(row["bus_i"], -1))
        if bus_i in isolated:
            continue

        name   = str(row.get("name", f"Bus_{bus_i}")).strip()
        vn_kv  = max(sf(row.get("base_kv", 100.0), 100.0), 0.1)
        ide    = int(sf(row.get("ide", 1), 1))
        in_svc = (ide != 4)
        zone   = int(sf(row.get("zone", 1), 1))

        hf_data = match_hf(name, hf_lookup)
        kf      = hf_data["kf"]
        ttr     = DEFAULT_TTR["bus"]

        rows.append({
            "name":            name,
            "index":           bus_i,
            "vn_kv":           round(vn_kv, 3),
            "longitude":       sf(row.get("longitude", 120.0 + bus_i * 0.001), 120.0),
            "latitude":        sf(row.get("latitude",  23.0  + bus_i * 0.001), 23.0),
            "zone":            zone,
            "max_vm_pu":       1.15,
            "min_vm_pu":       0.85,
            "in_service":      in_svc,
            "fragility_curve": DEFAULT_FC["bus"],
            "kf":              round(kf, 4),
            "resilienceFull":  ttr["resilienceFull"],
            "weatherTTR":      ttr["weatherTTR"],
            "normalTTR":       ttr["normalTTR"],
        })

    df = pd.DataFrame(rows)
    log.info(f"Bus sheet: {len(df)} rows")
    return df


def build_line_sheet(branches_df: pd.DataFrame, bus_map: dict,
                     kv_lookup: dict, hf_lookup: dict) -> pd.DataFrame:
    """
    reXplan line sheet columns:
    name, from_bus, to_bus, length_km,
    r_ohm_per_km, x_ohm_per_km, c_nf_per_km,
    max_i_ka, in_service, parallel, max_loading_percent,
    fragility_curve, kf, resilienceFull, weatherTTR, normalTTR
    """
    omega = 2.0 * np.pi * FREQ_HZ
    rows  = []

    for _, row in branches_df.iterrows():
        fb = int(sf(row["from_bus"], -1))
        tb = int(sf(row["to_bus"],   -1))
        if fb not in bus_map or tb not in bus_map or fb == tb:
            continue

        r_pu = sf(row.get("r_pu", 0.0), 0.0)
        x_pu = sf(row.get("x_pu", 1e-4), 1e-4)
        if abs(x_pu) < 1e-9:
            x_pu = 1e-6
        b_pu = sf(row.get("b_pu", 0.0), 0.0)

        fkv    = kv_lookup.get(fb, 100.0)
        z_base = (fkv ** 2) / BASE_MVA
        r_ohm  = r_pu * z_base
        x_ohm  = x_pu * z_base
        c_nf   = ((b_pu / z_base) / (omega * 1e-9)) if b_pu > 0.0 else 0.0

        rate     = sf(row.get("rate_a_mva", 0.0), 0.0)
        max_i_ka = (9.9 if rate <= 0.001
                    else max(rate / (fkv * np.sqrt(3)), 200.0 / (fkv * np.sqrt(3))))

        in_svc = (int(sf(row.get("status", 1), 1)) == 1)
        name   = f"Line_{fb}_{tb}_{row.get('ckt', '1')}"

        hf_data = match_hf(name, hf_lookup)
        kf      = hf_data["kf"]
        ttr     = DEFAULT_TTR["line"]

        rows.append({
            "name":               name,
            "from_bus":           fb,
            "to_bus":             tb,
            "length_km":          1.0,
            "r_ohm_per_km":       max(round(r_ohm, 6), 0.0),
            "x_ohm_per_km":       max(round(x_ohm, 6), 1e-6),
            "c_nf_per_km":        max(round(c_nf,  6), 0.0),
            "r0_ohm_per_km":      max(round(r_ohm * 3, 6), 0.0),
            "x0_ohm_per_km":      max(round(x_ohm * 3, 6), 1e-6),
            "c0_nf_per_km":       max(round(c_nf  * 0.5, 6), 0.0),
            "max_i_ka":           round(max_i_ka, 4),
            "parallel":           1,
            "in_service":         in_svc,
            "max_loading_percent": 100.0,
            "df":                 1.0,
            "fragility_curve":    DEFAULT_FC["line"],
            "kf":                 round(kf, 4),
            "resilienceFull":     ttr["resilienceFull"],
            "weatherTTR":         ttr["weatherTTR"],
            "normalTTR":          ttr["normalTTR"],
        })

    df = pd.DataFrame(rows)
    log.info(f"Line sheet: {len(df)} rows")
    return df


def build_trafo_sheet(trafo_df: pd.DataFrame, bus_map: dict,
                      kv_lookup: dict, hf_lookup: dict) -> pd.DataFrame:
    """
    reXplan trafo sheet — standard pandapower trafo parameters
    + reXplan resilience fields
    """
    rows = []
    for _, row in trafo_df.iterrows():
        fb = int(sf(row["from_bus"], -1))
        tb = int(sf(row["to_bus"],   -1))
        if fb not in bus_map or tb not in bus_map or fb == tb:
            continue

        vn_fb = kv_lookup.get(fb, 345.0)
        vn_tb = kv_lookup.get(tb, 161.0)

        # Skip same-kV (bus couplers) — buat sebagai line biasa nanti
        if abs(vn_fb - vn_tb) < 0.1:
            continue

        # Pastikan HV > LV
        if vn_fb < vn_tb:
            fb, tb     = tb, fb
            vn_fb, vn_tb = vn_tb, vn_fb

        sn_mva  = max(sf(row.get("sbase12_mva", BASE_MVA), BASE_MVA), 1.0)
        r12     = sf(row.get("r12_pu", 0.0),  0.0)
        x12     = sf(row.get("x12_pu", 0.1),  0.1)
        ang_deg = sf(row.get("ang1_deg", 0.0), 0.0)
        in_svc  = (int(sf(row.get("status", 1), 1)) == 1)

        vk_pct  = float(np.clip(abs(x12) * 100.0, 0.5, 30.0))
        vkr_pct = float(np.clip(abs(r12) * 100.0, 0.0,  5.0))
        if vkr_pct >= vk_pct:
            vkr_pct = vk_pct * 0.99

        name    = f"Trafo_{fb}_{tb}_{row.get('ckt', '1')}"
        hf_data = match_hf(name, hf_lookup)
        kf      = hf_data["kf"]
        ttr     = DEFAULT_TTR["trafo"]

        rows.append({
            "name":               name,
            "node_p":             fb,    # HV bus (primary)
            "node_s":             tb,    # LV bus (secondary)
            "vn_hv_kv":          round(vn_fb, 3),
            "vn_lv_kv":          round(vn_tb, 3),
            "sn_mva":            round(sn_mva, 3),
            "vk_percent":        round(vk_pct, 4),
            "vkr_percent":       round(vkr_pct, 4),
            "pfe_kw":            0.0,
            "i0_percent":        0.0,
            "shift_degree":      round(ang_deg, 2),
            "tap_side":          "hv",
            "tap_neutral":       0,
            "tap_min":           -2,
            "tap_max":           2,
            "tap_step_percent":  1.25,
            "tap_step_degree":   0.0,
            "tap_phase_shifter": 0,
            "tap_pos":           0,
            "parallel":          1,
            "in_service":        in_svc,
            "max_loading_percent": 100.0,
            "df":                1.0,
            "fragility_curve":   DEFAULT_FC["trafo"],
            "kf":                round(kf, 4),
            "resilienceFull":    ttr["resilienceFull"],
            "weatherTTR":        ttr["weatherTTR"],
            "normalTTR":         ttr["normalTTR"],
        })

    df = pd.DataFrame(rows)
    log.info(f"Trafo sheet: {len(df)} rows")
    return df


def build_load_sheet(loads_df: pd.DataFrame, bus_map: dict,
                     hf_lookup: dict) -> pd.DataFrame:
    """reXplan load sheet"""
    rows = []
    for _, row in loads_df.iterrows():
        bus_i = int(sf(row["bus_i"], -1))
        if bus_i not in bus_map:
            continue
        pl = sf(row.get("pl_mw", 0.0), 0.0)
        ql = sf(row.get("ql_mvar", 0.0), 0.0)
        if pl == 0.0 and ql == 0.0:
            continue

        in_svc  = (int(sf(row.get("status", 1), 1)) == 1)
        name    = f"Load_{bus_i}_{row.get('id', '1')}"
        hf_data = match_hf(name, hf_lookup)
        ttr     = DEFAULT_TTR["load"]

        rows.append({
            "name":            name,
            "node":            bus_i,
            "p_mw":            max(round(pl, 4), 0.0),
            "q_mvar":          round(ql, 4),
            "const_z_percent": 0.0,
            "const_i_percent": 0.0,
            "scaling":         1.0,
            "in_service":      in_svc,
            "controllable":    False,
            "fragility_curve": DEFAULT_FC["load"],
            "kf":              round(hf_data["kf"], 4),
            "resilienceFull":  ttr["resilienceFull"],
            "weatherTTR":      ttr["weatherTTR"],
            "normalTTR":       ttr["normalTTR"],
        })

    df = pd.DataFrame(rows)
    log.info(f"Load sheet: {len(df)} rows  (total {df['p_mw'].sum():.0f} MW)")
    return df


def build_gen_sheet(gens_df: pd.DataFrame, bus_map: dict,
                    name_lookup: dict, cost_lookup: dict,
                    hf_lookup: dict) -> pd.DataFrame:
    """reXplan gen sheet — termasuk cost polynomial dari step03"""
    rows = []
    for _, row in gens_df.iterrows():
        bus_i = int(sf(row["bus_i"], -1))
        if bus_i not in bus_map:
            continue

        pmax_raw  = sf(row.get("pmax_mw", 0.0), 0.0)
        mbase_mva = sf(row.get("mbase_mva", 100.0), 100.0)
        pmax = (mbase_mva if pmax_raw >= PSSE_SENTINEL else max(pmax_raw, 0.0))
        if pmax <= 0.0:
            continue

        pmin_raw = sf(row.get("pmin_mw", 0.0), 0.0)
        pmin = 0.0 if pmin_raw <= -PSSE_SENTINEL else max(pmin_raw, 0.0)
        pmin = min(pmin, pmax)

        pg   = float(np.clip(sf(row.get("pg_mw", 0.0), 0.0), pmin, pmax))
        vm   = float(np.clip(sf(row.get("vs_pu", 1.0), 1.0), 0.85, 1.15))

        qmax = sf(row.get("qt_mvar", pmax * 0.6), pmax * 0.6)
        qmin = sf(row.get("qb_mvar", -pmax * 0.4), -pmax * 0.4)
        if qmax <= qmin:
            qmax, qmin = pmax * 0.6, -pmax * 0.4

        in_svc   = (int(sf(row.get("status", 1), 1)) == 1)
        bus_name = name_lookup.get(bus_i, "")
        name     = f"Gen_{bus_i}_{row.get('gen_id', '1')}"

        costs   = match_cost(bus_name, cost_lookup)
        hf_data = match_hf(bus_name, hf_lookup)
        ttr     = DEFAULT_TTR["gen"]

        rows.append({
            "name":              name,
            "node":              bus_i,
            "p_mw":              round(pg, 4),
            "vm_pu":             round(vm, 4),
            "max_p_mw":          round(pmax, 3),
            "min_p_mw":          round(pmin, 3),
            "max_q_mvar":        round(qmax, 3),
            "min_q_mvar":        round(qmin, 3),
            "controllable":      True,
            "in_service":        in_svc,
            "cp2_eur_per_mw2":   costs["cp2_eur_per_mw2"],
            "cp1_eur_per_mw":    costs["cp1_eur_per_mw"],
            "cp0_eur":           costs["cp0_eur"],
            "fragility_curve":   DEFAULT_FC["gen"],
            "kf":                round(hf_data["kf"], 4),
            "resilienceFull":    ttr["resilienceFull"],
            "weatherTTR":        ttr["weatherTTR"],
            "normalTTR":         ttr["normalTTR"],
        })

    df = pd.DataFrame(rows)
    log.info(f"Gen sheet: {len(df)} rows  "
             f"(total Pmax={df[df['in_service']]['max_p_mw'].sum():.0f} MW)")
    return df


def build_extgrid_sheet(slack_bus: int, buses_df: pd.DataFrame) -> pd.DataFrame:
    """reXplan ext_grid sheet — slack bus"""
    slack_row = buses_df[buses_df["bus_i"] == slack_bus]
    vm = 1.0
    if not slack_row.empty and "vm" in slack_row.columns:
        vm = float(np.clip(sf(slack_row.iloc[0]["vm"], 1.0), 0.85, 1.15))

    df = pd.DataFrame([{
        "name":        f"SlackGrid_{slack_bus}",
        "node":        slack_bus,
        "vm_pu":       round(vm, 4),
        "va_degree":   0.0,
        "s_sc_max_mva": 10000.0,
        "s_sc_min_mva": 5000.0,
        "rx_max":       0.1,
        "rx_min":       0.1,
        "r0x0_max":     0.1,
        "x0x_max":      1.0,
        "slack_weight": 1.0,
        "in_service":   True,
    }])
    log.info(f"ExtGrid sheet: 1 row (slack bus {slack_bus})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    REXPLAN_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load step01 data ──────────────────────────────────────────────────────
    log.info("Loading step01 CSV data...")
    buses_df    = pd.read_csv(STEP01_DIR / "buses.csv")
    gens_df     = pd.read_csv(STEP01_DIR / "generators.csv")
    branches_df = pd.read_csv(STEP01_DIR / "branches.csv")
    trafo_df    = pd.read_csv(STEP01_DIR / "transformers.csv")
    loads_df    = pd.read_csv(STEP01_DIR / "loads.csv")

    with open(STEP01_DIR / "network_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    slack_bus      = int(meta["slack_bus"])
    isolated_buses = set(meta.get("isolated_bus_list", []))

    log.info(f"Buses: {len(buses_df)}, isolated: {len(isolated_buses)}, slack: {slack_bus}")

    # ── Build bus lookup maps ─────────────────────────────────────────────────
    active = buses_df[~buses_df["bus_i"].isin(isolated_buses)]
    bus_map    = {int(r["bus_i"]): i for i, (_, r) in enumerate(active.iterrows())}
    kv_lookup  = {int(r["bus_i"]): max(sf(r["base_kv"], 100.0), 0.1)
                  for _, r in active.iterrows()}
    name_lookup = {int(r["bus_i"]): str(r.get("name", "")).strip()
                   for _, r in active.iterrows()}

    # ── Load step03 cost data ─────────────────────────────────────────────────
    cost_lookup = load_cost_lookup(STEP03_DIR / "cost_parameters.csv")
    if cost_lookup:
        log.info(f"Cost data loaded: {len(cost_lookup)} units")
    else:
        log.warning("cost_parameters.csv not found — using default costs")

    # ── Load step04 hazard factors ────────────────────────────────────────────
    hf_lookup = load_hazard_lookup(STEP04_DIR / "hazard_factors.csv")
    if not hf_lookup:
        log.warning("hazard_factors.csv not found — using default kf=1.0")

    # ── Build sheets ──────────────────────────────────────────────────────────
    log.info("Building reXplan network sheets...")
    bus_df  = build_bus_sheet(buses_df, isolated_buses, hf_lookup)
    line_df = build_line_sheet(branches_df, bus_map, kv_lookup, hf_lookup)
    trf_df  = build_trafo_sheet(trafo_df, bus_map, kv_lookup, hf_lookup)
    load_df = build_load_sheet(loads_df, bus_map, hf_lookup)
    gen_df  = build_gen_sheet(gens_df, bus_map, name_lookup, cost_lookup, hf_lookup)
    ext_df  = build_extgrid_sheet(slack_bus, buses_df)

    # ── Write network.xlsx ────────────────────────────────────────────────────
    log.info(f"Writing: {OUT_FILE}")
    with pd.ExcelWriter(str(OUT_FILE), engine="openpyxl") as writer:
        bus_df.to_excel(writer,  sheet_name="bus",      index=False)
        line_df.to_excel(writer, sheet_name="line",     index=False)
        trf_df.to_excel(writer,  sheet_name="trafo",    index=False)
        load_df.to_excel(writer, sheet_name="load",     index=False)
        gen_df.to_excel(writer,  sheet_name="gen",      index=False)
        ext_df.to_excel(writer,  sheet_name="ext_grid", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  REXPLAN NETWORK CONVERTER — Complete")
    print("=" * 65)
    print(f"  Output      : {OUT_FILE}")
    print(f"  Buses       : {len(bus_df)}")
    print(f"  Lines       : {len(line_df)}")
    print(f"  Transformers: {len(trf_df)}")
    print(f"  Loads       : {len(load_df)}  "
          f"({load_df['p_mw'].sum():.0f} MW total)")
    print(f"  Generators  : {len(gen_df)}  "
          f"({gen_df[gen_df['in_service']]['max_p_mw'].sum():.0f} MW Pmax)")
    print(f"  Ext grid    : {len(ext_df)}")
    print(f"\n  Siap dipakai di notebook:")
    print(f"    network = rx.network.Network('taipower')")
    print("=" * 65)

    return OUT_FILE


if __name__ == "__main__":
    main()