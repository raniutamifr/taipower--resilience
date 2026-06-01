"""
Step 03 — Parse Generator Cost Data (發電處_各機組成本資料.xlsx)
================================================================
Reads the ACTUAL Taipower cost XLSX (verified from real data):

Sheet 1 — 起動成本 (Startup Costs): 86 generator units
  Columns: 機組名稱, 裝置容量(kWh), 暖機啟動成本($/次)
  Capacity note: column header says kWh but values are in kW (rated capacity)
  Cost unit: NT$ per warm-start

Sheet 2 — 各機組曲線 (Heat Rate Curves): 37 units
  Columns: 機組, 熱耗率曲線
  Format: "y = ax2 + bx + c"  [kcal/kWh, x in MW]
  Excel shows "2E-07x2" = 2e-7 * x^2 (scientific notation parsed correctly)

AC-OPF Cost Polynomial (NT$/hr):
  Input fuel cost: HR(P)[kcal/kWh] * P[MW] * 1000[kWh/MWh] * fp[NT$/kcal]
  pandapower poly_cost format: [c2, c1, c0] where cost = c2*P^2 + c1*P + c0
    c2 = a * 1000 * fp  [NT$/hr/MW^2]
    c1 = b * 1000 * fp  [NT$/hr/MW]
    c0 = c * 1000 * fp  [NT$/hr]  (no-load intercept)

Fuel prices (NT$/kcal):
  coal (台中, 協和, 林口, 大林): 0.00055
  gas  (大潭, 通霄, 興達CC, 南部): 0.00120
  oil  (協和 backup): 0.00180
"""

import re
import json
import logging
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = Path(r"C:\reXplan-repo\Project Taipower\Data")
OUT_DIR   = Path(r"C:\reXplan-repo\Project Taipower\Results\step03")
COST_FILE = DATA_DIR / "發電處_各機組成本資料.xlsx"

FUEL_TYPE_MAP = {
    "協和": "oil",
    "林口": "coal",
    "大潭": "gas",
    "通霄": "gas",
    "台中": "coal",
    "興達": "gas",
    "南部": "gas",
    "大林": "coal",
}

FUEL_PRICE = {"coal": 0.00055, "gas": 0.00120, "oil": 0.00180}
VOLL_NTD_PER_MWH = 6_000_000


def get_fuel(name: str) -> str:
    for prefix, ft in FUEL_TYPE_MAP.items():
        if str(name).startswith(prefix):
            return ft
    log.warning(f"Unknown fuel type for unit '{name}', defaulting to 'gas'")
    return "gas"


def parse_startup_costs(wb) -> pd.DataFrame:
    ws   = wb["起動成本"]
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        name, cap, cost = row[0], row[1], row[2]
        if name is None:
            continue
        try:
            cap_kw = float(cap) if cap not in (None, "-", "") else None
        except (ValueError, TypeError):
            cap_kw = None
        try:
            cost_ntd = float(cost) if cost not in (None, "-", "") else 0.0
        except (ValueError, TypeError):
            cost_ntd = 0.0
        rows.append({
            "unit_name":        str(name).strip(),
            "capacity_kw":      cap_kw,
            "capacity_mw":      round(cap_kw / 1000.0, 3) if cap_kw else None,
            "startup_cost_ntd": cost_ntd,
            "fuel_type":        get_fuel(name),
        })
    df = pd.DataFrame(rows)
    log.info(f"Startup costs: {len(df)} units parsed")
    return df


def parse_heat_rate_formula(formula: str):
    """
    Parse "y = ax2 + bx + c" → (a, b, c) floats.
    Handles standard notation, scientific notation (e.g. 2E-07x2),
    negative coefficients, and leading terms without an explicit sign.

    Fix 1: Strip all whitespace from the expression BEFORE running the
            main regex so that tokens like "2E-07x2" are not split.
    Fix 2: Use a regex that matches optional leading sign independently,
            allowing the first term (no preceding operator) to be captured.
    Fix 3: Correctly handle the scientific-notation sign (e.g. 2E-07)
            without confusing the minus in the exponent with a subtraction
            operator — achieved by requiring the variable part to follow
            directly after the numeric literal.
    """
    # Strip "y =" prefix and ALL whitespace in one pass before tokenising.
    s = re.sub(r"^y\s*=\s*", "", formula.strip())
    s = s.replace(" ", "")

    # Pattern explanation:
    #   ([+\-]?)          – optional leading sign (+ or -)
    #   ([\d]+(?:\.\d+)?  – integer or decimal mantissa
    #    (?:[Ee][+\-]?\d+)?) – optional scientific exponent (handles 2E-07)
    #   (x2|x)?           – optional variable part: x² or x
    #
    # Using re.finditer so we can distinguish the very first token (no sign
    # required) from subsequent tokens that need an explicit + or -.
    pattern = re.compile(
        r"([+\-]?)([\d]+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)(x2|x)?"
    )

    a = b = c = 0.0
    for m in pattern.finditer(s):
        sign_s, num_s, var = m.group(1), m.group(2), m.group(3)
        # Skip if the match starts mid-exponent (e.g. the "-07" in "2E-07")
        # by checking that the character before the match (if any) is not
        # an 'E' or 'e'.
        start = m.start()
        if start > 0 and s[start - 1] in ("E", "e"):
            continue
        try:
            coef = float((sign_s or "+") + num_s)
        except ValueError:
            continue
        if var == "x2":
            a = coef
        elif var == "x":
            b = coef
        else:
            c = coef

    return a, b, c


def parse_heat_rate_curves(wb) -> pd.DataFrame:
    ws   = wb["各機組曲線"]
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 2:
            continue    # skip header + blank row
        name, formula = row[0], row[1]
        if name is None or formula is None:
            continue
        a, b, c = parse_heat_rate_formula(str(formula))
        fuel = get_fuel(str(name))
        fp   = FUEL_PRICE[fuel]
        rows.append({
            "unit_name":           str(name).strip(),
            "formula_raw":         str(formula).strip(),
            "hr_a":                a,
            "hr_b":                b,
            "hr_c":                c,
            "fuel_type":           fuel,
            "fuel_price_ntd_kcal": fp,
            # AC-OPF poly_cost: cost(P) = c2*P^2 + c1*P + c0  [NT$/hr]
            "opf_c2": a * 1000.0 * fp,
            "opf_c1": b * 1000.0 * fp,
            "opf_c0": c * 1000.0 * fp,
        })
    df = pd.DataFrame(rows)
    log.info(f"Heat rate curves: {len(df)} units parsed")
    return df


def merge_cost_tables(startup_df: pd.DataFrame, hr_df: pd.DataFrame) -> pd.DataFrame:
    merged = startup_df.merge(hr_df, on="unit_name", how="left",
                              suffixes=("", "_hr"))

    # Default heat rate coefficients (a, b, c) for units missing a curve.
    DEFAULT_HR = {
        "coal": (0.0, 2.0, 150.0),
        "gas":  (0.0, 1.4, 200.0),
        "oil":  (0.0, 2.2, 120.0),
    }

    # Ensure poly_cost columns exist even when the right side had no matches.
    for col in ("opf_c2", "opf_c1", "opf_c0", "fuel_price_ntd_kcal"):
        if col not in merged.columns:
            merged[col] = float("nan")

    # FIX: use merged.at[idx, col] to read Series values, not row.get().
    for idx, row in merged.iterrows():
        if pd.isna(row["opf_c2"]):
            fuel = row["fuel_type"]
            fp   = FUEL_PRICE.get(fuel, 0.0012)
            a, b, c = DEFAULT_HR.get(fuel, (0.0, 1.5, 180.0))
            merged.at[idx, "opf_c2"] = a * 1000 * fp
            merged.at[idx, "opf_c1"] = b * 1000 * fp
            merged.at[idx, "opf_c0"] = c * 1000 * fp
            merged.at[idx, "fuel_price_ntd_kcal"] = fp

    return merged


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Reading: {COST_FILE}")
    wb = load_workbook(COST_FILE, read_only=True)
    startup_df = parse_startup_costs(wb)
    hr_df      = parse_heat_rate_curves(wb)
    wb.close()
    merged_df  = merge_cost_tables(startup_df, hr_df)

    startup_df.to_csv(OUT_DIR / "startup_costs.csv", index=False, encoding="utf-8-sig")
    hr_df.to_csv(OUT_DIR / "heat_rate_curves.csv",   index=False, encoding="utf-8-sig")
    merged_df.to_csv(OUT_DIR / "cost_parameters.csv", index=False, encoding="utf-8-sig")

    meta = {
        "n_startup_units":   len(startup_df),
        "n_hr_curve_units":  len(hr_df),
        "n_merged":          len(merged_df),
        "fuel_counts":       startup_df["fuel_type"].value_counts().to_dict(),
        "voll_ntd_mwh":      VOLL_NTD_PER_MWH,
        "fuel_prices":       FUEL_PRICE,
        "units_with_hr_curve": hr_df["unit_name"].tolist(),
    }
    (OUT_DIR / "cost_meta.json").write_text(
        json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("  COST DATA SUMMARY — Step 03 (AC-OPF ready)")
    print("=" * 65)
    print(f"  Startup cost units   : {len(startup_df)}")
    print(f"  Heat rate curves     : {len(hr_df)}")
    print(f"  Fuel breakdown       : {startup_df['fuel_type'].value_counts().to_dict()}")
    print(f"  VOLL                 : {VOLL_NTD_PER_MWH:,.0f} NT$/MWh")
    print(f"  poly_cost format     : [c2, c1, c0]  (NT$/hr, MW units)")
    print(f"  Outputs → {OUT_DIR}")
    print("=" * 65)
    return merged_df


if __name__ == "__main__":
    main()