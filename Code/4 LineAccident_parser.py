"""
Step 04 — Parse Line/Transformer Accident Data (線路變壓器事故.xlsx)
====================================================================
Reads the ACTUAL Taipower accident XLSX (verified from real data):

Sheet 1 — 變壓器事故 (Transformer Accidents): 34 rows
  Columns: 災害名稱, 區處, 變壓器, 電壓別, 事故時間, 修復時間
  Voltage codes: G=161kV, S=69kV, D=distribution, P=power, C=subtransmission

Sheet 2 — 線路事故 (Line Accidents): 187 rows
  Columns: 災害名稱, 區處, 線路名稱, 相別, 事故時間, 修復時間
  Phase codes: R, S, T (3-phase), RST (all phases), combined strings

Disasters (with real dates from file):
  丹娜絲 (Typhoon Dana)    — July 6-8, 2025   — severity weight: 2.0
  康芮  (Typhoon Kong-rey) — Oct 31-Nov 1, 2024 — severity weight: 3.0
  凱米  (Typhoon Gaemi)    — July 24-25, 2024  — severity weight: 2.5
  小犬  (Typhoon Koinu)    — Oct 4-8, 2023     — severity weight: 1.5

Hazard Factor Calculation:
  For each unique line/transformer, compute total outage hours per disaster.
  HF = (base_lambda + storm_additional_lambda) / base_lambda
  Where storm_additional_lambda is proportional to disaster severity and
  fraction of year affected.

  Baseline failure rates (outages/year):
    G (161kV): λ_base = 0.015
    S (69kV):  λ_base = 0.025
    D (dist):  λ_base = 0.050
    Line:      λ_base = 0.100 per 100km (approximated)

  HF capped at 10.0 for extremely vulnerable components.
"""

import json
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR      = Path(r"C:\reXplan-repo\Project Taipower\Data")
OUT_DIR       = Path(r"C:\reXplan-repo\Project Taipower\Results\step04")
ACCIDENT_FILE = DATA_DIR / "線路變壓器事故.xlsx"

# Real disasters from the XLSX data
DISASTER_INFO = {
    "丹娜絲": {
        "en_name":   "Typhoon Dana",
        "year":      2025,
        "severity":  2.0,
        "period_hr": 48,
    },
    "康芮": {
        "en_name":   "Typhoon Kong-rey",
        "year":      2024,
        "severity":  3.0,
        "period_hr": 36,
    },
    "凱米": {
        "en_name":   "Typhoon Gaemi",
        "year":      2024,
        "severity":  2.5,
        "period_hr": 30,
    },
    "小犬": {
        "en_name":   "Typhoon Koinu",
        "year":      2023,
        "severity":  1.5,
        "period_hr": 96,
    },
}

VOLTAGE_BASE_LAMBDA = {
    "G": 0.015,
    "P": 0.015,
    "S": 0.025,
    "D": 0.050,
    "C": 0.035,
}
LINE_BASE_LAMBDA = 0.10
HF_CAP = 10.0

# Storm lambda scaling: outage fraction of year × severity × base_λ × scale
# scale = 8760 / period_hr gives the annualized rate; use per-disaster period_hr
# so each disaster's contribution is correctly normalized to one year.
STORM_SCALE = 8760.0   # hours per year, used in lambda normalization


def read_accidents(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both sheets, return (trafo_df, line_df).

    FIX 1: `header_rows` parameter lets each sheet declare how many leading
    rows to skip (变压器 has 1 header row; 線路 has 2 per the docstring).
    FIX 2: Guard against sheets with fewer columns than expected by padding
    each row with None before zipping so no column is silently dropped.
    """
    wb = load_workbook(filepath, read_only=True)

    def sheet_to_df(sheet_name: str, columns: list[str], header_rows: int = 1) -> pd.DataFrame:
        ws = wb[sheet_name]
        rows = []
        n_cols = len(columns)
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i < header_rows:
                continue
            if row[0] is None:
                # Blank sentinel row — skip rather than treat as data
                continue
            # Pad short rows so zip always produces all expected columns
            padded = list(row) + [None] * max(0, n_cols - len(row))
            rows.append(dict(zip(columns, padded[:n_cols])))
        return pd.DataFrame(rows)

    trafo_cols = ["disaster", "district", "transformer", "voltage_code",
                  "fault_time", "restore_time"]
    line_cols  = ["disaster", "district", "line_name", "phase",
                  "fault_time", "restore_time"]

    # 變壓器事故 has 1 header row; 線路事故 has 2 (header + blank separator)
    trafo_df = sheet_to_df("變壓器事故", trafo_cols, header_rows=1)
    line_df  = sheet_to_df("線路事故",   line_cols,  header_rows=2)

    wb.close()
    log.info(f"Transformer accidents: {len(trafo_df)} events")
    log.info(f"Line accidents       : {len(line_df)} events")
    return trafo_df, line_df


def compute_outage_hours(df: pd.DataFrame) -> float:
    """Sum total outage hours from fault_time / restore_time columns.

    FIX: use >= 0 so zero-duration (instant restore) events are counted
    rather than silently dropped. Also log a warning for negative durations
    (data entry errors) rather than silently ignoring them.
    """
    total = 0.0
    for _, row in df.iterrows():
        ft = row["fault_time"]
        rt = row["restore_time"]
        if isinstance(ft, datetime) and isinstance(rt, datetime):
            dur = (rt - ft).total_seconds() / 3600.0
            if dur >= 0:
                total += dur
            else:
                log.warning(
                    f"Negative outage duration ({dur:.2f}h) for "
                    f"'{row.get('transformer', row.get('line_name', '?'))}' "
                    f"fault={ft} restore={rt} — skipped"
                )
    return total


def compute_hazard_factors(trafo_df: pd.DataFrame,
                           line_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-component hazard factor.
    HF = 1 + storm_lambda / base_lambda

    FIX: storm_lambda is now normalized by the disaster's own period_hr so
    each typhoon's contribution is scaled to its actual duration rather than
    relying on a magic constant. The formula is:
        storm_λ = severity × (outage_hr / period_hr) × base_λ
    i.e., "what fraction of this storm's window was the component down,
    weighted by storm severity, expressed as a multiple of the base rate".
    """
    records = []

    # ── Transformers ──────────────────────────────────────────────────────
    for (disaster, voltage, trafo), grp in trafo_df.groupby(
            ["disaster", "voltage_code", "transformer"]):
        info      = DISASTER_INFO.get(disaster, {"severity": 1.0, "period_hr": 24,
                                                  "en_name": disaster})
        severity  = info["severity"]
        period_hr = info["period_hr"]
        base_λ    = VOLTAGE_BASE_LAMBDA.get(str(voltage), 0.035)
        n_events  = len(grp)
        outage_hr = compute_outage_hours(grp)

        # Fraction of the storm window the component was offline × severity
        storm_fraction = outage_hr / period_hr
        storm_λ = severity * storm_fraction * base_λ

        hf = min(1.0 + storm_λ / base_λ, HF_CAP)
        records.append({
            "comp_type":    "transformer",
            "comp_name":    str(trafo).strip(),
            "disaster":     disaster,
            "disaster_en":  info.get("en_name", disaster),
            "voltage_code": str(voltage),
            "district":     grp["district"].iloc[0],
            "n_events":     n_events,
            "outage_hours": round(outage_hr, 2),
            "base_lambda":  base_λ,
            "storm_lambda": round(storm_λ, 6),
            "hazard_factor": round(hf, 4),
        })

    # ── Lines ─────────────────────────────────────────────────────────────
    for (disaster, line_name), grp in line_df.groupby(["disaster", "line_name"]):
        info      = DISASTER_INFO.get(disaster, {"severity": 1.0, "period_hr": 24,
                                                  "en_name": disaster})
        severity  = info["severity"]
        period_hr = info["period_hr"]
        n_events  = len(grp)
        outage_hr = compute_outage_hours(grp)

        storm_fraction = outage_hr / period_hr
        storm_λ = severity * storm_fraction * LINE_BASE_LAMBDA

        hf = min(1.0 + storm_λ / LINE_BASE_LAMBDA, HF_CAP)
        records.append({
            "comp_type":    "line",
            "comp_name":    str(line_name).strip(),
            "disaster":     disaster,
            "disaster_en":  info.get("en_name", disaster),
            "voltage_code": "L",
            "district":     grp["district"].iloc[0],
            "n_events":     n_events,
            "outage_hours": round(outage_hr, 2),
            "base_lambda":  LINE_BASE_LAMBDA,
            "storm_lambda": round(storm_λ, 6),
            "hazard_factor": round(hf, 4),
        })

    hf_df = pd.DataFrame(records)
    log.info(f"Hazard factors computed: {len(hf_df)} component-disaster pairs")
    return hf_df


def aggregate_component_hf(hf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across all disasters: max HF per component.

    FIX: group by (comp_type, comp_name, voltage_code) so that a transformer
    and a line that happen to share a name are never merged, and so that
    base_lambda is always taken from the correct voltage tier rather than
    whichever row appeared first.
    """
    agg = (hf_df.groupby(["comp_type", "comp_name", "voltage_code"])
           .agg(
               max_hf=("hazard_factor", "max"),
               sum_storm_lambda=("storm_lambda", "sum"),
               base_lambda=("base_lambda", "first"),
               n_disasters=("disaster", "nunique"),
               total_outage_hr=("outage_hours", "sum"),
               disasters=("disaster", lambda x: ", ".join(sorted(x.unique()))),
           )
           .reset_index()
    )
    agg["composite_hf"] = (
        (1.0 + agg["sum_storm_lambda"] / agg["base_lambda"])
        .clip(upper=HF_CAP)
        .round(4)
    )
    return agg


def make_disaster_stats(trafo_df: pd.DataFrame,
                        line_df: pd.DataFrame) -> dict:
    """
    FIX: Guard against empty DataFrames before renaming and concatenating.
    An empty sheet produces an empty DataFrame with no columns, so the
    column-rename and subset-select would raise a KeyError.
    """
    frames = []
    if not trafo_df.empty:
        frames.append(
            trafo_df.rename(columns={"transformer": "comp_name"})
            [["disaster", "comp_name", "fault_time", "restore_time"]]
        )
    if not line_df.empty:
        frames.append(
            line_df.rename(columns={"line_name": "comp_name"})
            [["disaster", "comp_name", "fault_time", "restore_time"]]
        )

    if not frames:
        log.warning("No accident data found — disaster stats will be empty")
        return {}

    all_events = pd.concat(frames, ignore_index=True)
    stats = {}
    for disaster, grp in all_events.groupby("disaster"):
        info      = DISASTER_INFO.get(disaster, {})
        outage_hr = compute_outage_hours(grp)
        stats[disaster] = {
            "en_name":         info.get("en_name", disaster),
            "n_events":        len(grp),
            "n_unique_comps":  grp["comp_name"].nunique(),
            "total_outage_hr": round(outage_hr, 2),
            "severity_weight": info.get("severity", 1.0),
        }
    return stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Reading: {ACCIDENT_FILE}")
    trafo_df, line_df = read_accidents(ACCIDENT_FILE)

    # Save raw parsed tables
    trafo_df.to_csv(OUT_DIR / "transformer_accidents.csv", index=False, encoding="utf-8-sig")
    line_df.to_csv(OUT_DIR / "line_accidents.csv",         index=False, encoding="utf-8-sig")

    # Combined
    trafo_df["comp_type"] = "transformer"
    line_df["comp_type"]  = "line"
    all_df = pd.concat([trafo_df, line_df], ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_accidents.csv", index=False, encoding="utf-8-sig")

    # Compute hazard factors
    hf_detail = compute_hazard_factors(trafo_df, line_df)
    hf_agg    = aggregate_component_hf(hf_detail)

    hf_detail.to_csv(OUT_DIR / "hazard_factors_detail.csv", index=False, encoding="utf-8-sig")
    hf_agg.to_csv(OUT_DIR / "hazard_factors.csv",           index=False, encoding="utf-8-sig")

    # Disaster-level stats
    dis_stats = make_disaster_stats(trafo_df, line_df)
    (OUT_DIR / "disaster_statistics.json").write_text(
        json.dumps(dis_stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    # System defaults for components not in accident data
    system_defaults = {
        "default_line_lambda": LINE_BASE_LAMBDA,
        "default_trafo_lambda_by_voltage": VOLTAGE_BASE_LAMBDA,
        "hf_cap": HF_CAP,
        "disasters_covered": list(DISASTER_INFO.keys()),
    }
    (OUT_DIR / "system_hazard_defaults.json").write_text(
        json.dumps(system_defaults, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Print summary
    print("\n" + "=" * 65)
    print("  ACCIDENT DATA SUMMARY — Step 04")
    print("=" * 65)
    print(f"  Total accident events: {len(all_df)}")
    print(f"    Transformer: {len(trafo_df)}")
    print(f"    Line:        {len(line_df)}")
    print(f"\n  Disasters covered:")
    for dis, info in dis_stats.items():
        print(f"    {dis} ({info['en_name']}): "
              f"{info['n_events']} events, "
              f"{info['n_unique_comps']} components, "
              f"{info['total_outage_hr']:.1f}hr total outage")
    if not hf_agg.empty:
        print(f"\n  Hazard factor stats (composite, across all disasters):")
        print(f"    Max HF  : {hf_agg['composite_hf'].max():.3f}")
        print(f"    Mean HF : {hf_agg['composite_hf'].mean():.3f}")
        print(f"    >1.5 HF : {(hf_agg['composite_hf'] > 1.5).sum()} components")
    print(f"  Outputs → {OUT_DIR}")
    print("=" * 65)
    return hf_agg


if __name__ == "__main__":
    main()