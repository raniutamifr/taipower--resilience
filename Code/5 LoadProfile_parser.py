"""
Step 05 — Parse Hourly Load Profile (114年淨發電量(小時平均).xlsx)
=================================================================
Sheet: 即時資料 (Real-time data)
Columns: 名稱 (name), 日期 (date YYYMMDD Minguo), 時間 (time HH:MM), VALUE (MW)

Data: Net generation (淨發電量) = system load proxy
     8760 hourly data points for Minguo Year 114 (Jan-Dec 2025)

Output:
  - load_profile_hourly.csv  : datetime, load_mw, load_pu (normalized 0-1)
  - load_statistics.csv      : seasonal and daily pattern stats
  - load_profile_8760.npy    : numpy array for fast loading in SMC engine
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import openpyxl

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
LOAD_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\114年淨發電量(小時平均).xlsx")
OUT_DIR   = Path(r"C:\reXplan-repo\Project Taipower\Results\step05")

MINGUO_OFFSET = 1911  # CE year = Minguo year + 1911


# ──────────────────────────────────────────────────────────────────────────────
# Date parser for Minguo calendar
# ──────────────────────────────────────────────────────────────────────────────
def parse_minguo_date(date_val, time_val) -> datetime:
    """
    Parse Minguo date like '1140101' (YYYmmDD) and time '00:00'.
    Minguo 114 year 1 month 1 day = 2025-01-01.
    """
    date_str = str(date_val).strip()
    time_str = str(time_val).strip() if time_val else "00:00"

    if len(date_str) == 7:
        # Format: YYYmmDD (Minguo 3-digit year)
        minguo_year = int(date_str[:3])
        month       = int(date_str[3:5])
        day         = int(date_str[5:7])
        ce_year     = minguo_year + MINGUO_OFFSET
    elif len(date_str) == 8:
        # Format: YYYYMMDD (already CE)
        ce_year = int(date_str[:4])
        month   = int(date_str[4:6])
        day     = int(date_str[6:8])
    else:
        raise ValueError(f"Unknown date format: {date_str}")

    hour, minute = map(int, time_str.split(":"))
    return datetime(ce_year, month, day, hour, minute)


# ──────────────────────────────────────────────────────────────────────────────
# Parse load data
# ──────────────────────────────────────────────────────────────────────────────
def parse_load_profile(load_file: Path) -> pd.DataFrame:
    """
    Parse net generation data as system load profile.
    Returns DataFrame with columns: datetime, load_mw
    """
    if not load_file.exists():
        log.error(f"Load file not found: {load_file}")
        return _generate_synthetic_load()

    wb = openpyxl.load_workbook(str(load_file))
    ws = wb.active

    records = []
    for row in ws.iter_rows(values_only=True):
        if row[0] is None:
            continue
        name_col = str(row[0]).strip()
        if name_col in ("名稱", "name", ""):
            continue  # skip header

        date_val  = row[1]
        time_val  = row[2]
        value_mw  = row[3]

        if value_mw is None:
            continue
        try:
            value_mw = float(value_mw)
        except (TypeError, ValueError):
            continue

        try:
            dt = parse_minguo_date(date_val, time_val)
        except (ValueError, TypeError):
            continue

        records.append({"datetime": dt, "load_mw": value_mw})

    df = pd.DataFrame(records)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop_duplicates("datetime").reset_index(drop=True)
    log.info(f"Parsed {len(df)} hourly load data points")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic load if file not available
# ──────────────────────────────────────────────────────────────────────────────
def _generate_synthetic_load() -> pd.DataFrame:
    """Generate realistic Taiwan load profile if real data unavailable."""
    log.warning("Generating synthetic Taiwan load profile")
    hours = pd.date_range("2025-01-01", periods=8760, freq="h")
    h = np.arange(8760)

    # Base load: 25,000 MW
    # Annual variation: summer peak (Jul-Aug) ~40,000 MW
    # Daily pattern: morning/evening peaks
    day_of_year = h // 24
    hour_of_day = h % 24

    # Annual seasonality (peak in summer, trough in spring/fall)
    seasonal = 5000 * np.sin(2 * np.pi * (day_of_year - 170) / 365)

    # Daily pattern
    daily = (
        2000 * np.sin(np.pi * (hour_of_day - 6) / 12) +
        1500 * np.sin(np.pi * (hour_of_day - 17) / 6) *
        np.clip(hour_of_day - 14, 0, 1) *
        np.clip(22 - hour_of_day, 0, 1)
    )
    daily = np.clip(daily, -3000, 3000)

    # Weekend reduction
    dow = (np.arange(8760) // 24) % 7  # 0=Monday
    weekend = np.where(dow >= 5, -2000, 0)

    load_mw = 28000 + seasonal + daily + weekend
    load_mw = np.clip(load_mw, 18000, 42000)  # Taiwan system bounds

    df = pd.DataFrame({"datetime": hours, "load_mw": load_mw})
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Pad / interpolate to ensure exactly 8760 hours
# ──────────────────────────────────────────────────────────────────────────────
def ensure_8760_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to exactly 8760 hourly observations for the year 2025.
    Interpolate any missing hours.
    """
    full_index = pd.date_range("2025-01-01 00:00", periods=8760, freq="h")
    df = df.set_index("datetime")
    df = df.reindex(full_index)
    df["load_mw"] = df["load_mw"].interpolate(method="linear").fillna(method="bfill").fillna(25000)
    df = df.reset_index().rename(columns={"index": "datetime"})
    log.info(f"Load profile padded/interpolated to {len(df)} hours")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Compute derived features
# ──────────────────────────────────────────────────────────────────────────────
def enrich_load_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features and normalized load."""
    df = df.copy()
    df["hour_of_day"]  = df["datetime"].dt.hour
    df["day_of_week"]  = df["datetime"].dt.dayofweek   # 0=Monday
    df["month"]        = df["datetime"].dt.month
    df["day_of_year"]  = df["datetime"].dt.dayofyear
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    peak_load    = df["load_mw"].max()
    min_load     = df["load_mw"].min()
    avg_load     = df["load_mw"].mean()

    # Normalized [0, 1] for FCNN feature input
    df["load_pu"] = (df["load_mw"] - min_load) / (peak_load - min_load)

    # Load scaling factor relative to annual peak (for SMC per-hour dispatch)
    df["load_scale"] = df["load_mw"] / peak_load

    log.info(f"Load profile: peak={peak_load:.0f} MW, "
             f"min={min_load:.0f} MW, avg={avg_load:.0f} MW")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Load statistics
# ──────────────────────────────────────────────────────────────────────────────
def compute_load_statistics(df: pd.DataFrame) -> dict:
    """Compute key statistics for OPF model setup."""
    stats = {
        "peak_load_mw":     float(df["load_mw"].max()),
        "min_load_mw":      float(df["load_mw"].min()),
        "avg_load_mw":      float(df["load_mw"].mean()),
        "load_factor":      float(df["load_mw"].mean() / df["load_mw"].max()),
        "annual_energy_gwh": float(df["load_mw"].sum() / 1000),  # GWh
        "peak_hour":        int(df["load_mw"].idxmax()),
        "peak_datetime":    str(df.loc[df["load_mw"].idxmax(), "datetime"]),
    }

    # Monthly statistics
    monthly = df.groupby("month")["load_mw"].agg(
        ["mean", "max", "min"]
    ).round(1)
    stats["monthly"] = monthly.to_dict()

    # Hourly average (load curve shape)
    hourly_avg = df.groupby("hour_of_day")["load_mw"].mean().round(1)
    stats["hourly_avg_mw"] = hourly_avg.to_dict()

    log.info(f"Load factor: {stats['load_factor']:.3f}")
    log.info(f"Annual energy: {stats['annual_energy_gwh']:.0f} GWh")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
def print_load_summary(df: pd.DataFrame, stats: dict):
    print("\n" + "=" * 65)
    print("  LOAD PROFILE SUMMARY — Step 05")
    print("=" * 65)
    print(f"  Data points     : {len(df)} hours")
    print(f"  Peak load       : {stats['peak_load_mw']:>10.1f} MW")
    print(f"    at: {stats['peak_datetime']}")
    print(f"  Minimum load    : {stats['min_load_mw']:>10.1f} MW")
    print(f"  Average load    : {stats['avg_load_mw']:>10.1f} MW")
    print(f"  Load factor     : {stats['load_factor']:>10.3f}")
    print(f"  Annual energy   : {stats['annual_energy_gwh']:>10.0f} GWh")
    print("\n  Monthly peak loads (MW):")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        mx = stats["monthly"].get("max", {}).get(m, 0)
        av = stats["monthly"].get("mean", {}).get(m, 0)
        print(f"    {month_names[m-1]}: peak={mx:.0f}, avg={av:.0f} MW")
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = parse_load_profile(LOAD_FILE)
    df_8760 = ensure_8760_hours(raw_df)
    df_full = enrich_load_profile(df_8760)

    stats = compute_load_statistics(df_full)
    print_load_summary(df_full, stats)

    # Save outputs
    df_full.to_csv(OUT_DIR / "load_profile_hourly.csv", index=False, encoding="utf-8")

    # Save numpy arrays for fast SMC loading
    np.save(str(OUT_DIR / "load_mw_8760.npy"),    df_full["load_mw"].values)
    np.save(str(OUT_DIR / "load_scale_8760.npy"),  df_full["load_scale"].values)
    np.save(str(OUT_DIR / "load_pu_8760.npy"),     df_full["load_pu"].values)

    # Monthly stats CSV
    monthly_df = df_full.groupby("month")["load_mw"].agg(["mean","max","min","std"]).round(2)
    monthly_df.to_csv(OUT_DIR / "load_monthly_stats.csv", encoding="utf-8")

    # Summary JSON
    (OUT_DIR / "load_statistics.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )

    log.info(f"Load profile saved to: {OUT_DIR}")
    return df_full, stats


if __name__ == "__main__":
    main()