"""
typhoon data CWA.py  (v3 — IBTrACS)
=====================================
Fetches historical typhoon best-track data from IBTrACS (NOAA).
No API key needed. Works offline after first download.

IBTrACS = International Best Track Archive for Climate Stewardship
Source  : https://www.ncei.noaa.gov/products/international-best-track-archive
Coverage: All western Pacific typhoons 1980–present, every 3 hours

Run on your LOCAL machine:
    python "typhoon data CWA.py"

Output:
    Data/typhoon_tracks/taipower_typhoon_tracks.json
"""

import json
import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

OUT_DIR = Path(r"C:\reXplan-repo\Project Taipower\Data\typhoon_tracks")

# IBTrACS western Pacific CSV (all storms, ~10MB)
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
    "stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv"
)

# Target typhoons from Step 04 accident data
TARGETS = [
    {"key": "kongrey_2024", "name": "KONG-REY", "zh": "康芮",  "year": 2024},
    {"key": "gaemi_2024",   "name": "GAEMI",    "zh": "凱米",  "year": 2024},
    {"key": "koinu_2023",   "name": "KOINU",    "zh": "小犬",  "year": 2023},
    {"key": "dana_2025",    "name": "DANA",     "zh": "丹娜絲","year": 2025},
]


def download_ibtracs(cache_path: Path) -> pd.DataFrame:
    """Download IBTrACS CSV and cache locally."""
    if cache_path.exists():
        print(f"  Using cached IBTrACS: {cache_path}")
        # Skip first 2 rows (header + units row)
        return pd.read_csv(cache_path, skiprows=[1], low_memory=False)

    print(f"  Downloading IBTrACS (~10MB)...")
    resp = requests.get(IBTRACS_URL, timeout=120)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8")
    print(f"  Saved: {cache_path}")
    return pd.read_csv(StringIO(resp.text), skiprows=[1], low_memory=False)


def find_typhoon(df: pd.DataFrame, name: str, year: int) -> pd.DataFrame:
    """
    Find a typhoon by name and year in IBTrACS DataFrame.
    IBTrACS name column: 'NAME', year from 'ISO_TIME'.
    """
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    mask = (
        (df["NAME"].str.upper() == name.upper()) &
        (df["ISO_TIME"].dt.year == year)
    )
    result = df[mask].copy()
    if result.empty:
        # Try partial match
        mask2 = (
            df["NAME"].str.upper().str.contains(name[:4].upper(), na=False) &
            (df["ISO_TIME"].dt.year == year)
        )
        result = df[mask2].copy()
    return result


def parse_ibtracs_track(df_storm: pd.DataFrame, name: str, year: int) -> list:
    """
    Convert IBTrACS rows for one storm to list of fix dicts.

    IBTrACS key columns:
        ISO_TIME   — datetime UTC
        LAT        — latitude
        LON        — longitude
        WMO_WIND   — max wind [knots]  (WMO standard)
        WMO_PRES   — central pressure [hPa]
        REUNION_R35_NE/SE/SW/NW  — 35kt radius per quadrant [nm]
        REUNION_R64_NE/SE/SW/NW  — 64kt radius per quadrant [nm]
    """
    KT_TO_MS = 0.514444
    NM_TO_KM = 1.852

    fixes = []
    for _, row in df_storm.iterrows():
        try:
            wind_kt  = float(row.get("WMO_WIND", 0) or 0)
            pres     = float(row.get("WMO_PRES", 1010) or 1010)

            # Radius of 34kt winds (≈17.5 m/s ≈ 15 m/s tier)
            r34_cols = ["USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW"]
            r34_vals = [float(row.get(c, 0) or 0) for c in r34_cols]
            r34_km   = max(r34_vals) * NM_TO_KM if any(v > 0 for v in r34_vals) else None

            # Radius of 64kt winds (≈33 m/s ≈ 25 m/s tier)
            r64_cols = ["USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW"]
            r64_vals = [float(row.get(c, 0) or 0) for c in r64_cols]
            r64_km   = max(r64_vals) * NM_TO_KM if any(v > 0 for v in r64_vals) else None

            if wind_kt <= 0:
                continue

            fixes.append({
                "datetime":       str(row["ISO_TIME"]),
                "lat":            float(row["LAT"]),
                "lon":            float(row["LON"]),
                "wind_ms":        round(wind_kt * KT_TO_MS, 1),
                "gust_ms":        round(wind_kt * KT_TO_MS * 1.25, 1),  # estimate
                "pressure_hpa":   pres,
                "radius_15ms_km": round(r34_km, 1) if r34_km else None,
                "radius_25ms_km": round(r64_km, 1) if r64_km else None,
            })
        except (ValueError, TypeError):
            continue

    return fixes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "ibtracs_WP_cache.csv"

    print("=" * 60)
    print("  Typhoon Track Fetcher — IBTrACS (NOAA)")
    print("=" * 60)

    # Download / load IBTrACS
    try:
        df = download_ibtracs(cache)
        print(f"  IBTrACS loaded: {len(df):,} rows, "
              f"{df['NAME'].nunique()} unique storm names")
    except Exception as e:
        print(f"  ✗ Failed to load IBTrACS: {e}")
        return

    all_tracks = {}

    for t in TARGETS:
        print(f"\nSearching: {t['name']} ({t['zh']}) {t['year']}...")

        df_storm = find_typhoon(df, t["name"], t["year"])

        if df_storm.empty:
            print(f"  ✗ Not found in IBTrACS")
            # Show nearby storms that year
            yr_storms = df[df["ISO_TIME"].dt.year == t["year"]]["NAME"].dropna().unique()
            print(f"    Storms in {t['year']}: {sorted(set(yr_storms))[:20]}")
            continue

        fixes = parse_ibtracs_track(df_storm, t["name"], t["year"])

        if not fixes:
            print(f"  ✗ No valid track points")
            continue

        winds = [f["wind_ms"] for f in fixes]
        r15s  = [f["radius_15ms_km"] for f in fixes if f["radius_15ms_km"]]

        peak_wind   = max(winds)
        min_pres    = min(f["pressure_hpa"] for f in fixes)
        max_r15     = max(r15s) if r15s else "N/A"

        print(f"  ✓ Found: {len(fixes)} track points")
        print(f"    Period     : {fixes[0]['datetime'][:10]} → {fixes[-1]['datetime'][:10]}")
        print(f"    Peak wind  : {peak_wind} m/s")
        print(f"    Min press  : {min_pres} hPa")
        print(f"    Max R15ms  : {max_r15} km")

        all_tracks[t["key"]] = {
            "meta": {
                "name":          t["name"],
                "zh_name":       t["zh"],
                "year":          t["year"],
                "track_points":  len(fixes),
                "peak_wind_ms":  peak_wind,
                "min_pressure":  min_pres,
                "max_radius_15ms_km": max_r15,
                "start_time":    fixes[0]["datetime"],
                "end_time":      fixes[-1]["datetime"],
                "source":        "IBTrACS v04r01 (NOAA)",
            },
            "fixes": fixes,
        }

    # Save
    out_path = OUT_DIR / "taipower_typhoon_tracks.json"
    out_path.write_text(
        json.dumps(all_tracks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"  Done: {len(all_tracks)}/4 typhoons")
    print(f"  Output: {out_path}")
    if len(all_tracks) == 4:
        print(f"\n  Next: send taipower_typhoon_tracks.json to Claude")
        print(f"        to update convert_typhoon_track.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()