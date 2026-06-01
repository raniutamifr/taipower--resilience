"""
fix_dana.py — Add DANAS (丹娜絲) 2025 to taipower_typhoon_tracks.json
Run: python "C:\reXplan-repo\Project Taipower\Code\fix_dana.py"
"""
import json
import pandas as pd
from pathlib import Path

TRACKS_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\typhoon_tracks\taipower_typhoon_tracks.json")
CACHE_FILE  = Path(r"C:\reXplan-repo\Project Taipower\Data\typhoon_tracks\ibtracs_WP_cache.csv")

KT_TO_MS = 0.514444
NM_TO_KM = 1.852

df = pd.read_csv(CACHE_FILE, skiprows=[1], low_memory=False)
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

mask = (df["NAME"].str.upper() == "DANAS") & (df["ISO_TIME"].dt.year == 2025)
df_storm = df[mask].copy()
print(f"DANAS 2025 rows: {len(df_storm)}")

# Show available wind columns to debug
wind_cols = [c for c in df_storm.columns if "WIND" in c.upper() or "PRES" in c.upper()]
print(f"Wind/pressure columns: {wind_cols}")
for c in wind_cols[:6]:
    non_null = df_storm[c].replace(' ', None).dropna()
    non_zero = non_null[non_null.astype(str).str.strip() != '0']
    print(f"  {c}: {len(non_zero)} non-zero values, sample={non_zero.values[:3]}")

fixes = []
for _, row in df_storm.iterrows():
    try:
        # Try multiple wind columns in priority order
        wind_kt = 0.0
        for col in ["WMO_WIND", "USA_WIND", "TOKYO_WIND", "CMA_WIND", "REUNION_WIND"]:
            v = str(row.get(col, "") or "").strip()
            if v and v not in ("", " ", "nan", "0"):
                try:
                    wind_kt = float(v)
                    if wind_kt > 0:
                        break
                except ValueError:
                    continue

        if wind_kt <= 0:
            continue

        # Pressure
        pres = 1010.0
        for col in ["WMO_PRES", "USA_PRES", "TOKYO_PRES", "CMA_PRES"]:
            v = str(row.get(col, "") or "").strip()
            if v and v not in ("", " ", "nan", "0"):
                try:
                    pres = float(v)
                    if pres > 800:
                        break
                except ValueError:
                    continue

        # Radius 34kt (≈ 15 m/s)
        r34_vals = []
        for col in ["USA_R34_NE","USA_R34_SE","USA_R34_SW","USA_R34_NW"]:
            v = str(row.get(col, "") or "").strip()
            try:
                r34_vals.append(float(v))
            except ValueError:
                r34_vals.append(0.0)
        r15 = max(r34_vals) * NM_TO_KM if max(r34_vals) > 0 else None

        # Radius 64kt (≈ 25 m/s)
        r64_vals = []
        for col in ["USA_R64_NE","USA_R64_SE","USA_R64_SW","USA_R64_NW"]:
            v = str(row.get(col, "") or "").strip()
            try:
                r64_vals.append(float(v))
            except ValueError:
                r64_vals.append(0.0)
        r25 = max(r64_vals) * NM_TO_KM if max(r64_vals) > 0 else None

        fixes.append({
            "datetime":       str(row["ISO_TIME"]),
            "lat":            float(row["LAT"]),
            "lon":            float(row["LON"]),
            "wind_ms":        round(wind_kt * KT_TO_MS, 1),
            "gust_ms":        round(wind_kt * KT_TO_MS * 1.25, 1),
            "pressure_hpa":   pres,
            "radius_15ms_km": round(r15, 1) if r15 else None,
            "radius_25ms_km": round(r25, 1) if r25 else None,
        })
    except (ValueError, TypeError) as e:
        print(f"  Skipped row: {e}")
        continue

print(f"\nValid fixes: {len(fixes)}")
if not fixes:
    print("Still no valid fixes — check wind column names above and tell Claude")
    exit(1)

winds = [f["wind_ms"] for f in fixes]
print(f"Peak wind  : {max(winds)} m/s")
print(f"Min press  : {min(f['pressure_hpa'] for f in fixes)} hPa")
print(f"Period     : {fixes[0]['datetime'][:10]} → {fixes[-1]['datetime'][:10]}")

# Add to existing tracks
with open(TRACKS_FILE, encoding="utf-8") as f:
    tracks = json.load(f)

tracks["dana_2025"] = {
    "meta": {
        "name": "DANA", "zh_name": "丹娜絲", "year": 2025,
        "track_points":  len(fixes),
        "peak_wind_ms":  max(winds),
        "min_pressure":  min(f["pressure_hpa"] for f in fixes),
        "start_time":    fixes[0]["datetime"],
        "end_time":      fixes[-1]["datetime"],
        "source":        "IBTrACS v04r01 (NOAA) — stored as DANAS",
    },
    "fixes": fixes,
}

TRACKS_FILE.write_text(
    json.dumps(tracks, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"\n✓ Saved 4/4 typhoons to: {TRACKS_FILE}")
print(f"  Keys: {list(tracks.keys())}")