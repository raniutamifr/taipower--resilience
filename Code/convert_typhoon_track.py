"""
convert_typhoon_track.py  (v2 — real IBTrACS data)
=====================================================
Converts taipower_typhoon_tracks.json (from IBTrACS) into reXplan
trajectory CSV files for use with:

    network.event.hazardFromTrajectory(
        filename      = 'typhoon_kongrey_2024.csv',
        max_intensity = 51.4,
        max_radius    = 416.7,
        sdate         = dt_date(2024, 10, 29),
        edate         = dt_date(2024, 10, 31),
        geodata1      = rx.network.GeoData(16.5, 120.3),
        geodata2      = rx.network.GeoData(24.2, 128.4),
        delta_km      = 10,
        frequency     = '1H',
    )

reXplan trajectory CSV columns:
    time       : datetime string  "YYYY-MM-DD HH:MM:SS"
    lat        : typhoon centre latitude
    lon        : typhoon centre longitude
    intensity  : wind speed / max_intensity  [0–1 per unit]
    radius     : radius / max_radius         [0–1 per unit]

Run: python convert_typhoon_track.py
Output: file/input/taipower/hazards/typhoon_*.csv
        file/input/taipower/hazards/rexplan_params.json
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TRACKS_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\typhoon_tracks\taipower_typhoon_tracks.json")
HAZARD_DIR  = Path(r"file\input\taipower\hazards")


# ─────────────────────────────────────────────────────────────────────────────
# Core converter: fixes list → reXplan trajectory DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def fixes_to_trajectory(
    fixes:          list,
    peak_wind_ms:   float,
    peak_radius_km: float,
    interpolate_hourly: bool = True,
) -> pd.DataFrame:
    """
    Convert IBTrACS fix list to reXplan trajectory DataFrame.

    Parameters
    ----------
    fixes          : list of dicts with datetime, lat, lon, wind_ms, radius_15ms_km
    peak_wind_ms   : denominator for per-unit intensity (actual peak wind)
    peak_radius_km : denominator for per-unit radius    (actual max radius)
    interpolate_hourly : resample to 1-hour intervals

    Returns
    -------
    DataFrame with columns: time, lat, lon, intensity, radius
    """
    rows = []
    for f in fixes:
        radius_km = f.get("radius_15ms_km") or peak_radius_km * 0.5
        rows.append({
            "time":      pd.to_datetime(f["datetime"]),
            "lat":       float(f["lat"]),
            "lon":       float(f["lon"]),
            "intensity": float(f["wind_ms"])   / peak_wind_ms,
            "radius":    float(radius_km)      / peak_radius_km,
        })

    df = pd.DataFrame(rows).set_index("time")
    df[["intensity", "radius"]] = df[["intensity", "radius"]].clip(0.0, 1.0)

    if interpolate_hourly:
        full_range = pd.date_range(
            start = df.index.min(),
            end   = df.index.max(),
            freq  = "1h",
        )
        df = df.reindex(df.index.union(full_range)).interpolate("time")
        df = df.reindex(full_range)

    df = df.reset_index().rename(columns={"index": "time"})
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def get_bounding_box(fixes: list, pad_deg: float = 2.0) -> tuple:
    """Return (sw_lat, sw_lon, ne_lat, ne_lon) with padding."""
    lats = [f["lat"] for f in fixes]
    lons = [f["lon"] for f in fixes]
    sw_lat = max(15.0, min(lats) - pad_deg)
    sw_lon = max(115.0, min(lons) - pad_deg)
    ne_lat = min(30.0, max(lats) + pad_deg)
    ne_lon = min(130.0, max(lons) + pad_deg)
    return round(sw_lat, 1), round(sw_lon, 1), round(ne_lat, 1), round(ne_lon, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    HAZARD_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRACKS_FILE, encoding="utf-8") as f:
        tracks = json.load(f)

    rexplan_params = {}

    print("\n" + "=" * 65)
    print("  TYPHOON TRACK CONVERTER — reXplan format")
    print("  Source: IBTrACS v04r01 (NOAA) + CWA")
    print("=" * 65)

    for key, tc in tracks.items():
        meta   = tc["meta"]
        fixes  = tc["fixes"]

        peak_wind   = float(meta["peak_wind_ms"])
        r15s        = [f["radius_15ms_km"] for f in fixes if f.get("radius_15ms_km")]
        peak_radius = float(max(r15s)) if r15s else 200.0

        sw_lat, sw_lon, ne_lat, ne_lon = get_bounding_box(fixes)
        sdate = meta["start_time"][:10]
        edate = meta["end_time"][:10]

        # Build trajectory CSV
        df = fixes_to_trajectory(fixes, peak_wind, peak_radius)
        csv_name = f"typhoon_{key}.csv"
        csv_path = HAZARD_DIR / csv_name
        df.to_csv(csv_path, index=False)

        # Build reXplan parameter dict
        params = {
            "filename":       csv_name,
            "max_intensity":  peak_wind,
            "max_radius":     peak_radius,
            "sdate":          sdate,
            "edate":          edate,
            "geodata1_lat":   sw_lat,
            "geodata1_lon":   sw_lon,
            "geodata2_lat":   ne_lat,
            "geodata2_lon":   ne_lon,
            "delta_km":       10,
            "frequency":      "1H",
        }
        rexplan_params[key] = params

        print(f"\n  [{meta['name']} / {meta['zh_name']} {meta['year']}]")
        print(f"  Track    : {len(df)} hourly points  ({sdate} → {edate})")
        print(f"  Peak wind: {peak_wind} m/s  |  Max radius: {peak_radius:.0f} km")
        print(f"  BBox     : SW({sw_lat}, {sw_lon}) → NE({ne_lat}, {ne_lon})")
        print(f"  Saved    : {csv_path}")
        print(f"\n  # Notebook code:")
        print(f"  network.event.hazardFromTrajectory(")
        print(f"      '{params['filename']}',")
        print(f"      max_intensity = {params['max_intensity']},")
        print(f"      max_radius    = {params['max_radius']:.1f},")
        print(f"      sdate = dt_date({sdate.replace('-', ', ')}),")
        print(f"      edate = dt_date({edate.replace('-', ', ')}),")
        print(f"      geodata1 = rx.network.GeoData({sw_lat}, {sw_lon}),")
        print(f"      geodata2 = rx.network.GeoData({ne_lat}, {ne_lon}),")
        print(f"      delta_km = 10, frequency = '1H',")
        print(f"  )")

    # Save params JSON for notebook import
    params_path = HAZARD_DIR / "rexplan_params.json"
    params_path.write_text(
        json.dumps(rexplan_params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*65}")
    print(f"  All 4 trajectory CSVs saved to: {HAZARD_DIR.resolve()}")
    print(f"  reXplan params saved to       : {params_path}")
    print(f"\n  In your notebook, load params like this:")
    print(f"  import json")
    print(f"  params = json.load(open('file/input/taipower/hazards/rexplan_params.json'))")
    print(f"  network.event.hazardFromTrajectory(**{{")
    print(f"      k: v for k, v in params['kongrey_2024'].items()")
    print(f"      if k not in ('geodata1_lat','geodata1_lon','geodata2_lat','geodata2_lon')")
    print(f"  }}, geodata1=rx.network.GeoData(params['kongrey_2024']['geodata1_lat'],")
    print(f"                                   params['kongrey_2024']['geodata1_lon']),")
    print(f"     geodata2=rx.network.GeoData(params['kongrey_2024']['geodata2_lat'],")
    print(f"                                   params['kongrey_2024']['geodata2_lon']))")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()