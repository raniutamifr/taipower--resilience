"""
audit_rexplan_readiness.py
===========================
Before we convert anything to reXplan format, we need to know EXACTLY:
  1. Apa yang kamu sudah punya dari step 1-5
  2. Apa yang reXplan butuhkan (wajib vs optional)
  3. Gap apa yang harus diisi
  4. Dependencies mana yang perlu install (Julia PowerModels, R SamplingStrata)

Run: python audit_rexplan_readiness.py
Output: audit_report.txt dengan roadmap actionable

This is a READ-ONLY diagnostic. Does not modify any files.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths — adjust to your actual locations
# ─────────────────────────────────────────────────────────────────────────────
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
DATA_BASE   = Path(r"C:\reXplan-repo\Project Taipower\Data")
REXPLAN_REPO = Path(r"C:\reXplan-repo")   # where you cloned reXplan-repo

OUT = []
def log(msg=""):
    print(msg)
    OUT.append(msg)

def section(title):
    log("")
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)

def check(label, ok, detail=""):
    mark = "✓" if ok else "✗"
    log(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Section A: Input data you already have
# ─────────────────────────────────────────────────────────────────────────────
section("A.  DATA INVENTORY — apa yang sudah ada dari step 1-5")

step01 = RESULT_BASE / "step01"
step02 = RESULT_BASE / "step02"
step03 = RESULT_BASE / "step03"
step04 = RESULT_BASE / "step04"
step05 = RESULT_BASE / "step05"

have = {}

log("\n  Step 01 (PSSE parse):")
have["buses"]      = check("buses.csv",       (step01 / "buses.csv").exists())
have["gens"]       = check("generators.csv",  (step01 / "generators.csv").exists())
have["lines"]      = check("branches.csv",    (step01 / "branches.csv").exists())
have["trafos"]     = check("transformers.csv",(step01 / "transformers.csv").exists())
have["loads"]      = check("loads.csv",       (step01 / "loads.csv").exists())
have["net_meta"]   = check("network_meta.json", (step01 / "network_meta.json").exists())

log("\n  Step 02 (Generator FOR):")
have["for_data"]   = check("generator_for.csv", (step02 / "generator_for.csv").exists())

log("\n  Step 03 (Cost data):")
have["cost"]       = check("cost_parameters.csv", (step03 / "cost_parameters.csv").exists())

log("\n  Step 04 (Accident / hazard factors):")
have["hf"]         = check("hazard_factors.csv", (step04 / "hazard_factors.csv").exists())
have["accidents"]  = check("all_accidents.csv",  (step04 / "all_accidents.csv").exists())

log("\n  Step 05 (Load profile):")
have["load_profile"] = check("load_profile_hourly.csv", (step05 / "load_profile_hourly.csv").exists())
have["load_npy"]     = check("load_scale_8760.npy",      (step05 / "load_scale_8760.npy").exists())

log("\n  Raw data files:")
have["psse"]    = check("PSSE RAW", (DATA_BASE / "11507DP_base(108).raw").exists())
have["typhoon_tracks"] = check("typhoon_tracks.json",
    (DATA_BASE / "typhoon_tracks" / "taipower_typhoon_tracks.json").exists(),
    "from IBTrACS")


# ─────────────────────────────────────────────────────────────────────────────
# Section B: Geo-coordinates — CRITICAL for reXplan hazard model
# ─────────────────────────────────────────────────────────────────────────────
section("B.  GEO-COORDINATES — reXplan HARUS punya lat/lon per bus")

geo_ok = False
if have["buses"]:
    import pandas as pd
    buses = pd.read_csv(step01 / "buses.csv")
    cols = buses.columns.tolist()
    log(f"\n  Columns di buses.csv: {cols}")
    has_lat = any(c.lower() in ("latitude", "lat", "y") for c in cols)
    has_lon = any(c.lower() in ("longitude", "lon", "x") for c in cols)
    check("Latitude column present",  has_lat)
    check("Longitude column present", has_lon)

    if has_lat and has_lon:
        lat_col = next(c for c in cols if c.lower() in ("latitude", "lat", "y"))
        lon_col = next(c for c in cols if c.lower() in ("longitude", "lon", "x"))
        nz_lat = (buses[lat_col].abs() > 0.1).sum()
        nz_lon = (buses[lon_col].abs() > 0.1).sum()
        check(f"Buses with non-zero coords", nz_lat > 0,
              f"{nz_lat}/{len(buses)} lat, {nz_lon}/{len(buses)} lon")
        geo_ok = (nz_lat > len(buses) * 0.8)

if not geo_ok:
    log("\n  ⚠ CRITICAL: reXplan's Hazard.get_intensity(lon, lat) butuh koordinat.")
    log("    Tanpa ini, fragility curve projection TIDAK BISA jalan.")
    log("    → Kamu butuh mapping bus_name → (lat, lon) dari Taipower substation DB.")


# ─────────────────────────────────────────────────────────────────────────────
# Section C: reXplan-specific inputs yang BELUM ADA
# ─────────────────────────────────────────────────────────────────────────────
section("C.  reXplan INPUT FILES — yang harus kamu bikin")

log("\n  reXplan expects input at: file/input/<simulationName>/...")
log("  (path dari config.py: path.networkFile, path.fragilityCurveFolder, etc.)")

rex_input_dir = None
# Check common locations
for candidate in [
    Path("file/input"),
    REXPLAN_REPO / "reXplan-repo" / "file" / "input",
    REXPLAN_REPO / "file" / "input",
    Path.cwd() / "file" / "input",
]:
    if candidate.exists():
        rex_input_dir = candidate
        break

log(f"\n  reXplan input directory: {rex_input_dir or 'NOT FOUND (will need to create)'}")

log("\n  WAJIB ada (reXplan tidak jalan tanpa ini):")
check("network.xlsx (dengan 10+ sheets)",          False, "BUILD from step 1-5 data")
check("fragilityCurves/*.csv (>= 1 curve)",        False, "BUILD from failure rate data")
check("returnPeriods/*.csv (>= 1 return period)",  False, "BUILD from typhoon data")
check("hazards/*.nc (netCDF hazard field)",        False, "BUILD from typhoon tracks")

log("\n  network.xlsx butuh 10 sheet:")
sheets_needed = [
    ("network",      "system params (f_hz, sn_mva)"),
    ("nodes",        "dari buses.csv + lat/lon"),
    ("transformers", "dari transformers.csv"),
    ("tr_type",      "transformer types (vk%, vkr%, etc)"),
    ("lines",        "dari branches.csv + from/to coords"),
    ("ln_type",      "line types (r/x/c per km, max_i_ka)"),
    ("loads",        "dari loads.csv"),
    ("external_gen", "ext_grid (slack bus)"),
    ("generators",   "dari generators.csv"),
    ("cost",         "dari cost_parameters.csv"),
    ("simulation",   "startTime, duration, hazardStartTime, hazardDuration"),
    ("profiles",     "time-series load + renewables"),
    ("crews",        "repair crew info"),
    ("switches",     "switchgear (optional)"),
]
for name, desc in sheets_needed:
    log(f"    [ ] {name:<15} — {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# Section D: Python environment dependencies
# ─────────────────────────────────────────────────────────────────────────────
section("D.  PYTHON DEPENDENCIES — yang reXplan import")

py_deps = [
    ("pandapower",      "core grid engine"),
    ("netCDF4",         "hazard .nc files"),
    ("xarray",          "hazard data arrays"),
    ("pygam",           "fragility curve GAM fitting"),
    ("rpy2",            "R interface for SamplingStrata"),
    ("mpl_toolkits.basemap", "geo plotting"),
    ("PIL",             "hazard GIF generation"),
    ("openpyxl",        "excel read/write"),
]

log()
for mod, desc in py_deps:
    try:
        __import__(mod)
        check(f"{mod}", True, desc)
    except ImportError:
        check(f"{mod}", False, f"{desc} — pip install needed")


# ─────────────────────────────────────────────────────────────────────────────
# Section E: Julia + PowerModels.jl
# ─────────────────────────────────────────────────────────────────────────────
section("E.  JULIA + PowerModels.jl — reXplan's OPF engine")

julia_ok = False
try:
    result = subprocess.run(["julia", "--version"], capture_output=True, timeout=10, text=True)
    if result.returncode == 0:
        check("Julia installed", True, result.stdout.strip())
        julia_ok = True
    else:
        check("Julia installed", False, "julia command exists but returned error")
except (subprocess.TimeoutExpired, FileNotFoundError):
    check("Julia installed", False, "run `where julia` — tidak ditemukan di PATH")

if julia_ok:
    log("\n  Checking Julia packages (akan butuh ~30s)...")
    try:
        check_pkg = subprocess.run(
            ["julia", "-e",
             'using Pkg; ps = Pkg.dependencies(); '
             'for (u,p) in ps; if p.name in ["PowerModels","PandaModels","Ipopt"]; '
             'println("$(p.name) $(p.version)"); end; end'],
            capture_output=True, timeout=60, text=True)
        installed = check_pkg.stdout.strip().split("\n") if check_pkg.stdout.strip() else []
        pkg_names = {line.split()[0] for line in installed if line}
        check("PowerModels.jl",  "PowerModels"  in pkg_names)
        check("PandaModels.jl",  "PandaModels"  in pkg_names)
        check("Ipopt.jl",        "Ipopt"        in pkg_names)
    except Exception as e:
        log(f"    Could not check Julia packages: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Section F: R + SamplingStrata
# ─────────────────────────────────────────────────────────────────────────────
section("F.  R + SamplingStrata — reXplan's stratified sampling (optional)")

r_ok = False
try:
    result = subprocess.run(["R", "--version"], capture_output=True, timeout=10, text=True)
    if result.returncode == 0:
        first_line = result.stdout.split("\n")[0]
        check("R installed", True, first_line)
        r_ok = True
except (subprocess.TimeoutExpired, FileNotFoundError):
    check("R installed", False, "Needed for initialize_model_rp (stratified SMC)")

if r_ok:
    try:
        res = subprocess.run(["R", "-e", 'if (require("SamplingStrata")) cat("YES") else cat("NO")'],
                              capture_output=True, timeout=30, text=True)
        has_ss = "YES" in res.stdout
        check("SamplingStrata R package", has_ss,
              "install via: install.packages('SamplingStrata')" if not has_ss else "")
    except Exception:
        check("SamplingStrata R package", False, "couldn't test")


# ─────────────────────────────────────────────────────────────────────────────
# Section G: reXplan package itself
# ─────────────────────────────────────────────────────────────────────────────
section("G.  reXplan PACKAGE — installed & importable?")

rex_ok = False
try:
    import reXplan
    rex_ok = True
    check("reXplan importable", True, reXplan.__file__ if hasattr(reXplan, '__file__') else "yes")
    from reXplan import simulation, network, hazard, fragilitycurve
    check("  submodules", True, "simulation, network, hazard, fragilitycurve")
except ImportError as e:
    check("reXplan importable", False, str(e))
    log("  → kamu perlu: cd ke reXplan-repo, lalu 'pip install -e .'")


# ─────────────────────────────────────────────────────────────────────────────
# Section H: Roadmap based on findings
# ─────────────────────────────────────────────────────────────────────────────
section("H.  ROADMAP — apa yang harus dilakukan berikutnya")

log("""
  Berdasarkan hasil audit, task list (urutan critical → optional):

  ┌─ WAJIB untuk reXplan jalan ─────────────────────────────────┐
  │                                                              │
  │  1. Install dependencies yang missing (lihat section D, E)  │
  │  2. Install reXplan package (pip install -e .)              │
  │  3. Dapatkan geo-koordinat per bus (lat/lon)                │
  │     — reXplan Hazard model WAJIB butuh ini                  │
  │  4. Build converter: step 1-5 CSVs → network.xlsx           │
  │  5. Build fragilityCurves/*.csv                             │
  │     — dari FOR data (step 02) + literature curves           │
  │  6. Build returnPeriods/*.csv                               │
  │     — dari typhoon_tracks.json                              │
  │  7. Build hazards/*.nc                                      │
  │     — dari typhoon tracks pakai Hazard.epicenterTrajectory  │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  ┌─ OPTIONAL (bisa skip kalau belum punya) ────────────────────┐
  │                                                              │
  │  8. R + SamplingStrata → untuk initialize_model_rp()        │
  │     Alternative: pakai initialize_model_sh() (no R needed)  │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  ⚠ IMPORTANT: semua task 1-7 di atas TIDAK BISA di-parallel.
    Task 4 butuh task 3 dulu, dll. Estimasi realistis: 3-5 hari kerja
    kalau semua data sudah lengkap, 1-2 minggu kalau koordinat bus
    belum ada.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Write report
# ─────────────────────────────────────────────────────────────────────────────
out_path = Path.cwd() / "audit_report.txt"
out_path.write_text("\n".join(OUT), encoding="utf-8")
print(f"\n\n  Full report saved: {out_path}")