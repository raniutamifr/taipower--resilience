"""
build_real_cost_v4.py
=====================
Final cost builder v4. Adds two plants that the audit revealed v3 missed:

  - 南火 (Nanbu Huoli, Southern Thermal) — 10 thermal gens that v3 sent
    to fallback. They look like oil/gas peakers from kV/capacity profile.
  - 台中新 (Taichung Xin, Taichung New) — 6 combined-cycle GAS gens that
    v3 incorrectly mapped to 台中's COAL curve.

Also adds 中龍 as cogen-industrial (matched but tagged separately so the
log is honest about which gens come from a non-Taipower source).

Pattern ordering matters: longest/most specific first so '台中新' wins
over '台中', and '中火' wins over generic '中'.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandapower as pp
from pathlib import Path

NET_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")
OUT_FILE = Path(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network_realcost.json")

# ── Plant patterns — order matters, most specific first ─────────────────────
PLANT_PATTERNS = [
    # Taichung complex — '台中新' (gas CC) before '台中', and '中火' separately
    ("台中新", "台中新"),       # Taichung New = combined cycle GAS
    ("中火",   "台中"),          # legacy Taichung coal units
    ("台中",   "台中"),          # safety net
    # Newly added: Southern Thermal Plant (different from 南部 CC)
    ("南火",   "南火"),
    # China Steel cogen (industrial, not Taipower fleet)
    ("中龍",   "中龍"),
    # Existing thermal
    ("協和",   "協和"),
    ("林口",   "林口"),
    ("大潭",   "大潭"),
    ("通霄",   "通霄"),
    ("興新",   "興達"),          # new 興達 CC blocks (興新複)
    ("興達",   "興達"),
    ("南部",   "南部"),
    ("大林",   "大林"),
    # Nuclear
    ("核一",   "核一"), ("核二", "核二"), ("核三", "核三"),
    ("龍門",   "核四"),
    # Hydro
    ("明潭", "明潭"), ("大觀", "大觀"), ("碧海", "碧海"),
    ("石門", "石門"), ("天輪", "天輪"), ("德基", "德基"),
    ("青山", "青山"), ("谷關", "谷關"), ("萬大", "萬大"),
    ("卓蘭", "卓蘭"), ("立霧", "立霧"),
]

PLANT_FUEL = {
    "協和": "oil", "林口": "coal", "大潭": "gas", "通霄": "gas",
    "台中": "coal",                   # legacy 中火#1-#10
    "台中新": "gas",                  # new combined-cycle blocks
    "南火": "oil",                    # Southern Thermal — peaker/oil
    "中龍": "cogen",                  # industrial cogen, treated separately
    "興達": "gas", "南部": "gas", "大林": "coal",
    "核一": "nuclear", "核二": "nuclear", "核三": "nuclear", "核四": "nuclear",
    "明潭": "hydro", "大觀": "hydro", "碧海": "hydro", "石門": "hydro",
    "天輪": "hydro", "德基": "hydro", "青山": "hydro", "谷關": "hydro",
    "萬大": "hydro", "卓蘭": "hydro", "立霧": "hydro",
}

# ── Heat-rate curves (Gcal/hr) ──────────────────────────────────────────────
HR = {
    "協和#3":(0.0005,1.8688,125.08), "協和#4":(-0.0002,2.1977,84.411),
    "林口#1":(-0.0011,2.9407,97.561),"林口#2":(-0.0012,3.1284,59.903),
    "林口#3":(-0.0012,3.0931,59.248),
    "大潭CC#1":(-2e-7,1.2345,400.38),"大潭CC#2":(-5e-7,1.2501,383.72),
    "大潭CC#3":(2e-7,1.4027,206.18), "大潭CC#4":(-4e-7,1.4168,209.53),
    "大潭CC#5":(1e-7,1.4921,165.73), "大潭CC#6":(-7e-7,1.4434,195.46),
    "通霄CC#1":(0.0004,0.7728,402.46),"通霄CC#2":(0.0004,0.8253,356.86),
    "通霄CC#3":(0.0002,1.0794,289.01),"通霄CC#6":(0.001,0.9939,191.5),
    "台中#1":(0.0004,1.8826,157.29), "台中#2":(0.0003,2.0392,137.46),
    "台中#3":(0.0003,2.0392,137.46), "台中#4":(0.0004,1.8268,157.82),
    "台中#5":(0.0004,1.8186,157.62), "台中#6":(0.0004,1.8647,155.28),
    "台中#7":(0.0004,1.8544,160.3),  "台中#8":(0.0005,1.7564,179.46),
    "台中#9":(0.0006,1.6459,230.78), "台中#10":(0.0005,1.7296,216.03),
    "興達CC#1":(3e-5,1.5701,131.3),  "興達CC#2":(3e-5,1.5701,131.3),
    "興達CC#3":(5e-5,1.6295,119.82), "興達CC#4":(2e-5,1.7072,99.993),
    "興達CC#5":(0.0001,1.5163,133.84),
    "南部CC#1":(0.0024,0.3968,236.75),"南部CC#2":(0.0006,1.3457,134.03),
    "南部CC#3":(0.0021,0.426,264.0), "南部CC#4":(0.0031,0.1048,207.84),
    "大林#1":(5e-5,1.8711,182.21),   "大林#2":(4e-5,1.8765,174.76),
    "大林#6":(0.0002,1.9843,139.82),
}

# Build plant -> list of curves (average within plant)
PLANT_CURVES = {}
for k,(a,b,c) in HR.items():
    for prefix, plant in PLANT_PATTERNS:
        if k.startswith(prefix):
            PLANT_CURVES.setdefault(plant,[]).append((a,b,c)); break

# Per-fuel averages
FUEL_CURVES = {}
for plant, lst in PLANT_CURVES.items():
    f = PLANT_FUEL.get(plant, "gas")
    for c in lst:
        FUEL_CURVES.setdefault(f,[]).append(c)
FUEL_AVG = {f: tuple(np.mean(np.array(v),axis=0)) for f,v in FUEL_CURVES.items()}
# Synthetic flat curves where we have no heat-rate data
FUEL_AVG["nuclear"] = (0.0, 0.3, 0.0)
FUEL_AVG["hydro"]   = (0.0, 0.0, 0.0)
FUEL_AVG["cogen"]   = FUEL_AVG.get("gas", (1e-4, 1.4, 200.))   # industrial, gas-equiv

FUEL_PRICE = {"coal":550., "gas":1200., "oil":1800.,
              "nuclear":1.0, "hydro":1.0, "cogen":900., "default":1000.}
VOLL = 50_000.0


def detect_plant(bus_name: str) -> str | None:
    s = str(bus_name)
    for prefix, plant in PLANT_PATTERNS:
        if prefix in s:
            return plant
    return None


def pick_curve(plant: str):
    """Plant-average curve, falling back to fuel average."""
    f = PLANT_FUEL.get(plant, "gas")
    if plant in PLANT_CURVES:
        arr = np.array(PLANT_CURVES[plant])
        return tuple(arr.mean(axis=0)), f
    # 南火, 中龍, 台中新, etc — no direct curve, use fuel average
    return FUEL_AVG.get(f, FUEL_AVG["gas"]), f


def sec(t): print("\n"+"="*70+f"\n  {t}\n"+"="*70)


# ── Build ──────────────────────────────────────────────────────────────────
net = pp.from_json(str(NET_FILE))

sec("STEP 1 : DETECT PLANT FOR EACH GENERATOR")
gb = net.gen["bus"].values
bus_names = net.bus.loc[gb, "name"].astype(str).values
plants = [detect_plant(b) for b in bus_names]
n_match = sum(p is not None for p in plants)
print(f"\n  generators identified : {n_match} / {len(plants)}")
print(f"  fallback              : {len(plants)-n_match}")

counts = pd.Series([p for p in plants if p]).value_counts()
print("\n  Generators per plant:")
for plant, n in counts.items():
    f = PLANT_FUEL.get(plant, "?")
    print(f"    {plant:<6} ({f:<8}) : {n} gen(s)")


sec("STEP 2 : BUILD pandapower poly_cost")
net.poly_cost = net.poly_cost.iloc[0:0].copy()
shed = net.sgen["name"].astype(str).str.startswith("LoadShed_")

log = []
for idx, g in net.gen.iterrows():
    cap = float(g["max_p_mw"])
    plant = plants[idx]
    if plant is not None:
        (a,b,c), fuel = pick_curve(plant)
        how = f"bus:{plant}"
    else:
        a,b,c = FUEL_AVG["gas"]; fuel = "gas"
        how = "fallback:gas"
    phi = FUEL_PRICE.get(fuel, 1000.)
    cp2, cp1, cp0 = phi*a, phi*b, phi*c
    pp.create_poly_cost(net, element=idx, et="gen",
                        cp0_eur=cp0, cp1_eur_per_mw=cp1, cp2_eur_per_mw2=cp2)
    log.append((idx, str(g["name"]), bus_names[idx], plant or "-",
                fuel, cap, how, cp2, cp1, cp0))

for ei in net.ext_grid.index:
    a,b,c = FUEL_AVG["gas"]; phi = FUEL_PRICE["gas"]
    pp.create_poly_cost(net, element=ei, et="ext_grid",
                        cp0_eur=0., cp1_eur_per_mw=phi*b, cp2_eur_per_mw2=phi*a)
for si in net.sgen.index[shed]:
    pp.create_poly_cost(net, element=si, et="sgen",
                        cp0_eur=0., cp1_eur_per_mw=VOLL, cp2_eur_per_mw2=0.)


sec("STEP 3 : RESULT")
df = pd.DataFrame(log, columns=["gen","name","bus_name","plant","fuel",
                                "cap_mw","how","cp2","cp1","cp0"])
print("\n  Fuel mix:")
print(df.fuel.value_counts().to_string())
print("\n  cp1 (NT$/MWh) by fuel:")
print(df.groupby("fuel")["cp1"].agg(["min","mean","max","count"]).round(0).to_string())

# Specifically confirm the two corrected plants
print("\n  Verify 南火 (should be oil now, not fallback gas):")
print(df[df.plant=="南火"][["name","bus_name","cap_mw","fuel","cp1"]].to_string(index=False))
print("\n  Verify 台中新 (should be gas now, not 台中 coal):")
print(df[df.plant=="台中新"][["name","bus_name","cap_mw","fuel","cp1"]].to_string(index=False))

log_csv = OUT_FILE.with_name("cost_match_log_v4.csv")
df.to_csv(log_csv, index=False, encoding="utf-8-sig")
print(f"\n  detailed log -> {log_csv}")

net.gen["controllable"] = True
net.sgen.loc[shed, "controllable"] = True
pp.to_json(net, str(OUT_FILE))
print(f"  network      -> {OUT_FILE}")
print("\n  Cost build is now complete. Ready for Monte Carlo.")