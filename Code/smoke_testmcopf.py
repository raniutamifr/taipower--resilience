"""
smoke_test_mc_opf_v4.py
=======================
Incremental improvement over v3:
  - Flushes output per hour (so Ctrl+C doesn't lose progress)
  - Handles NaN in demand profile gracefully
  - Saves partial results every 5 hours so a stop is recoverable
  - Prints v4 plant signatures right at the top so the run log is
    self-documenting (useful when sending to professor)
"""

import warnings, time, sys
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd
import pandapower as pp
from pathlib import Path

PROJECT_ROOT = Path(r"C:\reXplan-repo\Project Taipower")
NET_FILE = PROJECT_ROOT / "Results" / "step06" / "taipower_network_realcost.json"
LOG_FILE = PROJECT_ROOT / "Results" / "step06" / "cost_match_log_v4.csv"
OUT_DIR  = PROJECT_ROOT / "Results" / "step_mc_opf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_HOURS = 24
TEST_DAY_OFFSET = 24 * 30   # Feb 1
PARTIAL_SAVE_EVERY = 5

LADDER = [
    ("L0_spec",          0.95, 1.05, 1.00),
    ("L1_std",           0.90, 1.10, 1.00),
    ("L2_thermal_150",   0.90, 1.10, 1.50),
    ("L3_wide",          0.85, 1.15, 2.00),
    ("L4_unconstrained", 0.80, 1.20, 1e6),
]


def sec(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70, flush=True)


def find_demand_file() -> Path:
    for c in [
        PROJECT_ROOT / "Data" / "114年淨發電量(小時平均).xlsx",
        PROJECT_ROOT / "Data" / "114年淨發電量(小時平均).xlsx",
        PROJECT_ROOT / "Data" / "114年淨發電量_小時平均_.xlsx",
    ]:
        if c.exists():
            return c
    hits = list(PROJECT_ROOT.rglob("*淨發電量*.xlsx"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No demand profile under {PROJECT_ROOT}")


# ── PHASE 0 : cost audit ────────────────────────────────────────────────────
sec("PHASE 0 : COST BUILD AUDIT")
log = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
print(f"  cost log     : {LOG_FILE.name}")
print(f"  rows         : {len(log)}")
print(f"  plant-matched: {(log.how != 'fallback:gas').sum()}")
print(f"  fallback     : {(log.how == 'fallback:gas').sum()}")
print(f"  fuel mix     : {dict(log.fuel.value_counts())}")
nf = log[log.plant == "南火"]
tn = log[log.plant == "台中新"]
assert len(nf) > 0 and (nf.fuel == "oil").all(), "南火 wrong fuel"
assert len(tn) > 0 and (tn.fuel == "gas").all(), "台中新 wrong fuel"
print("  v4 signature : OK (南火=oil, 台中新=gas)", flush=True)


# ── PHASE 1 : network ──────────────────────────────────────────────────────
sec("PHASE 1 : LOAD NETWORK")
net0 = pp.from_json(str(NET_FILE))
print(f"  buses     : {len(net0.bus)}")
print(f"  gens      : {len(net0.gen)}")
print(f"  loads     : {len(net0.load)}  total P = {net0.load.p_mw.sum():.0f} MW")
print(f"  lines     : {len(net0.line)}")
print(f"  poly_cost : {len(net0.poly_cost)} entries", flush=True)
nominal_p = net0.load["p_mw"].values.copy()
nominal_q = net0.load["q_mvar"].values.copy()


# ── PHASE 2 : demand profile (with NaN handling) ────────────────────────────
sec("PHASE 2 : LOAD HOURLY DEMAND PROFILE")
GEN_FILE = find_demand_file()
print(f"  found file : {GEN_FILE}")
df = pd.read_excel(GEN_FILE)
df.columns = ["name", "date", "time", "value"]
df["value"] = pd.to_numeric(df["value"], errors="coerce")
hourly_raw = df["value"].values
nan_count = pd.Series(hourly_raw).isna().sum()
print(f"  rows total     : {len(df)}")
print(f"  NaN in profile : {nan_count}")

# Drop NaN, then take the test window
hourly = pd.Series(hourly_raw).dropna().values
print(f"  valid values   : {len(hourly)}")
if len(hourly) >= 8760:
    print(f"  range MW       : [{hourly.min():.0f}, {hourly.max():.0f}] mean {hourly.mean():.0f}")
else:
    print(f"  WARNING: fewer than 8760 valid hours — profile may be incomplete")

start = TEST_DAY_OFFSET
window = hourly[start:start + N_HOURS]
total_nominal = nominal_p.sum()
scales = window / total_nominal
print(f"  test window    : hours {start}..{start + N_HOURS - 1} "
      f"(load {window.min():.0f}–{window.max():.0f} MW)")
print(f"  load scale     : {scales.min():.3f}–{scales.max():.3f}", flush=True)


# ── PHASE 3 : OPF sweep ────────────────────────────────────────────────────
sec(f"PHASE 3 : SMOKE TEST — {N_HOURS} OPF RUNS  (Ctrl+C safe; partial saves every {PARTIAL_SAVE_EVERY}h)")
results = []
t0 = time.time()
shed_mask = net0.sgen["name"].astype(str).str.startswith("LoadShed_")

try:
    for h in range(N_HOURS):
        scale = scales[h]
        net = copy.deepcopy(net0)
        net.load["p_mw"] = nominal_p * scale
        net.load["q_mvar"] = nominal_q * scale

        converged = False
        for tier_name, vmin, vmax, line_mult in LADDER:
            try:
                n = copy.deepcopy(net)
                n.bus["min_vm_pu"] = vmin
                n.bus["max_vm_pu"] = vmax
                if "max_loading_percent" in n.line.columns:
                    n.line["max_loading_percent"] = 100. * line_mult
                for tbl in ["res_bus", "res_gen", "res_line", "res_load",
                            "res_ext_grid", "res_sgen"]:
                    if hasattr(n, tbl):
                        getattr(n, tbl).drop(getattr(n, tbl).index, inplace=True)
                pp.runopp(n, init="flat", numba=False,
                          calculate_voltage_angles=True)
                cost = n.res_cost if hasattr(n, "res_cost") else float("nan")
                shed = (float(n.res_sgen.loc[shed_mask, "p_mw"].sum())
                        if shed_mask.any() else 0.)
                results.append({
                    "hour": h, "scale": scale, "load_mw": float(window[h]),
                    "tier": tier_name, "vmin_set": vmin, "vmax_set": vmax,
                    "cost_ntd_per_hr": cost,
                    "gen_dispatch_mw": float(n.res_gen.p_mw.sum()),
                    "vbus_min": float(n.res_bus.vm_pu.min()),
                    "vbus_max": float(n.res_bus.vm_pu.max()),
                    "shed_mw": shed,
                })
                converged = True
                print(f"  h={h:02d} load={window[h]:7.0f}MW "
                      f"-> {tier_name:<16} cost={cost / 1e6:7.2f}M NT$/h  "
                      f"V=[{n.res_bus.vm_pu.min():.3f},{n.res_bus.vm_pu.max():.3f}]  "
                      f"shed={shed:.1f}MW", flush=True)
                break
            except Exception:
                continue

        if not converged:
            results.append({"hour": h, "scale": scale,
                            "load_mw": float(window[h]),
                            "tier": "FAILED",
                            "vmin_set": None, "vmax_set": None,
                            "cost_ntd_per_hr": None, "gen_dispatch_mw": None,
                            "vbus_min": None, "vbus_max": None, "shed_mw": None})
            print(f"  h={h:02d} load={window[h]:7.0f}MW -> FAILED ALL TIERS",
                  flush=True)

        if (h + 1) % PARTIAL_SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(
                OUT_DIR / "smoke_test_24h_partial.csv",
                index=False, encoding="utf-8-sig")

except KeyboardInterrupt:
    print(f"\n  Interrupted by user after {len(results)} hour(s). Saving partial.",
          flush=True)


# ── PHASE 4 : summary ──────────────────────────────────────────────────────
elapsed = time.time() - t0
sec(f"PHASE 4 : RESULTS  ({len(results)}/{N_HOURS} hours, {elapsed:.0f}s total)")
res = pd.DataFrame(results)
print("\n  Tier distribution:")
print(res.tier.value_counts().to_string(), flush=True)

ok = res[res.tier != "FAILED"]
if len(ok):
    print(f"\n  Cost (M NT$/h): mean {ok.cost_ntd_per_hr.mean() / 1e6:.2f}  "
          f"min {ok.cost_ntd_per_hr.min() / 1e6:.2f}  "
          f"max {ok.cost_ntd_per_hr.max() / 1e6:.2f}")
    print(f"  Voltage range : [{ok.vbus_min.min():.3f}, {ok.vbus_max.max():.3f}]")
    print(f"  Total shed    : {ok.shed_mw.sum():.1f} MW over {len(ok)}h "
          f"({(ok.shed_mw > 0.1).sum()} hours with shed)")

out_csv = OUT_DIR / "smoke_test_24h.csv"
res.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n  saved: {out_csv}", flush=True)


sec("VERDICT")
n_fail = (res.tier == "FAILED").sum()
n_strict = res.tier.isin(["L0_spec", "L1_std"]).sum()
n_unconstr = (res.tier == "L4_unconstrained").sum()
total = len(res)
if total == 0:
    print("  No results.")
elif n_fail == 0 and n_strict >= total * 0.8:
    print(f"  EXCELLENT - {n_strict}/{total} hours at strict bounds (L0/L1).")
    print("  -> Proceed to full Monte Carlo (100 samples x 168 hours).")
elif n_fail == 0 and n_unconstr >= total * 0.5:
    print(f"  WARNING - {n_unconstr}/{total} hours need unconstrained tier.")
    print("  -> Investigate voltage profile before scaling up.")
elif n_fail == 0:
    print(f"  OK - all {total} converged, mix of tiers.")
    print("  -> Proceed; expect similar tier mix at full scale.")
else:
    print(f"  PROBLEM - {n_fail} hour(s) failed all tiers.")