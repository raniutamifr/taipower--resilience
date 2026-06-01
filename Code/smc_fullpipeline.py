"""
================================================================================
 STEP 07 + 09  |  FULL SEQUENTIAL MONTE CARLO  ->  AC-OPF DATASET GENERATOR
                 (mode-driven: BASELINE generator outages  +  TYPHOON line/gen outages)
================================================================================
Project : reXplan-repo / Project Taipower  (Power System Resilience)
Network : Taipower 2797-bus pandapower model (Step 06 output, AC-PF healthy)
Purpose : Produce the GROUND-TRUTH OPF datasets that the FCNN / GNN surrogates
          (Step 10) are trained and benchmarked against.

TWO DATASETS, ONE SCRIPT
------------------------
  SCENARIO_MODE = "baseline"
      Generator forced outages only (FOR ~5 %, sequential Markov).
      Topology is intact every hour -> high convergence, smooth ENS signal.
      Output: Results/step07_09_mc_opf_baseline/

  SCENARIO_MODE = "typhoon"
      Bounded N-k LINE outages during a landfall window (k in [1, 25]) with an
      MTTR repair tail, PLUS elevated generator FOR (0.35-0.70) on an exposed
      subset. This is the resilience-stress dataset.
      Output: Results/step07_09_mc_opf_typhoon/

WHY NOT "FOR 35-70 % on every line"
-----------------------------------
On a 2797-bus grid, knocking out even ~10 % of the 2210 lines at random causes
massive islanding -> most OPF runs do not converge -> the dataset is unusable.
A real typhoon is a *bounded* N-k event (a few to a few dozen lines in the storm
corridor), not half the grid. We therefore sample a bounded number of line
outages from an EXPOSED subset. The 0.35-0.70 calibration range is applied as a
GENERATOR FOR on the exposed plants, matching the Step-02 (per-unit) parse.
Plug real fragility output in via VULN_LINES_CSV / EXPOSED_GENS_CSV.

KEY ENGINEERING DECISIONS
-------------------------
  * SMC trajectories sampled UPFRONT (cheap, single-process). Only the OPF
    solves are parallelised -- independent across (sample, hour).
  * Workers MUTATE-AND-RESTORE one cost-loaded net per process (NO per-scenario
    deepcopy -- deepcopy of a 2797-bus net dominated the smoke-test runtime).
  * Cost build is loaded ONCE from the existing real-cost JSON (NOT recomputed
    16,800x); pandapower.from_json restores poly_cost together with the net.
  * Ladder starts at L1 (L0 skipped: base case never reaches L0; contingencies
    only push the system to higher tiers).
  * Per-sample checkpointing -> Ctrl+C safe; restart skips finished samples, so
    the FIRST completed sample acts as your live pilot.

INPUT NETWORK
-------------
The runner loads the SAME cost-applied pandapower JSON the smoke test uses:
    Results/step06/taipower_network_realcost.json
No prerequisite step is required -- pandapower.from_json restores poly_cost
together with the rest of the network, byte-identical to what the smoke test
sees. This guarantees the OPF objective is the same as the validated run.
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("mc_opf")

import pandapower as pp
from pandapower.optimal_powerflow import OPFNotConverged


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
SCENARIO_MODE = "baseline"     # <-- "baseline"  or  "typhoon"

RESULT_BASE   = Path(r"C:\reXplan-repo\Project Taipower\Results")
DATA_BASE     = Path(r"C:\reXplan-repo\Project Taipower\Data")
STEP06_DIR    = RESULT_BASE / "step06"

# Cost-loaded network (same file the validated smoke test loads via pp.from_json).
COSTED_NET    = STEP06_DIR / "taipower_network_realcost.json"
LOAD_PROFILE  = DATA_BASE / "114年淨發電量(小時平均).xlsx"

# Optional inputs from Step 02 / fragility side-track ------------------------
RELIABILITY_CSV  = RESULT_BASE / "step02" / "gen_reliability.csv"   # gen_name, FOR, MTTR_h
VULN_LINES_CSV   = RESULT_BASE / "step02" / "vulnerable_lines.csv"  # line_idx [, exposure]
EXPOSED_GENS_CSV = RESULT_BASE / "step02" / "exposed_gens.csv"      # gen_name [, exposure]

# --- Monte Carlo scope -------------------------------------------------------
N_SAMPLES      = 100
HOURS_PER_WEEK = 168
WEEK_START_HR  = 720
BASE_TOTAL_MW  = 42360.0
SEED           = 42

# --- Baseline reliability defaults ------------------------------------------
DEFAULT_FOR    = 0.05
DEFAULT_MTTR_H = 48.0

# --- Typhoon event parameters (only used when SCENARIO_MODE == "typhoon") ----
EVENT_FOR_LOW    = 0.35        # calibrated range from real Taipower typhoons
EVENT_FOR_HIGH   = 0.70        #   (Gaemi ~0.35  ...  Kong-rey ~0.70)
LANDFALL_HOURS   = 18          # contiguous high-hazard window within the week
MIN_LINES_OUT    = 1           # bounded N-k: lines lost per event
MAX_LINES_OUT    = 25
MTTR_LINE_H      = 36.0        # transmission line repair tail
EXPOSED_GEN_FRAC = 0.15        # fallback fraction of generators in the storm path
EXPOSED_LINE_FRAC = 0.20       # fallback exposed-line pool (lines are then sampled k from it)

# --- OPF tier ladder (MUST match the validated smoke test) -------------------
TIER_LADDER = [
    ("L1_std",     0.90, 1.10, 100.0),
    ("L2_thermal", 0.90, 1.10, 150.0),
    ("L3_wide",    0.85, 1.15, 200.0),
    ("L4_unconst", 0.00, 5.00, 1e6),
]
TIER_RANK = {"L0_strict": 0, "L1_std": 1, "L2_thermal": 2, "L3_wide": 3, "L4_unconst": 4}
VOLL_NTD_PER_MWH = 6_000_000.0

# --- Compute -----------------------------------------------------------------
N_WORKERS     = max(1, (os.cpu_count() or 2) - 1)
CHUNKSIZE     = 4
SAVE_NODE_VM  = True
SAVE_GEN_P    = True

# --- Derived output paths (per mode) ----------------------------------------
OUT_DIR        = RESULT_BASE / f"step07_09_mc_opf_{SCENARIO_MODE}"
STATE_DIR      = OUT_DIR / "mc_states"
CKPT_FILE      = OUT_DIR / "_completed_samples.json"
FLAT_PARQUET   = OUT_DIR / "mc_opf_results.parquet"
GRAPH_TEMPLATE = OUT_DIR / "graph_template.npz"


# ==============================================================================
# 2. RESULT CONTAINER
# ==============================================================================
@dataclass
class ScenarioResult:
    sample:          int
    hour:            int
    load_scale:      float
    n_gen_failed:    int
    n_line_failed:   int
    cap_out_mw:      float
    converged:       bool
    tier:            str
    relax_level:     int
    cost_ntd_hr:     float
    shed_mw:         float
    ens_mwh:         float
    min_vm_pu:       float
    max_vm_pu:       float
    max_loading_pct: float
    p_loss_mw:       float
    solve_ms:        float


# ==============================================================================
# 3. SEQUENTIAL MONTE CARLO SAMPLERS  (Step 07)
# ==============================================================================
def hourly_transition_rates(forr: float, mttr_h: float) -> tuple[float, float]:
    """FOR + MTTR -> hourly two-state Markov transition probabilities."""
    forr = float(np.clip(forr, 1e-6, 0.95))
    mu = 1.0 / max(mttr_h, 1.0)
    lam = mu * forr / (1.0 - forr)
    return 1.0 - np.exp(-lam), 1.0 - np.exp(-mu)


def sample_gen_trajectory(n_gen: int, forr: np.ndarray, mttr: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """One 168-h generator availability trajectory. True = in service."""
    p_fail = np.empty(n_gen); p_rep = np.empty(n_gen)
    for g in range(n_gen):
        p_fail[g], p_rep[g] = hourly_transition_rates(forr[g], mttr[g])
    status = np.ones((HOURS_PER_WEEK, n_gen), dtype=bool)
    status[0] = rng.random(n_gen) > forr
    for h in range(1, HOURS_PER_WEEK):
        up = status[h - 1]; draw = rng.random(n_gen); nxt = up.copy()
        nxt[up]  = draw[up]  >= p_fail[up]
        nxt[~up] = draw[~up] <  p_rep[~up]
        status[h] = nxt
    return status


def sample_line_trajectory(n_line: int, exposed_pool: np.ndarray,
                           rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """
    One bounded N-k typhoon line-outage trajectory.
    True = line in service. Returns (status[168, n_line], event_k).
      * choose k in [MIN_LINES_OUT, MAX_LINES_OUT] distinct exposed lines
      * each fails at a random hour inside the landfall window
      * each stays down for an MTTR_LINE_H tail (clamped to week end)
    """
    status = np.ones((HOURS_PER_WEEK, n_line), dtype=bool)
    k = int(rng.integers(MIN_LINES_OUT, MAX_LINES_OUT + 1))
    k = min(k, len(exposed_pool))
    if k == 0:
        return status, 0
    t0 = int(rng.integers(0, max(1, HOURS_PER_WEEK - LANDFALL_HOURS)))
    failing = rng.choice(exposed_pool, size=k, replace=False)
    for li in failing:
        hf = int(rng.integers(t0, t0 + LANDFALL_HOURS))
        repair = min(HOURS_PER_WEEK, hf + int(MTTR_LINE_H))
        status[hf:repair, li] = False
    return status, k


# ==============================================================================
# 4. NETWORK PRE-PROCESSING (main process)
# ==============================================================================
def load_costed_net():
    if not COSTED_NET.exists():
        log.error("Cost-loaded network not found: %s", COSTED_NET)
        log.error("Expected the same JSON the smoke test loads (Step 06 output).")
        sys.exit(1)
    net = pp.from_json(str(COSTED_NET))
    log.info("Cost-loaded network: %d buses, %d gens, %d sgens, %d lines, %d poly_cost",
             len(net.bus), len(net.gen), len(net.sgen), len(net.line), len(net.poly_cost))
    return net


def split_gen_masks(net):
    """
    Real generators are ALL entries of net.gen (816 in Taipower).
    Load-shedding entities live in net.sgen as LoadShed_* controllable sgens.
    Returns (real_gen_idx, shed_sgen_idx).
    """
    real_idx = net.gen.index.to_numpy()
    shed_mask = net.sgen["name"].astype(str).str.startswith("LoadShed_")
    shed_idx  = net.sgen.index[shed_mask].to_numpy()
    return real_idx, shed_idx


def load_demand_profile():
    df = pd.read_excel(LOAD_PROFILE)
    val_col = df.select_dtypes("number").columns[-1]
    mw = pd.to_numeric(df[val_col], errors="coerce").to_numpy()
    window = mw[WEEK_START_HR: WEEK_START_HR + HOURS_PER_WEEK]
    if np.isnan(window).any():
        window = pd.Series(window).interpolate(limit_direction="both").to_numpy()
    scale = window / BASE_TOTAL_MW
    log.info("Demand window: %.0f-%.0f MW, scale %.3f-%.3f",
             window.min(), window.max(), scale.min(), scale.max())
    return scale


def load_gen_reliability(net, real_idx):
    n = len(real_idx)
    forr = np.full(n, DEFAULT_FOR); mttr = np.full(n, DEFAULT_MTTR_H)
    names = net.gen.loc[real_idx, "name"].astype(str).to_numpy()
    if RELIABILITY_CSV.exists():
        tab = pd.read_csv(RELIABILITY_CSV)
        lut = {str(r.get("gen_name", "")): (r.get("FOR"), r.get("MTTR_h")) for _, r in tab.iterrows()}
        hit = 0
        for i, nm in enumerate(names):
            if nm in lut and pd.notna(lut[nm][0]):
                forr[i], mttr[i] = float(lut[nm][0]), float(lut[nm][1]); hit += 1
        log.info("Reliability table matched %d / %d generators", hit, n)
    else:
        log.warning("No reliability table -> defaults FOR=%.2f MTTR=%.0fh", DEFAULT_FOR, DEFAULT_MTTR_H)

    # Typhoon mode: raise FOR on the EXPOSED generator subset to the event range.
    if SCENARIO_MODE == "typhoon":
        rng = np.random.default_rng(SEED + 7)
        if EXPOSED_GENS_CSV.exists():
            exp_names = set(pd.read_csv(EXPOSED_GENS_CSV)["gen_name"].astype(str))
            exp_mask = np.array([nm in exp_names for nm in names])
        else:
            exp_mask = rng.random(n) < EXPOSED_GEN_FRAC
            log.warning("No exposed-gens file -> random %.0f%% exposed subset",
                        100 * EXPOSED_GEN_FRAC)
        forr[exp_mask] = rng.uniform(EVENT_FOR_LOW, EVENT_FOR_HIGH, exp_mask.sum())
        log.info("Typhoon: %d generators elevated to FOR %.2f-%.2f",
                 int(exp_mask.sum()), EVENT_FOR_LOW, EVENT_FOR_HIGH)
    return forr, mttr


def exposed_line_pool(net):
    """Index pool (positional into net.line) from which typhoon outages are drawn."""
    n_line = len(net.line)
    if VULN_LINES_CSV.exists():
        col = pd.read_csv(VULN_LINES_CSV)["line_idx"].to_numpy()
        pos = {b: i for i, b in enumerate(net.line.index)}
        pool = np.array([pos[x] for x in col if x in pos], dtype=int)
        log.info("Exposed-line pool from file: %d lines", len(pool))
    else:
        rng = np.random.default_rng(SEED + 11)
        size = max(MAX_LINES_OUT, int(EXPOSED_LINE_FRAC * n_line))
        pool = rng.choice(n_line, size=size, replace=False)
        log.warning("No vulnerable-lines file -> random exposed pool of %d lines", len(pool))
    return pool


def save_graph_template(net):
    bus_pos = {b: i for i, b in enumerate(net.bus.index)}
    n_bus = len(net.bus)
    lf = net.line["from_bus"].map(bus_pos).to_numpy()
    lt = net.line["to_bus"].map(bus_pos).to_numpy()
    src = list(lf); dst = list(lt)
    er = list(net.line["r_ohm_per_km"] * net.line["length_km"])
    ex = list(net.line["x_ohm_per_km"] * net.line["length_km"])
    etype = [0] * len(lf)
    if len(net.trafo):
        tf = net.trafo["hv_bus"].map(bus_pos).to_numpy(); tt = net.trafo["lv_bus"].map(bus_pos).to_numpy()
        src += list(tf); dst += list(tt); er += [0.0]*len(tf); ex += list(net.trafo["vk_percent"]/100.0); etype += [1]*len(tf)
    real_idx, _ = split_gen_masks(net)
    np.savez_compressed(
        GRAPH_TEMPLATE, n_bus=n_bus,
        bus_vn_kv=net.bus["vn_kv"].to_numpy(np.float32),
        edge_index=np.array([src, dst], dtype=np.int64),
        edge_attr=np.array([er, ex, etype], dtype=np.float32).T,
        gen_bus=net.gen.loc[real_idx, "bus"].map(bus_pos).to_numpy(np.int64),
        load_bus=net.load["bus"].map(bus_pos).to_numpy(np.int64),
        gen_pmax=net.gen.loc[real_idx, "max_p_mw"].to_numpy(np.float32),
        load_p_base=net.load["p_mw"].to_numpy(np.float32),
        n_line=len(net.line),
    )
    log.info("Graph template saved: %d nodes, %d edges", n_bus, len(src))


# ==============================================================================
# 5. WORKER  (one cost-loaded net per process; mutate-and-restore)
# ==============================================================================
_W = {}

def _worker_init(costed_net_path):
    net = pp.from_json(str(costed_net_path))
    real_idx, shed_idx = split_gen_masks(net)
    _W.update(
        net=net,
        real_idx=real_idx,
        shed_idx=shed_idx,
        line_idx=net.line.index.to_numpy(),
        load_p0=net.load["p_mw"].to_numpy().copy(),
        load_q0=net.load["q_mvar"].to_numpy().copy(),
    )


def _apply_tier(net, vmin, vmax, smax):
    net.bus["min_vm_pu"] = vmin; net.bus["max_vm_pu"] = vmax
    if "max_loading_percent" in net.line:
        net.line["max_loading_percent"] = smax
    if len(net.trafo):
        net.trafo["max_loading_percent"] = smax


def _solve_scenario(task):
    """task = (sample, hour, load_scale, gen_status[bool n_real], line_status[bool n_line])"""
    sample, hour, load_scale, gen_status, line_status = task
    net = _W["net"]; real_idx = _W["real_idx"]; shed_idx = _W["shed_idx"]; line_idx = _W["line_idx"]
    t0 = time.perf_counter()

    # apply state -- mirror the smoke test exactly: scale loads only.
    # LoadShed sgens keep their nameplate; they only inject if OPF needs them.
    net.load["p_mw"]   = _W["load_p0"] * load_scale
    net.load["q_mvar"] = _W["load_q0"] * load_scale
    failed = ~gen_status
    net.gen.loc[real_idx, "in_service"] = gen_status
    cap_out = float(net.gen.loc[real_idx[failed], "max_p_mw"].sum()) if failed.any() else 0.0
    n_line_out = 0
    if line_status is not None and not line_status.all():
        net.line.loc[line_idx, "in_service"] = line_status
        n_line_out = int((~line_status).sum())

    res = dict(converged=False, tier="FAILED", relax=99, cost=0.0, shed=0.0,
               vmin=np.nan, vmax=np.nan, load=np.nan, ploss=np.nan)
    node_vm = node_gp = None
    for tier, vmin, vmax, smax in TIER_LADDER:
        _apply_tier(net, vmin, vmax, smax)
        try:
            pp.runopp(net, calculate_voltage_angles=True, init="flat",
                      numba=True, suppress_warnings=True)
        except (OPFNotConverged, Exception):
            continue
        if not net.OPF_converged:
            continue
        shed_mw = float(net.res_sgen.loc[shed_idx, "p_mw"].clip(lower=0).sum()) \
                  if len(shed_idx) else 0.0
        res.update(converged=True, tier=tier, relax=TIER_RANK[tier],
                   cost=float(net.res_cost), shed=shed_mw,
                   vmin=float(net.res_bus["vm_pu"].min()), vmax=float(net.res_bus["vm_pu"].max()),
                   load=float(max(net.res_line["loading_percent"].max() if len(net.res_line) else 0,
                                  net.res_trafo["loading_percent"].max() if len(net.res_trafo) else 0)),
                   ploss=float(net.res_line["pl_mw"].sum() if len(net.res_line) else 0.0))
        if SAVE_NODE_VM: node_vm = net.res_bus["vm_pu"].to_numpy(np.float32)
        if SAVE_GEN_P:   node_gp = net.res_gen.loc[real_idx, "p_mw"].to_numpy(np.float32)
        break

    # restore
    net.gen.loc[real_idx, "in_service"] = True
    if n_line_out:
        net.line.loc[line_idx, "in_service"] = True

    r = ScenarioResult(
        sample=sample, hour=hour, load_scale=float(load_scale),
        n_gen_failed=int(failed.sum()), n_line_failed=n_line_out, cap_out_mw=cap_out,
        converged=res["converged"], tier=res["tier"], relax_level=res["relax"],
        cost_ntd_hr=res["cost"], shed_mw=res["shed"], ens_mwh=res["shed"],
        min_vm_pu=res["vmin"], max_vm_pu=res["vmax"], max_loading_pct=res["load"],
        p_loss_mw=res["ploss"], solve_ms=(time.perf_counter()-t0)*1000.0)
    out = asdict(r)
    out["_node_vm"] = node_vm
    out["_node_gp"] = node_gp
    out["_gen_status"]  = np.packbits(gen_status)
    out["_line_status"] = np.packbits(line_status) if line_status is not None else None
    return out


# ==============================================================================
# 6. ORCHESTRATOR
# ==============================================================================
def load_checkpoint():
    return set(json.loads(CKPT_FILE.read_text())) if CKPT_FILE.exists() else set()

def save_checkpoint(done):
    CKPT_FILE.write_text(json.dumps(sorted(done)))

def flush_sample(sample, rows, n_real, n_line):
    flat = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    if FLAT_PARQUET.exists():
        flat = pd.concat([pd.read_parquet(FLAT_PARQUET), flat], ignore_index=True)
    flat.to_parquet(FLAT_PARQUET, index=False)

    rs = sorted(rows, key=lambda r: r["hour"])
    arrays = dict(
        gen_status=np.array([np.unpackbits(r["_gen_status"])[:n_real].astype(bool) for r in rs]),
        load_scale=np.array([r["load_scale"] for r in rs], dtype=np.float32),
    )
    if rs[0]["_line_status"] is not None:
        arrays["line_status"] = np.array([np.unpackbits(r["_line_status"])[:n_line].astype(bool) for r in rs])
    if SAVE_NODE_VM and rs[0]["_node_vm"] is not None:
        arrays["node_vm"] = np.array([r["_node_vm"] if r["_node_vm"] is not None
                                      else np.full_like(rs[0]["_node_vm"], np.nan) for r in rs])
    if SAVE_GEN_P and rs[0]["_node_gp"] is not None:
        arrays["node_gp"] = np.array([r["_node_gp"] if r["_node_gp"] is not None
                                      else np.full_like(rs[0]["_node_gp"], np.nan) for r in rs])
    np.savez_compressed(STATE_DIR / f"sample_{sample:04d}.npz", **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("MODE=%s | %d samples x %d h = %d solves | workers=%d",
             SCENARIO_MODE, args.samples, HOURS_PER_WEEK, args.samples*HOURS_PER_WEEK, args.workers)
    log.info("output -> %s", OUT_DIR)
    log.info("=" * 70)

    # 1. setup
    net = load_costed_net()
    real_idx, _ = split_gen_masks(net)
    n_real, n_line = len(real_idx), len(net.line)
    load_scale = load_demand_profile()
    forr, mttr = load_gen_reliability(net, real_idx)
    pool = exposed_line_pool(net) if SCENARIO_MODE == "typhoon" else None
    if not GRAPH_TEMPLATE.exists():
        save_graph_template(net)
    del net

    # 2. sample SMC trajectories upfront
    rng = np.random.default_rng(SEED)
    log.info("Sampling %d SMC trajectories (mode=%s)...", args.samples, SCENARIO_MODE)
    gen_traj, line_traj = {}, {}
    for s in range(args.samples):
        gen_traj[s] = sample_gen_trajectory(n_real, forr, mttr, rng)
        if SCENARIO_MODE == "typhoon":
            line_traj[s], _ = sample_line_trajectory(n_line, pool, rng)
    avg_g = np.mean([(~gen_traj[s]).sum(1).mean() for s in gen_traj])
    log.info("Mean generators offline/hour: %.1f", avg_g)
    if SCENARIO_MODE == "typhoon":
        avg_l = np.mean([(~line_traj[s]).sum(1).max() for s in line_traj])
        log.info("Mean PEAK lines offline/event: %.1f", avg_l)

    # 3. parallel OPF
    done = load_checkpoint()
    if done: log.info("Resuming: %d samples already complete", len(done))
    import multiprocessing as mp
    t_start = time.time(); n_run0 = len(done)
    with mp.Pool(args.workers, initializer=_worker_init, initargs=(str(COSTED_NET),)) as pool_:
        for s in range(args.samples):
            if s in done: continue
            tasks = [(s, h, float(load_scale[h]), gen_traj[s][h],
                      line_traj[s][h] if SCENARIO_MODE == "typhoon" else None)
                     for h in range(HOURS_PER_WEEK)]
            rows = list(pool_.imap_unordered(_solve_scenario, tasks, chunksize=CHUNKSIZE))
            flush_sample(s, rows, n_real, n_line)
            done.add(s); save_checkpoint(done)
            conv = sum(r["converged"] for r in rows)
            n_run = len(done) - n_run0
            elapsed = time.time() - t_start
            eta = (elapsed / max(n_run, 1)) * (args.samples - len(done))
            log.info("sample %3d/%d | conv %3d/%d | shed>0 %2dh | maxLineOut %2d | "
                     "elapsed %5.0fs | ETA ~%.1fh",
                     s+1, args.samples, conv, HOURS_PER_WEEK,
                     sum(r["shed_mw"] > 1e-3 for r in rows),
                     max((r["n_line_failed"] for r in rows), default=0),
                     elapsed, eta/3600)

    # 4. summary
    df = pd.read_parquet(FLAT_PARQUET)
    log.info("=" * 70)
    log.info("DONE [%s]. %d scenarios | %.1f%% converged", SCENARIO_MODE, len(df), 100*df["converged"].mean())
    log.info("Tier distribution:\n%s", df["tier"].value_counts().to_string())
    log.info("Cost (M NT$/h): mean %.1f min %.1f max %.1f",
             df["cost_ntd_hr"].mean()/1e6, df["cost_ntd_hr"].min()/1e6, df["cost_ntd_hr"].max()/1e6)
    log.info("Total ENS: %.1f MWh | hours with shed: %d", df["ens_mwh"].sum(), int((df["shed_mw"]>1e-3).sum()))
    log.info("Flat dataset : %s", FLAT_PARQUET)
    log.info("=" * 70)


if __name__ == "__main__":
    main()


# ==============================================================================
# README  |  RUN ORDER + STEP 10 CONSUMPTION
# ==============================================================================
"""
RUN ORDER (two separate datasets, one script)
---------------------------------------------
  1) SCENARIO_MODE = "baseline"  ->  python step07_09_full_mc_opf.py --samples 100
     Output: Results/step07_09_mc_opf_baseline/
  2) SCENARIO_MODE = "typhoon"   ->  python step07_09_full_mc_opf.py --samples 100
     Output: Results/step07_09_mc_opf_typhoon/
  Each gets its own parquet + mc_states + graph_template (identical topology).

WATCH SAMPLE #1 (your live pilot)
---------------------------------
  baseline: conv should be ~168/168, tiers mostly L1 with occasional L2 at peak,
            shed>0 rare. If you see many FAILED -> stop, diagnose.
  typhoon : maxLineOut in [1,25], conv high but lower than baseline, shed>0 in
            several hours, tiers spread into L2/L3. That spread IS the signal.

PLUG IN REAL FRAGILITY DATA (recommended for the typhoon dataset)
-----------------------------------------------------------------
  vulnerable_lines.csv : column line_idx (pandapower line index of exposed lines)
  exposed_gens.csv     : column gen_name (plants in the storm corridor)
  Drop them in Results/step02/ and rerun -- the random fallbacks are replaced by
  your IBTrACS/CWA fragility selection.

STEP 10 (FCNN vs GNN vs AC-OPF ground truth)
--------------------------------------------
  Flat parquet -> FCNN feature vectors (load_scale, n_gen_failed, n_line_failed,
                  cap_out_mw -> predict cost_ntd_hr / ens_mwh).
  graph_template.npz + mc_states/*.npz -> GNN graphs:
      node X : load (load_p_base * load_scale on load_bus), gen avail + pmax on
               gen_bus, bus_vn_kv ; edges masked by line_status (typhoon).
      y      : cost_ntd_hr (+ ens_mwh) ; optional node label node_vm.
  Train BOTH on the SAME split, predict on held-out scenarios, compare:
      accuracy (R2/MAE/RMSE) | survivability-curve fidelity vs OPF | inference
      speed vs Ipopt | hybrid (X% OPF + rest surrogate) NTD savings.
  Reference: IEEE 14-bus GNN R2=0.9997. The Taipower gap is itself a finding.
"""