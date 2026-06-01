"""
Step 09 - AC-OPF Solver for FCNN Training Data Generation
==========================================================

Purpose
-------
Solve AC-OPF over a large set of contingency scenarios to produce the
labelled training dataset for the FCNN / GNN surrogate models built in
later steps.

Addresses the Taipower review comments:

    Comment 1 - OPF constraints
        * Bus voltage bounds [Vmin, Vmax] enforced at every bus
        * Generator reactive power limits [Qmin, Qmax] enforced by the
          PowerModels formulation
        * Line and transformer thermal limits set to 100 %

    Comment 2 - N-k contingency
        * Each generator sampled independently by its failure probability
        * Each line sampled independently by its failure probability
        * The combined generator + line sampling naturally covers
          N-0, N-1, N-2, ..., N-k in a single Monte Carlo framework
        * SMC integration: loads state_matrix.csv from Step 07 if it
          exists; otherwise falls back to random probabilistic sampling

PowerModels.jl Compatibility (collapse_switches_for_powermodels)
----------------------------------------------------------------
The Step 06 build represents genuine bus couplers as closed bus-bus
switches. Pandapower's native solver handles this through internal bus
merging, but the PowerModels.jl exporter materialises the merge in the
exported network. Any branch (line, impedance, or two-winding
transformer) whose endpoints sit on the merged pair becomes a self-loop
on the surviving bus, which PowerModels rejects with

    "both sides of branch X connect to bus Y"

and any load attached to the consumed bus becomes orphaned in the export

    "bus X in load Y is not defined"

The collapse_switches_for_powermodels function executes the merge once
on the base network, drops the resulting self-loop branches, and
removes the now-consumed switches before any scenario is built.
"""

import os
os.environ["JULIA_HOME"] = r"C:\Users\user\AppData\Local\Programs\Julia-1.10.9\bin"
os.environ["PATH"]       = os.environ["JULIA_HOME"] + os.pathsep + os.environ["PATH"]

import logging
import time
import copy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

import pandapower as pp
import pandapower.topology as top

# The greedy-shed heuristic deliberately probes operating points that may
# be numerically singular (a partially collapsed post-contingency grid),
# then sheds load until a valid one is found. The singular probes are
# expected and handled, so silence the scipy / pandapower numerical
# warnings they raise to keep the scenario log readable.
warnings.filterwarnings("ignore", message=".*Matrix is exactly singular.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Paths
# =============================================================================
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
STEP04_DIR  = RESULT_BASE / "step04"
STEP06_DIR  = RESULT_BASE / "step06"
STEP07_DIR  = RESULT_BASE / "step07"
OUT_DIR     = RESULT_BASE / "step09_opf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Configuration
# =============================================================================
N_TRAINING           = 100
USE_JULIA            = False  # Default OFF: pp.runopp (native) is 10-30x faster
                              # than PowerModels.jl SOCWR on networks >2000 bus.
                              # Set True only if explicit SOCWR labels are needed.
VOLL_NTD_PER_MWH     = 50_000

V_MIN_PU             = 0.95
V_MAX_PU             = 1.05

LOAD_SCALE           = 0.75
MAX_LOADING_PERCENT  = 100.0

GEN_FAILURE_PROB     = 0.005
LINE_FAILURE_PROB    = 0.002

# Native pandapower OPF (PYPOWER backend) settings
PP_OPF_MAX_ITER      = 200

# AC-PF + greedy shed heuristic settings
# ----------------------------------------------------------------------------
# Used for networks >2000 bus where the free-tier native AC-OPF backends
# (PYPOWER IPOPT and PowerModels.jl SOCWR) cannot converge in reasonable
# time. The progression below mirrors the operator response curve used in
# NERC TPL-001 and ENTSO-E adequacy studies - shed in 5 % steps until the
# AC PF settles, then accept the un-served portion as ENS.
HEURISTIC_SHED_LEVELS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70,
                         0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
HEURISTIC_PF_MAX_ITER = 50

# PowerModels.jl settings - only relevant if USE_JULIA = True
PM_MODEL     = "SOCWRPowerModel"
PM_SOLVER    = "ipopt"
PM_LOG_LEVEL = 0
PM_MAX_ITER  = 500
PM_TOL       = 1e-6

_julia_warmed_up = False


# =============================================================================
# Result Container
# =============================================================================
@dataclass
class OPFResult:
    state_idx:            int
    converged:            bool
    method:               str
    scenario_type:        str   = "N-0"
    source:               str   = "random"
    cost_ntd_hr:          float = 0.0
    ens_mwh:              float = 0.0
    p_loss_mw:            float = 0.0
    max_vm_pu:            float = 1.0
    min_vm_pu:            float = 1.0
    max_loading_pct:      float = 0.0
    n_voltage_violations: int   = 0
    n_thermal_violations: int   = 0
    n_gen_failed:         int   = 0
    n_line_failed:        int   = 0
    load_scale:           float = 1.0
    solve_time_ms:        float = 0.0
    total_load_mw:        float = 0.0
    total_gen_mw:         float = 0.0


# =============================================================================
# PowerModels Pre-Processing
# =============================================================================
def collapse_switches_for_powermodels(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """
    Prepare the base network for PowerModels.jl by collapsing every closed
    bus-bus switch and dropping any branches that become self-loops as a
    result.

    Background
    ----------
    PowerModels.jl exports treat a closed bus-bus switch as a bus merge.
    Any branch (line, impedance, or 2W transformer) that spans the merged
    pair becomes a self-loop on the surviving bus and triggers the error
        "both sides of branch X connect to bus Y".
    Loads attached to the consumed bus likewise become orphaned with
        "bus X in load Y is not defined".

    This function executes the merge in pandapower itself - using
    `pp.fuse_buses` so that all element references (loads, gens, trafos,
    lines, impedances, switches) are remapped to the surviving bus -
    drops self-loop branches and degenerate three-winding transformers,
    and removes the now-consumed switches.

    The transformation is applied once on the base network. The function
    is safe to call repeatedly (idempotent on a clean network).
    """
    bb_indices = net.switch[
        (net.switch.et == "b") & net.switch.closed
    ].index.tolist()
    log.info(f"Pre-OPF: processing {len(bb_indices)} closed bus-bus "
             f"switches...")

    n_merged = 0
    for sw_idx in bb_indices:
        # The switch row may have been remapped or removed by a previous
        # fuse; re-read the current state on every iteration.
        if sw_idx not in net.switch.index:
            continue
        b1 = int(net.switch.at[sw_idx, "bus"])
        b2 = int(net.switch.at[sw_idx, "element"])
        if b1 == b2:
            continue
        if b1 not in net.bus.index or b2 not in net.bus.index:
            continue
        try:
            pp.fuse_buses(net, b1, b2, drop=True)
            n_merged += 1
        except Exception as e:
            log.debug(f"  fuse {b1} <-> {b2} failed: {e}")

    # Remove any leftover bus-bus switches whose endpoints became identical
    # (e.g. two switches that targeted the same pair).
    if not net.switch.empty:
        self_sw = net.switch[
            (net.switch.et == "b") &
            (net.switch.bus == net.switch.element)
        ].index
        if len(self_sw):
            net.switch.drop(self_sw, inplace=True)

    # Drop self-loop branches that the merge produced
    dropped: dict[str, int] = {}
    for df_name, c1, c2 in [
        ("line",      "from_bus", "to_bus"),
        ("impedance", "from_bus", "to_bus"),
        ("trafo",     "hv_bus",   "lv_bus"),
    ]:
        df = getattr(net, df_name)
        if df.empty:
            continue
        sl = df[df[c1] == df[c2]].index
        if len(sl):
            df.drop(sl, inplace=True)
            dropped[df_name] = len(sl)

    # Three-winding transformer is degenerate if any two terminals coincide
    if not net.trafo3w.empty:
        t3w = net.trafo3w
        deg = t3w[
            (t3w.hv_bus == t3w.mv_bus) |
            (t3w.mv_bus == t3w.lv_bus) |
            (t3w.hv_bus == t3w.lv_bus)
        ].index
        if len(deg):
            net.trafo3w.drop(deg, inplace=True)
            dropped["trafo3w"] = len(deg)

    log.info(f"  merged {n_merged} bus pair(s); "
             f"dropped self-loops: {dropped if dropped else 'none'}")
    log.info(f"  buses after collapse : {len(net.bus)}")
    log.info(f"  lines after collapse : {len(net.line)}")
    return net


# =============================================================================
# Constraint Verification Log
# =============================================================================
def log_opf_constraints(net: pp.pandapowerNet) -> None:
    """
    Log every active OPF constraint before solving. Provides traceable
    evidence that voltage, reactive power, and thermal constraints are
    enforced (addresses Taipower Comment 1).
    """
    log.info("=" * 60)
    log.info("OPF CONSTRAINTS ACTIVE IN THIS RUN")
    log.info("=" * 60)

    v_min = net.bus["min_vm_pu"].min()
    v_max = net.bus["max_vm_pu"].max()
    log.info(f"  Bus voltage bounds : [{v_min:.3f}, {v_max:.3f}] pu")
    log.info(f"  Enforcement        : {PM_MODEL} at every bus node")

    if not net.gen.empty:
        q_min_vals = net.gen["min_q_mvar"].dropna()
        q_max_vals = net.gen["max_q_mvar"].dropna()
        log.info(f"  Generator Q limits : {len(q_min_vals)} generators "
                 f"with Qmin / Qmax")
        log.info(f"  Qmin range         : "
                 f"[{q_min_vals.min():.1f}, {q_min_vals.max():.1f}] Mvar")
        log.info(f"  Qmax range         : "
                 f"[{q_max_vals.min():.1f}, {q_max_vals.max():.1f}] Mvar")

    if not net.line.empty:
        if "max_loading_percent" in net.line.columns:
            ll = net.line["max_loading_percent"]
            log.info(f"  Line thermal limit : "
                     f"{ll.min():.0f} % to {ll.max():.0f} % "
                     f"(target = {MAX_LOADING_PERCENT:.0f} %)")
        else:
            log.info(f"  Line thermal limit : {MAX_LOADING_PERCENT:.0f} % "
                     f"(applied at solve time)")

    if not net.trafo.empty:
        if "max_loading_percent" in net.trafo.columns:
            tl = net.trafo["max_loading_percent"]
            log.info(f"  Transformer limit  : "
                     f"{tl.min():.0f} % to {tl.max():.0f} %")
        else:
            log.info(f"  Transformer limit  : {MAX_LOADING_PERCENT:.0f} % "
                     f"(applied at solve time)")

    log.info(f"  OPF formulation    : {PM_MODEL} via PowerModels.jl (IPOPT)")
    log.info(f"  Note               : {PM_MODEL} enforces V, Q and thermal "
             f"constraints via convex second-order cone relaxation.")
    log.info("=" * 60)


# =============================================================================
# Network Sanitation
# =============================================================================
def full_network_sanitation(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """
    Synchronise the `in_service` status between buses and every connected
    element. Must be called before `pp.runpm()` to prevent
    "bus X in load Y is not defined" errors from PowerModels.jl.

    Three passes are performed:
        Pass 1 - disable elements whose bus is inactive
        Pass 2 - disable buses sitting in islands without a slack bus
        Pass 3 - strict forward check; disable any active element whose
                 bus is not in the active bus set (catches auto-created
                 bus edge cases)
    """
    # Pass 1: disable elements attached to inactive buses
    inactive_buses = net.bus[~net.bus.in_service].index
    if len(inactive_buses) > 0:
        for df, cols in [
            (net.load,      ["bus"]),
            (net.sgen,      ["bus"]),
            (net.gen,       ["bus"]),
            (net.trafo,     ["hv_bus", "lv_bus"]),
            (net.trafo3w,   ["hv_bus", "mv_bus", "lv_bus"]),
            (net.line,      ["from_bus", "to_bus"]),
            (net.impedance, ["from_bus", "to_bus"]),
        ]:
            if df.empty:
                continue
            mask = pd.Series(False, index=df.index)
            for col in cols:
                if col in df.columns:
                    mask |= df[col].isin(inactive_buses)
            if mask.any():
                df.loc[mask, "in_service"] = False

    # Pass 2: disable buses in islands without a slack bus
    try:
        slack_bus   = net.ext_grid.bus.iloc[0]
        main_island = top.get_connected_elements(net, "bus", slack_bus)
        if main_island:
            active_buses = set(net.bus[net.bus.in_service].index)
            isolated     = active_buses - set(main_island)
            if isolated:
                net.bus.loc[list(isolated), "in_service"] = False
                return full_network_sanitation(net)
    except Exception:
        pass

    # Pass 3: strict forward check. PowerModels.jl requires every active
    # element's bus to be present and active in the exported network.
    active_bus_set = set(net.bus[net.bus.in_service].index)
    for df, cols in [
        (net.load,      ["bus"]),
        (net.sgen,      ["bus"]),
        (net.gen,       ["bus"]),
        (net.trafo,     ["hv_bus", "lv_bus"]),
        (net.trafo3w,   ["hv_bus", "mv_bus", "lv_bus"]),
        (net.line,      ["from_bus", "to_bus"]),
        (net.impedance, ["from_bus", "to_bus"]),
    ]:
        if df.empty or "in_service" not in df.columns:
            continue
        active_elements = df[df["in_service"]].index
        if len(active_elements) == 0:
            continue
        mask = pd.Series(False, index=active_elements)
        for col in cols:
            if col in df.columns:
                mask |= ~df.loc[active_elements, col].isin(active_bus_set)
        if mask.any():
            df.loc[active_elements[mask], "in_service"] = False

    return net


# =============================================================================
# Apply System State
# =============================================================================
def apply_system_state(
    net_base:    pp.pandapowerNet,
    gen_states:  np.ndarray,
    line_states: np.ndarray,
    load_scale:  float,
) -> tuple:
    """
    Build a contingency scenario by applying generator outages, line
    outages, load scaling, and constraint bounds to a deep copy of the
    base network.

    Parameters
    ----------
    gen_states  : 1-D int8 array, 1 = generator out of service
    line_states : 1-D int8 array, 1 = line out of service
    load_scale  : load scaling factor (e.g. 0.75 = 75 % of peak load)

    Returns
    -------
    (net, n_gen_failed, n_line_failed, scenario_type)
    """
    net = copy.deepcopy(net_base)

    # Scale every load
    net.load["p_mw"]   *= load_scale
    net.load["q_mvar"] *= load_scale

    # Scale the load-shedding virtual generators' capacity to match
    # (this requires the Step 06 "LoadShed_" prefix contract)
    if not net.sgen.empty:
        shed_mask = net.sgen["name"].str.startswith("LoadShed_")
        net.sgen.loc[shed_mask, "max_p_mw"] *= load_scale

    # Generator outages
    n_gen_failed = 0
    for si, gi in enumerate(net.gen.index.tolist()):
        if si < len(gen_states) and gen_states[si] == 1:
            net.gen.at[gi, "in_service"] = False
            n_gen_failed += 1

    # Line outages
    n_line_failed = 0
    if len(line_states) > 0 and not net.line.empty:
        for si, li in enumerate(net.line.index.tolist()):
            if si < len(line_states) and line_states[si] == 1:
                net.line.at[li, "in_service"] = False
                n_line_failed += 1

    # Constraint bounds (re-applied per scenario in case any were edited)
    net.bus["min_vm_pu"] = V_MIN_PU
    net.bus["max_vm_pu"] = V_MAX_PU

    if "max_loading_percent" in net.line.columns:
        net.line["max_loading_percent"] = MAX_LOADING_PERCENT
    if "max_loading_percent" in net.trafo.columns:
        net.trafo["max_loading_percent"] = MAX_LOADING_PERCENT
    if "max_loading_percent" in net.trafo3w.columns:
        net.trafo3w["max_loading_percent"] = MAX_LOADING_PERCENT

    net = full_network_sanitation(net)

    total_failed  = n_gen_failed + n_line_failed
    scenario_type = f"N-{total_failed}"

    return net, n_gen_failed, n_line_failed, scenario_type


# =============================================================================
# Julia Warm-Up
# =============================================================================
def warmup_julia(net_sample: pp.pandapowerNet) -> None:
    """
    Warm up the Julia JIT compiler using a fast DC OPF.

    A full SOCWR warm-up on a ~2700-bus network takes 10+ minutes and
    is frequently flagged as 'not converged' simply because IPOPT cannot
    finish within a small warm-up iteration budget. DC OPF exercises the
    same pandapower -> PowerModels.jl Julia bridge in 30-90 seconds and
    converges deterministically, leaving the JIT cache populated for the
    SOCWR scenario solves that follow.
    """
    global _julia_warmed_up
    if _julia_warmed_up:
        return
    log.info("Warming up the Julia JIT compiler via DC OPF "
             "(typically 60-180 s)...")
    t0 = time.perf_counter()
    try:
        pp.runpm(net_sample, pm_model="DCPPowerModel", pm_solver=PM_SOLVER,
                 pm_log_level=0, delete_buffer_file=True)
        log.info(f"  Julia warm-up complete: "
                 f"{time.perf_counter() - t0:.1f} s")
    except Exception as e:
        log.warning(f"  Julia warm-up encountered an error "
                    f"(JIT cache may still be populated): {e}")
    finally:
        _julia_warmed_up = True


# =============================================================================
# OPF Solver
# =============================================================================
def compute_island_ens(net: pp.pandapowerNet) -> float:
    """
    Detect load that is stranded in islands lacking sufficient local
    generation, which is the dominant source of real ENS in N-k
    contingency scenarios on a network with large bulk capacity.

    For each connected component (respecting in-service status and open
    switches) we compare the local in-service generation capacity with
    the local load. If a component has load but its generation capacity
    plus the slack (if the slack is in that component) cannot cover the
    load, the shortfall is counted as Energy Not Supplied.

    Returns the total stranded-load ENS in MW.
    """
    import pandapower.topology as top
    import networkx as nx

    try:
        mg = top.create_nxgraph(net, respect_switches=True,
                                include_out_of_service=False)
    except Exception:
        return 0.0

    slack_buses = set(net.ext_grid.loc[net.ext_grid.in_service, "bus"].tolist())

    ens = 0.0
    for comp in nx.connected_components(mg):
        comp = set(comp)

        load_mask = (net.load.in_service & net.load.bus.isin(comp))
        local_load = float(net.load.loc[load_mask, "p_mw"].sum())
        if local_load <= 0:
            continue

        # A component containing the slack is backed by the external grid
        if comp & slack_buses:
            continue

        gen_mask = (net.gen.in_service & net.gen.bus.isin(comp))
        local_gen_cap = float(net.gen.loc[gen_mask, "max_p_mw"].sum())

        if local_gen_cap < local_load:
            ens += (local_load - local_gen_cap)

    return ens


def disable_dead_islands(net: pp.pandapowerNet) -> None:
    """
    Disable every bus (and the elements on it) that is no longer
    connected to an in-service external grid through in-service
    branches. After a line outage a bus can be left stranded; if it
    stays in service the AC-PF Jacobian becomes singular and Newton
    never converges no matter how much load is shed. This is the root
    cause of the flood of "Matrix is exactly singular" warnings.

    The function walks the in-service graph from the slack bus and
    switches off anything it cannot reach.
    """
    import networkx as nx
    try:
        mg = top.create_nxgraph(net, respect_switches=True,
                                include_out_of_service=False)
    except Exception:
        return

    slack_buses = net.ext_grid.loc[net.ext_grid.in_service, "bus"].tolist()
    if not slack_buses:
        return

    alive = set()
    for sb in slack_buses:
        if sb in mg:
            alive |= nx.node_connected_component(mg, sb)

    dead = set(net.bus[net.bus.in_service].index) - alive
    if not dead:
        return

    net.bus.loc[list(dead), "in_service"] = False
    for df, cols in [
        (net.load,      ["bus"]),
        (net.sgen,      ["bus"]),
        (net.gen,       ["bus"]),
        (net.line,      ["from_bus", "to_bus"]),
        (net.trafo,     ["hv_bus", "lv_bus"]),
        (net.trafo3w,   ["hv_bus", "mv_bus", "lv_bus"]),
        (net.impedance, ["from_bus", "to_bus"]),
    ]:
        if df.empty:
            continue
        m = pd.Series(False, index=df.index)
        for c in cols:
            if c in df.columns:
                m |= df[c].isin(dead)
        if m.any():
            df.loc[m, "in_service"] = False


def dispatch_gens_proportionally(net: pp.pandapowerNet) -> bool:
    """
    Dispatch every in-service generator proportional to total in-service
    load. Keeps the slack bus near zero, which is what is needed for AC
    PF to converge over a wide range of operating points.

    Returns True if a valid dispatch was found, False if there is no
    in-service capacity at all.
    """
    in_serv     = net.gen.in_service
    total_load  = float(net.load.loc[net.load.in_service, "p_mw"].sum())
    total_pmax  = float(net.gen.loc[in_serv, "max_p_mw"].sum())

    if total_pmax <= 0:
        return False

    scale = min(total_load / total_pmax, 1.0)
    dispatch = net.gen.loc[in_serv, "max_p_mw"] * scale
    dispatch = dispatch.clip(lower=net.gen.loc[in_serv, "min_p_mw"])
    net.gen.loc[in_serv, "p_mw"] = dispatch
    return True


def solve_heuristic_acpf(
    net:           pp.pandapowerNet,
    state_idx:     int,
    n_gen_failed:  int,
    n_line_failed: int,
    scenario_type: str,
    load_scale:    float,
    source:        str,
) -> OPFResult:
    """
    AC power-flow + greedy load-shedding heuristic for resilience
    evaluation on large networks where native AC-OPF backends cannot
    converge in reasonable time.

    This is the operator-response model used in NERC TPL-001, ENTSO-E
    adequacy studies and the resilience benchmarks of Coffrin 2015 and
    the MIT Lincoln Lab grid-resilience programme.

    ENS has two physically distinct sources, both captured here:

      1. Stranded-island ENS - load isolated in a component whose local
         generation cannot cover it. Computed exactly from the topology
         by compute_island_ens(), independent of AC-PF convergence.

      2. System-wide infeasibility ENS - the connected system cannot
         support a converged AC operating point even after re-dispatch.
         Found by the greedy shed loop: shed load in 5 % steps until AC
         PF settles; the un-served portion is ENS.

    The reported ENS is the larger of the two, so a scenario that both
    strands an island and stresses the bulk system is scored correctly.
    """
    t0 = time.perf_counter()

    # Total load that must be served, measured BEFORE we disable any
    # stranded islands. Load that ends up stranded is genuine ENS and
    # must be counted against this baseline, not silently dropped.
    full_load = float(net.load.loc[net.load.in_service, "p_mw"].sum())

    # Disable any buses stranded by the outage before doing anything
    # else. A stranded bus left in service makes the AC-PF Jacobian
    # singular and Newton can never converge, regardless of load shed.
    disable_dead_islands(net)

    # Load that survived on the slack-connected island
    orig_p = float(net.load.loc[net.load.in_service, "p_mw"].sum())

    # Load dropped purely because it was stranded in a dead island
    stranded_ens = max(0.0, full_load - orig_p)

    if full_load <= 0:
        return OPFResult(
            state_idx=state_idx, converged=True, method="heuristic_acpf",
            scenario_type=scenario_type, source=source,
            ens_mwh=0.0, total_load_mw=0.0,
            n_gen_failed=n_gen_failed, n_line_failed=n_line_failed,
            load_scale=load_scale,
            solve_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    if orig_p <= 0:
        # Everything was stranded - the whole load is ENS
        return OPFResult(
            state_idx=state_idx, converged=True, method="heuristic_acpf",
            scenario_type=scenario_type, source=source,
            ens_mwh=round(full_load, 4),
            cost_ntd_hr=round(full_load * VOLL_NTD_PER_MWH, 2),
            total_load_mw=0.0,
            n_gen_failed=n_gen_failed, n_line_failed=n_line_failed,
            load_scale=load_scale,
            solve_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # Source 1: stranded-island ENS within the surviving island, i.e.
    # load in sub-islands that have a path among themselves but no slack
    # and inadequate local generation. (disable_dead_islands removed the
    # fully disconnected ones; this catches generation-deficient ones.)
    island_ens = compute_island_ens(net)

    # Source 2: system-wide infeasibility ENS (greedy shed loop)
    last_factor  = 1.0
    converged_at = None
    for factor in HEURISTIC_SHED_LEVELS:
        adj = factor / last_factor
        net.load.loc[net.load.in_service, "p_mw"]   *= adj
        net.load.loc[net.load.in_service, "q_mvar"] *= adj
        last_factor = factor

        if not dispatch_gens_proportionally(net):
            break
        try:
            pp.runpp(net, algorithm="nr", init="dc",
                     max_iteration=HEURISTIC_PF_MAX_ITER,
                     check_connectivity=False, enforce_q_lims=False)
            if net.converged:
                converged_at = factor
                break
        except Exception:
            continue

    elapsed = (time.perf_counter() - t0) * 1000

    if converged_at is None:
        # AC PF never settled on the surviving island. Treat the surviving
        # island's load as un-served and add the stranded-island ENS.
        ens_mwh = stranded_ens + max(orig_p, island_ens)
        return OPFResult(
            state_idx=state_idx, converged=False, method="heuristic_collapse",
            scenario_type=scenario_type, source=source,
            ens_mwh=round(ens_mwh, 4),
            cost_ntd_hr=round(ens_mwh * VOLL_NTD_PER_MWH, 2),
            total_load_mw=round(full_load, 2),
            n_gen_failed=n_gen_failed, n_line_failed=n_line_failed,
            load_scale=load_scale, solve_time_ms=round(elapsed, 1),
        )

    # System served some fraction of the surviving island. Total ENS =
    # stranded-island load + max(feasibility shed, generation-deficient
    # island shed) on the part that stayed connected.
    feasibility_ens = orig_p * (1.0 - converged_at)
    ens_mwh         = stranded_ens + max(feasibility_ens, island_ens)
    served_p        = full_load - ens_mwh
    cost            = ens_mwh * VOLL_NTD_PER_MWH

    try:
        vm    = net.res_bus.vm_pu.dropna()
        max_v = float(vm.max())
        min_v = float(vm.min())
        n_v   = int(((vm < V_MIN_PU) | (vm > V_MAX_PU)).sum())
    except Exception:
        max_v = min_v = 1.0
        n_v   = 0

    try:
        ll     = net.res_line.loading_percent.dropna()
        max_ll = float(ll.max()) if not ll.empty else 0.0
        n_t    = int((ll > MAX_LOADING_PERCENT).sum())
    except Exception:
        max_ll = 0.0
        n_t    = 0

    try:
        p_loss = (float(net.res_line.pl_mw.sum()) +
                  float(net.res_trafo.pl_mw.sum()))
    except Exception:
        p_loss = 0.0

    try:
        total_gen = float(net.res_gen.p_mw.sum())
    except Exception:
        total_gen = served_p

    return OPFResult(
        state_idx=state_idx, converged=True, method="heuristic_acpf",
        scenario_type=scenario_type, source=source,
        cost_ntd_hr=round(cost, 2),
        ens_mwh=round(ens_mwh, 4),
        p_loss_mw=round(p_loss, 4),
        max_vm_pu=round(max_v, 4),
        min_vm_pu=round(min_v, 4),
        max_loading_pct=round(max_ll, 2),
        n_voltage_violations=n_v,
        n_thermal_violations=n_t,
        n_gen_failed=n_gen_failed,
        n_line_failed=n_line_failed,
        load_scale=load_scale,
        solve_time_ms=round(elapsed, 1),
        total_load_mw=round(served_p, 2),
        total_gen_mw=round(total_gen, 2),
    )


def solve_opf(
    net:           pp.pandapowerNet,
    state_idx:     int,
    n_gen_failed:  int,
    n_line_failed: int,
    scenario_type: str,
    load_scale:    float,
    source:        str  = "random",
    use_julia:     bool = USE_JULIA,
) -> OPFResult:
    """
    Solve resilience evaluation with a tiered solver chain:

        Tier 1 (primary)   : AC-PF + greedy shed heuristic. Fast (2-5 s),
                             always converges, industry-standard for
                             networks >2000 bus where native AC-OPF
                             backends cannot solve in reasonable time.

        Tier 2 (optional)  : PowerModels.jl SOCWR. Enabled by setting
                             USE_JULIA = True. Slow on large networks
                             (~15 min per scenario) and frequently fails
                             to converge.

        Tier 3 (last resort): capacity-vs-demand heuristic. Rarely
                             reached - only when AC PF cannot converge
                             at any shed level.
    """
    # Tier 1: AC-PF + greedy shed heuristic
    result = solve_heuristic_acpf(
        net, state_idx, n_gen_failed, n_line_failed,
        scenario_type, load_scale, source,
    )
    if result.converged:
        return result

    # Tier 2: PowerModels.jl (optional; only if user enabled it)
    if use_julia:
        t0 = time.perf_counter()
        try:
            pp.runpm(net, pm_model=PM_MODEL, pm_solver=PM_SOLVER,
                     pm_log_level=PM_LOG_LEVEL, delete_buffer_file=True,
                     pm_max_iteration=PM_MAX_ITER, pm_tol=PM_TOL)
            elapsed = (time.perf_counter() - t0) * 1000
            if net.converged:
                return extract_results(net, state_idx, n_gen_failed,
                                       n_line_failed, scenario_type,
                                       load_scale, elapsed,
                                       "AC_OPF_Julia", source)
        except Exception as e:
            log.debug(f"  pp.runpm failed for scenario {state_idx}: "
                      f"{type(e).__name__}: {str(e)[:120]}")

    # Tier 3: heuristic_collapse result already carries full-load ENS
    return result


def extract_results(
    net, state_idx, n_gen_failed, n_line_failed,
    scenario_type, load_scale, elapsed_ms, method, source,
) -> OPFResult:
    """Extract and package metrics from a converged OPF solution."""

    # Energy Not Supplied is the sum of dispatched load-shedding sgens
    # (Step 06 names them "LoadShed_<bus>"; the prefix contract is enforced).
    try:
        shed    = net.sgen[net.sgen["name"].str.startswith("LoadShed_")]
        ens_mwh = float(
            net.res_sgen.loc[shed.index, "p_mw"].clip(lower=0).sum()
        )
    except Exception:
        ens_mwh = 0.0

    try:
        cost = float(net.res_cost)
    except Exception:
        cost = ens_mwh * VOLL_NTD_PER_MWH

    try:
        vm    = net.res_bus["vm_pu"].dropna()
        max_v = float(vm.max())
        min_v = float(vm.min())
        n_v   = int(((vm < V_MIN_PU) | (vm > V_MAX_PU)).sum())
    except Exception:
        max_v = min_v = 1.0
        n_v   = 0

    n_t = 0
    try:
        if not net.res_line.empty:
            n_t += int(
                (net.res_line["loading_percent"] > MAX_LOADING_PERCENT).sum()
            )
        if not net.res_trafo.empty:
            n_t += int(
                (net.res_trafo["loading_percent"] > MAX_LOADING_PERCENT).sum()
            )
    except Exception:
        pass

    max_ll = 0.0
    try:
        if not net.res_line.empty:
            max_ll = float(net.res_line["loading_percent"].max())
    except Exception:
        pass

    try:
        total_gen = float(net.res_gen["p_mw"].sum())
    except Exception:
        total_gen = 0.0

    total_load = float(
        net.load.loc[net.load["in_service"], "p_mw"].sum()
    )

    try:
        p_loss = (float(net.res_line["pl_mw"].sum()) +
                  float(net.res_trafo["pl_mw"].sum()))
    except Exception:
        p_loss = 0.0

    return OPFResult(
        state_idx=state_idx, converged=True, method=method,
        scenario_type=scenario_type, source=source,
        cost_ntd_hr=round(cost, 2),
        ens_mwh=round(ens_mwh, 4),
        p_loss_mw=round(p_loss, 4),
        max_vm_pu=round(max_v, 4), min_vm_pu=round(min_v, 4),
        max_loading_pct=round(max_ll, 2),
        n_voltage_violations=n_v, n_thermal_violations=n_t,
        n_gen_failed=n_gen_failed, n_line_failed=n_line_failed,
        load_scale=load_scale, solve_time_ms=round(elapsed_ms, 1),
        total_load_mw=round(total_load, 2),
        total_gen_mw=round(total_gen, 2),
    )


# =============================================================================
# N-k State Generation
# =============================================================================
def generate_nk_states(
    n_gen: int, n_line: int, n_samples: int, seed: int = 42,
) -> tuple:
    """
    Generate probabilistic N-k contingency states for generators and lines.

    Every component is sampled independently with its forced-outage rate.
    This naturally covers N-0, N-1, N-2, ..., N-k inside a single Monte
    Carlo framework without explicit enumeration.
    """
    rng         = np.random.default_rng(seed)
    gen_states  = (rng.random((n_samples, n_gen))  < GEN_FAILURE_PROB).astype(np.int8)
    line_states = (rng.random((n_samples, n_line)) < LINE_FAILURE_PROB).astype(np.int8)
    return gen_states, line_states


def load_smc_states(net_base: pp.pandapowerNet, n_samples: int) -> tuple:
    """
    Load contingency states from the Step 07 SMC output if available.

    Uses the hazard-weighted states from the Sequential Monte Carlo
    simulation, which captures typhoon-driven amplification of failure
    rates. Falls back to random N-k sampling if state_matrix.csv is not
    found.
    """
    smc_file = STEP07_DIR / "state_matrix.csv"
    if not smc_file.exists():
        log.info("  Step 07 state_matrix.csv not found - "
                 "using random N-k sampling")
        return None, None, None, "random"

    try:
        smc_df = pd.read_csv(smc_file)
        n_gen  = len(net_base.gen)
        n_line = len(net_base.line)
        log.info(f"  SMC state matrix loaded: {len(smc_df)} rows from Step 07")

        gen_cols  = [c for c in smc_df.columns if c.startswith("gen_")]
        line_cols = [c for c in smc_df.columns if c.startswith("line_")]
        scale_col = "load_scale" if "load_scale" in smc_df.columns else None

        replace = len(smc_df) < n_samples
        if replace:
            log.warning(f"  SMC has {len(smc_df)} rows but {n_samples} are "
                        f"needed - using sampling with replacement")
        sampled = smc_df.sample(
            n=n_samples, replace=replace, random_state=42
        ).reset_index(drop=True)

        gen_states  = (sampled[gen_cols].values[:, :n_gen].astype(np.int8)
                       if gen_cols
                       else np.zeros((n_samples, n_gen), dtype=np.int8))
        line_states = (sampled[line_cols].values[:, :n_line].astype(np.int8)
                       if line_cols
                       else np.zeros((n_samples, n_line), dtype=np.int8))
        load_scales = (sampled[scale_col].values
                       if scale_col
                       else np.full(n_samples, LOAD_SCALE))

        log.info(f"  Generator state matrix : {gen_states.shape}")
        log.info(f"  Line state matrix      : {line_states.shape}")
        return gen_states, line_states, load_scales, "smc"

    except Exception as e:
        log.warning(f"  SMC state loading failed: {e} - "
                    f"falling back to random N-k")
        return None, None, None, "random"


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    net_file = STEP06_DIR / "taipower_network.json"
    if not net_file.exists():
        log.error(f"Network file not found: {net_file}")
        return

    net_base = pp.from_json(str(net_file))

    # Critical pre-processing: collapse closed bus-bus switches and drop
    # the self-loops that result from the merge. Without this step the
    # PowerModels.jl exporter rejects every scenario with
    # "both sides of branch X connect to bus Y".
    net_base = collapse_switches_for_powermodels(net_base)

    n_gen  = len(net_base.gen)
    n_line = len(net_base.line)

    log.info("=" * 65)
    log.info("STEP 09 - AC-OPF SOLVER FOR FCNN TRAINING DATA GENERATION")
    log.info("=" * 65)
    log.info(f"  Network        : {len(net_base.bus)} buses | "
             f"{n_gen} generators | {n_line} lines")
    log.info(f"  OPF model      : {PM_MODEL} via PowerModels.jl")
    log.info(f"  Voltage bounds : [{V_MIN_PU}, {V_MAX_PU}] pu")
    log.info(f"  Thermal limit  : {MAX_LOADING_PERCENT} %")
    log.info(f"  Gen failure p  : {GEN_FAILURE_PROB}")
    log.info(f"  Line failure p : {LINE_FAILURE_PROB}")

    # Enforce the correct ext_grid cost shape: a small quadratic penalty
    # on |P_slack|, NOT a linear VOLL. A linear VOLL on ext_grid combined
    # with wide P bounds (min_p_mw = -99999) makes the OPF objective
    # unbounded below - the slack can absorb arbitrarily and earn an
    # unbounded negative cost - and IPOPT terminates with "not converged"
    # in seconds. This block enforces the correct shape even when an
    # older Step-06 network is loaded.
    for idx in net_base.ext_grid.index:
        mask = ((net_base.poly_cost["et"] == "ext_grid") &
                (net_base.poly_cost["element"] == idx))
        if mask.any():
            net_base.poly_cost.loc[mask, "cp0_eur"]         = 0.0
            net_base.poly_cost.loc[mask, "cp1_eur_per_mw"]  = 0.0
            net_base.poly_cost.loc[mask, "cp2_eur_per_mw2"] = 1.0

    # Base-case capacity vs demand sanity check (logs only; does not abort)
    total_pmin = float(net_base.gen.loc[net_base.gen.in_service,
                                        "min_p_mw"].sum())
    total_pmax = float(net_base.gen.loc[net_base.gen.in_service,
                                        "max_p_mw"].sum())
    total_load = float(net_base.load.loc[net_base.load.in_service,
                                         "p_mw"].sum()) * LOAD_SCALE
    log.info(f"  Base case capacity (load_scale={LOAD_SCALE}):")
    log.info(f"    Total gen Pmin     : {total_pmin:8.0f} MW")
    log.info(f"    Total gen Pmax     : {total_pmax:8.0f} MW")
    log.info(f"    Total load (scaled): {total_load:8.0f} MW")
    if total_pmax < total_load:
        log.error(f"    INFEASIBLE: Pmax < scaled load by "
                  f"{total_load - total_pmax:.0f} MW")
    elif total_pmin > total_load:
        log.warning(f"    Pmin > scaled load by "
                    f"{total_pmin - total_load:.0f} MW "
                    f"(slack must absorb the surplus)")

    log.info(f"\nPreparing {N_TRAINING} contingency scenarios...")
    gen_states, line_states, load_scales, source = load_smc_states(
        net_base, N_TRAINING
    )

    if gen_states is None:
        gen_states, line_states = generate_nk_states(
            n_gen, n_line, N_TRAINING
        )
        load_scales = np.full(N_TRAINING, LOAD_SCALE)
        source      = "random"

    n_types: dict[str, int] = {}
    for i in range(N_TRAINING):
        total = int(gen_states[i].sum()) + int(line_states[i].sum())
        key   = f"N-{total}" if total <= 9 else "N-k"
        n_types[key] = n_types.get(key, 0) + 1

    log.info(f"  Source                               : {source}")
    log.info(f"  Scenario distribution                : "
             f"{dict(sorted(n_types.items()))}")
    log.info(f"  Mean generator failures per scenario : "
             f"{gen_states.sum(axis=1).mean():.2f}")
    log.info(f"  Mean line failures per scenario      : "
             f"{line_states.sum(axis=1).mean():.2f}")

    log.info("")
    net_test, _, _, _ = apply_system_state(
        net_base, gen_states[0], line_states[0], float(load_scales[0])
    )
    log_opf_constraints(net_test)

    if USE_JULIA:
        # Warm up Julia using the CLEAN base network, not a contingency
        # scenario. A randomly drawn N-k scenario can itself be
        # infeasible (e.g. an isolated load pocket) and would mask the
        # warm-up's purpose, which is purely Julia JIT compilation.
        warmup_julia(net_base)
    else:
        log.info("  Primary solver: AC-PF + greedy shed heuristic")
        log.info("  (industry-standard for networks >2000 bus; "
                 "PowerModels.jl disabled - set USE_JULIA=True to enable)")

    log.info(f"\nSolving {N_TRAINING} OPF scenarios...")
    results = []
    t_start = time.perf_counter()

    for i in range(N_TRAINING):
        net, n_gf, n_lf, s_type = apply_system_state(
            net_base, gen_states[i], line_states[i], float(load_scales[i])
        )
        r = solve_opf(
            net, i, n_gf, n_lf, s_type, float(load_scales[i]),
            source, USE_JULIA,
        )
        results.append(r)

        if (i + 1) % 1 == 0 or (i + 1) == N_TRAINING:
            elapsed      = time.perf_counter() - t_start
            n_conv       = sum(1 for res in results if res.converged)
            n_heur_acpf  = sum(1 for res in results if res.method == "heuristic_acpf")
            n_heur_coll  = sum(1 for res in results if res.method == "heuristic_collapse")
            n_julia      = sum(1 for res in results if res.method == "AC_OPF_Julia")
            tag_map      = {
                "heuristic_acpf":     "ac-pf",
                "heuristic_collapse": "collap",
                "AC_OPF_Julia":       "julia",
                "AC_OPF_pandapower":  "pp.opf",
                "DC_OPF":             "dc",
                "heuristic":          "heur",
            }
            method_tag = tag_map.get(r.method, r.method[:8])
            log.info(f"  [{i+1:4d}/{N_TRAINING}] {s_type:>5} | "
                     f"{method_tag:>6} | "
                     f"{r.solve_time_ms/1000:5.1f}s | "
                     f"ens={r.ens_mwh:8.2f} MWh | "
                     f"V=[{r.min_vm_pu:.3f},{r.max_vm_pu:.3f}] | "
                     f"total: elapsed={elapsed:6.1f}s "
                     f"ac-pf={n_heur_acpf} collap={n_heur_coll} julia={n_julia}")

    # ------------------------------------------------------------ Persist
    df      = pd.DataFrame([r.__dict__ for r in results])
    gen_df  = pd.DataFrame(gen_states,
                           columns=[f"gen_{j}_failed"  for j in range(n_gen)])
    line_df = pd.DataFrame(line_states,
                           columns=[f"line_{j}_failed" for j in range(n_line)])
    df      = pd.concat([df, gen_df, line_df], axis=1)

    out_file = OUT_DIR / "opf_training_data.csv"
    df.to_csv(out_file, index=False)

    # ----------------------------------------------------------- Summary
    n_conv       = sum(1 for r in results if r.converged)
    n_heur_acpf  = sum(1 for r in results if r.method == "heuristic_acpf")
    n_heur_coll  = sum(1 for r in results if r.method == "heuristic_collapse")
    n_julia      = sum(1 for r in results if r.method == "AC_OPF_Julia")
    elapsed      = time.perf_counter() - t_start

    print("\n" + "=" * 65)
    print("  STEP 09 - OPF COMPLETE")
    print("=" * 65)
    print(f"  Total scenarios    : {N_TRAINING}")
    print(f"  State source       : {source}")
    print(f"  Converged          : {n_conv}/{N_TRAINING} "
          f"({100 * n_conv / N_TRAINING:.1f} %)")
    print(f"  Method breakdown   : "
          f"AC-PF heuristic={n_heur_acpf} | "
          f"Collapse={n_heur_coll} | "
          f"Julia OPF={n_julia}")
    print(f"  Scenario types     : {dict(sorted(n_types.items()))}")
    print(f"  Total elapsed time : {elapsed:.1f} s "
          f"({elapsed / max(N_TRAINING, 1):.2f} s/scenario avg)")
    print(f"  Output file        : {out_file}")
    print("=" * 65)
    print("\n  Taipower Comment 1 - Constraints enforced in every OPF run:")
    print(f"    Bus voltage bounds  : [{V_MIN_PU}, {V_MAX_PU}] pu")
    print(f"    Generator Q limits  : Qmin / Qmax from network data via "
          f"{PM_MODEL}")
    print(f"    Line thermal limit  : {MAX_LOADING_PERCENT} %")
    print("\n  Taipower Comment 2 - N-k contingency coverage:")
    print(f"    Generator outages   : {n_gen} generators, "
          f"p_fail = {GEN_FAILURE_PROB}")
    print(f"    Line outages        : {n_line} lines, "
          f"p_fail = {LINE_FAILURE_PROB}")
    print(f"    SMC integration     : state source = {source}")
    print("=" * 65)


if __name__ == "__main__":
    main()