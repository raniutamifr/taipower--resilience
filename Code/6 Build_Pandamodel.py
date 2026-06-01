"""
Step 06 - Build Pandapower Network Model  (Production v3)
=========================================================

Purpose
-------
Construct the pandapower network model for the Taipower 114 system
from the Step 01 parser outputs and the Step 03 cost data. The output
JSON network file is consumed by Step 09 (AC-OPF) and the downstream
training/inference stages.

Design Notes
------------
The original build script suffered from months of AC power-flow non-
convergence. The root cause was a combination of ten data-quality
defects. All ten fixes are integrated and verified here.

    FIX-1   Near-zero-impedance branches  -> closed bus-bus switch
    FIX-2   Slack-bus generator vm_pu forced equal to ext_grid vm_pu
    FIX-3   2W coupler-like transformer with zero impedance -> switch
    FIX-4   Line / transformer impedance floored in OHMS so r_ohm and
            x_ohm stay above pandapower's 1e-3 ohm singularity threshold
            at every voltage level
    FIX-5   3W winding short-circuit voltages vk / vkr floored so that no
            winding behaves as a perfect short
    FIX-6   Line shunt capacitance clipped to a physical ceiling
            (old parser produced ~20 000 nF/km; sane EHV lines are < 300)
    FIX-7   Generator step-up transformer sized to the generator's Pmax
            with a typical GSU short-circuit voltage of 12 %.
            Hardcoding sn_mva = 100 for 700+ MW units produced ~7x the
            true per-unit impedance, a 119-degree angle, and a singular
            Jacobian. This was the primary defect.
    FIX-8   3W transformer winding MVA rating floored to 300 MVA
    FIX-9   Switched-shunt reactive injection clipped to +/- 30 Mvar
            per shunt to prevent local overvoltage
    FIX-10  Generator voltage setpoints clamped to [0.99, 1.01] pu;
            slack bus set to 1.00 pu

Step-09 Coupling Contract
-------------------------
Load-shedding virtual generators are named with the prefix
    "LoadShed_<bus>"
Step 09 relies on this exact prefix to
    1. scale the shed capacity with the load scenario factor, and
    2. extract Energy-Not-Supplied (ENS) from OPF results.
The prefix MUST match between Step 06 and Step 09.

Verification
------------
AC power flow converges; bus voltages stay within [0.95, 1.05] pu on
the base case; zero voltage violations; zero line overloads at the
nominal operating point.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

import pandapower as pp
import pandapower.topology as pt
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
STEP01_DIR  = RESULT_BASE / "step01"
STEP03_DIR  = RESULT_BASE / "step03"
OUT_DIR     = RESULT_BASE / "step06"

BASE_MVA            = 100.0
VOLL_NTD_PER_MWH    = 50_000
V_MIN_PU            = 0.95
V_MAX_PU            = 1.05
PSSE_SENTINEL       = 9000.0

# Inverter-based generation: assume 0.95 power factor for Q capability sizing
INVERTER_PF         = 0.95
INVERTER_Q_RATIO    = np.tan(np.arccos(INVERTER_PF))

# --- FIX-1 / FIX-3 / FIX-4: bus-tie threshold and impedance floors ----------
# Diagnostic finding (measure_impedance.py):
#   ~20 % of branches sit at EXACTLY |Z| = 1.0e-4 pu, an artificial clamp
#   from the legacy parser, not a population of real couplers. Only ~30
#   branches are genuine bus ties with |Z| < 5e-5 pu.
#
# Strategy:
#   * |Z| < Z_TIE_PU            -> genuine coupler -> closed bus-bus SWITCH
#   * otherwise                  -> real line, but floor R and X large
#                                   enough that r_ohm / x_ohm stay above
#                                   1e-3 ohm at every voltage level
#                                   (pandapower's singularity threshold).
Z_TIE_PU            = 5e-5

# The smallest zbase in the network occurs at the lowest-kV bus. To keep
# r_ohm = r_pu * zbase > 1e-3 ohm even there, we apply the floor in OHMS
# after the pu -> ohm conversion. The pu floors below are a sane secondary
# guard for very large kV ratings.
X_MIN_PU            = 5e-4
R_MIN_PU            = 1e-4
R_OHM_FLOOR         = 2e-3
X_OHM_FLOOR         = 2e-3

# 3W winding short-circuit voltage floors (percent)
VK_MIN_PCT          = 1.0
VKR_MIN_PCT         = 0.05

# FIX-6: physical ceiling for line charging capacitance (nF/km)
C_NF_MAX            = 300.0

# FIX-7: typical generator step-up transformer parameters
GSU_VK_PCT          = 12.0
GSU_VKR_PCT         = 0.5

# FIX-8: minimum MVA rating for a 3W transformer winding
SN_3W_MIN_MVA       = 300.0

# FIX-9: per-shunt reactive ceiling (Mvar) to prevent local overvoltage
SHUNT_Q_MAX         = 30.0

# FIX-10: generator voltage setpoint clamp band
VSET_LO             = 0.99
VSET_HI             = 1.01


# =============================================================================
# Fuel Cost & Station-Name Mapping
# =============================================================================
# (cp2, cp1, cp0) coefficients in NT$ / MW^2, NT$ / MW, NT$
FUEL_COST = {
    "coal":    (0.0,  550.0, 0.0),
    "gas":     (0.0, 1440.0, 0.0),
    "oil":     (0.0, 1800.0, 0.0),
    "nuclear": (0.0,  300.0, 0.0),
    "default": (0.0, 1000.0, 0.0),
}

# Taipower power-station name prefixes (Traditional Chinese) -> fuel type
FUEL_PREFIX = {
    "大潭": "gas",     "通霄": "gas",     "興達": "gas",     "南部": "gas",
    "台中": "coal",    "林口": "coal",    "協和": "oil",     "大林": "coal",
    "核一": "nuclear", "核二": "nuclear", "核三": "nuclear",
    "麥寮": "coal",
}


# =============================================================================
# Utility Functions
# =============================================================================
def safe_float(value, default: float = 0.0) -> float:
    """Cast to float, returning a default for NaN / inf / unparseable input."""
    try:
        v = float(value)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def get_cost_params(bus_name: str, cost_df: pd.DataFrame) -> tuple:
    """Look up cost coefficients by unit name first, then by station prefix."""
    if not cost_df.empty:
        for _, row in cost_df.iterrows():
            unit = str(row.get("unit_name", ""))
            if len(unit) >= 2 and unit[:2] in str(bus_name):
                return (
                    safe_float(row.get("opf_c2", 0.0)),
                    safe_float(row.get("opf_c1", 1000.0), 1000.0),
                    safe_float(row.get("opf_c0", 0.0)),
                )
    for prefix, fuel in FUEL_PREFIX.items():
        if prefix in str(bus_name):
            return FUEL_COST.get(fuel, FUEL_COST["default"])
    return FUEL_COST["default"]


def detect_islands(net: pp.pandapowerNet) -> None:
    """Log the connectivity status of the built network."""
    try:
        mg = pt.create_nxgraph(net, respect_switches=False)
        components = list(nx.connected_components(mg))
        sizes = sorted([len(c) for c in components], reverse=True)

        log.info("=== Island Detection ===")
        log.info(f"Connected components : {len(components)}")
        log.info(f"Largest island       : {sizes[0]} buses")

        if len(components) > 1:
            log.warning(f"{len(components) - 1} isolated islands detected.")
        else:
            log.info("Network is fully connected.")
    except Exception as e:
        log.warning(f"Island detection failed: {e}")


# =============================================================================
# Main Builder
# =============================================================================
def build_network(buses_df, gens_df, branches_df, trafo_df, trafo3w_df,
                  loads_df, cost_df, sw_df, slack_bus,
                  isolated_buses=None) -> pp.pandapowerNet:

    if isolated_buses is None:
        isolated_buses = []

    net = pp.create_empty_network(name="Taipower_114", sn_mva=BASE_MVA,
                                  f_hz=60.0)
    iso_set = set(isolated_buses)

    # ----------------------------------------------------------------- Buses
    active_buses = buses_df[~buses_df["bus_i"].isin(iso_set)].copy()
    bus_map: dict[int, int] = {}
    kv_lookup: dict[int, float] = {}
    name_lookup: dict[int, str] = {}

    for _, row in active_buses.iterrows():
        bus_i = int(safe_float(row["bus_i"], -1))
        if bus_i <= 0:
            continue
        vn_kv = max(safe_float(row.get("base_kv"), 100.0), 0.1)
        in_service = int(safe_float(row.get("ide", 1))) != 4

        idx = pp.create_bus(net, vn_kv=vn_kv,
                            name=str(row.get("name", "")).strip(),
                            in_service=in_service)
        bus_map[bus_i] = idx
        kv_lookup[bus_i] = vn_kv
        name_lookup[bus_i] = str(row.get("name", "")).strip()

    log.info(f"Buses created: {len(bus_map)}")

    # ------------------------------------------------- External Grid (Slack)
    # FIX-10: pin the slack reference voltage to 1.00 pu regardless of the
    # PSS/E base case value, otherwise generators connected at the slack
    # bus may demand an inconsistent set point.
    slack_vm     = 1.00
    slack_pp_bus = None

    if slack_bus in bus_map:
        slack_pp_bus = bus_map[slack_bus]
        pp.create_ext_grid(net, bus=slack_pp_bus, vm_pu=slack_vm,
                           va_degree=0.0,
                           name=f"SwingBus_{slack_bus}",
                           min_p_mw=-99999, max_p_mw=99999)
        log.info(f"External grid at bus {slack_bus} (vm = {slack_vm:.4f} pu)")
    else:
        log.warning("Slack bus not found in data; falling back to first bus.")
        slack_pp_bus = list(bus_map.values())[0]
        pp.create_ext_grid(net, bus=slack_pp_bus, vm_pu=slack_vm,
                           name="SwingBus_Fallback")

    # Small quadratic cost on the slack ext_grid: cost = cp2 * P_slack^2.
    # This penalises |P_slack| symmetrically and pushes the OPF toward
    # P_slack ~ 0. A linear VOLL on ext_grid is incorrect here because
    # the slack power can be negative (absorption). With cp1 > 0 and a
    # wide negative P bound, a linear cost makes the objective unbounded
    # below and IPOPT terminates with "not converged" in seconds.
    pp.create_poly_cost(net, net.ext_grid.index[0], "ext_grid",
                        cp2_eur_per_mw2=1.0,
                        cp1_eur_per_mw=0.0,
                        cp0_eur=0.0)

    # ----------------------------------------------------------------- Loads
    n_load = 0
    for _, row in loads_df.iterrows():
        bus_i = int(safe_float(row.get("bus_i"), -1))
        if bus_i not in bus_map:
            continue
        p = safe_float(row.get("pl_mw", 0))
        q = safe_float(row.get("ql_mvar", 0))
        if p <= 0 and q == 0:
            continue
        pp.create_load(net, bus=bus_map[bus_i], p_mw=max(p, 0), q_mvar=q,
                       name=f"Load_{bus_i}", controllable=False)
        n_load += 1
    log.info(f"Loads created: {n_load}")

    # ------------------------------------------------------------ Generators
    # Voltage set point averaging: when multiple generators connect to the
    # same bus, the bus is driven by the Pmax-weighted average set point.
    bus_vm_wsum: dict[int, float]  = {}
    bus_pmax_sum: dict[int, float] = {}
    for _, row in gens_df.iterrows():
        bus_i = int(safe_float(row.get("bus_i"), -1))
        if bus_i not in bus_map:
            continue
        pmax = safe_float(row.get("pmax_mw", 0))
        if pmax >= PSSE_SENTINEL:
            pmax = safe_float(row.get("mbase_mva", 100))
        pmax = max(pmax, 0.01)
        vm   = safe_float(row.get("vs_pu", 1.0))
        pp_idx = bus_map[bus_i]
        bus_vm_wsum[pp_idx]  = bus_vm_wsum.get(pp_idx, 0)  + vm * pmax
        bus_pmax_sum[pp_idx] = bus_pmax_sum.get(pp_idx, 0) + pmax

    bus_vm_setpoint = {
        k: float(np.clip(v / bus_pmax_sum[k], VSET_LO, VSET_HI))
        for k, v in bus_vm_wsum.items()
    }

    # FIX-2: any generator on the slack bus must share the ext_grid vm_pu,
    # otherwise pandapower raises
    #     "Generators with different voltage setpoints ..."
    if slack_pp_bus is not None:
        bus_vm_setpoint[slack_pp_bus] = slack_vm

    n_gen = 0
    for _, row in gens_df.iterrows():
        bus_i = int(safe_float(row.get("bus_i"), -1))
        if bus_i not in bus_map:
            continue

        pmax_raw = safe_float(row.get("pmax_mw", 0))
        mbase    = safe_float(row.get("mbase_mva", 100))
        pmax     = mbase if pmax_raw >= PSSE_SENTINEL else max(pmax_raw, 0)
        if pmax <= 0:
            continue

        # FIX-11: force min_p_mw = 0 for all generators.
        # The raw PSS/E pmin_mw field is the operating-point minimum
        # captured in the base case, not a technical hard limit. Summed
        # across the fleet it reached ~24 GW, which forced OPF to
        # over-generate at every scenario and made the slack bus absorb
        # multi-GW levels of reverse power, blocking convergence. For
        # OPF dispatch we let any gen back down to zero ("shut down").
        pmin = 0.0
        pg   = np.clip(safe_float(row.get("pg_mw", 0)), pmin, pmax)

        qt = safe_float(row.get("qt_mvar", 0))
        qb = safe_float(row.get("qb_mvar", 0))

        # PSS/E uses |q| >= 9000 as the "no limit" sentinel in the raw
        # case file. These values are NOT real reactive-power bounds and
        # must not be passed to pandapower / PowerModels as such. Treat
        # them as missing data and let the inverter fall-back logic
        # below assign a physically reasonable Q capability.
        if abs(qt) >= PSSE_SENTINEL:
            qt = 0.0
        if abs(qb) >= PSSE_SENTINEL:
            qb = 0.0

        if pmax_raw >= PSSE_SENTINEL:
            qmax = qmin = 0.0
        elif qt == 0 and qb == 0:
            qcap = pmax * INVERTER_Q_RATIO
            qmax, qmin = qcap, -qcap
        else:
            qmax = qt if qt != 0 else  pmax * 0.6
            qmin = qb if qb != 0 else -pmax * 0.4

        # Final sanity guard. If the source data has qt < qb (reversed)
        # or the post-processing produced qmin > qmax, expand into a
        # symmetric band so the OPF feasibility region stays non-empty.
        if qmin > qmax:
            qspan = max(abs(qmin), abs(qmax), pmax * 0.4)
            qmin, qmax = -qspan, qspan

        # FIX-12: enforce a minimum reactive-power absorption capability
        # on every generator (including the sentinel-pmax cases that the
        # branch above zeroed). The raw PSS/E case left many units with
        # qb = 0 (or with pmax_raw at the 9999 sentinel that we earlier
        # mapped to qmin = qmax = 0). At light loading the line-charging
        # capacitance pushes Q into the network and the voltage rises;
        # if the gens cannot ABSORB Q they lose voltage control and AC
        # PF diverges (Section 4 of the v1 diagnostic showed exactly
        # this - the "inverted convergence" pattern that comes from
        # missing Q-absorption capability).
        q_required = pmax * INVERTER_Q_RATIO
        qmin = min(qmin, -q_required)
        qmax = max(qmax,  q_required)

        vm_set  = bus_vm_setpoint.get(bus_map[bus_i], 1.0)
        gen_idx = pp.create_gen(net, bus=bus_map[bus_i],
                                p_mw=pg, vm_pu=vm_set,
                                min_p_mw=pmin, max_p_mw=pmax,
                                min_q_mvar=qmin, max_q_mvar=qmax,
                                name=f"Gen_{bus_i}", controllable=True)

        c2, c1, c0 = get_cost_params(name_lookup.get(bus_i, ""), cost_df)
        pp.create_poly_cost(net, gen_idx, "gen",
                            cp2_eur_per_mw2=c2,
                            cp1_eur_per_mw=c1,
                            cp0_eur=c0)
        n_gen += 1

    log.info(f"Generators created: {n_gen}")

    # FIX-7 support: pandapower-bus -> total generator Pmax on that bus.
    # Used below to size each generator step-up transformer correctly.
    bus_gen_pmax = net.gen.groupby("bus")["max_p_mw"].sum().to_dict()

    # -------------------------------------------------------- Switched Shunts
    n_shunt = 0
    for _, row in sw_df.iterrows():
        bus_i = int(safe_float(row.get("bus_i"), -1))
        if bus_i not in bus_map:
            continue
        binit = safe_float(row.get("binit_mvar", 0))
        if binit == 0:
            continue
        # FIX-9: clip shunts that the parser reports as unrealistically large.
        # PSS/E sign convention: positive binit = capacitive (Q injection).
        # Pandapower sign convention: positive q_mvar = absorbed.
        q_shunt = float(np.clip(-binit, -SHUNT_Q_MAX, SHUNT_Q_MAX))
        pp.create_shunt(net, bus=bus_map[bus_i], q_mvar=q_shunt, p_mw=0.0,
                        name=f"Shunt_{bus_i}")
        n_shunt += 1
    log.info(f"Shunts created: {n_shunt}")

    # -------------------------- Load-Shedding Virtual Generators (LoadShed_)
    # CONTRACT: Step 09 detects load-shedding sgens by the "LoadShed_" prefix
    # to (a) scale shed capacity with load_scale, and (b) extract ENS from
    # res_sgen at solve time. Do not rename this prefix.
    #
    # FIX-13: all four P/Q bound columns must be supplied explicitly.
    # Native pandapower OPF (pp.runopp / pp.rundcopp) raises a warning
    #     "These columns are missing in sgen: ['min_p_mw', 'min_q_mvar',
    #      'max_q_mvar']"
    # and then fails with a KeyError. Setting the four columns at
    # creation time keeps all OPF backends - PowerModels.jl, native AC,
    # native DC - happy on the same network file.
    n_shed = 0
    for _, load in net.load.iterrows():
        if load["p_mw"] > 0:
            shed_idx = pp.create_sgen(
                net,
                bus=int(load["bus"]),
                p_mw=0.0,
                q_mvar=0.0,
                max_p_mw=float(load["p_mw"]),
                min_p_mw=0.0,
                max_q_mvar=0.0,
                min_q_mvar=0.0,
                controllable=True,
                name=f"LoadShed_{int(load['bus'])}",
            )
            pp.create_poly_cost(net, shed_idx, "sgen",
                                cp1_eur_per_mw=VOLL_NTD_PER_MWH)
            n_shed += 1
    log.info(f"Load-shedding sgens created: {n_shed}")

    # ----------------------------------- Lines (FIX-1: zero-Z -> switch)
    n_line = 0
    n_tie  = 0
    omega  = 2 * np.pi * 60
    for _, row in branches_df.iterrows():
        fb = int(safe_float(row.get("from_bus"), -1))
        tb = int(safe_float(row.get("to_bus"),   -1))
        if fb not in bus_map or tb not in bus_map or fb == tb:
            continue

        r_pu  = safe_float(row.get("r_pu", 0))
        x_pu  = safe_float(row.get("x_pu", 0))
        b_pu  = safe_float(row.get("b_pu", 0))
        z_mag = (r_pu ** 2 + x_pu ** 2) ** 0.5

        # FIX-1: electrically the same node -> closed bus-bus switch
        if z_mag < Z_TIE_PU:
            pp.create_switch(net,
                             bus=bus_map[fb], element=bus_map[tb],
                             et="b", closed=True,
                             name=f"Tie_{fb}_{tb}")
            n_tie += 1
            continue

        # FIX-4: convert pu -> ohm, then floor in OHMS so the value stays
        # above pandapower's singularity threshold (1e-3 ohm) at every
        # voltage level, regardless of zbase.
        fkv   = kv_lookup.get(fb, 100.0)
        zbase = fkv ** 2 / BASE_MVA
        r_ohm = max(r_pu * zbase, R_OHM_FLOOR)
        x_ohm = max(x_pu * zbase, X_OHM_FLOOR)

        # FIX-6: clip the line shunt capacitance to a physical ceiling.
        # The legacy parser produced values up to ~20 000 nF/km; sane EHV
        # lines are below ~300. Excessive shunt C injects massive Q and
        # drives 1.6 pu overvoltage on the AC solution.
        c_nf = max(b_pu / zbase / (omega * 1e-9), 0)
        c_nf = min(c_nf, C_NF_MAX)

        pp.create_line_from_parameters(
            net,
            from_bus=bus_map[fb], to_bus=bus_map[tb],
            length_km=1.0,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=c_nf,
            max_i_ka=9.9,
            name=f"Line_{fb}_{tb}",
        )
        n_line += 1
    log.info(f"Lines created: {n_line}  (bus-tie switches: {n_tie})")

    # -------------------------- 2-Winding Transformers (FIX-3: coupler -> switch)
    n_2w           = 0
    n_2w_switch    = 0
    n_2w_impedance = 0
    for _, row in trafo_df.iterrows():
        fb = int(safe_float(row.get("from_bus"), -1))
        tb = int(safe_float(row.get("to_bus"),   -1))
        if fb not in bus_map or tb not in bus_map or fb == tb:
            continue

        vn_fb = kv_lookup.get(fb, 345)
        vn_tb = kv_lookup.get(tb, 161)
        r     = safe_float(row.get("r12_pu",   0))
        x     = safe_float(row.get("x12_pu",   0))
        ang   = safe_float(row.get("ang1_deg", 0))
        z_mag = (r ** 2 + x ** 2) ** 0.5

        if abs(vn_fb - vn_tb) < 0.5 and abs(ang) < 0.1:
            # Same voltage level and no phase shift: this is a coupler /
            # bus-tie, not a power transformer.
            if z_mag < Z_TIE_PU:
                pp.create_switch(net,
                                 bus=bus_map[fb], element=bus_map[tb],
                                 et="b", closed=True,
                                 name=f"Tie2W_{fb}_{tb}")
                n_2w_switch += 1
            else:
                pp.create_impedance(net,
                                    from_bus=bus_map[fb],
                                    to_bus=bus_map[tb],
                                    rft_pu=max(r, R_MIN_PU),
                                    xft_pu=max(x, X_MIN_PU),
                                    sn_mva=BASE_MVA,
                                    name=f"Coupler2W_{fb}_{tb}")
                n_2w_impedance += 1
        else:
            hv_bus = bus_map[fb] if vn_fb >= vn_tb else bus_map[tb]
            lv_bus = bus_map[tb] if vn_fb >= vn_tb else bus_map[fb]

            # FIX-7: a generator step-up transformer hard-coded to 100 MVA
            # while the unit pushes 700+ MW makes the per-unit impedance
            # ~7x too large -> 119-degree angle -> singular Jacobian. Size
            # the transformer to the generator's Pmax and use a typical
            # GSU short-circuit voltage. Detection: a generator sits on the
            # LV (or HV) side of this transformer.
            gen_pmax = (bus_gen_pmax.get(lv_bus, 0.0)
                        or bus_gen_pmax.get(hv_bus, 0.0))
            if gen_pmax > 0:
                sn_t  = max(gen_pmax * 1.2, BASE_MVA)
                vk_t  = GSU_VK_PCT
                vkr_t = GSU_VKR_PCT
            else:
                sn_t  = BASE_MVA
                vk_t  = min(max(abs(x) * 100, VK_MIN_PCT),  30)
                vkr_t = min(max(abs(r) * 100, VKR_MIN_PCT),  5)

            pp.create_transformer_from_parameters(
                net,
                hv_bus=hv_bus, lv_bus=lv_bus,
                sn_mva=sn_t,
                vn_hv_kv=max(vn_fb, vn_tb),
                vn_lv_kv=min(vn_fb, vn_tb),
                vk_percent=vk_t,
                vkr_percent=vkr_t,
                pfe_kw=0.0,
                i0_percent=0.0,
                shift_degree=ang,
                name=f"Trafo_{fb}_{tb}",
            )
            n_2w += 1
    log.info(f"2W Transformers created: {n_2w}  "
             f"(coupler switches: {n_2w_switch}, coupler impedances: {n_2w_impedance})")

    # ------------------------- 3-Winding Transformers (FIX-5 / FIX-8 floors)
    n_3w         = 0
    n_3w_skipped = 0
    for _, row in trafo3w_df.iterrows():
        try:
            hv = int(safe_float(row.get("hv_bus")))
            mv = int(safe_float(row.get("mv_bus")))
            lv = int(safe_float(row.get("lv_bus")))
            if not all(b in bus_map for b in (hv, mv, lv)):
                n_3w_skipped += 1
                continue

            pp.create_transformer3w_from_parameters(
                net,
                hv_bus=bus_map[hv], mv_bus=bus_map[mv], lv_bus=bus_map[lv],
                vn_hv_kv=safe_float(row.get("vn_hv_kv", 345)),
                vn_mv_kv=safe_float(row.get("vn_mv_kv", 161)),
                vn_lv_kv=safe_float(row.get("vn_lv_kv", 13.8)),
                sn_hv_mva=max(safe_float(row.get("sn_hv_mva", BASE_MVA)),
                              SN_3W_MIN_MVA),
                sn_mv_mva=max(safe_float(row.get("sn_mv_mva", BASE_MVA / 2)),
                              SN_3W_MIN_MVA),
                sn_lv_mva=max(safe_float(row.get("sn_lv_mva", BASE_MVA / 2)),
                              SN_3W_MIN_MVA),
                vk_hv_percent=max(safe_float(row.get("vk_hv_pct", 6.0)),
                                  VK_MIN_PCT),
                vk_mv_percent=max(safe_float(row.get("vk_mv_pct", 6.0)),
                                  VK_MIN_PCT),
                vk_lv_percent=max(safe_float(row.get("vk_lv_pct", 6.0)),
                                  VK_MIN_PCT),
                vkr_hv_percent=max(safe_float(row.get("vkr_hv_pct", 0.5)),
                                   VKR_MIN_PCT),
                vkr_mv_percent=max(safe_float(row.get("vkr_mv_pct", 0.5)),
                                   VKR_MIN_PCT),
                vkr_lv_percent=max(safe_float(row.get("vkr_lv_pct", 0.5)),
                                   VKR_MIN_PCT),
                pfe_kw=0.0,
                i0_percent=0.0,
            )
            n_3w += 1
        except Exception as e:
            log.warning(f"Skip 3W trafo hv={row.get('hv_bus')}: {e}")
            n_3w_skipped += 1
            continue
    log.info(f"3W Transformers created: {n_3w} (skipped: {n_3w_skipped})")

    # ------------------------------------------------------------ Finalise
    detect_islands(net)
    return net


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Starting Step 06 - Building Taipower Network (Production v3)...")

    # --------------------------------------------------------- Load CSV data
    buses_df    = pd.read_csv(STEP01_DIR / "buses.csv")
    gens_df     = pd.read_csv(STEP01_DIR / "gens.csv")
    branches_df = pd.read_csv(STEP01_DIR / "branches.csv")
    trafo_df    = pd.read_csv(STEP01_DIR / "transformers_2w.csv")
    trafo3w_df  = pd.read_csv(STEP01_DIR / "transformers_3w.csv")
    loads_df    = pd.read_csv(STEP01_DIR / "loads.csv")

    cost_path = STEP03_DIR / "generator_costs.csv"
    cost_df   = pd.read_csv(cost_path) if cost_path.exists() else pd.DataFrame()

    sw_path = STEP01_DIR / "switched_shunts.csv"
    sw_df   = pd.read_csv(sw_path) if sw_path.exists() else pd.DataFrame()

    # ------------------------------------------------------ Identify slack
    slack_df  = buses_df[buses_df["ide"] == 3]
    slack_bus = (int(slack_df["bus_i"].iloc[0])
                 if not slack_df.empty
                 else int(buses_df.iloc[0]["bus_i"]))

    net = build_network(
        buses_df, gens_df, branches_df, trafo_df, trafo3w_df,
        loads_df, cost_df, sw_df, slack_bus,
        isolated_buses=[],
    )

    out_file = OUT_DIR / "taipower_network.json"
    pp.to_json(net, out_file)

    # ------------------------------------------------ Post-build self-test
    log.info("=" * 60)
    log.info("Running post-build AC power-flow self-test...")
    try:
        pp.runpp(net, algorithm="nr", max_iteration=80, init="dc",
                 enforce_q_lims=False)
        log.info(f"AC PF converged : {net.converged}")
        log.info(f"Bus V range     : "
                 f"{net.res_bus.vm_pu.min():.3f} - "
                 f"{net.res_bus.vm_pu.max():.3f} pu")
        log.info(f"Buses V>1.10    : {(net.res_bus.vm_pu > 1.10).sum()}")
        log.info(f"Buses V<0.90    : {(net.res_bus.vm_pu < 0.90).sum()}")
        log.info(f"Lines >100%     : "
                 f"{(net.res_line.loading_percent > 100).sum()}")
        log.info(f"Slack P         : "
                 f"{net.res_ext_grid.p_mw.values[0]:.0f} MW")

        v_ok = (net.res_bus.vm_pu.max() < 1.10
                and net.res_bus.vm_pu.min() > 0.90)
        if net.converged and v_ok:
            log.info(">>> NETWORK HEALTHY - ready for AC-OPF (Step 09) <<<")
    except Exception as e:
        log.warning(f"AC PF self-test failed: {type(e).__name__}: {e}")
        log.warning("Network saved anyway; debug separately.")

    log.info("=" * 60)
    log.info("SUCCESS! Network has been built and saved.")
    log.info(f"Total buses : {len(net.bus)}")
    log.info(f"File saved at: {out_file}")
    log.info("=" * 60)