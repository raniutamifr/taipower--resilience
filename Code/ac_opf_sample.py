"""
Step 09 - DC-OPF Ultra Simple - Only for Training Data
=======================================================
"""

import os
import logging
import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict

import pandapower as pp

# =============================================================================
# Setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

# Paths
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
STEP06_DIR = RESULT_BASE / "step06"
OUT_DIR = RESULT_BASE / "step09_opf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings - SANGAT SEDERHANA UNTUK TRAINING DATA
N_TRAINING = 100  # Banyak scenarios untuk training
LOAD_SCALE = 0.75
VOLL = 50000

# Probabilitas failure - realistic untuk training
GEN_FAIL_PROB = 0.01   # 1% chance per generator
LINE_FAIL_PROB = 0.005 # 0.5% chance per line

@dataclass
class TrainingSample:
    """Training data sample"""
    scenario_id: int
    n_gen_failed: int
    n_line_failed: int
    total_failures: int
    total_load_mw: float
    available_gen_mw: float
    ens_mwh: float
    cost_ntd: float
    load_scale: float
    
    # Features for training
    mean_gen_failure_prob: float
    mean_line_failure_prob: float

# =============================================================================
# Simple Functions
# =============================================================================

def create_scenario_simple(net_base, gen_failures, line_failures, load_scale):
    """Create scenario by only disabling generators and lines - NO BUS MODIFICATION"""
    net = copy.deepcopy(net_base)
    
    # Scale loads
    net.load["p_mw"] *= load_scale
    net.load["q_mvar"] *= load_scale
    
    # Disable generators
    n_gen_failed = 0
    for i, gen_idx in enumerate(net.gen.index):
        if i < len(gen_failures) and gen_failures[i] == 1:
            net.gen.at[gen_idx, "in_service"] = False
            n_gen_failed += 1
    
    # Disable lines
    n_line_failed = 0
    for i, line_idx in enumerate(net.line.index):
        if i < len(line_failures) and line_failures[i] == 1:
            net.line.at[line_idx, "in_service"] = False
            n_line_failed += 1
    
    # CRITICAL: DO NOT modify buses or switches
    # Keep ext_grid as is - it's the slack bus
    
    return net, n_gen_failed, n_line_failed

def calculate_heuristic_solution(net):
    """Simple heuristic: available generation vs demand"""
    
    # Total load
    total_load = net.load["p_mw"].sum() if not net.load.empty else 0
    
    # Available generation (sum of max_p_mw of in-service generators)
    available_gen = 0
    if not net.gen.empty:
        available_gen = net.gen[net.gen["in_service"]]["max_p_mw"].sum()
    
    # External grid can supply unlimited power (but with cost)
    if not net.ext_grid.empty and net.ext_grid["in_service"].any():
        # External grid can cover any deficit
        ens = max(0, total_load - available_gen)
    else:
        ens = max(0, total_load - available_gen)
    
    return ens

def generate_training_scenarios(n_gen, n_line, n_samples, seed=42):
    """Generate random failure patterns for training"""
    np.random.seed(seed)
    
    gen_failures = np.random.random((n_samples, n_gen)) < GEN_FAIL_PROB
    line_failures = np.random.random((n_samples, n_line)) < LINE_FAIL_PROB
    
    # Convert to int
    gen_failures = gen_failures.astype(np.int8)
    line_failures = line_failures.astype(np.int8)
    
    return gen_failures, line_failures

# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 09 - GENERATING TRAINING DATA (Heuristic-based)")
    print("="*70)
    
    # Load network
    net_file = STEP06_DIR / "taipower_network.json"
    if not net_file.exists():
        log.error(f"Network file not found: {net_file}")
        return
    
    log.info(f"Loading network: {net_file.name}")
    net_base = pp.from_json(str(net_file))
    
    n_gen = len(net_base.gen)
    n_line = len(net_base.line)
    total_load_original = net_base.load["p_mw"].sum()
    total_gen_capacity = net_base.gen["max_p_mw"].sum() if not net_base.gen.empty else 0
    
    log.info(f"Network statistics:")
    log.info(f"  Buses: {len(net_base.bus)}")
    log.info(f"  Generators: {n_gen} (capacity: {total_gen_capacity:.0f} MW)")
    log.info(f"  Lines: {n_line}")
    log.info(f"  Original load: {total_load_original:.0f} MW")
    log.info(f"  Scaled load (x{LOAD_SCALE}): {total_load_original * LOAD_SCALE:.0f} MW")
    
    # Generate training scenarios
    log.info(f"\nGenerating {N_TRAINING} training scenarios...")
    gen_failures, line_failures = generate_training_scenarios(n_gen, n_line, N_TRAINING)
    
    # Calculate statistics
    failures_per_scenario = gen_failures.sum(axis=1) + line_failures.sum(axis=1)
    log.info(f"  Average failures per scenario: {failures_per_scenario.mean():.2f}")
    log.info(f"  Max failures: {failures_per_scenario.max()}")
    log.info(f"  Min failures: {failures_per_scenario.min()}")
    
    # Distribution
    unique, counts = np.unique(failures_per_scenario, return_counts=True)
    log.info(f"  Failure distribution:")
    for k, count in zip(unique, counts):
        log.info(f"    N-{k}: {count} scenarios ({100*count/N_TRAINING:.1f}%)")
    
    # Process each scenario
    log.info(f"\nProcessing scenarios...")
    samples = []
    t_start = time.perf_counter()
    
    for i in range(N_TRAINING):
        # Create scenario
        net, n_gf, n_lf = create_scenario_simple(
            net_base, gen_failures[i], line_failures[i], LOAD_SCALE
        )
        
        # Calculate solution (heuristic)
        ens = calculate_heuristic_solution(net)
        
        # Get load after scaling
        total_load = net.load["p_mw"].sum()
        available_gen = net.gen[net.gen["in_service"]]["max_p_mw"].sum() if not net.gen.empty else 0
        
        # Create training sample
        sample = TrainingSample(
            scenario_id=i,
            n_gen_failed=int(n_gf),
            n_line_failed=int(n_lf),
            total_failures=int(n_gf + n_lf),
            total_load_mw=round(total_load, 2),
            available_gen_mw=round(available_gen, 2),
            ens_mwh=round(ens, 2),
            cost_ntd=round(ens * VOLL, 2),
            load_scale=LOAD_SCALE,
            mean_gen_failure_prob=GEN_FAIL_PROB,
            mean_line_failure_prob=LINE_FAIL_PROB
        )
        samples.append(sample)
        
        # Progress update
        if (i + 1) % max(1, N_TRAINING // 20) == 0:
            elapsed = time.perf_counter() - t_start
            log.info(f"  Processed {i+1:3d}/{N_TRAINING} scenarios | "
                    f"Time: {elapsed:.1f}s | "
                    f"Avg ENS: {np.mean([s.ens_mwh for s in samples[-20:]]):.1f} MWh")
    
    # Create DataFrame
    df = pd.DataFrame([asdict(s) for s in samples])
    
    # Add failure pattern features (for training)
    # Add first few failures as features (to keep file size reasonable)
    max_features = min(100, n_gen)  # Limit to 100 features
    for i in range(max_features):
        df[f'gen_{i}_failed'] = gen_failures[:, i]
    
    max_features_line = min(100, n_line)
    for i in range(max_features_line):
        df[f'line_{i}_failed'] = line_failures[:, i]
    
    # Save
    out_file = OUT_DIR / "training_data_heuristic.csv"
    df.to_csv(out_file, index=False)
    
    # Summary
    elapsed = time.perf_counter() - t_start
    total_ens = df['ens_mwh'].sum()
    total_cost = df['cost_ntd'].sum()
    
    print("\n" + "="*70)
    print("TRAINING DATA GENERATION COMPLETE")
    print("="*70)
    print(f"Total scenarios        : {N_TRAINING}")
    print(f"Total processing time  : {elapsed:.1f} seconds")
    print(f"Avg time per scenario  : {elapsed/N_TRAINING:.2f} seconds")
    print(f"\nEnergy Not Supplied (ENS) statistics:")
    print(f"  Total ENS: {total_ens:.0f} MWh")
    print(f"  Average ENS: {df['ens_mwh'].mean():.2f} MWh")
    print(f"  Max ENS: {df['ens_mwh'].max():.2f} MWh")
    print(f"  Zero ENS scenarios: {(df['ens_mwh'] == 0).sum()}/{N_TRAINING} ({100*(df['ens_mwh']==0).sum()/N_TRAINING:.1f}%)")
    print(f"\nCost statistics:")
    print(f"  Total cost: {total_cost:,.0f} NTD")
    print(f"  Average cost: {df['cost_ntd'].mean():,.0f} NTD")
    print(f"\nFailure statistics:")
    print(f"  Average generator failures: {df['n_gen_failed'].mean():.2f}")
    print(f"  Average line failures: {df['n_line_failed'].mean():.2f}")
    print(f"\nOutput file: {out_file}")
    print(f"File size: {out_file.stat().st_size / 1024:.1f} KB")
    print("="*70)
    
    # Show sample
    print("\nSample training data (first 5 scenarios):")
    print(df[['scenario_id', 'total_failures', 'total_load_mw', 
              'available_gen_mw', 'ens_mwh', 'cost_ntd']].head(10).to_string(index=False))
    
    print("\n✓ Training data generated successfully!")
    print("\nNext steps:")
    print("  1. Use this CSV file for training your FCNN/GNN model")
    print("  2. The heuristic solution provides the target ENS values")
    print("  3. Features include failure patterns + load information")
    
    return df

if __name__ == "__main__":
    try:
        df = main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()