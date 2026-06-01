"""
test_with_progress.py
======================
Wrap reXplan Network() with progress monitoring to distinguish:
  - Genuine hang (stuck)
  - Just slow (progressing, need to wait)

This patches pp.create_line_from_parameters to print a dot every 100 lines.
If you see dots progressing → just wait, it's slow but working.
If no dots for 60+ seconds → genuine hang, Ctrl+C OK.

Run:
    python test_with_progress.py
"""
import sys
import time
import warnings
from pathlib import Path

# Suppress the noisy "Missing scaling field" warnings
warnings.filterwarnings("ignore", message='Missing "scaling" field')

import pandapower as pp
import reXplan

# Monkey-patch pp.create_line_from_parameters to show progress
_original_create_line = pp.create_line_from_parameters
_line_counter = [0]
_last_print_time = [time.perf_counter()]
_start_time = [None]


def patched_create_line(*args, **kwargs):
    if _start_time[0] is None:
        _start_time[0] = time.perf_counter()
    result = _original_create_line(*args, **kwargs)
    _line_counter[0] += 1
    now = time.perf_counter()
    # Print every 50 lines or every 2 seconds
    if _line_counter[0] % 50 == 0 or (now - _last_print_time[0]) > 2:
        elapsed = now - _start_time[0]
        rate = _line_counter[0] / elapsed
        print(f"  [LINES] {_line_counter[0]} created | "
              f"elapsed {elapsed:.1f}s | rate {rate:.1f}/s", flush=True)
        _last_print_time[0] = now
    return result


pp.create_line_from_parameters = patched_create_line

# Same for transformer and load
_original_create_trafo = pp.create_transformer_from_parameters
_trafo_counter = [0]


def patched_create_trafo(*args, **kwargs):
    result = _original_create_trafo(*args, **kwargs)
    _trafo_counter[0] += 1
    if _trafo_counter[0] % 100 == 0:
        print(f"  [TRAFOS] {_trafo_counter[0]} created", flush=True)
    return result


pp.create_transformer_from_parameters = patched_create_trafo

_original_create_load = pp.create_load
_load_counter = [0]


def patched_create_load(*args, **kwargs):
    result = _original_create_load(*args, **kwargs)
    _load_counter[0] += 1
    if _load_counter[0] % 100 == 0:
        print(f"  [LOADS] {_load_counter[0]} created", flush=True)
    return result


pp.create_load = patched_create_load

# Set reXplan config path
import reXplan.config as cfg
cfg.path.inputFolder = r"C:\reXplan-repo\file\input"

# Try to load
from reXplan.network import Network

print("=" * 60)
print("Loading Network('taipower') with progress monitoring...")
print("=" * 60)

t_start = time.perf_counter()
try:
    net = Network("taipower")
    elapsed = time.perf_counter() - t_start
    print(f"\n✓ Network loaded in {elapsed:.1f} seconds")
    print(f"  Buses: {len(net.pp_network.bus)}")
    print(f"  Lines: {len(net.pp_network.line)}")
    print(f"  Trafos: {len(net.pp_network.trafo)}")
    print(f"  Loads: {len(net.pp_network.load)}")
    print(f"  Gens: {len(net.pp_network.gen)}")
    print(f"  Ext_grid: {len(net.pp_network.ext_grid)}")
except Exception as e:
    elapsed = time.perf_counter() - t_start
    print(f"\n✗ Failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()