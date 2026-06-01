"""
cleanup_old_cost_logs.py
========================
Archives obsolete cost match logs. Only cost_match_log_v4.csv reflects
the current, correct build (南火=oil, 台中新=gas, 174-192 plant matches).
The unversioned + v2 + v3 files were intermediate iterations.

Default mode = dry-run (prints what it would do). Set --apply to move them.
Usage:
  python cleanup_old_cost_logs.py              # dry-run
  python cleanup_old_cost_logs.py --apply      # move to _archive
"""

import sys, shutil
from pathlib import Path

STEP06 = Path(r"C:\reXplan-repo\Project Taipower\Results\step06")
ARCHIVE = STEP06 / "_archive"

OBSOLETE = [
    "cost_match_log.csv",
    "cost_match_log_v2.csv",
    "cost_match_log_v3.csv",
]
KEEP = ["cost_match_log_v4.csv", "bus_index_mapping.csv",
        "bus_index_mapping.xlsx", "taipower_network.json",
        "taipower_network_realcost.json"]

apply = "--apply" in sys.argv

print(f"\n  Scan : {STEP06}\n")
for f in OBSOLETE:
    p = STEP06 / f
    if p.exists():
        kb = p.stat().st_size // 1024
        if apply:
            ARCHIVE.mkdir(exist_ok=True)
            shutil.move(str(p), str(ARCHIVE / f))
            print(f"  MOVED  {f} ({kb} KB) -> _archive/")
        else:
            print(f"  WOULD MOVE  {f} ({kb} KB)")
    else:
        print(f"  not found   {f}")

print("\n  Kept (current):")
for f in KEEP:
    p = STEP06 / f
    if p.exists():
        kb = p.stat().st_size // 1024
        print(f"    {f} ({kb} KB)")

if not apply:
    print("\n  (dry-run; re-run with --apply to actually move files)")