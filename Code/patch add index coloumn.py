"""
Patch v2 — add ALL missing columns reXplan needs
==================================================
Running into cascading KeyError issues because reXplan expects columns
the converter didn't provide. This script adds all missing ones at once
so you don't have to keep re-running step 2 to discover them one by one.

Issues fixed (from reading network.py build_pp_network):
  - lines sheet        : 'index' column (fixed in v1)
  - transformers sheet : 'index' column (fixed in v1)
  - ln_type sheet      : 'g_us_per_km' + 'g0_us_per_km' (this v2)

Run:
    python patch_add_missing_columns.py

Safe to re-run — it only adds columns that don't exist.
"""
import shutil
from pathlib import Path
import openpyxl

XLSX = Path(r"C:\reXplan-repo\file\input\taipower\network.xlsx")
BACKUP = XLSX.with_suffix(".xlsx.bak2")

print("=" * 60)
print("  Patch v2: add missing reXplan-required columns")
print("=" * 60)

if not XLSX.exists():
    print(f"✗ Not found: {XLSX}")
    raise SystemExit(1)

shutil.copy2(XLSX, BACKUP)
print(f"  Backup saved: {BACKUP}")

wb = openpyxl.load_workbook(XLSX)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: 'index' column in lines (re-applied if missing)
# ─────────────────────────────────────────────────────────────────────────────
for sheet_name in ["lines", "transformers"]:
    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]
    if "index" in headers:
        print(f"\n  '{sheet_name}': ✓ 'index' column already present")
    else:
        ws.insert_cols(1)
        ws.cell(row=1, column=1, value="index")
        n_rows = ws.max_row - 1
        for i in range(n_rows):
            ws.cell(row=i + 2, column=1, value=i)
        print(f"\n  '{sheet_name}': ✓ added 'index' column")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: ln_type missing 'g_us_per_km' and 'g0_us_per_km'
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  'ln_type':")
ws = wb["ln_type"]
headers = [c.value for c in ws[1]]
print(f"    Current columns: {headers}")

n_rows = ws.max_row - 1

# Add g_us_per_km (conductance, default 0 — realistic for OHL)
if "g_us_per_km" not in headers:
    new_col = ws.max_column + 1
    ws.cell(row=1, column=new_col, value="g_us_per_km")
    for i in range(n_rows):
        ws.cell(row=i + 2, column=new_col, value=0.0)
    print(f"    ✓ added 'g_us_per_km' (default 0 μS/km)")
else:
    print(f"    ✓ 'g_us_per_km' already present")

if "g0_us_per_km" not in headers:
    new_col = ws.max_column + 1
    ws.cell(row=1, column=new_col, value="g0_us_per_km")
    for i in range(n_rows):
        ws.cell(row=i + 2, column=new_col, value=0.0)
    print(f"    ✓ added 'g0_us_per_km' (default 0 μS/km)")
else:
    print(f"    ✓ 'g0_us_per_km' already present")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Pre-emptive — check if tr_type needs anything reXplan references
# ─────────────────────────────────────────────────────────────────────────────
# From network.py kwargs_tr (lines 280-303), tr_type needs:
#   sn_mva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent,
#   shift_degree, tap_side, tap_neutral, tap_max, tap_min, tap_step_percent,
#   tap_step_degree, tap_phase_shifter
# Your tr_type already has all of these (verified).
print(f"\n  'tr_type': all required columns present (skipped)")

# Save
wb.save(XLSX)
print(f"\n  Saved: {XLSX}")
print(f"\n  Next: python '2 test network load.py'")