"""Root-cause diagnostic: detect bad network parameters."""
import pandapower as pp
import pandas as pd
from pathlib import Path

NET_PATH = r'C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json'
STEP01   = Path(r'C:\reXplan-repo\Project Taipower\Results\step01')
net = pp.from_json(NET_PATH)

# ══════════════════════════════════════════════════════════════════════
# CHECK 1: Are 3W transformer params all DEFAULTS? (column-name bug)
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("CHECK 1: 3W TRANSFORMER PARAMETERS (detect all-default bug)")
print("=" * 65)
t3 = net.trafo3w
for col in ['vn_hv_kv', 'vn_mv_kv', 'vn_lv_kv',
            'vk_hv_percent', 'vk_mv_percent', 'vk_lv_percent',
            'vkr_hv_percent', 'vkr_mv_percent', 'vkr_lv_percent']:
    if col in t3.columns:
        v = t3[col]
        flag = "  <-- ALL SAME (likely default!)" if v.nunique() == 1 else ""
        print(f"{col:16s}: min={v.min():8.3f}  median={v.median():8.3f}  "
              f"max={v.max():8.3f}  n_unique={v.nunique()}{flag}")

# ══════════════════════════════════════════════════════════════════════
# CHECK 2: What columns does transformers_3w.csv ACTUALLY have?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("CHECK 2: transformers_3w.csv ACTUAL COLUMN NAMES")
print("=" * 65)
df3 = pd.read_csv(STEP01 / "transformers_3w.csv", nrows=2)
print(list(df3.columns))

# ══════════════════════════════════════════════════════════════════════
# CHECK 3: 2W transformer params (same potential bug)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("CHECK 3: 2W TRANSFORMER PARAMETERS")
print("=" * 65)
t2 = net.trafo
for col in ['vn_hv_kv', 'vn_lv_kv', 'vk_percent', 'vkr_percent']:
    if col in t2.columns:
        v = t2[col]
        flag = "  <-- ALL SAME" if v.nunique() == 1 else ""
        print(f"{col:14s}: min={v.min():8.3f}  median={v.median():8.3f}  "
              f"max={v.max():8.3f}  n_unique={v.nunique()}{flag}")

# ══════════════════════════════════════════════════════════════════════
# CHECK 4: pandapower built-in structural diagnostic
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("CHECK 4: pandapower diagnostic (impedance / voltage scan)")
print("=" * 65)
try:
    pp.diagnostic(net, report_style='compact')
except Exception as e:
    print(f"diagnostic error: {e}")