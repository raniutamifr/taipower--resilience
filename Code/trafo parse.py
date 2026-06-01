"""
parse_transformers_fix.py
==========================
Drop-in replacement for parse_transformers() in 1_PSSEparser_to_network.py

Key fix:
  - Original only handles 2-winding (4-line records)
  - PSSE v32 also has 3-winding transformers (5-line records)
  - 3W detection: K != 0 on line 1 (vs K = 0 for 2W)
  - Taipower gen step-ups (345/13.8 kV) are 3W → missing from original parse!

Integration:
  1. Copy the parse_transformers() function below
  2. REPLACE the function with the same name in 1_PSSEparser_to_network.py
  3. Re-run: python 1_PSSEparser_to_network.py
  4. Verify: transformer count increases, 3W records appear

PSSE v32 3-winding transformer format:
  Line 1: I, J, K, CKT, CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, O1, F1, ...
          → K != 0 marks this as 3W
  Line 2: R1-2, X1-2, SBASE1-2, R2-3, X2-3, SBASE2-3, R3-1, X3-1, SBASE3-1, VMSTAR, ANSTAR
  Line 3: WINDV1, NOMV1, ANG1, RATA1, RATB1, RATC1, COD1, CONT1, RMA1, RMI1, ...
  Line 4: WINDV2, NOMV2, ANG2, RATA2, RATB2, RATC2, COD2, CONT2, RMA2, RMI2, ...
  Line 5: WINDV3, NOMV3, ANG3, RATA3, RATB3, RATC3, COD3, CONT3, RMA3, RMI3, ...
"""

import re
import logging
import pandas as pd

log = logging.getLogger(__name__)

X_ZERO_THRESHOLD = 1e-9


def decode_big5(s: str) -> str:
    """Decode a latin-1 string that contains Big5-encoded Chinese characters."""
    try:
        return s.encode("latin-1").decode("big5")
    except Exception:
        return s


def parse_transformers(lines: list) -> pd.DataFrame:
    """
    Parse both 2-winding (4 lines) and 3-winding (5 lines) PSSE v32 transformers.
    
    Returns DataFrame with:
      - For 2W: from_bus, to_bus, status, r_pu, x_pu, sbase_mva, vn_hv_kv, vn_lv_kv, ...
      - For 3W: split into 3 separate 2W-equivalent records (star-model)
        hv-star, mv-star, lv-star so downstream code can handle uniformly.
    
    3W to 3×2W conversion uses PSSE star-bus equivalent:
      Z12, Z23, Z31 (pu on own SBASE) → Zp, Zs, Zt (pu on system base)
      where Zp = (Z12 + Z31 - Z23) / 2, etc.
    """
    records_2w = []
    records_3w = []
    i = 0
    line_count = 0
    
    while i < len(lines):
        line1 = lines[i].strip()
        if not line1 or line1.startswith("@"):
            i += 1
            continue
        
        # Parse line 1: I, J, K, 'CKT', CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, ...
        m1 = re.match(
            r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*'([^']*)'\s*,(.+)", line1
        )
        if not m1:
            i += 1
            continue
        
        from_bus = int(m1.group(1))    # I
        to_bus   = int(m1.group(2))    # J
        k_bus    = int(m1.group(3))    # K (0 for 2W, bus number for 3W)
        ckt      = m1.group(4).strip()
        rest1    = m1.group(5).split(",")
        
        # Extract NAME and STAT from rest
        try:
            cw   = int(rest1[0])
            cz   = int(rest1[1])
            cm   = int(rest1[2])
            mag1 = float(rest1[3])
            mag2 = float(rest1[4])
            nmetr = int(rest1[5])
            # Field 6 is the 'NAME' (quoted) — already split, so skip
            # STAT is after NAME
            # Find STAT: look for an integer after the name field
            stat = 1
            for k in range(6, min(10, len(rest1))):
                val = rest1[k].strip().strip("'")
                if val.isdigit():
                    stat = int(val)
                    break
        except (IndexError, ValueError):
            i += 1
            continue
        
        # Detect 2W vs 3W
        is_3w = (k_bus != 0)
        
        if is_3w:
            # ── 3-WINDING: 5 lines total ─────────────────────────────────────
            if i + 4 >= len(lines):
                break
            
            # Line 2: R12, X12, SBASE12, R23, X23, SBASE23, R31, X31, SBASE31, ...
            parts2 = lines[i + 1].strip().split(",")
            try:
                r12     = float(parts2[0])
                x12     = float(parts2[1])
                sbase12 = float(parts2[2])
                r23     = float(parts2[3])
                x23     = float(parts2[4])
                sbase23 = float(parts2[5])
                r31     = float(parts2[6])
                x31     = float(parts2[7])
                sbase31 = float(parts2[8])
            except (IndexError, ValueError):
                i += 5
                continue
            
            # Lines 3, 4, 5: winding data (WINDV_i, NOMV_i, ANG_i, RATA_i, RATB_i, RATC_i, ...)
            parts3 = lines[i + 2].strip().split(",")
            parts4 = lines[i + 3].strip().split(",")
            parts5 = lines[i + 4].strip().split(",")
            
            try:
                nomv1 = float(parts3[1])   # HV winding kV
                rata1 = float(parts3[3])   # HV rating MVA
                nomv2 = float(parts4[1])   # MV winding kV
                rata2 = float(parts4[3])   # MV rating MVA
                nomv3 = float(parts5[1])   # LV winding kV
                rata3 = float(parts5[3])   # LV rating MVA
            except (IndexError, ValueError):
                nomv1 = nomv2 = nomv3 = 0.0
                rata1 = rata2 = rata3 = 0.0
            
            # Fix zero reactance
            if abs(x12) < X_ZERO_THRESHOLD: x12 = 1e-6
            if abs(x23) < X_ZERO_THRESHOLD: x23 = 1e-6
            if abs(x31) < X_ZERO_THRESHOLD: x31 = 1e-6
            
            # Convert 3W Z12/Z23/Z31 (pipe-model) to star-model Zp/Zs/Zt:
            # Zp = (Z12 + Z31 - Z23) / 2
            # Zs = (Z12 + Z23 - Z31) / 2
            # Zt = (Z23 + Z31 - Z12) / 2
            rp = (r12 + r31 - r23) / 2
            xp = (x12 + x31 - x23) / 2
            rs = (r12 + r23 - r31) / 2
            xs = (x12 + x23 - x31) / 2
            rt = (r23 + r31 - r12) / 2
            xt = (r23 + r31 - r12) / 2
            
            # Record the 3W data using pandapower's 3W trafo schema
            records_3w.append({
                "hv_bus": from_bus,       # winding 1 (highest voltage typically)
                "mv_bus": to_bus,         # winding 2
                "lv_bus": k_bus,          # winding 3
                "ckt": ckt,
                "status": stat,
                "vn_hv_kv": nomv1,
                "vn_mv_kv": nomv2,
                "vn_lv_kv": nomv3,
                "sn_hv_mva": rata1,
                "sn_mv_mva": rata2,
                "sn_lv_mva": rata3,
                # Star-equivalent impedances
                "r_hv_pu": rp, "x_hv_pu": xp,
                "r_mv_pu": rs, "x_mv_pu": xs,
                "r_lv_pu": rt, "x_lv_pu": xt,
                # Original Z (for reference)
                "r12_pu": r12, "x12_pu": x12, "sbase12_mva": sbase12,
                "r23_pu": r23, "x23_pu": x23, "sbase23_mva": sbase23,
                "r31_pu": r31, "x31_pu": x31, "sbase31_mva": sbase31,
            })
            
            i += 5
            line_count += 5
            
        else:
            # ── 2-WINDING: 4 lines total ─────────────────────────────────────
            if i + 3 >= len(lines):
                break
            
            # Line 2: R1-2, X1-2, SBASE1-2
            parts2 = lines[i + 1].strip().split(",")
            try:
                r12     = float(parts2[0])
                x12     = float(parts2[1])
                sbase12 = float(parts2[2])
            except (IndexError, ValueError):
                r12 = x12 = sbase12 = 0.0
            
            if abs(x12) < X_ZERO_THRESHOLD: x12 = 1e-6
            
            # Line 3: winding 1
            parts3 = lines[i + 2].strip().split(",")
            try:
                windv1 = float(parts3[0])
                nomv1  = float(parts3[1])
                ang1   = float(parts3[2])
                rata1  = float(parts3[3])
                ratb1  = float(parts3[4])
                ratc1  = float(parts3[5])
            except (IndexError, ValueError):
                windv1 = 1.0; nomv1 = 0.0; ang1 = 0.0
                rata1 = ratb1 = ratc1 = 0.0
            
            # Line 4: winding 2
            parts4 = lines[i + 3].strip().split(",")
            try:
                windv2 = float(parts4[0])
                nomv2  = float(parts4[1])
            except (IndexError, ValueError):
                windv2 = 1.0; nomv2 = 0.0
            
            records_2w.append({
                "from_bus": from_bus, "to_bus": to_bus,
                "k_bus": 0, "ckt": ckt, "status": stat,
                "r12_pu": r12, "x12_pu": x12, "sbase12_mva": sbase12,
                "windv1": windv1, "nomv1": nomv1, "ang1_deg": ang1,
                "rata1_mva": rata1, "ratb1_mva": ratb1, "ratc1_mva": ratc1,
                "windv2": windv2, "nomv2": nomv2,
            })
            
            i += 4
            line_count += 4
    
    df_2w = pd.DataFrame(records_2w)
    df_3w = pd.DataFrame(records_3w)
    
    log.info(f"Parsed {len(df_2w)} 2-winding transformers")
    log.info(f"Parsed {len(df_3w)} 3-winding transformers  ← previously lost!")
    
    return df_2w, df_3w


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    RAW_FILE = Path(r"C:\reXplan-repo\Project Taipower\Data\11507DP_base(108).raw")
    raw_text = RAW_FILE.read_text(encoding="latin-1")
    all_lines = raw_text.splitlines()
    
    # Find transformer section
    br_end = tr_end = -1
    for i, line in enumerate(all_lines):
        if "0 /End of Branch data" in line:
            br_end = i
        if "0 /End of Transformer data" in line:
            tr_end = i
            break
    
    tr_lines = all_lines[br_end + 1 : tr_end]
    print(f"Transformer section: {len(tr_lines)} lines from file")
    print()
    
    df_2w, df_3w = parse_transformers(tr_lines)
    
    print(f"\n2-winding transformers: {len(df_2w)}")
    print(f"3-winding transformers: {len(df_3w)}  ← KEY FINDING!")
    print()
    
    if len(df_3w) > 0:
        print("Sample 3-winding trafos (first 10):")
        print(df_3w[["hv_bus", "mv_bus", "lv_bus", "vn_hv_kv", "vn_mv_kv",
                     "vn_lv_kv", "sn_hv_mva"]].head(10).to_string())
        print()
        
        # Key insight: 345 kV → ~20 kV (gen step-up)
        gen_stepups = df_3w[(df_3w["vn_hv_kv"] >= 340) & (df_3w["vn_lv_kv"] < 30)]
        print(f"3W trafos with HV=345kV and LV<30kV (likely gen step-ups): {len(gen_stepups)}")
        if len(gen_stepups) > 0:
            print("These were PREVIOUSLY MISSED by old parser — this is why 345 kV backbone")
            print("appeared to have 0 generators!")
    
    # Save
    OUT = Path(r"C:\reXplan-repo\Project Taipower\Results\step01")
    OUT.mkdir(parents=True, exist_ok=True)
    df_2w.to_csv(OUT / "transformers_2w.csv", index=False)
    df_3w.to_csv(OUT / "transformers_3w.csv", index=False)
    print(f"\nSaved to:")
    print(f"  {OUT / 'transformers_2w.csv'}")
    print(f"  {OUT / 'transformers_3w.csv'}")