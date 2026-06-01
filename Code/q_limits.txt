import pandapower as pp
import pandas as pd

net = pp.from_json(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")

# Cek gen
print("=== GEN Q LIMITS ===")
print(net.gen[['name','p_mw','min_q_mvar','max_q_mvar']].to_string())

# Cek sgen
print("\n=== SGEN Q LIMITS ===")
print(net.sgen[['name','p_mw','min_q_mvar','max_q_mvar']].to_string())