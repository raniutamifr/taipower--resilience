import pandapower as pp

net = pp.from_json(r"C:\reXplan-repo\Project Taipower\Results\step06\taipower_network.json")

# Jalankan OPF baseline (tanpa constraint) dulu
pp.runopp(net, numba=False)

# Lihat bus yang voltasenya paling ekstrem
hasil = net.res_bus[['vm_pu']].copy()
hasil['bus_name'] = net.bus['name']
hasil = hasil.sort_values('vm_pu')

print("=== 20 BUS VOLTASE TERENDAH ===")
print(hasil.head(20).to_string())

print("\n=== 20 BUS VOLTASE TERTINGGI ===")
print(hasil.tail(20).to_string())

print(f"\nTotal bus di bawah 0.95 pu : {(hasil.vm_pu < 0.95).sum()}")
print(f"Total bus di atas 1.05 pu  : {(hasil.vm_pu > 1.05).sum()}")