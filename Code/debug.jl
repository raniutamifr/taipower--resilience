using PyCall

@pyimport pandapower as pp
@pyimport pandapower.networks as pn

println("=== DEBUG: Inspecting net ===")
net = pn.case9()

# Lihat semua keys yang tersedia
println("Keys in net:")
for key in keys(net)
    println("  ", key)
end

# Cek tipe net
println("\nType of net: ", typeof(net))
println("Is net a PyObject? ", PyCall.ispy(net))

# Cek apakah poly_cost ada (mungkin dengan nama berbeda)
if haskey(net, "poly_cost")
    println("\n poly_cost exists as string key")
    println("   Length: ", length(net["poly_cost"]))
elseif haskey(net, :poly_cost)
    println("\n poly_cost exists as symbol key")
else
    println("\n poly_cost not found")
    println("   This network may not have cost data by default")
end