using PyCall

@pyimport pandapower as pp
@pyimport pandapower.networks as pn

println("pandapower: ", pp.__version__)
println("Testing Julia PandaModels AC-OPF...")

# Create network
net = pn.case9()


if length(net["poly_cost"]) == 0
    println("Adding poly_cost for generators...")
    n_gen = length(net["gen"])
    for idx in 0:(n_gen-1)
        pp.create_poly_cost(net, idx, "gen", cp1_eur_per_mw=40.0)
    end
    n_ext = length(net["ext_grid"])
    for idx in 0:(n_ext-1)
        pp.create_poly_cost(net, idx, "ext_grid", cp1_eur_per_mw=60.0)
    end
end

println("Running pp.runpm (Julia AC-OPF)...")
try
    pp.runpm(net,
        pm_model="ACPPowerModel",
        pm_solver="ipopt",
        pm_log_level=0,
        delete_buffer_file=true,
        verbose=false
    )
    # Akses dengan string key (dari debug, converged adalah string key)
    println("Converged: ", net["converged"])
    if net["converged"]
        println(" SUCCESS! Julia PandaModels AC-OPF is working!")
        println("   Cost: ", net["res_cost"])
    else
        println(" OPF did not converge")
    end
catch e
    println(" runpm failed: ", e)
    
    # Check installed packages
    println("\nChecking Julia packages...")
    for pkg in ["PowerModels", "Ipopt", "PandaModels", "JuMP"]
        try
            @eval using $pkg
            println("   $pkg installed")
        catch
            println("   $pkg NOT installed")
        end
    end
end