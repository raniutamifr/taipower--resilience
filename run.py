from reXplan.simulation import Sim

if __name__ == "__main__":
    # Beri nama untuk simulasi
    simulation_name = "MyFirstSimulation"
    print(f"Creating Sim instance with name: {simulation_name}")
    
    sim = Sim(simulationName=simulation_name)
    print("Sim instance created:", sim)
    
    # Cek method yang tersedia
    print("\nChecking available methods...")
    methods = [m for m in dir(sim) if not m.startswith('_')]
    print("Methods available:", methods)
    
    # Coba cari method yang umum untuk menjalankan simulasi
    if hasattr(sim, 'run'):
        print("\nCalling sim.run()...")
        sim.run()
    elif hasattr(sim, 'simulate'):
        print("\nCalling sim.simulate()...")
        sim.simulate()
    elif hasattr(sim, 'execute'):
        print("\nCalling sim.execute()...")
        sim.execute()
    elif hasattr(sim, 'start'):
        print("\nCalling sim.start()...")
        sim.start()
    else:
        print("\nNo common run method found. Please check the methods listed above.")
        print("You might need to call a specific method based on the simulation workflow.")
