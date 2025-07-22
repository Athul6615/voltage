

import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

def check_convergence(dss):
    return dss.solution.converged

def reset_load_models(dss):
    for name in dss.loads.names:
        if name and name.lower() != "none":
            dss.loads.name = name
            dss.loads.model = 1

def main():
    print("Initializing OpenDSS...")
    dss = py_dss_interface.DSS()
    script_path = os.path.dirname(os.path.abspath(__file__))
    paths = [
        pathlib.Path(script_path).joinpath("feeders", "123Bus", "Run_IEEE123Bus.DSS"),
        pathlib.Path(script_path).joinpath("123Bus", "Run_IEEE123Bus.DSS"),
        pathlib.Path(script_path).joinpath("Run_IEEE123Bus.DSS"),
        pathlib.Path(script_path).joinpath("feeders", "Run_IEEE123Bus.DSS")
    ]
    for path in paths:
        if path.exists():
            print("Compiling:", path)
            dss.text(f'compile "{str(path)}"')
            break
    else:
        raise FileNotFoundError("123-bus DSS file not found")

    dss.text("set mode=snapshot")
    dss.text("set controlmode=static")
    dss.text("set maxiterations=100")
    dss.text("set tolerance=0.001")
    reset_load_models(dss)

    dss.text("solve")
    if not check_convergence(dss):
        print("Initial solve failed. Exiting.")
        return

    load_names = dss.loads.names
    # Get original load values
    original_loads = {}
    for name in load_names:
        dss.loads.name = name
        original_loads[name] = {
            'kw': dss.loads.kw,
            'kvar': dss.loads.kvar
        }

    # Find all live buses (voltage > 0.4 kV at base case)
    base_voltages = {}
    for bus in dss.circuit.buses_names:
        dss.circuit.set_active_bus(bus)
        v = dss.bus.voltages
        if v and len(v) >= 2:
            mag = (v[0]**2 + v[1]**2)**0.5 / 1000
            base_voltages[bus] = mag
    live_buses = [bus for bus, v in base_voltages.items() if v > 0.4]

    # Set all loads to maximum multiplier
    max_multiplier = 5
    reset_load_models(dss)
    total_kw = 0
    for name in load_names:
        dss.loads.name = name
        orig = original_loads[name]
        dss.loads.kw = orig['kw'] * max_multiplier
        dss.loads.kvar = orig['kvar'] * max_multiplier
        total_kw += orig['kw'] * max_multiplier

    dss.text("solve")
    converged = check_convergence(dss)

    # Store voltages for all live buses
    maxload_voltages = {}
    for bus in live_buses:
        dss.circuit.set_active_bus(bus)
        v = dss.bus.voltages
        if v and len(v) >= 2:
            mag = (v[0]**2 + v[1]**2)**0.5 / 1000
            maxload_voltages[bus] = mag
        else:
            maxload_voltages[bus] = float('nan')

    # Find weakest bus
    weakest_bus = min(maxload_voltages, key=lambda b: maxload_voltages[b])
    weakest_v = maxload_voltages[weakest_bus]

    # Plot each bus as a point (different color)
    plt.figure(figsize=(14, 6))
    buses = list(maxload_voltages.keys())
    voltages = [maxload_voltages[b] for b in buses]
    colors = ['red' if b == weakest_bus else 'C0' for b in buses]
    plt.bar(buses, voltages, color=colors)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel('Bus')
    plt.ylabel('Voltage (kV)')
    plt.title(f'Voltage Profile at Maximum Load Multiplier = {max_multiplier} (PQ Model)')
    plt.tight_layout()
    plt.show()

    print("\n=== MAX LOADING ANALYSIS ===")
    print(f"Converged at maximum load? {'YES' if converged else 'NO'}")
    print(f"Total load applied: {total_kw:.2f} kW")
    print(f"Weakest bus: {weakest_bus}  |  Voltage: {weakest_v:.4f} kV")
    print("Lowest voltage buses:")
    for b, v in sorted(maxload_voltages.items(), key=lambda x: x[1])[:10]:
        print(f"  {b}: {v:.4f} kV")

if __name__ == "__main__":
    main()
How to Use/Adapt:
This script only analyzes the max loading case (multiplier = 5), but you can easily adjust max_multiplier as needed.

It will highlight the weakest bus in red and show all voltages together, for quick inspection.

Prints out the 10 lowest-voltage buses.

If you want this as a ready-to-download .py file, just ask!
Or say if you want this logic added to your multi-step load scaling script.








You said:

not like this i want one plot that highlights total losses vs load  multipliers ,one with voltage profile of each and every bus with vs load multipliers 2)my second plot like the one posted above , it will also be good if i know which bus will become unstable first etcc 

ChatGPT said:
Perfect, you want a two-part analysis:

1. Multi-bus Profile and System Losses (overview plots):
Plot 1: Total system losses vs load multiplier.

Plot 2: Voltage profile vs load multiplier for all buses (every line a bus, color each differently).

2. Critical/Nose Bus Analysis (detailed instability plot):
Find the first/weakest bus to approach instability (lowest minimum voltage as load increases).

For that bus, plot PV (nose) curve (voltage vs multiplier), highlight the collapse point, and clearly annotate the critical bus.

Here's a robust, ready-to-use script plan:
python
Copy
Edit
import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

def check_convergence(dss):
    return dss.solution.converged

def reset_load_models(dss):
    for name in dss.loads.names:
        if name and name.lower() != "none":
            dss.loads.name = name
            dss.loads.model = 1

def main():
    dss = py_dss_interface.DSS()
    script_path = os.path.dirname(os.path.abspath(__file__))
    paths = [
        pathlib.Path(script_path).joinpath("feeders", "123Bus", "Run_IEEE123Bus.DSS"),
        pathlib.Path(script_path).joinpath("123Bus", "Run_IEEE123Bus.DSS"),
        pathlib.Path(script_path).joinpath("Run_IEEE123Bus.DSS"),
        pathlib.Path(script_path).joinpath("feeders", "Run_IEEE123Bus.DSS")
    ]
    for path in paths:
        if path.exists():
            dss.text(f'compile "{str(path)}"')
            break
    else:
        raise FileNotFoundError("123-bus DSS file not found")

    dss.text("set mode=snapshot")
    dss.text("set controlmode=static")
    dss.text("set maxiterations=100")
    dss.text("set tolerance=0.001")
    reset_load_models(dss)

    dss.text("solve")
    if not check_convergence(dss):
        print("Initial solve failed. Exiting.")
        return

    load_names = dss.loads.names
    original_loads = {}
    for name in load_names:
        dss.loads.name = name
        original_loads[name] = {
            'kw': dss.loads.kw,
            'kvar': dss.loads.kvar
        }

    # Only include "live" buses
    base_voltages = {}
    for bus in dss.circuit.buses_names:
        dss.circuit.set_active_bus(bus)
        v = dss.bus.voltages
        if v and len(v) >= 2:
            mag = (v[0]**2 + v[1]**2)**0.5 / 1000
            base_voltages[bus] = mag
    live_buses = [bus for bus, v in base_voltages.items() if v > 0.4]

    multipliers = np.linspace(0, 10, 80)  # Larger range for collapse visualization
    results = {
        'multiplier': [],
        'total_load': [],
        'total_losses': [],
        'bus_voltages': {bus: [] for bus in live_buses},
        'converged': []
    }

    for m in multipliers:
        reset_load_models(dss)
        total_kw = 0
        for name in load_names:
            dss.loads.name = name
            orig = original_loads[name]
            dss.loads.kw = orig['kw'] * m
            dss.loads.kvar = orig['kvar'] * m
            total_kw += orig['kw'] * m

        dss.text("solve")
        converged = check_convergence(dss)
        results['converged'].append(converged)

        if not converged:
            # After first failure, fill the rest with NaN
            for bus in live_buses:
                results['bus_voltages'][bus].append(np.nan)
            results['multiplier'].append(m)
            results['total_load'].append(total_kw)
            results['total_losses'].append(np.nan)
            break

        results['multiplier'].append(m)
        results['total_load'].append(total_kw)
        losses = dss.circuit.losses[0] / 1000
        results['total_losses'].append(abs(losses))
        for bus in live_buses:
            dss.circuit.set_active_bus(bus)
            v = dss.bus.voltages
            if v and len(v) >= 2:
                mag = (v[0]**2 + v[1]**2)**0.5 / 1000
                results['bus_voltages'][bus].append(mag)
            else:
                results['bus_voltages'][bus].append(np.nan)

    # --- Plot 1: Losses and voltage profile ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.plot(results['multiplier'], results['total_losses'], 'r-', linewidth=2)
    ax1.set_title('Total System Losses vs Load Multiplier', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Load Multiplier (Lambda)', fontsize=12)
    ax1.set_ylabel('Total System Losses (kW)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Voltage profile for all buses (highlight weakest)
    weakest_bus = None
    min_overall = float('inf')
    for bus in live_buses:
        v_list = results['bus_voltages'][bus]
        min_v = np.nanmin(v_list)
        if min_v < min_overall:
            min_overall = min_v
            weakest_bus = bus
    for bus in live_buses:
        ax2.plot(results['multiplier'], results['bus_voltages'][bus], 
                 label=bus if bus==weakest_bus else None, 
                 color='red' if bus==weakest_bus else None, 
                 linewidth=2 if bus==weakest_bus else 0.8, 
                 alpha=1 if bus==weakest_bus else 0.45)
    ax2.set_title('Voltage Profile vs Load Multiplier (All Buses)', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Load Multiplier (Lambda)', fontsize=12)
    ax2.set_ylabel('Bus Voltage (kV)', fontsize=12)
    if weakest_bus:
        ax2.legend(title="Critical Bus", loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: PV (Nose) Curve for Weakest Bus ---
    v_weakest = results['bus_voltages'][weakest_bus]
    mult = np.array(results['multiplier'])
    v_weakest = np.array(v_weakest)
    # Detect collapse point as the last point before NaN or minimum
    collapse_idx = np.where(np.isnan(v_weakest))[0]
    if len(collapse_idx) > 0:
        collapse_idx = collapse_idx[0] - 1
    else:
        collapse_idx = np.argmin(v_weakest)
    collapse_lambda = mult[collapse_idx]
    collapse_voltage = v_weakest[collapse_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(mult, v_weakest, 'b-', linewidth=2)
    plt.plot(collapse_lambda, collapse_voltage, 'ro', markersize=10, label="Voltage Collapse Point")
    plt.title(f'PV (Nose) Curve at Critical Bus: {weakest_bus}', fontweight='bold', fontsize=16)
    plt.xlabel('Lambda (Load Multiplier)', fontsize=13)
    plt.ylabel(f'Voltage at bus {weakest_bus} (kV)', fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nAnalysis Complete")
    print("=====================")
    print(f"Critical (weakest) Bus: {weakest_bus}")
    print(f"Estimated Collapse Multiplier: {collapse_lambda:.2f}")
    print(f"Voltage at Collapse: {collapse_voltage:.4f} kV")
    print("=====================")

if __name__ == "__main__":
    main()
