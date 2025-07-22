

import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

def check_convergence(dss):
    return dss.solution.converged

def reset_load_models(dss):
    # Set all loads to PQ model
    for name in dss.loads.names:
        if name and name.lower() != "none":
            dss.loads.name = name
            dss.loads.model = 1

def reset_controls(dss):
    if dss.regcontrols.count > 0 and dss.regcontrols.names[0].lower() != "none":
        for name in dss.regcontrols.names:
            if name and name.lower() != "none":
                dss.regcontrols.name = name
                dss.regcontrols.enabled = True
    if dss.capcontrols.count > 0 and dss.capcontrols.names[0].lower() != "none":
        for name in dss.capcontrols.names:
            if name and name.lower() != "none":
                dss.capcontrols.name = name
                dss.capcontrols.enabled = True

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
    reset_controls(dss)

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

    print("Live (monitored) buses:", live_buses[:10], "..." if len(live_buses) > 10 else "")

    multipliers = np.linspace(0, 5, 200)
    results = {
        'multiplier': [],
        'total_load': [],
        'total_losses': [],
        'bus_voltages': {bus: [] for bus in live_buses}
    }

    for m in multipliers:
        # Reset all load models before applying new values
        reset_load_models(dss)
        total_kw = 0
        for name in load_names:
            dss.loads.name = name
            orig = original_loads[name]
            dss.loads.kw = orig['kw'] * m
            dss.loads.kvar = orig['kvar'] * m
            total_kw += orig['kw'] * m

        dss.text("solve")
        if not check_convergence(dss):
            print(f"  Solve failed at multiplier {m:.2f}. Stopping analysis here.")
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
                results['bus_voltages'][bus].append(float('nan'))

        if m in [0, 1, 2, 3, 4, 5]:
            print(f"Multiplier {m:.2f}: Total load = {total_kw:.2f} kW, Losses = {losses:.2f} kW")
            print(" Sample bus voltages:", {bus: results['bus_voltages'][bus][-1] for bus in live_buses[:3]})

    # Find weakest bus and collapse point among live buses
    weakest_bus = None
    weakest_voltage = float('inf')
    for bus in live_buses:
        min_v = min(results['bus_voltages'][bus])
        if min_v < weakest_voltage:
            weakest_voltage = min_v
            weakest_bus = bus

    collapse_idx = np.argmin([min([results['bus_voltages'][bus][i] for bus in live_buses]) for i in range(len(results['multiplier']))])
    collapse_multiplier = results['multiplier'][collapse_idx]

    # Plots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(results['multiplier'], results['total_losses'], 'r-', linewidth=2)
    ax1.set_title('System Losses vs Load Multiplier (PQ Model)')
    ax1.set_xlabel('Load Multiplier (× base load)')
    ax1.set_ylabel('Total System Losses (kW)')
    ax1.grid(True)

    for bus in live_buses:
        ax2.plot(results['multiplier'], results['bus_voltages'][bus], label=bus, alpha=0.8 if bus==weakest_bus else 0.3, linewidth=2 if bus==weakest_bus else 0.7)
    ax2.set_title('Voltage Profile vs Load Multiplier (Live Buses)')
    ax2.set_xlabel('Load Multiplier (× base load)')
    ax2.set_ylabel('Voltage (kV)')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    fig2, ax = plt.subplots(figsize=(10, 6))
    for bus in live_buses:
        ax.plot(results['multiplier'], results['bus_voltages'][bus], label=bus if bus==weakest_bus else None, color='red' if bus==weakest_bus else 'gray', linewidth=2 if bus==weakest_bus else 0.7, alpha=0.9 if bus==weakest_bus else 0.3)
    ax.axvline(collapse_multiplier, linestyle='--', color='blue', label=f'Nose Tip ≈ {collapse_multiplier:.2f}×')
    ax.set_title('Weakest Bus and Voltage Collapse Point (Live Buses)')
    ax.set_xlabel('Load Multiplier (× base load)')
    ax.set_ylabel('Voltage (kV)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nAnalysis Summary")
    print("=====================")
    print(f"Weakest Bus: {weakest_bus}")
    print(f"Minimum Voltage at Weakest Bus: {weakest_voltage:.4f} kV")
    print(f"Estimated Collapse Multiplier: {collapse_multiplier:.2f}")
    print("=====================")

if __name__ == "__main__":
    main()
