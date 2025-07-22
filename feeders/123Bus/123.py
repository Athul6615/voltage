
import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

def check_convergence(dss):
    return dss.solution.converged

def reset_load_models_to_cvr(dss):
    # Set all loads to CVR model (model=5)
    for name in dss.loads.names:
        if name and name.lower() != "none":
            dss.loads.name = name
            dss.loads.model = 5  # CVR model

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
    reset_load_models_to_cvr(dss)

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

    multipliers = np.linspace(0, 10, 80)  # Large range to see CVR behavior
    results = {
        'multiplier': [],
        'total_load': [],
        'total_losses': [],
        'bus_voltages': {bus: [] for bus in live_buses},
        'converged': []
    }

    for m in multipliers:
        reset_load_models_to_cvr(dss)
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
    ax1.set_title('Total System Losses vs Load Multiplier (CVR Model)', fontweight='bold', fontsize=14)
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
    ax2.set_title('Voltage Profile vs Load Multiplier (All Buses, CVR Model)', fontweight='bold', fontsize=14)
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
    plt.title(f'PV (Nose) Curve at Critical Bus: {weakest_bus} (CVR Model)', fontweight='bold', fontsize=16)
    plt.xlabel('Lambda (Load Multiplier)', fontsize=13)
    plt.ylabel(f'Voltage at bus {weakest_bus} (kV)', fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nAnalysis Complete (CVR Model)")
    print("=====================")
    print(f"Critical (weakest) Bus: {weakest_bus}")
    print(f"Estimated Collapse Multiplier: {collapse_lambda:.2f}")
    print(f"Voltage at Collapse: {collapse_voltage:.4f} kV")
    print("=====================")

if __name__ == "__main__":
    main()


