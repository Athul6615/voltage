

import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

def check_convergence_detailed(dss):
    converged = dss.solution.converged
    return converged, {
        'converged': converged,
        'iterations': dss.solution.iterations,
        'max_iterations': dss.solution.max_iterations,
        'tolerance': dss.solution.tolerance,
        'control_iterations': dss.solution.control_iterations,
        'max_control_iterations': dss.solution.max_control_iterations,
        'solution_mode': dss.solution.mode,
        'algorithm': dss.solution.algorithm
    }

def improve_convergence(dss, increment, attempt=1):
    strategies = [
        {
            'name': 'PQ Model - Increased Iterations',
            'commands': [
                "set loadmodel=1",
                "set maxiterations=300",
                "set tolerance=0.0001",
                "solve"
            ]
        },
        {
            'name': 'PQ Model - Newton-Raphson',
            'commands': [
                "set algorithm=newton",
                "set maxiterations=200",
                "set tolerance=0.001",
                "solve"
            ]
        }
    ]
    if attempt <= len(strategies):
        for cmd in strategies[attempt - 1]['commands']:
            dss.text(cmd)
        return check_convergence_detailed(dss)
    return False, None

def reset_convergence_parameters(dss):
    dss.text("set maxiterations=100")
    dss.text("set tolerance=0.001")
    dss.text("set algorithm=normal")
    dss.text("set loadmodel=1")
    for name in dss.loads.names:
        if name and name.lower() != "none":
            dss.loads.name = name
            dss.loads.model = 1
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
    reset_convergence_parameters(dss)
    dss.text("solve")
    converged, _ = check_convergence_detailed(dss)
    if not converged:
        for attempt in range(1, 3):
            converged, _ = improve_convergence(dss, 0, attempt)
            if converged:
                break
        if not converged:
            raise RuntimeError("Initial convergence failed")
    load_names = dss.loads.names
    # ---- FIXED: Safely build original_loads dict ----
    original_loads = {}
    for name in load_names:
        dss.loads.name = name
        original_loads[name] = {
            'kw': dss.loads.kw,
            'kvar': dss.loads.kvar
        }
    multipliers = np.linspace(0, 5, 200)
    results = {
        'multiplier': [],
        'total_load': [],
        'total_losses': [],
        'bus_voltages': {}
    }
    for m in multipliers:
        total_kw = 0
        for name in load_names:
            dss.loads.name = name
            orig = original_loads[name]
            dss.loads.kw = orig['kw'] * m
            dss.loads.kvar = orig['kvar'] * m
            total_kw += orig['kw'] * m
        reset_convergence_parameters(dss)
        dss.text("solve")
        converged, _ = check_convergence_detailed(dss)
        if not converged:
            for attempt in range(1, 3):
                converged, _ = improve_convergence(dss, m, attempt)
                if converged:
                    break
            if not converged:
                continue
        losses = dss.circuit.losses[0] / 1000
        bus_voltages = {}
        for bus in dss.circuit.buses_names:
            if "sourcebus" in bus.lower():
                continue
            dss.circuit.set_active_bus(bus)
            v = dss.bus.voltages
            if v and len(v) >= 2:
                voltage_mag = (v[0]**2 + v[1]**2)**0.5 / 1000
                bus_voltages[bus] = voltage_mag
        results['multiplier'].append(m)
        results['total_load'].append(total_kw)
        results['total_losses'].append(abs(losses))
        for bus, voltage in bus_voltages.items():
            if bus not in results['bus_voltages']:
                results['bus_voltages'][bus] = []
            results['bus_voltages'][bus].append(voltage)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(results['multiplier'], results['total_losses'], 'r-', linewidth=2)
    ax1.set_title('System Losses vs Load Multiplier (PQ Model)')
    ax1.set_xlabel('Load Multiplier (× base load)')
    ax1.set_ylabel('Total System Losses (kW)')
    ax1.grid(True)
    for bus, volts in results['bus_voltages'].items():
        if len(volts) == len(results['multiplier']):
            ax2.plot(results['multiplier'], volts, label=bus)
    ax2.set_title('Voltage Profile vs Load Multiplier (All Buses)')
    ax2.set_xlabel('Load Multiplier (× base load)')
    ax2.set_ylabel('Voltage (kV)')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
    weakest_at_end = min({b: v[-1] for b, v in results['bus_voltages'].items()}.items(), key=lambda x: x[1])
    weakest_bus = weakest_at_end[0]
    collapse_idx = np.argmin([min([v[i] for v in results['bus_voltages'].values()]) for i in range(len(results['multiplier']))])
    collapse_multiplier = results['multiplier'][collapse_idx]
    fig2, ax = plt.subplots(figsize=(10, 6))
    for bus, voltages in results['bus_voltages'].items():
        if len(voltages) == len(results['multiplier']):
            color = 'red' if bus == weakest_bus else 'gray'
            ax.plot(results['multiplier'], voltages, label=bus if bus == weakest_bus else None, color=color, linewidth=2 if bus == weakest_bus else 0.8, alpha=1 if bus == weakest_bus else 0.3)
    ax.axvline(collapse_multiplier, linestyle='--', color='blue', label=f'Nose Tip ≈ {collapse_multiplier:.2f}×')
    ax.set_title('Weakest Bus and Voltage Collapse Point')
    ax.set_xlabel('Load Multiplier (× base load)')
    ax.set_ylabel('Voltage (kV)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    print("\\nAnalysis Summary")
    print("=====================")
    print(f"Weakest Bus: {weakest_bus}")
    print(f"Voltage at Max Load: {weakest_at_end[1]:.4f} kV")
    print(f"Estimated Collapse Multiplier: {collapse_multiplier:.2f}")
    print("=====================")

if __name__ == "__main__":
    main()
