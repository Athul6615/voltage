
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
                "batchedit load..* model=1",
                "set loadmodel=1",
                "set maxiterations=300",
                "set tolerance=0.0001",
                "solve"
            ]
        },
        {
            'name': 'PQ Model - Newton-Raphson',
            'commands': [
                "batchedit load..* model=1",
                "set algorithm=newton",
                "set maxiterations=200",
                "set tolerance=0.001",
                "solve"
            ]
        },
        {
            'name': 'PQ Model - Relaxed Tolerance',
            'commands': [
                "batchedit load..* model=1",
                "set algorithm=normal",
                "set maxiterations=500",
                "set tolerance=0.01",
                "solve"
            ]
        },
        {
            'name': 'Constant Impedance Model',
            'commands': [
                "batchedit load..* model=2",
                "set loadmodel=2",
                "set maxiterations=200",
                "set tolerance=0.001",
                "solve"
            ]
        },
        {
            'name': 'PQ Model - Disable Controls',
            'commands': [
                "batchedit load..* model=1",
                "batchedit regcontrol..* enabled=no",
                "batchedit capcontrol..* enabled=no",
                "set maxiterations=300",
                "set tolerance=0.001",
                "solve"
            ]
        },
        {
            'name': 'Very Relaxed Settings',
            'commands': [
                "batchedit load..* model=1",
                "set algorithm=normal",
                "set maxiterations=1000",
                "set tolerance=0.1",
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
    cmds = [
        "set maxiterations=100",
        "set tolerance=0.001",
        "set algorithm=normal",
        "set loadmodel=1",
        "batchedit load..* model=1",
        "batchedit regcontrol..* enabled=yes",
        "batchedit capcontrol..* enabled=yes"
    ]
    for cmd in cmds:
        dss.text(cmd)


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

    setup_cmds = [
        "set mode=snapshot",
        "set maxiterations=100",
        "set tolerance=0.001",
        "set maxcontroliter=50",
        "set controlmode=static",
        "set loadmodel=1",
        "batchedit load..* model=1"
    ]
    for cmd in setup_cmds:
        dss.text(cmd)

    dss.text("solve")
    converged, _ = check_convergence_detailed(dss)

    if not converged:
        for attempt in range(1, 7):
            converged, _ = improve_convergence(dss, 0, attempt)
            if converged:
                break
        if not converged:
            raise RuntimeError("Initial convergence failed")

    load_names = dss.loads.names
    original_loads = {name: {'kw': dss.loads.kw, 'kvar': dss.loads.kvar}
                      for name in load_names}

    load_increments = np.arange(0, 5000, 25)
    results = {
        'load_added': [],
        'total_load': [],
        'total_losses': [],
        'bus_voltages': {}
    }

    for increment in load_increments:
        total_kw = 0
        for name in load_names:
            dss.loads.name = name
            orig = original_loads[name]
            dss.loads.kw = orig['kw'] + increment
            dss.loads.kvar = orig['kvar'] + increment * 0.4
            total_kw += orig['kw'] + increment

        reset_convergence_parameters(dss)
        dss.text("solve")
        converged, _ = check_convergence_detailed(dss)

        if not converged:
            for attempt in range(1, 7):
                converged, _ = improve_convergence(dss, increment, attempt)
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

        results['load_added'].append(increment)
        results['total_load'].append(total_kw)
        results['total_losses'].append(abs(losses))

        for bus, voltage in bus_voltages.items():
            if bus not in results['bus_voltages']:
                results['bus_voltages'][bus] = []
            results['bus_voltages'][bus].append(voltage)

    # Post-analysis for weakest bus and voltage collapse
    voltage_at_max_load = {bus: voltages[-1] for bus, voltages in results['bus_voltages'].items()}
    weakest_bus = min(voltage_at_max_load, key=voltage_at_max_load.get)
    weakest_voltage = voltage_at_max_load[weakest_bus]

    min_voltage_points = [min([v[i] for v in results['bus_voltages'].values()]) for i in range(len(results['load_added']))]
    collapse_index = np.argmin(min_voltage_points)
    collapse_load = results['load_added'][collapse_index]
    collapse_voltage = min_voltage_points[collapse_index]

    print("\n--- Weakest Bus Analysis ---")
    print(f"Weakest Bus: {weakest_bus}")
    print(f"Voltage at Max Load: {weakest_voltage:.4f} kV")

    print("\n--- Voltage Collapse Estimate ---")
    print(f"Estimated Collapse Load Addition: {collapse_load} kW per bus")
    print(f"Minimum Bus Voltage at Collapse: {collapse_voltage:.4f} kV")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(results['load_added'], results['total_losses'], 'r-o')
    ax1.set_title('Total System Losses vs Load Addition')
    ax1.set_xlabel('Load Addition per Bus (kW)')
    ax1.set_ylabel('Total System Losses (kW)')
    ax1.grid(True)

    for bus, voltages in results['bus_voltages'].items():
        if len(voltages) == len(results['load_added']):
            ax2.plot(results['load_added'], voltages, label=bus)
    ax2.set_title('Voltage Nose Curve (All Buses)')
    ax2.set_xlabel('Load Addition per Bus (kW)')
    ax2.set_ylabel('Voltage (kV)')
    ax2.grid(True)
    ax2.axvline(collapse_load, color='red', linestyle='--', linewidth=1)
    ax2.text(collapse_load, collapse_voltage + 0.1, f'Nose Tip ~{collapse_load}kW', color='red')
    ax2.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
