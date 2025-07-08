import py_dss_interface
import os
import pathlib

# Initialize DSS
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = pathlib.Path(script_path).joinpath("feeders", "13bus", "IEEE13Nodeckt.dss")
dss = py_dss_interface.DSS()

# Compile and solve
dss.text(f'compile "{str(dss_file)}"')
dss.text("solve")

if dss.solution.converged:
    print("=== Original Load Values ===")

    # Get all load names
    load_names = dss.loads.names

    # Store original values and display
    original_loads = {}
    for load_name in load_names:
        dss.loads.name = load_name
        original_loads[load_name] = {
            'kw': dss.loads.kw,
            'kvar': dss.loads.kvar
        }
        print(f"{load_name}: {dss.loads.kw:.2f} kW, {dss.loads.kvar:.2f} kvar")

    # Apply multiplier
    multiplier = 1.5
    print(f"\n=== Applying {multiplier}x Multiplier ===")

    dss.loads.first
    for load_name in load_names:
        dss.loads.name = load_name
        # Multiply existing values
        dss.loads.kw = dss.loads.kw * multiplier
        dss.loads.kvar = dss.loads.kvar * multiplier
        print(f"{load_name}: {dss.loads.kw:.2f} kW, {dss.loads.kvar:.2f} kvar")

    # Solve with new loads
    dss.text("solve")

    if dss.solution.converged:
        print(f"\n=== Voltage Comparison (Multiplier = {multiplier}) ===")
        # Check voltage at key buses
        for bus_name in ["671", "645", "632"]:
            dss.circuit.set_active_bus(bus_name)
            voltages = dss.bus.vmag_angle
            print(f"Bus {bus_name}: {[f'{v:.1f}' for i, v in enumerate(voltages) if i % 2 == 0]} V")

    # Reset to original values
    print(f"\n=== Resetting to Original Values ===")
    for load_name in load_names:
        dss.loads.name = load_name
        dss.loads.kw = original_loads[load_name]['kw']
        dss.loads.kvar = original_loads[load_name]['kvar']

    dss.text("solve")
    print("Loads reset to original values")

else:
    print("Solution did not converge!")