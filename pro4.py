import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib


def configure_zip_loads(dss):
    """Configure ZIP loads while skipping non-load objects"""
    dss.loads.first()  # Move to first load
    while True:
        # Only configure actual load objects
        if dss.loads.name.lower().startswith('load.'):
            # Set ZIP model and coefficients
            dss.text(f'edit load.{dss.loads.name} model=8')
            dss.text(f'edit load.{dss.loads.name} zipv=[0.5, 0.5, 0, 0, 0, 0]')

        if not dss.loads.next():  # Move to next load or break if no more
            break


def run_power_system_analysis():
    """Main analysis function with proper object handling"""
    dss = py_dss_interface.DSS()

    # Load IEEE 13-bus system
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("feeders", "13bus", "IEEE13Nodeckt.dss")

    if not dss_file.exists():
        print(f"Error: File not found - {dss_file}")
        return

    dss.text(f'compile "{str(dss_file)}"')

    # Configure ZIP loads properly
    print("\nConfiguring ZIP load model...")
    configure_zip_loads(dss)

    # Disable regulators separately (not as loads)
    print("Disabling regulators...")
    dss.text("batchedit regcontrol..* enabled=no")

    # Configure system settings
    dss.text("set mode=snapshot")
    dss.text("set maxiterations=200")
    dss.text("set tolerance=0.0001")
    dss.text("set controlmode=static")

    # Get only actual load objects
    load_names = [name for name in dss.loads.names if name.lower().startswith('load.')]
    original_kws = []
    for load in load_names:
        dss.loads.name = load
        original_kws.append(dss.loads.kw)

    # Analysis parameters
    load_multipliers = np.linspace(0.5, 2.0, 50)
    results = {
        'load_multiplier': [],
        'total_load': [],
        'min_voltage': [],
        'losses': []
    }

    print("\nRunning PV analysis...")
    for mult in load_multipliers:
        dss.text(f"set loadmult={mult}")
        dss.text("solve")

        # Get system voltages
        min_v = 1.0
        for bus in dss.circuit.buses_names:
            dss.circuit.set_active_bus(bus)
            pu_voltages = dss.bus.pu_voltages
            if pu_voltages:
                min_v = min(min_v, min(abs(v) for v in pu_voltages))

        # Store results
        results['load_multiplier'].append(mult)
        results['total_load'].append(sum(original_kws) * mult)
        results['min_voltage'].append(min_v)
        results['losses'].append(abs(dss.circuit.losses[0]) / 1000)

        print(f"Load mult: {mult:.2f} | Min V: {min_v:.3f} pu")

        if min_v < 0.7:
            print("Voltage collapse detected!")
            break

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(results['total_load'], results['min_voltage'], 'b-')
    plt.title('PV Curve')
    plt.xlabel('Total Load (kW)')
    plt.ylabel('Min Voltage (pu)')
    plt.grid(True)

    plt.subplot(122)
    plt.plot(results['total_load'], results['losses'], 'r-')
    plt.title('System Losses')
    plt.xlabel('Total Load (kW)')
    plt.ylabel('Losses (kW)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_power_system_analysis()