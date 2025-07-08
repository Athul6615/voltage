import py_dss_interface
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

# Initialize DSS
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = pathlib.Path(script_path).joinpath("feeders", "13bus", "IEEE13Nodeckt.dss")
dss = py_dss_interface.DSS()

# Compile and solve
dss.text(f'compile "{str(dss_file)}"')
dss.text("solve")

if dss.solution.converged:
    # Get load names
    load_names = dss.loads.names
    print(f"Analyzing loads: {load_names}")

    # Store original values
    original_loads = {}
    for load in load_names:
        dss.loads.name = load
        original_loads[load] = {'kw': dss.loads.kw, 'kvar': dss.loads.kvar}

    # Define constant range
    constants = np.arange(0.2, 2.2, 0.2)

    # Get bus names for individual voltage tracking
    bus_names = dss.circuit.buses_names
    print(f"Monitoring buses: {bus_names}")

    # Storage for results
    results = {
        'constants': constants,
        'bus_voltages': {bus: [] for bus in bus_names},
        'total_kw': [],
        'total_kvar': [],
        'total_losses': []
    }

    # Test each constant value
    for constant in constants:
        print(f"Testing constant: {constant}")

        # Apply equation to all loads: P_new = P_old + constant, Q_new = Q_old + constant
        total_kw = 0
        total_kvar = 0

        for load_name in load_names:
            dss.loads.name = load_name
            new_kw = original_loads[load_name]['kw'] + constant
            new_kvar = original_loads[load_name]['kvar'] + constant

            dss.loads.kw = new_kw
            dss.loads.kvar = new_kvar

            total_kw += new_kw
            total_kvar += new_kvar

        # Solve and get individual bus voltages
        dss.text("solve")

        # Get voltage for each bus
        for bus_name in bus_names:
            try:
                dss.circuit.set_active_bus(bus_name)
                voltage = dss.bus.vmag_angle[0] if dss.bus.vmag_angle else 0
                results['bus_voltages'][bus_name].append(voltage)
            except:
                results['bus_voltages'][bus_name].append(0)

        # Get total system losses
        total_losses = dss.circuit.losses[0] / 1000  # Convert to kW

        # Store results
        results['total_kw'].append(total_kw)
        results['total_kvar'].append(total_kvar)
        results['total_losses'].append(total_losses)

    # Create first figure with 3 plots
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Load Analysis - Part 1: Voltages and Powers vs Constant', fontsize=16, fontweight='bold')

    # Colors for different buses
    colors = plt.cm.tab10(np.linspace(0, 1, len(bus_names)))

    # Plot 1: Individual Bus Voltages vs Constant
    for i, bus_name in enumerate(bus_names):
        if max(results['bus_voltages'][bus_name]) > 0:  # Only plot buses with valid data
            ax1.plot(constants, results['bus_voltages'][bus_name], 'o-',
                     linewidth=2, markersize=4, color=colors[i], label=f'Bus {bus_name}')

    ax1.set_xlabel('Constant Added (kW/kvar)')
    ax1.set_ylabel('Bus Voltage (V)')
    ax1.set_title('Individual Bus Voltages vs Constant')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 2: Total Real Power vs Constant
    ax2.plot(constants, results['total_kw'], 'ro-', linewidth=3, markersize=6)
    ax2.set_xlabel('Constant Added (kW/kvar)')
    ax2.set_ylabel('Total Real Power (kW)')
    ax2.set_title('Total Real Power vs Constant')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Total Reactive Power vs Constant
    ax3.plot(constants, results['total_kvar'], 'go-', linewidth=3, markersize=6)
    ax3.set_xlabel('Constant Added (kW/kvar)')
    ax3.set_ylabel('Total Reactive Power (kvar)')
    ax3.set_title('Total Reactive Power vs Constant')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # Make room for legend
    plt.show()

    # Create second figure with 3 plots
    fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Load Analysis - Part 2: Voltage-Power Relationships and Losses', fontsize=16, fontweight='bold')

    # Plot 4: Individual Bus Voltages vs Total Power
    for i, bus_name in enumerate(bus_names):
        if max(results['bus_voltages'][bus_name]) > 0:
            ax4.plot(results['total_kw'], results['bus_voltages'][bus_name], 'o-',
                     linewidth=2, markersize=4, color=colors[i], label=f'Bus {bus_name}')

    ax4.set_xlabel('Total Real Power (kW)')
    ax4.set_ylabel('Bus Voltage (V)')
    ax4.set_title('Individual Bus Voltages vs Total Power')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 5: Total Losses vs Constant
    ax5.plot(constants, results['total_losses'], 'co-', linewidth=3, markersize=6)
    ax5.set_xlabel('Constant Added (kW/kvar)')
    ax5.set_ylabel('Total System Losses (kW)')
    ax5.set_title('Total Losses vs Constant')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Losses vs Total Power
    ax6.plot(results['total_kw'], results['total_losses'], 'yo-', linewidth=3, markersize=6)
    ax6.set_xlabel('Total Real Power (kW)')
    ax6.set_ylabel('Total System Losses (kW)')
    ax6.set_title('Losses vs Total Power')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # Make room for legend
    plt.show()

    # Print simple summary
    print(f"\nSUMMARY:")
    print(f"Constant range: {constants[0]} to {constants[-1]}")

    # Show voltage changes for each bus
    print(f"\nVoltage changes by bus:")
    for bus_name in bus_names:
        if max(results['bus_voltages'][bus_name]) > 0:
            voltage_change = results['bus_voltages'][bus_name][-1] - results['bus_voltages'][bus_name][0]
            print(f"  Bus {bus_name}: {voltage_change:.2f} V")

    print(f"\nPower change: {results['total_kw'][-1] - results['total_kw'][0]:.2f} kW")
    print(f"Losses change: {results['total_losses'][-1] - results['total_losses'][0]:.2f} kW")

    # Reset loads to original values
    for load_name in load_names:
        dss.loads.name = load_name
        dss.loads.kw = original_loads[load_name]['kw']
        dss.loads.kvar = original_loads[load_name]['kvar']

    dss.text("solve")
    print("Loads reset to original values")

else:
    print("Power flow did not converge!")