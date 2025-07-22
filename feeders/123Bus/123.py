import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib


def check_convergence_detailed(dss):
    """
    Enhanced convergence checking with detailed diagnostics
    """
    converged = dss.solution.converged

    convergence_info = {
        'converged': converged,
        'iterations': dss.solution.iterations,
        'max_iterations': dss.solution.max_iterations,
        'tolerance': dss.solution.tolerance,
        'control_iterations': dss.solution.control_iterations,
        'max_control_iterations': dss.solution.max_control_iterations,
        'solution_mode': dss.solution.mode,
        'algorithm': dss.solution.algorithm
    }

    return converged, convergence_info


def improve_convergence(dss, increment, attempt=1):
    """
    Apply multiple convergence strategies systematically with detailed checking
    """
    print(f"    Convergence attempt {attempt} for {increment} kW increment...")

    strategies = [
        # Strategy 1: Increase iterations with tight tolerance (PQ model)
        {
            'name': 'PQ Model - Increased Iterations',
            'commands': [
                "batchedit load..* model=1",  # Ensure PQ model
                "set loadmodel=1",
                "set maxiterations=300",
                "set tolerance=0.0001",
                "solve"
            ]
        },

        # Strategy 2: Newton-Raphson with PQ model
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

        # Strategy 3: Relaxed tolerance with PQ model
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

        # Strategy 4: Change to constant impedance model temporarily
        {
            'name': 'Constant Impedance Model',
            'commands': [
                "batchedit load..* model=2",  # Constant impedance
                "set loadmodel=2",
                "set maxiterations=200",
                "set tolerance=0.001",
                "solve"
            ]
        },

        # Strategy 5: Disable controls with PQ model
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

        # Strategy 6: Very relaxed settings
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
        strategy = strategies[attempt - 1]
        print(f"      Trying strategy: {strategy['name']}")

        try:
            for command in strategy['commands']:
                dss.text(command)

            converged, conv_info = check_convergence_detailed(dss)

            if converged:
                print(f"      ✓ Converged with {strategy['name']}")
                print(f"      Iterations: {conv_info['iterations']}/{conv_info['max_iterations']}")
                print(f"      Tolerance: {conv_info['tolerance']}")
                print(f"      Algorithm: {conv_info['algorithm']}")
            else:
                print(f"      ❌ Failed with {strategy['name']}")
                print(f"      Iterations: {conv_info['iterations']}/{conv_info['max_iterations']}")
                print(f"      Final tolerance achieved: {conv_info['tolerance']}")

            return converged, conv_info

        except Exception as e:
            print(f"      Error in strategy {strategy['name']}: {e}")
            return False, None

    return False, None


def reset_convergence_parameters(dss):
    """
    Reset DSS to default convergence parameters with PQ load model
    """
    dss.text("set maxiterations=100")
    dss.text("set tolerance=0.001")
    dss.text("set algorithm=normal")
    dss.text("set loadmodel=1")  # PQ model (constant power) - 1=PQ, 2=Z, 5=I
    dss.text("batchedit load..* model=1")  # Ensure all loads use PQ model
    dss.text("batchedit regcontrol..* enabled=yes")  # Re-enable controls
    dss.text("batchedit capcontrol..* enabled=yes")


# Initialize DSS
print("Initializing OpenDSS...")
dss = py_dss_interface.DSS()

# Load IEEE 123-bus system
script_path = os.path.dirname(os.path.abspath(__file__))
# Try different possible paths for the 123-bus system
possible_paths = [
    pathlib.Path(script_path).joinpath("feeders", "123Bus", "Run_IEEE123Bus.DSS"),
    pathlib.Path(script_path).joinpath("123Bus", "Run_IEEE123Bus.DSS"),
    pathlib.Path(script_path).joinpath("Run_IEEE123Bus.DSS"),
    pathlib.Path(script_path).joinpath("feeders", "Run_IEEE123Bus.DSS")
]

dss_file = None
for path in possible_paths:
    if path.exists():
        dss_file = path
        print(f"Found DSS file at: {path}")
        break

if not dss_file.exists():
    print(f"Error: File not found - {dss_file}")
    exit()

dss.text(f'compile "{str(dss_file)}"')
print("IEEE 123-Bus System loaded successfully")

# Configure system with enhanced convergence settings and PQ load model
print("\nConfiguring system with enhanced convergence settings and PQ load model...")
dss.text("set mode=snapshot")
dss.text("set maxiterations=100")
dss.text("set tolerance=0.001")
dss.text("set maxcontroliter=50")  # Increased control iterations
dss.text("set controlmode=static")  # Static control mode for better convergence
dss.text("set loadmodel=1")  # Set default load model to PQ (constant power)
dss.text("batchedit load..* model=1")  # Ensure all existing loads use PQ model
print("  ✓ All loads set to PQ model (constant power)")

# Initial solve with convergence checking
print("\nPerforming initial solve with convergence diagnostics...")
dss.text("solve")
initial_converged, initial_info = check_convergence_detailed(dss)

if initial_converged:
    print("✓ Initial solution converged successfully")
    print(f"  Iterations: {initial_info['iterations']}/{initial_info['max_iterations']}")
    print(f"  Tolerance: {initial_info['tolerance']}")
else:
    print("❌ Initial solution failed to converge!")
    print(f"  Iterations: {initial_info['iterations']}/{initial_info['max_iterations']}")
    print("  Attempting to improve initial convergence...")

    # Try to get initial convergence
    for attempt in range(1, 7):  # Increased to 6 attempts
        conv_result, conv_info = improve_convergence(dss, 0, attempt)
        if conv_result:
            print("✓ Initial convergence achieved!")
            break
    else:
        print("❌ Could not achieve initial convergence after 6 attempts. Results may be unreliable.")

# Get load names
load_names = dss.loads.names
print(f"\nFound {len(load_names)} loads")
print(f"Load names (first 10): {load_names[:10]}")
if len(load_names) > 10:
    print(f"... and {len(load_names) - 10} more loads")

# Store original load values
original_loads = {}
for load in load_names:
    dss.loads.name = load
    original_loads[load] = {
        'kw': dss.loads.kw,
        'kvar': dss.loads.kvar
    }

# Analysis parameters - smaller increments for 123-bus system due to complexity
load_increments = np.arange(0, 5000, 25)  # Reduced range and smaller steps
print(f"\nTesting load increments: 0 to {max(load_increments)} kW in steps of 25 kW")

# Results storage - only what we need for the two plots
results = {
    'load_added': [],
    'total_load': [],
    'total_losses': [],
    'bus_voltages': {}  # Dictionary to store voltage for each bus
}

# Main analysis loop
print("\nRunning analysis...")
for i, increment in enumerate(load_increments):
    print(f"\nTest {i + 1}/{len(load_increments)}: Adding {increment} kW to each load")

    # Modify loads
    total_kw = 0
    for load in load_names:
        dss.loads.name = load
        orig = original_loads[load]
        new_kw = orig['kw'] + increment
        new_kvar = orig['kvar'] + (increment * 0.4)  # Assume 0.4 kvar/kw ratio

        dss.loads.kw = new_kw
        dss.loads.kvar = new_kvar
        total_kw += new_kw

    # Reset to default parameters before solving
    reset_convergence_parameters(dss)

    # Initial solve attempt
    dss.text("solve")
    converged, conv_info = check_convergence_detailed(dss)

    if not converged:
        print(f"  ❌ Initial solve failed, trying convergence strategies...")

        # Try different strategies (increased to 6 strategies)
        for attempt in range(1, 7):
            conv_result, conv_info = improve_convergence(dss, increment, attempt)

            if conv_result:
                converged = True
                print(f"  ✓ Converged after {attempt} attempts")
                break

        if not converged:
            print(f"  ❌ Failed to converge after 6 attempts, skipping this load level")
            continue

    # Reset parameters after successful convergence
    reset_convergence_parameters(dss)

    # Get losses
    try:
        losses = dss.circuit.losses
        if isinstance(losses, (list, tuple)):
            total_losses_kw = abs(losses[0]) / 1000
        else:
            total_losses_kw = abs(losses) / 1000
    except Exception as e:
        print(f"  Error getting losses: {e}")
        continue

    # Get voltage for each bus (limited to first 20 buses for visualization)
    bus_names = [bus for bus in dss.circuit.buses_names if "sourcebus" not in bus.lower()]

    # For 123-bus system, only track a subset of buses for plotting clarity
    selected_buses = bus_names[:20]  # First 20 buses
    current_bus_voltages = {}

    for bus_name in selected_buses:
        try:
            dss.circuit.set_active_bus(bus_name)
            bus_voltages = dss.bus.voltages
            if bus_voltages:
                voltage_mag = abs(complex(bus_voltages[0], bus_voltages[1] if len(bus_voltages) > 1 else 0))
                voltage_kv = voltage_mag / 1000
                current_bus_voltages[bus_name] = voltage_kv
        except:
            continue

    # Store results
    if current_bus_voltages:  # Only store if we have voltage data
        results['load_added'].append(increment)
        results['total_load'].append(total_kw)
        results['total_losses'].append(total_losses_kw)

        # Store voltage for each bus
        for bus_name, voltage in current_bus_voltages.items():
            if bus_name not in results['bus_voltages']:
                results['bus_voltages'][bus_name] = []
            results['bus_voltages'][bus_name].append(voltage)

        print(f"  Load: {total_kw:.0f} kW, Losses: {total_losses_kw:.1f} kW")
        if current_bus_voltages:
            print(
                f"  Voltage range (sample): {min(current_bus_voltages.values()):.2f} - {max(current_bus_voltages.values()):.2f} kV")

print(f"\nAnalysis complete! {len(results['total_losses'])} successful data points")

# Create the two specific plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Total System Losses vs Load Added
ax1.plot(results['load_added'], results['total_losses'], 'ro-', linewidth=2, markersize=6)
ax1.set_title('IEEE 123-Bus: Total System Losses vs Load Addition', fontweight='bold', fontsize=14)
ax1.set_xlabel('Load Addition per Bus (kW)', fontsize=12)
ax1.set_ylabel('Total System Losses (kW)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(results['load_added']) if results['load_added'] else 100)

# Plot 2: Individual Bus Voltages vs Load Added (subset of buses)
colors = plt.cm.tab20(np.linspace(0, 1, len(results['bus_voltages'])))
for i, (bus_name, voltages) in enumerate(results['bus_voltages'].items()):
    if len(voltages) == len(results['load_added']):  # Make sure we have complete data
        ax2.plot(results['load_added'], voltages, '-', linewidth=1.5,
                 label=bus_name, color=colors[i], marker='o', markersize=2)

ax2.set_title('IEEE 123-Bus: Bus Voltages vs Load Addition (Sample)', fontweight='bold', fontsize=14)
ax2.set_xlabel('Load Addition per Bus (kW)', fontsize=12)
ax2.set_ylabel('Bus Voltage (kV)', fontsize=12)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(results['load_added']) if results['load_added'] else 100)

plt.tight_layout()
plt.show()

# Simple summary
print("\n" + "=" * 70)
print("IEEE 123-BUS SYSTEM ANALYSIS SUMMARY (PQ LOAD MODEL)")
print("=" * 70)
print(f"Load Model Used: PQ (Constant Power)")
print(f"Total loads in system: {len(load_names)}")
print(f"Total successful data points: {len(results['total_losses'])}")
if results['total_losses']:
    print(f"Load range tested: {min(results['load_added'])} - {max(results['load_added'])} kW per bus")
    print(f"Total load range: {min(results['total_load']):.0f} - {max(results['total_load']):.0f} kW")
    print(f"Losses range: {min(results['total_losses']):.1f} - {max(results['total_losses']):.1f} kW")
    print(f"Loss percentage at max load: {(max(results['total_losses']) / max(results['total_load']) * 100):.2f}%")
    print(f"Number of buses monitored (sample): {len(results['bus_voltages'])}")
    print(f"Total buses in system: {len([bus for bus in dss.circuit.buses_names if 'sourcebus' not in bus.lower()])}")
    if results['bus_voltages']:
        all_voltages = [v for voltages in results['bus_voltages'].values() for v in voltages]
        print(f"Sample voltage range: {min(all_voltages):.2f} - {max(all_voltages):.2f} kV")
        print(f"Voltage variation: {((max(all_voltages) - min(all_voltages)) / max(all_voltages) * 100):.2f}%")
print("=" * 70)