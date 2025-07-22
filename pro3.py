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


def set_zip_load_model(dss, z_percent=50, i_percent=30, p_percent=20):
    """
    Configure ZIP load model for all loads in the system

    Parameters:
    z_percent: Percentage of constant impedance component (0-100)
    i_percent: Percentage of constant current component (0-100)
    p_percent: Percentage of constant power component (0-100)

    Note: z_percent + i_percent + p_percent should equal 100
    """
    # Convert percentages to fractions
    z_frac = z_percent / 100.0
    i_frac = i_percent / 100.0
    p_frac = p_percent / 100.0

    # Normalize if they don't add up to 1
    total = z_frac + i_frac + p_frac
    if abs(total - 1.0) > 0.001:
        z_frac /= total
        i_frac /= total
        p_frac /= total

    load_names = dss.loads.names
    print(f"Setting ZIP load model (Z:{z_percent}%, I:{i_percent}%, P:{p_percent}%) for {len(load_names)} loads")

    for load_name in load_names:
        dss.loads.name = load_name

        # Set ZIP coefficients for real power
        dss.loads.zipv = [z_frac, i_frac, p_frac, z_frac, i_frac, p_frac]

        # Alternative method using text commands for more control
        dss.text(
            f"load.{load_name}.zipv=[{z_frac:.3f},{i_frac:.3f},{p_frac:.3f},{z_frac:.3f},{i_frac:.3f},{p_frac:.3f}]")

        # Set load model to use ZIP coefficients
        dss.loads.model = 1  # Model 1 uses ZIP coefficients


def continuation_power_flow_step_zip(dss, lambda_val, base_loads, critical_bus_name, zip_coeffs):
    """
    Perform a continuation power flow step for given lambda value with ZIP load model
    """
    # Set loads based on lambda parameter with ZIP model characteristics
    for load_name, base_load in base_loads.items():
        dss.loads.name = load_name

        # Get current bus voltage for ZIP calculations
        try:
            dss.circuit.set_active_bus(dss.loads.bus1)
            bus_voltages = dss.bus.voltages
            if bus_voltages:
                voltage_mag = abs(complex(bus_voltages[0], bus_voltages[1] if len(bus_voltages) > 1 else 0))
                voltage_pu = voltage_mag / (dss.bus.kv_base * 1000)
            else:
                voltage_pu = 1.0
        except:
            voltage_pu = 1.0

        # Calculate ZIP load components
        # P_total = P_Z * (V/V0)^2 + P_I * (V/V0) + P_P
        # where P_Z, P_I, P_P are the Z, I, P components respectively

        base_kw = base_load['kw']
        base_kvar = base_load['kvar']

        # Apply loading factor and ZIP characteristics
        z_frac, i_frac, p_frac = zip_coeffs

        # For the continuation method, we scale the base load by (1 + lambda)
        # The ZIP model will naturally adjust based on voltage
        new_kw = base_kw * (1 + lambda_val)
        new_kvar = base_kvar * (1 + lambda_val)

        dss.loads.kw = new_kw
        dss.loads.kvar = new_kvar

    # Try multiple solution strategies with increased robustness for ZIP loads
    strategies = [
        {'algo': 'normal', 'max_iter': 150, 'tol': 0.0001},
        {'algo': 'normal', 'max_iter': 200, 'tol': 0.001},
        {'algo': 'newton', 'max_iter': 200, 'tol': 0.001},
        {'algo': 'normal', 'max_iter': 300, 'tol': 0.01},
        {'algo': 'newton', 'max_iter': 250, 'tol': 0.01},
        {'algo': 'normal', 'max_iter': 500, 'tol': 0.1}
    ]

    converged = False
    voltage_pu = None

    for strategy in strategies:
        # Reset and apply strategy
        dss.text(f"set algorithm={strategy['algo']}")
        dss.text(f"set maxiterations={strategy['max_iter']}")
        dss.text(f"set tolerance={strategy['tol']}")
        dss.text("set controlmode=static")

        # Solve
        dss.text("solve")
        converged, _ = check_convergence_detailed(dss)

        if converged:
            # Get voltage at critical bus
            try:
                dss.circuit.set_active_bus(critical_bus_name)
                bus_voltages = dss.bus.voltages
                if bus_voltages:
                    voltage_mag = abs(complex(bus_voltages[0], bus_voltages[1] if len(bus_voltages) > 1 else 0))
                    voltage_pu = voltage_mag / (dss.bus.kv_base * 1000)
                    break
            except:
                converged = False
                continue

    return converged, voltage_pu, lambda_val


def predictor_corrector_pv_curve_zip(dss, base_loads, critical_bus_name, zip_coeffs, max_lambda=5.0, initial_step=0.02):
    """
    Generate P-V curve using predictor-corrector continuation method with ZIP loads
    """
    lambda_values = []
    voltages = []
    converged_flags = []
    actual_loads = []  # Track actual power consumption

    # Start from base case
    lambda_val = 0.0
    step_size = initial_step
    min_step = 0.0005
    max_step = 0.05

    # Get base case voltage
    converged, voltage_pu, _ = continuation_power_flow_step_zip(dss, lambda_val, base_loads, critical_bus_name,
                                                                zip_coeffs)
    if converged:
        lambda_values.append(lambda_val)
        voltages.append(voltage_pu)
        converged_flags.append(True)

        # Calculate actual load consumed
        total_actual_load = 0
        for load_name in base_loads.keys():
            dss.loads.name = load_name
            total_actual_load += dss.loads.kw
        actual_loads.append(total_actual_load)

        prev_voltage = voltage_pu
        prev_lambda = lambda_val

        print(f"Base case: λ = {lambda_val:.3f}, V = {voltage_pu:.3f} pu, Load = {total_actual_load:.1f} kW")
    else:
        print("Base case failed to converge!")
        return lambda_values, voltages, converged_flags, actual_loads

    consecutive_failures = 0
    max_consecutive_failures = 3

    # Forward continuation to find the maximum loadability
    while lambda_val < max_lambda and consecutive_failures < max_consecutive_failures:
        # Predictor step
        lambda_val += step_size

        # Corrector step
        converged, voltage_pu, _ = continuation_power_flow_step_zip(dss, lambda_val, base_loads, critical_bus_name,
                                                                    zip_coeffs)

        if converged and voltage_pu is not None and voltage_pu > 0.05:
            lambda_values.append(lambda_val)
            voltages.append(voltage_pu)
            converged_flags.append(True)

            # Calculate actual load consumed with ZIP model
            total_actual_load = 0
            for load_name in base_loads.keys():
                dss.loads.name = load_name
                total_actual_load += dss.loads.kw
            actual_loads.append(total_actual_load)

            consecutive_failures = 0

            # Check if we've passed the nose (voltage starts increasing significantly)
            if len(voltages) > 2:
                voltage_trend = voltages[-1] - voltages[-2]
                prev_voltage_trend = voltages[-2] - voltages[-3] if len(voltages) > 2 else 0

                if voltage_trend > 0 and prev_voltage_trend < 0:
                    print(f"Potential nose point detected at λ = {lambda_val:.3f}, V = {voltage_pu:.3f} pu")
                    step_size = max(min_step, step_size * 0.3)

            # Adaptive step size control based on voltage change
            if len(voltages) > 1:
                voltage_change = abs(voltage_pu - prev_voltage)
                if voltage_change > 0.03:  # Large voltage change
                    step_size = max(min_step, step_size * 0.6)
                elif voltage_change < 0.005:  # Small voltage change
                    step_size = min(max_step, step_size * 1.3)

            prev_voltage = voltage_pu
            prev_lambda = lambda_val

            print(
                f"λ = {lambda_val:.3f}, V = {voltage_pu:.3f} pu, Load = {total_actual_load:.1f} kW, Step = {step_size:.4f}")

            # More aggressive continuation - only stop at very low voltages
            if voltage_pu < 0.1:
                print("Voltage critically low, stopping continuation")
                break

        else:
            print(f"Failed to converge at λ = {lambda_val:.3f}, reducing step size")
            lambda_val -= step_size  # Step back
            step_size = max(min_step, step_size * 0.4)
            consecutive_failures += 1

            if step_size <= min_step and consecutive_failures >= 2:
                print("Minimum step size reached with multiple failures, attempting final push...")
                # Try a few more very small steps
                for micro_step in [0.001, 0.0005, 0.0001]:
                    lambda_val += micro_step
                    converged, voltage_pu, _ = continuation_power_flow_step_zip(dss, lambda_val, base_loads,
                                                                                critical_bus_name, zip_coeffs)
                    if converged and voltage_pu is not None and voltage_pu > 0.05:
                        lambda_values.append(lambda_val)
                        voltages.append(voltage_pu)
                        converged_flags.append(True)
                        # Calculate actual load
                        total_actual_load = 0
                        for load_name in base_loads.keys():
                            dss.loads.name = load_name
                            total_actual_load += dss.loads.kw
                        actual_loads.append(total_actual_load)
                        print(
                            f"Final point: λ = {lambda_val:.3f}, V = {voltage_pu:.3f} pu, Load = {total_actual_load:.1f} kW")
                    else:
                        lambda_val -= micro_step  # Step back
                        break
                break

    return lambda_values, voltages, converged_flags, actual_loads


def find_critical_bus(dss):
    """
    Find the bus with the lowest voltage (most critical)
    """
    bus_names = [bus for bus in dss.circuit.buses_names if "sourcebus" not in bus.lower()]
    min_voltage = float('inf')
    critical_bus = None

    for bus_name in bus_names:
        try:
            dss.circuit.set_active_bus(bus_name)
            bus_voltages = dss.bus.voltages
            if bus_voltages:
                voltage_mag = abs(complex(bus_voltages[0], bus_voltages[1] if len(bus_voltages) > 1 else 0))
                voltage_pu = voltage_mag / (dss.bus.kv_base * 1000)
                if voltage_pu < min_voltage and voltage_pu > 0:
                    min_voltage = voltage_pu
                    critical_bus = bus_name
        except:
            continue

    return critical_bus


# Initialize DSS
print("Initializing OpenDSS...")
dss = py_dss_interface.DSS()

# Load IEEE 13-bus system
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = pathlib.Path(script_path).joinpath("feeders", "13bus", "IEEE13Nodeckt.dss")

if not dss_file.exists():
    print(f"Error: File not found - {dss_file}")
    print("Please ensure the IEEE 13-bus system files are in the correct directory.")
    exit()

dss.text(f'compile "{str(dss_file)}"')
print("System loaded successfully")

# Initial system configuration
print("\nConfiguring system...")
dss.text("set mode=snapshot")
dss.text("set maxiterations=150")
dss.text("set tolerance=0.001")
dss.text("set controlmode=static")

# Configure ZIP load model
# Typical ZIP coefficients: Z=40%, I=30%, P=30% for residential loads
Z_PERCENT = 40  # Constant impedance
I_PERCENT = 30  # Constant current
P_PERCENT = 30  # Constant power

print(f"\nConfiguring ZIP load model (Z:{Z_PERCENT}%, I:{I_PERCENT}%, P:{P_PERCENT}%)...")
set_zip_load_model(dss, Z_PERCENT, I_PERCENT, P_PERCENT)
zip_coeffs = (Z_PERCENT / 100, I_PERCENT / 100, P_PERCENT / 100)

# Initial solve
print("\nSolving base case with ZIP loads...")
dss.text("solve")
converged, conv_info = check_convergence_detailed(dss)

if not converged:
    print("Base case failed to converge!")
    print(f"Convergence info: {conv_info}")
    exit()

print("Base case converged successfully!")

# Get base load data
load_names = dss.loads.names
base_loads = {}
total_base_load = 0

for load_name in load_names:
    dss.loads.name = load_name
    base_loads[load_name] = {
        'kw': dss.loads.kw,
        'kvar': dss.loads.kvar
    }
    total_base_load += dss.loads.kw

print(f"Base system load: {total_base_load:.1f} kW")

# Find critical bus
critical_bus = find_critical_bus(dss)
print(f"Critical bus identified: {critical_bus}")

# Generate P-V curve using continuation method with ZIP loads
print(f"\nGenerating P-V curve using continuation power flow with ZIP load model...")
print("Pushing load to maximum possible value...")

lambda_values, voltages, converged_flags, actual_loads = predictor_corrector_pv_curve_zip(
    dss, base_loads, critical_bus, zip_coeffs, max_lambda=6.0, initial_step=0.01
)

print(f"\nP-V curve generation complete! {len(lambda_values)} points obtained")

if len(lambda_values) > 0:
    max_lambda = max(lambda_values)
    max_load = max(actual_loads) if actual_loads else 0
    print(f"Maximum λ achieved: {max_lambda:.3f}")
    print(f"Maximum load achieved: {max_load:.1f} kW ({max_load / total_base_load:.2f} times base load)")

# Create clean Voltage vs Lambda graph showing nose curve
plt.figure(figsize=(12, 8))

if len(voltages) > 0:
    # Plot the P-V curve with lambda on x-axis
    plt.plot(lambda_values, voltages, 'b-', linewidth=3, label='P-V Curve (ZIP Load Model)')
    plt.plot(lambda_values, voltages, 'bo', markersize=6, markerfacecolor='lightblue',
             markeredgecolor='blue', markeredgewidth=1.5)

    # Find and mark the nose point (minimum voltage)
    min_voltage_idx = np.argmin(voltages)
    nose_lambda = lambda_values[min_voltage_idx]
    nose_voltage = voltages[min_voltage_idx]

    plt.plot(nose_lambda, nose_voltage, 'ro', markersize=14, markerfacecolor='red',
             markeredgecolor='darkred', markeredgewidth=2,
             label=f'Nose Point (λ = {nose_lambda:.3f}, V = {nose_voltage:.3f} pu)', zorder=10)

    # Mark operating point (base case)
    plt.plot(lambda_values[0], voltages[0], 'go', markersize=12, markerfacecolor='green',
             markeredgecolor='darkgreen', markeredgewidth=2,
             label=f'Operating Point (λ = {lambda_values[0]:.3f}, V = {voltages[0]:.3f} pu)', zorder=10)

    # Add vertical line at nose point to show maximum loadability
    plt.axvline(x=nose_lambda, color='red', linestyle='--', alpha=0.6, linewidth=2)

    # Add horizontal voltage limit lines
    plt.axhline(y=0.95, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Normal Limit (0.95 pu)')
    plt.axhline(y=0.9, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Acceptable Limit (0.90 pu)')

    # Shade regions to show stability zones
    if len(lambda_values) > 1:
        # Stable region (left side of nose)
        stable_region = [i for i, lam in enumerate(lambda_values) if lam <= nose_lambda]
        if stable_region:
            plt.fill_between(lambda_values[:max(stable_region) + 1],
                             [0] * len(lambda_values[:max(stable_region) + 1]),
                             voltages[:max(stable_region) + 1],
                             alpha=0.1, color='green', label='Stable Region')

        # Unstable region (right side of nose)
        unstable_region = [i for i, lam in enumerate(lambda_values) if lam > nose_lambda]
        if unstable_region:
            plt.fill_between(lambda_values[min(unstable_region):],
                             [0] * len(lambda_values[min(unstable_region):]),
                             voltages[min(unstable_region):],
                             alpha=0.1, color='red', label='Unstable Region')

    # Add annotations for key regions
    mid_stable_idx = len([lam for lam in lambda_values if lam <= nose_lambda]) // 2
    if mid_stable_idx > 0:
        plt.annotate('STABLE\nREGION',
                     xy=(lambda_values[mid_stable_idx], voltages[mid_stable_idx]),
                     xytext=(lambda_values[mid_stable_idx] - 0.3, voltages[mid_stable_idx] + 0.1),
                     fontsize=12, fontweight='bold', color='green',
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))

    if len(lambda_values) > min_voltage_idx + 5:
        unstable_idx = min_voltage_idx + (len(lambda_values) - min_voltage_idx) // 2
        plt.annotate('UNSTABLE\nREGION',
                     xy=(lambda_values[unstable_idx], voltages[unstable_idx]),
                     xytext=(lambda_values[unstable_idx] + 0.2, voltages[unstable_idx] + 0.15),
                     fontsize=12, fontweight='bold', color='red',
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Add nose point annotation
    plt.annotate(f'NOSE POINT\n(Maximum Loadability)\nλ = {nose_lambda:.3f}\nV = {nose_voltage:.3f} pu',
                 xy=(nose_lambda, nose_voltage),
                 xytext=(nose_lambda - 0.4, nose_voltage + 0.25),
                 fontsize=11, fontweight='bold', color='darkred',
                 ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8, edgecolor='red'),
                 arrowprops=dict(arrowstyle='->', color='red', lw=3))

# Customize the plot
plt.title('Voltage Stability Analysis: P-V Curve (Nose Curve)\nZIP Load Model - Bus Voltage vs Loading Parameter',
          fontweight='bold', fontsize=16, pad=20)
plt.xlabel('Loading Parameter (λ)', fontsize=14, fontweight='bold')
plt.ylabel('Critical Bus Voltage (pu)', fontsize=14, fontweight='bold')

# Enhanced grid
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.2, linestyle=':', linewidth=0.3, which='minor')

# Legend
plt.legend(fontsize=11, loc='lower left', framealpha=0.9, fancybox=True, shadow=True)

# Set axis limits for better visibility of the nose
if lambda_values and voltages:
    lambda_range = max(lambda_values) - min(lambda_values)
    voltage_range = max(voltages) - min(voltages)

    plt.xlim(min(lambda_values) - 0.05 * lambda_range, max(lambda_values) + 0.05 * lambda_range)
    plt.ylim(min(voltages) - 0.05 * voltage_range, max(voltages) + 0.1 * voltage_range)

# Add subtle background
plt.gca().set_facecolor('#fafafa')

plt.tight_layout()
plt.show()

# Print comprehensive summary
print("\n" + "=" * 90)
print("ZIP LOAD MODEL P-V CURVE ANALYSIS SUMMARY")
print("=" * 90)
print(f"Load Model Configuration: Z={Z_PERCENT}%, I={I_PERCENT}%, P={P_PERCENT}%")
print(f"Total points in P-V curve: {len(lambda_values)}")
print(f"Base case voltage: {voltages[0]:.3f} pu" if voltages else "No data")
print(f"Base system load: {total_base_load:.1f} kW")

if len(voltages) > 1 and actual_loads:
    min_voltage_idx = np.argmin(voltages)
    max_load_factor = load_factors[min_voltage_idx]
    min_voltage = voltages[min_voltage_idx]
    max_actual_load = actual_loads[min_voltage_idx]

    print(f"\nMaximum Loadability Results:")
    print(f"  Maximum load factor (λ): {max_load_factor:.3f}")
    print(f"  Critical voltage: {min_voltage:.3f} pu")
    print(f"  Maximum load achieved: {max_actual_load:.1f} kW")
    print(f"  Load increase: {(max_load_factor - 1) * 100:.1f}%")
    print(f"  Load multiplication factor: {max_actual_load / total_base_load:.2f}x")

    # Calculate stability margins
    operating_voltage = voltages[0]
    voltage_margin = ((operating_voltage - min_voltage) / operating_voltage) * 100
    loading_margin = ((max_load_factor - 1) / 1) * 100

    print(f"\nStability Margins:")
    print(f"  Voltage stability margin: {voltage_margin:.1f}%")
    print(f"  Loading margin: {loading_margin:.1f}%")
    print(f"  Power transfer capability: {max_actual_load - total_base_load:.1f} kW additional")

print(f"\nCritical bus: {critical_bus}")
print(f"ZIP load model provides more realistic load behavior compared to constant power model")
print("=" * 90)