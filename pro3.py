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


def improve_convergence(dss, lambda_val, attempt=1):
    """
    Apply multiple convergence strategies systematically
    """
    print(f"    Convergence attempt {attempt} for λ = {lambda_val}")

    strategies = [
        # Strategy 1: Increase iterations with tight tolerance
        {
            'name': 'Increased Iterations',
            'commands': [
                "set maxiterations=200",
                "set tolerance=0.0001",
                "solve"
            ]
        },

        # Strategy 2: Newton-Raphson with moderate tolerance
        {
            'name': 'Newton-Raphson',
            'commands': [
                "set algorithm=newton",
                "set maxiterations=150",
                "set tolerance=0.001",
                "solve"
            ]
        },

        # Strategy 3: Relaxed tolerance with more iterations
        {
            'name': 'Relaxed Tolerance',
            'commands': [
                "set algorithm=normal",
                "set maxiterations=300",
                "set tolerance=0.01",
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
            else:
                print(f"      ❌ Failed with {strategy['name']}")

            return converged, conv_info

        except Exception as e:
            print(f"      Error in strategy {strategy['name']}: {e}")
            return False, None

    return False, None


def reset_convergence_parameters(dss):
    """
    Reset DSS to default convergence parameters
    """
    dss.text("set maxiterations=100")
    dss.text("set tolerance=0.001")
    dss.text("set algorithm=normal")
    dss.text("set loadmodel=1")
    dss.text("batchedit regcontrol..* enabled=yes")
    dss.text("batchedit capcontrol..* enabled=yes")


def get_critical_bus_voltage(dss):
    """
    Get the minimum voltage magnitude across all buses (critical bus)
    """
    bus_names = [bus for bus in dss.circuit.buses_names if "sourcebus" not in bus.lower()]
    min_voltage = float('inf')
    critical_bus = None

    for bus_name in bus_names:
        try:
            dss.circuit.set_active_bus(bus_name)
            bus_voltages = dss.bus.voltages
            if bus_voltages:
                # Calculate voltage magnitude
                voltage_mag = abs(complex(bus_voltages[0], bus_voltages[1] if len(bus_voltages) > 1 else 0))
                voltage_kv = voltage_mag / 1000

                if voltage_kv < min_voltage:
                    min_voltage = voltage_kv
                    critical_bus = bus_name
        except:
            continue

    return min_voltage, critical_bus


def run_pv_curve_analysis():
    """
    Run P-V curve analysis for voltage stability assessment
    """
    # Initialize DSS
    print("Initializing OpenDSS...")
    dss = py_dss_interface.DSS()

    # Load IEEE 13-bus system
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("feeders", "13bus", "IEEE13Nodeckt.dss")

    if not dss_file.exists():
        print(f"Error: File not found - {dss_file}")
        return

    dss.text(f'compile "{str(dss_file)}"')
    print("System loaded successfully")

    # Configure system
    dss.text("set mode=snapshot")
    dss.text("set maxiterations=100")
    dss.text("set tolerance=0.001")
    dss.text("set maxcontroliter=50")
    dss.text("set controlmode=static")

    # Initial solve
    dss.text("solve")
    initial_converged, initial_info = check_convergence_detailed(dss)

    if initial_converged:
        print("✓ Initial solution converged successfully")
    else:
        print("❌ Initial solution failed to converge!")
        return

    # Get load information
    load_names = dss.loads.names
    print(f"\nFound {len(load_names)} loads: {load_names}")

    # Store original load values
    original_loads = {}
    original_total_kw = 0

    for load in load_names:
        dss.loads.name = load
        original_loads[load] = {
            'kw': dss.loads.kw,
            'kvar': dss.loads.kvar
        }
        original_total_kw += dss.loads.kw

    print(f"Original total load: {original_total_kw:.0f} kW")

    # Analysis parameters - using lambda (λ) multiplier approach
    print("\nUsing multiplicative loading method")
    lambda_values = np.arange(0.1, 10.0, 0.1)  # λ from 0.1 to 10.0
    print(f"λ range: {min(lambda_values)} to {max(lambda_values)} in steps of 0.1")
    print(f"Total test points: {len(lambda_values)}")

    # Results storage
    results = {
        'lambda_values': [],
        'total_loads': [],
        'critical_voltages': [],
        'critical_bus': [],
        'converged_points': []
    }

    print("\nRunning λ-based loading analysis...")

    max_lambda = 0
    convergence_failures = 0

    for i, lambda_val in enumerate(lambda_values):
        print(f"\nTest {i + 1}/{len(lambda_values)}: λ = {lambda_val:.1f}")

        # Apply lambda multiplier to all loads
        total_kw = 0
        for load in load_names:
            dss.loads.name = load
            orig = original_loads[load]
            new_kw = orig['kw'] * lambda_val
            new_kvar = orig['kvar'] * lambda_val

            dss.loads.kw = new_kw
            dss.loads.kvar = new_kvar
            total_kw += new_kw

        # Reset convergence parameters
        reset_convergence_parameters(dss)

        # Solve
        dss.text("solve")
        converged, conv_info = check_convergence_detailed(dss)

        if not converged:
            print(f"  ❌ Initial solve failed, trying convergence strategies...")

            # Try convergence strategies
            for attempt in range(1, 4):
                conv_result, conv_info = improve_convergence(dss, lambda_val, attempt)
                if conv_result:
                    converged = True
                    break

            if not converged:
                print(f"  ❌ Failed to converge at λ = {lambda_val:.1f}")
                convergence_failures += 1
                print(f"  Maximum loading reached: λ = {max_lambda:.1f}")
                break

        # Get critical bus voltage
        min_voltage, critical_bus = get_critical_bus_voltage(dss)

        if min_voltage == float('inf'):
            print(f"  ❌ Could not read voltages")
            continue

        # Store results
        results['lambda_values'].append(lambda_val)
        results['total_loads'].append(total_kw)
        results['critical_voltages'].append(min_voltage)
        results['critical_bus'].append(critical_bus)
        results['converged_points'].append(converged)

        max_lambda = lambda_val

        # Check for voltage warnings
        if min_voltage < 0.9:  # Below 0.9 per unit (assuming 1 kV base)
            voltage_warning = f"⚠ Low voltage warning: {min_voltage:.2f} kV"
        else:
            voltage_warning = ""

        print(f"  ✓ Converged: Load = {total_kw:.0f} kW, Critical V = {min_voltage:.2f} kV")
        if voltage_warning:
            print(f"  {voltage_warning}")

    print(f"\nAnalysis complete! {len(results['lambda_values'])} successful data points")
    print(f"Maximum λ achieved: {max_lambda:.1f}")

    # Create P-V curve plot
    create_pv_curve_plot(results, original_total_kw)

    # Print summary
    print_pv_analysis_summary(results, original_total_kw, convergence_failures)


def create_pv_curve_plot(results, original_total_kw):
    """
    Create the P-V curve showing voltage stability characteristics
    """
    if not results['lambda_values']:
        print("No data to plot")
        return

    # Convert to per unit values
    P_pu = np.array(results['lambda_values'])  # λ values are already per unit
    V_pu = np.array(results['critical_voltages'])  # Assuming 1 kV base voltage

    # Create the P-V curve
    plt.figure(figsize=(12, 8))

    # Plot the P-V curve
    plt.plot(P_pu, V_pu, 'b-', linewidth=3, label='P-V Curve')
    plt.plot(P_pu, V_pu, 'bo', markersize=4, alpha=0.6)

    # Mark operating point (λ = 1.0)
    if 1.0 in results['lambda_values']:
        idx = results['lambda_values'].index(1.0)
        plt.plot(1.0, results['critical_voltages'][idx], 'go', markersize=12,
                 label='Operating Point', markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=2)

    # Mark maximum loadability point
    max_lambda = max(results['lambda_values'])
    max_idx = results['lambda_values'].index(max_lambda)
    plt.plot(max_lambda, results['critical_voltages'][max_idx], 'ro', markersize=12,
             label=f'Maximum Loadability Point\n(λ = {max_lambda:.1f})',
             markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)

    # Add voltage stability margin annotation
    if 1.0 in results['lambda_values']:
        plt.annotate('', xy=(max_lambda, 0.1), xytext=(1.0, 0.1),
                     arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        plt.text((1.0 + max_lambda) / 2, 0.05, 'Voltage Stability Margin',
                 ha='center', va='bottom', fontsize=12, color='purple', fontweight='bold')

    # Formatting
    plt.xlabel('Load Parameter (λ)', fontsize=14, fontweight='bold')
    plt.ylabel('Critical Bus Voltage (per unit)', fontsize=14, fontweight='bold')
    plt.title('P-V Curve - Voltage Stability Analysis\nIEEE 13-Bus Test System',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Set axis limits
    plt.xlim(0, max(P_pu) * 1.1)
    plt.ylim(0, max(V_pu) * 1.1)

    # Add stability regions
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Voltage Limit (0.9 pu)')
    plt.axvline(x=max_lambda, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def print_pv_analysis_summary(results, original_total_kw, convergence_failures):
    """
    Print comprehensive P-V analysis summary
    """
    print("\n" + "=" * 80)
    print("P-V CURVE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Loading Method: multiplicative (λ-based)")
    print(f"Original system load: {original_total_kw:.0f} kW")
    print(f"Successful data points: {len(results['lambda_values'])}")
    print(f"Convergence failures: {convergence_failures}")

    if results['lambda_values']:
        max_lambda = max(results['lambda_values'])
        min_voltage = min(results['critical_voltages'])

        print(f"λ range achieved: {min(results['lambda_values']):.1f} to {max_lambda:.1f}")
        print(f"Maximum loadability: {max_lambda:.1f} × base load")
        print(f"Maximum load achieved: {max(results['total_loads']):.0f} kW")
        print(f"Critical voltage range: {min_voltage:.2f} to {max(results['critical_voltages']):.2f} per unit")
        print(
            f"Voltage stability margin: {max_lambda - 1.0:.1f} per unit" if max_lambda > 1.0 else "System unstable at base load")

    print("=" * 80)


if __name__ == "__main__":
    run_pv_curve_analysis()