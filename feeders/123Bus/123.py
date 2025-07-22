
import py_dss_interface
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

def check_convergence(dss):
    return dss.solution.converged

def run_analysis(dss, load_model, multipliers):
    for name in dss.loads.names:
        if name and name.lower() != "none":
            dss.loads.name = name
            dss.loads.model = load_model

    dss.text("solve")
    if not check_convergence(dss):
        print(f"Initial solve failed for model {load_model}. Exiting.")
        return None

    load_names = dss.loads.names
    original_loads = {}
    for name in load_names:
        dss.loads.name = name
        original_loads[name] = {
            'kw': dss.loads.kw,
            'kvar': dss.loads.kvar
        }

    base_voltages = {}
    for bus in dss.circuit.buses_names:
        dss.circuit.set_active_bus(bus)
        v = dss.bus.voltages
        if v and len(v) >= 2:
            mag = (v[0]**2 + v[1]**2)**0.5 / 1000
            base_voltages[bus] = mag
    live_buses = [bus for bus, v in base_voltages.items() if v > 0.4]

    results = {
        'multiplier': [],
        'total_load': [],
        'total_losses': [],
        'bus_voltages': {bus: [] for bus in live_buses},
        'converged': []
    }

    for m in multipliers:
        for name in dss.loads.names:
            if name and name.lower() != "none":
                dss.loads.name = name
                dss.loads.model = load_model

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
    return results, live_buses

def get_weakest_bus(results, live_buses):
    weakest_bus = None
    min_overall = float('inf')
    for bus in live_buses:
        v_list = results['bus_voltages'][bus]
        min_v = np.nanmin(v_list)
        if min_v < min_overall:
            min_overall = min_v
            weakest_bus = bus
    return weakest_bus

def get_collapse_point(mult, v_curve):
    v_curve = np.array(v_curve)
    mult = np.array(mult)
    n = min(len(mult), len(v_curve))
    v_curve = v_curve[:n]
    mult = mult[:n]
    collapse_idx = np.where(np.isnan(v_curve))[0]
    if len(collapse_idx) > 0:
        collapse_idx = collapse_idx[0] - 1
    else:
        collapse_idx = np.argmin(v_curve)
    collapse_lambda = mult[collapse_idx]
    collapse_voltage = v_curve[collapse_idx]
    return collapse_lambda, collapse_voltage

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

    multipliers = np.linspace(0, 10, 80)

    # PQ Analysis
    pq_results, live_buses = run_analysis(dss, 1, multipliers)
    # CVR Analysis (re-compile DSS file to reset)
    dss.text(f'compile "{str(path)}"')
    dss.text("set mode=snapshot")
    dss.text("set controlmode=static")
    dss.text("set maxiterations=100")
    dss.text("set tolerance=0.001")
    cvr_results, _ = run_analysis(dss, 5, multipliers)

    # Use only converged range for each case!
    pq_mult = pq_results['multiplier']
    cvr_mult = cvr_results['multiplier']

    weakest_bus = get_weakest_bus(pq_results, live_buses)
    v_curve_pq = pq_results['bus_voltages'][weakest_bus]
    v_curve_cvr = cvr_results['bus_voltages'][weakest_bus]
    cl_pq, v_pq = get_collapse_point(pq_mult, v_curve_pq)
    cl_cvr, v_cvr = get_collapse_point(cvr_mult, v_curve_cvr)

    # Plot system losses comparison
    plt.figure(figsize=(10,6))
    plt.plot(pq_mult, pq_results['total_losses'], label='PQ Model', linewidth=2)
    plt.plot(cvr_mult, cvr_results['total_losses'], label='CVR Model', linewidth=2)
    plt.title("Total System Losses vs Load Multiplier\n(PQ vs CVR)", fontsize=15)
    plt.xlabel('Load Multiplier (Lambda)', fontsize=12)
    plt.ylabel('Total System Losses (kW)', fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot voltage nose curve for weakest bus
    plt.figure(figsize=(10,6))
    plt.plot(pq_mult, v_curve_pq, label='PQ Model', linewidth=2)
    plt.plot(cvr_mult, v_curve_cvr, label='CVR Model', linewidth=2)
    plt.plot(cl_pq, v_pq, 'ro', label="PQ Collapse Point")
    plt.plot(cl_cvr, v_cvr, 'mo', label="CVR Collapse Point")
    plt.title(f'PV (Nose) Curve at Critical Bus: {weakest_bus}\n(PQ vs CVR)', fontsize=15)
    plt.xlabel('Lambda (Load Multiplier)', fontsize=12)
    plt.ylabel(f'Voltage at bus {weakest_bus} (kV)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nComparison Summary")
    print("=====================")
    print(f"Critical (weakest) Bus: {weakest_bus}")
    print(f"PQ Collapse Multiplier: {cl_pq:.2f}, Voltage at Collapse: {v_pq:.4f} kV")
    print(f"CVR Collapse Multiplier: {cl_cvr:.2f}, Voltage at Collapse: {v_cvr:.4f} kV")
    print("=====================")

if __name__ == "__main__":
    main()
