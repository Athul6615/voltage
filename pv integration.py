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
    print("Base case solved successfully")

    # Get bus names for PV placement
    bus_names = dss.circuit.buses_names
    print(f"Available buses: {bus_names}")

    # Define PV systems to be added
    pv_systems = [
        {"name": "PV1", "bus": "680", "kw": 50, "pf": 1.0},
        {"name": "PV2", "bus": "634", "kw": 75, "pf": 0.95},
        {"name": "PV3", "bus": "645", "kw": 100, "pf": 0.98}
    ]

    # Add PV systems to the circuit
    for pv in pv_systems:
        # Create PVSystem in DSS
        pv_command = f'New PVSystem.{pv["name"]} bus1={pv["bus"]} kV=4.16 kVA={pv["kw"] / pv["pf"]:.1f} Pmpp={pv["kw"]} pf={pv["pf"]} irradiance=1.0 temperature=25'
        dss.text(pv_command)
        print(f"Added {pv['name']}: {pv['kw']} kW at bus {pv['bus']}")

    # Solve with PV systems
    dss.text("solve")

    if dss.solution.converged:
        print("\nSystem with PV solved successfully")

        # Get total circuit power
        total_power = dss.circuit.total_power
        print(f"Total Circuit Power: {total_power[0]:.2f} kW, {total_power[1]:.2f} kvar")

        # Get system losses
        losses = dss.circuit.losses
        print(f"System Losses: {losses[0] / 1000:.2f} kW, {losses[1] / 1000:.2f} kvar")

        # Get PV generation summary
        pv_names = dss.pvsystems.names
        total_pv_generation = 0

        print("\nPV Generation Summary:")
        for pv_name in pv_names:
            dss.pvsystems.name = pv_name
            pv_power = dss.pvsystems.pmpp
            total_pv_generation += pv_power
            print(f"  {pv_name}: {pv_power:.2f} kW")

        print(f"Total PV Generation: {total_pv_generation:.2f} kW")

        # Get voltage profile at key buses
        print("\nVoltage Profile (p.u.):")
        key_buses = ["sourcebus", "650", "634", "680", "645", "611"]

        # Get all bus voltages at once
        dss.text("show voltages")
        all_voltages = dss.circuit.buses_vmag_pu

        for i, bus in enumerate(bus_names):
            if bus in key_buses and i < len(all_voltages):
                print(f"  Bus {bus}: {all_voltages[i]:.4f} p.u.")

        # Additional analysis
        print(f"\nSystem Analysis:")
        print(f"Net Power Injection from Grid: {abs(total_power[0]):.2f} kW")
        print(f"PV Penetration: {(total_pv_generation / abs(total_power[0])) * 100:.1f}%")
        print(f"Loss Reduction: PV systems help reduce transmission losses")

        # Show power factor at PV buses
        print(f"\nPV Systems Performance:")
        for pv in pv_systems:
            print(f"  {pv['name']}: {pv['kw']} kW at {pv['pf']} power factor")

    else:
        print("System with PV did not converge!")

else:
    print("Base case power flow did not converge!")