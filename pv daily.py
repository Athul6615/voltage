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
    print("Base case solved successfully")

    # Create LoadShape for PV (solar irradiance pattern)
    # Simple daily solar pattern - you can modify this
    solar_pattern = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 12AM-6AM: No sun
        0.1, 0.3, 0.5, 0.7, 0.8, 0.9,  # 6AM-12PM: Rising sun
        1.0, 1.0, 0.9, 0.8, 0.7, 0.5,  # 12PM-6PM: Peak and declining
        0.3, 0.1, 0.0, 0.0, 0.0, 0.0  # 6PM-12AM: Evening/night
    ]

    # Create LoadShape in DSS
    solar_mult_str = '[' + ' '.join([str(x) for x in solar_pattern]) + ']'
    dss.text(f'New Loadshape.SolarPattern npts=24 interval=1 mult={solar_mult_str}')
    print(f"Created solar irradiance pattern for 24 hours")

    # Define PV systems with daily variation
    pv_systems = [
        {"name": "PV1", "bus": "680", "kw": 50, "pf": 1.0},
        {"name": "PV2", "bus": "634", "kw": 75, "pf": 0.95},
        {"name": "PV3", "bus": "645", "kw": 100, "pf": 0.98}
    ]

    # Add PV systems with LoadShape
    for pv in pv_systems:
        pv_command = (f'New PVSystem.{pv["name"]} bus1={pv["bus"]} kV=4.16 '
                      f'kVA={pv["kw"] / pv["pf"]:.1f} Pmpp={pv["kw"]} pf={pv["pf"]} '
                      f'irradiance=1.0 temperature=25 daily=SolarPattern')
        dss.text(pv_command)
        print(f"Added {pv['name']}: {pv['kw']} kW at bus {pv['bus']} with daily variation")

    # Set solution mode to Daily
    dss.text("Set Mode=Daily")
    dss.text("Set Number=24")  # 24 hours
    dss.text("Set Stepsize=1h")  # 1 hour steps

    print(f"\nRunning Daily simulation (24 hours)...")

    # Initialize storage for results
    hours = []
    total_powers = []
    total_losses = []
    pv_generations = []
    bus_voltages = {bus: [] for bus in ["680", "634", "645"]}  # Monitor PV buses

    # Run daily simulation
    dss.text("Solve")

    # Collect results for each hour
    for hour in range(24):
        dss.text(f"Set hour={hour}")
        dss.text("Solve")

        if dss.solution.converged:
            # Get results for this hour
            total_power = dss.circuit.total_power
            losses = dss.circuit.losses

            # Calculate total PV generation for this hour
            total_pv_gen = 0
            pv_names = dss.pvsystems.names
            for pv_name in pv_names:
                dss.pvsystems.name = pv_name
                total_pv_gen += dss.pvsystems.pmpp * solar_pattern[hour]  # Scale by irradiance

            # Store results
            hours.append(hour)
            total_powers.append(total_power[0])
            total_losses.append(losses[0] / 1000)
            pv_generations.append(total_pv_gen)

            # Get voltages at PV buses
            all_voltages = dss.circuit.buses_vmag_pu
            bus_names = dss.circuit.buses_names

            for bus in bus_voltages.keys():
                if bus in bus_names:
                    bus_idx = bus_names.index(bus)
                    if bus_idx < len(all_voltages):
                        bus_voltages[bus].append(all_voltages[bus_idx])
                    else:
                        bus_voltages[bus].append(1.0)
                else:
                    bus_voltages[bus].append(1.0)

        else:
            print(f"Solution did not converge at hour {hour}")

    # Print daily summary
    print(f"\nDaily Analysis Summary:")
    print(f"Peak PV Generation: {max(pv_generations):.2f} kW at hour {pv_generations.index(max(pv_generations))}")
    print(f"Total Daily PV Energy: {sum(pv_generations):.2f} kWh")
    print(f"Peak Power Demand: {max([abs(p) for p in total_powers]):.2f} kW")
    print(f"Daily Energy Losses: {sum(total_losses):.2f} kWh")

    # Create daily plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Daily PV System Analysis (24 Hours)', fontsize=16, fontweight='bold')

    # Plot 1: Solar Irradiance Pattern
    ax1.plot(hours, solar_pattern, 'yo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Solar Irradiance (p.u.)')
    ax1.set_title('Solar Irradiance Pattern')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 23)

    # Plot 2: PV Generation vs Time
    ax2.plot(hours, pv_generations, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('PV Generation (kW)')
    ax2.set_title('Total PV Generation')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 23)

    # Plot 3: System Power vs Time
    ax3.plot(hours, total_powers, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Net System Power (kW)')
    ax3.set_title('Net Power Flow')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 23)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 4: Bus Voltages
    colors = ['r', 'g', 'b']
    for i, (bus, voltages) in enumerate(bus_voltages.items()):
        if voltages:  # Only plot if we have data
            ax4.plot(hours[:len(voltages)], voltages, f'{colors[i]}o-',
                     linewidth=2, markersize=4, label=f'Bus {bus}')

    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Voltage (p.u.)')
    ax4.set_title('PV Bus Voltages')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(0, 23)

    plt.tight_layout()
    plt.show()

    # Print hourly details for peak hours
    print(f"\nDetailed Results for Peak Hours:")
    peak_hours = [6, 12, 18]  # Morning, noon, evening
    for hour in peak_hours:
        if hour < len(hours):
            print(f"\nHour {hour}:00")
            print(f"  Solar Irradiance: {solar_pattern[hour]:.1f} p.u.")
            print(f"  PV Generation: {pv_generations[hour]:.2f} kW")
            print(f"  System Power: {total_powers[hour]:.2f} kW")
            print(f"  System Losses: {total_losses[hour]:.2f} kW")

else:
    print("Base case power flow did not converge!")