import py_dss_interface
import os
import pathlib


# Define the path to the DSS file
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = pathlib.Path(script_path).joinpath("feeders", "13bus", "IEEE13Nodeckt.dss")

# Verify the file exists
if not dss_file.exists():
    print(f"Error: File not found at {dss_file}")
    exit(1)

# Initialize DSS interface
dss = py_dss_interface.DSS()

# Compile the DSS file FIRST
dss.text(f'compile "{str(dss_file)}"')

# Solve the circuit
dss.text("solve")

# Check convergence (property, not method)
if dss.solution.converged:
    # Now access bus information
    dss.circuit.set_active_bus("671")
    print("671:", dss.bus.vmag_angle)

    dss.circuit.set_active_bus("645")
    print("645:", dss.bus.vmag_angle)
else:
    print("Solution did not converge!")