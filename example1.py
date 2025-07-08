import py_dss_interface
import pathlib
dss= py_dss_interface.DSS(r"C:\Program Files\OpenDSS")

dss_file = r"C:\Program Files\OpenDSS\IEEETestCases\13Bus\IEEE13Nodeckt.dss"

dss.text(f"compile [{dss_file}]")

dss.text("solve")

dss.text("show voltages LN nodes")
