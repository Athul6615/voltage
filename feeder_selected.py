from doctest import script_from_examples

import py_dss_interface
import os
import pathlib
script_path = os.path.dirname(os.path.abspath(__file__))

dss_file = pathlib.Path(script_path).joinpath("feeders","13bus","IEEE13Nodeckt.dss")

dss=py_dss_interface.DSS()


dss.text(f'compile "{str(dss_file)}"')


dss.text("show voltages")

print("here")

















