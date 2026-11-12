"""conda-build test: check that the data and settings files are embedded in the package."""
import os

import pyCGM2

expected = [
    os.path.join(pyCGM2.NORMATIVE_DATABASE_PATH, "Schwartz2008.json"),
    os.path.join(pyCGM2.NORMATIVE_DATABASE_PATH, "emg", "normalActivation.json"),
    os.path.join(pyCGM2.NORMATIVE_DATABASE_PATH, "stp", "pyCGM2_GaitRite_normaldata.xlsx"),
    os.path.join(pyCGM2.PYCGM2_APPDATA_PATH, "intellEventModels", "version0", "ic_intellevent.onnx"),
    os.path.join(pyCGM2.PYCGM2_APPDATA_PATH, "intellEventModels", "version0", "fo_intellevent.onnx"),
    os.path.join(pyCGM2.PYCGM2_SETTINGS_FOLDER, "emg.settings"),
    os.path.join(pyCGM2.PYCGM2_SETTINGS_FOLDER, "CGM2_3-pyCGM2.settings"),
]

missing = [f for f in expected if not os.path.isfile(f)]
if missing:
    raise FileNotFoundError("missing packaged files:\n" + "\n".join(missing))
print("pyCGM2", pyCGM2.__version__, "- packaged data OK")
