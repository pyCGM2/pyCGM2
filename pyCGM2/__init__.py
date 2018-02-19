import CONFIG
import json


thirdPartyLibraries= json.loads(open(str(CONFIG.PYCGM2_APPDATA_PATH+"thirdPartyLibraries")).read())

# vicon nexus
CONFIG.addNexusPythonSdk()

# openMA
if not thirdPartyLibraries["OpenMA"]: CONFIG.addOpenma()

#btk
if not thirdPartyLibraries["Btk"]: CONFIG.addBtk()

#opensim
if not thirdPartyLibraries["OpenSim"]: CONFIG.addOpensim3()
