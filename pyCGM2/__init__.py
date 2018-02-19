import CONFIG
import json


thirdPartyLibraries= json.loads(open(str(CONFIG.PYCGM2_APPDATA_PATH+"thirdPartyLibraries")).read())

# vicon nexus
CONFIG.addNexusPythonSdk()

# openMA
if thirdPartyLibraries["OpenMA"]: CONFIG.addOpenma()

#btk
if thirdPartyLibraries["Btk"]: CONFIG.addBtk()

#opensim
if thirdPartyLibraries["OpenSim"]: CONFIG.addOpensim3()
