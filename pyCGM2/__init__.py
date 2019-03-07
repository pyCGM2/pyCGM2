import sys
import glob
import re
import os
import logging

def getLastNexusVersion():
    nexusDir = "C:\Program Files (x86)\Vicon"
    dirs = os.listdir(nexusDir)
    li =[]
    for it in dirs:
        if "Nexus2" in it:
            version = int(it[it.find(".")+1:])
            li.append(version)
    last = max(li)
    return "Nexus2."+str(last)

try:
    NEXUS_VERSION = getLastNexusVersion()
    if not "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Python" in sys.path:
        sys.path.append( "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Python")

    if not "C:/Program Files (x86)/Vicon/Nexus"+NEXUS_VERSION+"/SDK/Win32" in sys.path:
        sys.path.append( "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Win32")
except Exception, errormsg:
    logging.error (errormsg)

# CONSTANTS
cased_path = glob.glob(re.sub(r'([^:])(?=[/\\]|$)', r'[\1]', __file__))[0]
MAIN_PYCGM2_PATH = os.path.abspath(os.path.join(os.path.dirname(cased_path), os.pardir)) + "\\"

#  [Optional] setting folder
PYCGM2_SETTINGS_FOLDER = MAIN_PYCGM2_PATH+"pyCGM2\Settings\\"


#  [Optional]programData
if  os.path.isdir(os.getenv("PROGRAMDATA")+"\\pyCGM2"):
    PYCGM2_APPDATA_PATH = os.getenv("PROGRAMDATA")+"\\pyCGM2\\"
else:
    PYCGM2_APPDATA_PATH = PYCGM2_SETTINGS_FOLDER
    



# [Optional]: Apps path
MAIN_PYCGM2_APPS_PATH = MAIN_PYCGM2_PATH+"Apps\\"

# [Optional] path to embbbed Normative data base.
NORMATIVE_DATABASE_PATH = MAIN_PYCGM2_PATH +"pyCGM2\\Data\\normativeData\\"  # By default, use pyCGM2-embedded normative data ( Schartz - Pinzone )

# [Optional] main folder containing osim model
OPENSIM_PREBUILD_MODEL_PATH = PYCGM2_APPDATA_PATH + "opensim\\"


# [Optional] path pointing at Data Folders used for Tests
TEST_DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\"
MAIN_BENCHMARK_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-benchmarks\\Gait patterns\\"

# [optional] path pointing pyCGM2-Nexus tools
NEXUS_PYCGM2_TOOLS_PATH = MAIN_PYCGM2_PATH + "pyCGM2\\Nexus\\"
