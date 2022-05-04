import sys
import glob
import re
import os
from . import log
import yaml
import yamlordereddictloader


LOGGER = log.pyCGM2_Logger(__name__)
NEXUS_VERSION = None

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
    if NEXUS_VERSION is None:
        NEXUS_VERSION = getLastNexusVersion()

    version = float(NEXUS_VERSION[5:])
    if version<2.12:
        LOGGER.logger.error ("This version of pyCGM2 is only compatible Nexus 2.12+. Nexus Apps will fail if you call them ")
    else :
        if not "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Win64/Python/viconnexusapi" in sys.path:
            sys.path.append( "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Win64/Python/viconnexusapi")

        if not "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Win64/Python/viconnexusutils" in sys.path:
            sys.path.append( "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Win64/Python/viconnexusutils")

except Exception:
    # integration on Linux not needed
    # LOGGER.logger.error ("Nexus Integration failed")
    pass


ENCODER = "latin-1"

# CONSTANTS
MAIN_PYCGM2_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + "\\"

#  [Optional] setting folder
PYCGM2_SETTINGS_FOLDER = MAIN_PYCGM2_PATH+"pyCGM2\Settings\\"

EMG_CHANNELS = list()
try:
    for key in yaml.load(open((PYCGM2_SETTINGS_FOLDER+"emg.settings")).read(),Loader=yamlordereddictloader.Loader)["CHANNELS"].keys():
        EMG_CHANNELS.append(key)
except:
    LOGGER.logger.error ("EMG_CHANNELS is empty ")



#  [Optional]programData
if (os.getenv("PROGRAMDATA") is not None) and \
   os.path.isdir(os.getenv("PROGRAMDATA")+"\\pyCGM2"):
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

TEST_DATA_PATH = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\"
TEST_DATA_PATH_OUT = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests-OUT\\"
MAIN_BENCHMARK_PATH = "C:\\Users\\fleboeuf\\Documents\\DATA\\Gait patterns\\"

# [optional] path pointing pyCGM2-Nexus tools
NEXUS_PYCGM2_TOOLS_PATH = MAIN_PYCGM2_PATH + "pyCGM2\\Nexus\\"

PYCGM2_SCRIPTS_PATH = MAIN_PYCGM2_PATH+"Scripts\\"
