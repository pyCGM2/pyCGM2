import sys
import glob
import re
import os
from . import log
import yaml
import yamlordereddictloader
from dotenv import load_dotenv

__version__= "4.4rc2"


LOGGER = log.pyCGM2_Logger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))



# CONSTANTS
MAIN_PYCGM2_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + "\\"

MAIN_PYCGM2_TESTS_PATH = MAIN_PYCGM2_PATH+"Tests\\"

#opensim plugin binaries
sys.path.append( MAIN_PYCGM2_PATH +"pyCGM2\\opensim4\\KSlibaries\\lib\\bin")
OPENSIM_KSLIB_PATH = MAIN_PYCGM2_PATH +"pyCGM2\\opensim4\\KSlibaries\\lib\\bin\\" 


#  [Optional] setting folder
PYCGM2_SETTINGS_FOLDER = MAIN_PYCGM2_PATH+"pyCGM2\Settings\\"

EMG_CHANNELS = []
try:
    for key in yaml.load(open((PYCGM2_SETTINGS_FOLDER+"emg.settings")).read(),Loader=yamlordereddictloader.Loader)["CHANNELS"].keys():
        EMG_CHANNELS.append(key)
except:
    LOGGER.logger.error ("EMG_CHANNELS is empty ")

PYCGM2_APPDATA_PATH = MAIN_PYCGM2_PATH +"Data\\" 

# #  [Optional]programData
# if (os.getenv("PROGRAMDATA") is not None) and \
#    os.path.isdir(os.getenv("PROGRAMDATA")+"\\pyCGM2"):
#     PYCGM2_APPDATA_PATH = os.getenv("PROGRAMDATA")+"\\pyCGM2\\"
# else:
#     PYCGM2_APPDATA_PATH = PYCGM2_SETTINGS_FOLDER




# [Optional]: Apps path
MAIN_PYCGM2_APPS_PATH = MAIN_PYCGM2_PATH+"Apps\\"

# [Optional] path to embbbed Normative data base.
NORMATIVE_DATABASE_PATH = MAIN_PYCGM2_PATH +"Data\\normativeData\\"  # By default, use pyCGM2-embedded normative data ( Schartz - Pinzone )

# [Optional] main folder containing osim model
OPENSIM_PREBUILD_MODEL_PATH = PYCGM2_SETTINGS_FOLDER + "opensim\\"

# [Optional] path pointing at Data Folders used for Tests

TEST_DATA_PATH = os.getenv("TEST_DATA_PATH") 
TEST_DATA_PATH_OUT = os.getenv("TEST_DATA_PATH_OUT") 


# [optional] path pointing pyCGM2-Nexus tools
NEXUS_PYCGM2_TOOLS_PATH = MAIN_PYCGM2_PATH + "pyCGM2\\Nexus\\"


# [optional] moveck path (défini dans le fichier .env)
MOVECKPATH = os.getenv("MOVECK_API_FOLDER")
if MOVECKPATH is None:
    LOGGER.logger.error("No moveck path defined in .env file, moveck related implementation will not work.")
else:
    sys.path.append(MOVECKPATH)


