import sys
import glob
import re
import os
from . import log
import yaml
import yamlordereddictloader

__version__= "4.4rc2"


LOGGER = log.pyCGM2_Logger(__name__)



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

TEST_DATA_PATH = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\"
TEST_DATA_PATH_OUT = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests-OUT\\"


# [optional] path pointing pyCGM2-Nexus tools
NEXUS_PYCGM2_TOOLS_PATH = MAIN_PYCGM2_PATH + "pyCGM2\\Nexus\\"

# [optional] moveck path
MOVECKPATH = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\moveck\\"
sys.path.append(MOVECKPATH+"Moveck_pipe-2024.1.0-win64-pipeline_install\\packages")

#----- CONSTANTES -----

MUSCLES_LABELS = ['glut_med1_r', 'glut_med2_r', 'glut_med3_r', 'bifemlh_r', 'bifemsh_r', 'sar_r', 'add_mag2_r', 'tfl_r', 'pect_r', 'grac_r', 'glut_max1_r', 'glut_max2_r', 'glut_max3_r', 'iliacus_r', 'psoas_r', 'quad_fem_r', 'gem_r', 'peri_r', 'rect_fem_r', 'vas_int_r', 'med_gas_r', 'soleus_r', 'tib_post_r', 'tib_ant_r', 'glut_med1_l', 'glut_med2_l', 'glut_med3_l', 'bifemlh_l', 'bifemsh_l', 'sar_l', 'add_mag2_l', 'tfl_l', 'pect_l', 'grac_l', 'glut_max1_l', 'glut_max2_l', 'glut_max3_l', 'iliacus_l', 'psoas_l', 'quad_fem_l', 'gem_l', 'peri_l', 'rect_fem_l', 'vas_int_l', 'med_gas_l', 'soleus_l', 'tib_post_l', 'tib_ant_l', 'ercspn_r', 'ercspn_l', 'intobl_r', 'intobl_l', 'extobl_r', 'extobl_l']


#--- eclipse db
ECLISPE_DB_PATH = "\\\\Bio2012006\\Vicon Data\\Donnees_Nexus\\eclipseDB.db"
ECLIPSE_DB_LOCAL = MAIN_PYCGM2_PATH+"DB\\eclipseDB.db"