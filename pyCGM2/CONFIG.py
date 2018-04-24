# -*- coding: utf-8 -*-
import logging
import sys
import os
import generatePipeline
import json
import shutil

# ------------------- CONSTANTS ------------------------------------------------

# [REQUIRED - if Vicon USer] configure if you want to run processes from Nexus
#NEXUS_SDK_WIN32 = 'C:/Program Files (x86)/Vicon/Nexus2.6/SDK/Win32'
#NEXUS_SDK_PYTHON = 'C:/Program Files (x86)/Vicon/Nexus2.6/SDK/Python'
#PYTHON_NEXUS = 'C:\\Program Files (x86)\\Vicon\\Nexus2.6\\Python'


APPDATA_FOLDER = os.getenv("PROGRAMDATA")
PYCGM2_APPDATA_PATH = APPDATA_FOLDER+"\\pyCGM2\\"

if not os.path.exists(PYCGM2_APPDATA_PATH[:-1]):
    os.makedirs(PYCGM2_APPDATA_PATH[:-1])

# convenient if you want to regenerate it without running the initial setup
dirname = APPDATA_FOLDER+"\\pyCGM2\\viconPipelines"
if not os.path.exists(dirname):
    os.makedirs(dirname)

dirname = APPDATA_FOLDER+"\\pyCGM2\\translators"
if not os.path.exists(dirname):
    os.makedirs(dirname)

dirname = APPDATA_FOLDER+"\\pyCGM2\\IkWeightSets"
if not os.path.exists(dirname):
    os.makedirs(dirname)

# [OPTIONAL] ----------------------------------
MAIN_PYCGM2_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "\\" #C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\" # path toward your pyCGM2 folder ( dont forget \\ ending)


# [Optional]: Apps path
MAIN_PYCGM2_APPS_PATH = MAIN_PYCGM2_PATH+"Apps\\"


# [Optional]: openMA binding
THIRDPARTY_PATH = MAIN_PYCGM2_PATH + "thirdParty\\" # By default, use openMA distribution included in third party folder

# [Optional] path to embbbed Normative data base.
NORMATIVE_DATABASE_PATH = MAIN_PYCGM2_PATH +"Data\\normativeData\\"  # By default, use pyCGM2-embedded normative data ( Schartz - Pinzone )

# [Optional] main folder containing osim model
OPENSIM_PREBUILD_MODEL_PATH = MAIN_PYCGM2_PATH + "Extern\\opensim\\"

# [Optional] path pointing at Data Folders used for Tests
TEST_DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\"
MAIN_BENCHMARK_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-benchmarks\\Gait patterns\\"

# [optional] path pointing pyCGM2-Nexus tools
NEXUS_PYCGM2_TOOLS_PATH = MAIN_PYCGM2_PATH + "pyCGM2\\Nexus\\"

# [optional]  setting folder
PYCGM2_SESSION_SETTINGS_FOLDER = MAIN_PYCGM2_PATH+"SessionSettings\\"


# setting files

cgmFileGlobalSettings = "CGM1-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM1_1-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM2_1-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM2_2-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM2_3-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM2_4-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM2_2-Expert-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM2_3-Expert-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)

cgmFileGlobalSettings = "CGM2_4-Expert-pyCGM2.settings"
if not os.path.isfile( PYCGM2_APPDATA_PATH + cgmFileGlobalSettings):
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+cgmFileGlobalSettings, PYCGM2_APPDATA_PATH + cgmFileGlobalSettings)



# pipeline generation
# cgm1
generatePipeline.pipeline_pyCGM2_CGM1_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM1_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# cgm1.1
generatePipeline.pipeline_pyCGM2_CGM1_1_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM1_1_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# cgm2.1
generatePipeline.pipeline_pyCGM2_CGM2_1_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM2_1_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# cgm2.2
generatePipeline.pipeline_pyCGM2_CGM2_2_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM2_2_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# cgm2.2-Expert
generatePipeline.pipeline_pyCGM2_CGM2_2_Expert_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM2_2_Expert_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# cgm2.3
generatePipeline.pipeline_pyCGM2_CGM2_3_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM2_3_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# cgm2.3-Expert
generatePipeline.pipeline_pyCGM2_CGM2_3_Expert_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM2_3_Expert_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")

# cgm2.3
generatePipeline.pipeline_pyCGM2_CGM2_4_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM2_4_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# cgm2.4-Expert
generatePipeline.pipeline_pyCGM2_CGM2_4_Expert_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
generatePipeline.pipeline_pyCGM2_CGM2_4_Expert_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")

# Sara
generatePipeline.pipeline_pyCGM2_SARA_kneeCalibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")
# 2DOF
generatePipeline.pipeline_pyCGM2_2dof_kneeCalibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")

# Event detector
generatePipeline.pipeline_pyCGM2_eventDetector(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")

# MoGapFill
generatePipeline.pipeline_pyCGM2_mogapfill(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH+"viconPipelines\\")



# translators
src_files = os.listdir(PYCGM2_SESSION_SETTINGS_FOLDER+"translators")
for file_name in src_files:
    full_file_name = os.path.join(PYCGM2_SESSION_SETTINGS_FOLDER+"translators", file_name)
    if not (os.path.isfile( PYCGM2_APPDATA_PATH +"translators\\"+file_name)):
        shutil.copy(full_file_name, PYCGM2_APPDATA_PATH +"translators\\"+file_name)

# IkWeightSets
src_files = os.listdir(PYCGM2_SESSION_SETTINGS_FOLDER+"IkWeightSets")
for file_name in src_files:
    full_file_name = os.path.join(PYCGM2_SESSION_SETTINGS_FOLDER+"IkWeightSets", file_name)
    if not (os.path.isfile(PYCGM2_APPDATA_PATH +"IkWeightSets\\"+file_name)):
        shutil.copy(full_file_name, PYCGM2_APPDATA_PATH +"IkWeightSets\\"+file_name)


# ------------------- METHODS ------------------------------------------------


def addNexusPythonSdk():

    nexusPaths = json.loads(open(str(PYCGM2_APPDATA_PATH+"nexusPaths")).read())

    if nexusPaths["NEXUS_SDK_WIN32"]  not in sys.path:
        sys.path.append( nexusPaths["NEXUS_SDK_WIN32"])
        #print NEXUS_SDK_WIN32 + " added to the python path"
    if nexusPaths["NEXUS_SDK_PYTHON"]   not in sys.path:
        sys.path.append( nexusPaths["NEXUS_SDK_PYTHON"])
        #print NEXUS_SDK_WIN32 + " added to the python path"

    NEXUS_PYTHON_USE = True if nexusPaths["PYTHON_NEXUS"]  in sys.path else False
    if NEXUS_PYTHON_USE:
        raise Exception("untick Use nexus Python in your python pipeline operation. pyCCGM2 apps recommand anaconda Packages ")


def addOpenma(branch=None):

    if branch is None:
        sys.path.append(THIRDPARTY_PATH + "openma")
        sys.path.append(THIRDPARTY_PATH + "openma\\ma")
    else:
        if branch=="master":
            sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\distribuable\\OpenMA\\openma")
            sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\distribuable\\OpenMA\\openma\\ma")

            # method 1 : with path definition
            # need definition in the PATH, I appended these two folders:
            # C:\Users\HLS501\Documents\Programming\openMA\Build\master\bin;
            # C:\Users\HLS501\Documents\Programming\openMA\Build\master\bin\swig\python\openma\ma;
            #sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\Build\\master\\bin\\swig\\python\\openma")
            #sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\Build\\master\\bin\\swig\\python\\openma\\ma")

            # method 2 : with openMA distribution made manually
            sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\distribuable\\OpenMA\\openma")
            sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\distribuable\\OpenMA\\openma\\ma")

        elif branch=="plugin-gait-kad":

            # method 2 : with openMA distribution made manually
            sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\distribuable\\OpenMA-KAD\\openma")
            sys.path.append("C:\\Users\\HLS501\\Documents\\Programming\\openMA\\distribuable\\OpenMA-KAD\\openma\\ma")


def addBtk():
    sys.path.append(THIRDPARTY_PATH + "btk")

def addOpensim3():
    sys.path.append(THIRDPARTY_PATH + "opensim3")
