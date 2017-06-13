# -*- coding: utf-8 -*-
import logging
import sys
import os
import generatePipeline
import generateFiles   

# ------------------- CONSTANTS ------------------------------------------------

# [REQUIRED - if Vicon USer] configure if you want to run processes from Nexus   
NEXUS_SDK_WIN32 = 'C:/Program Files (x86)/Vicon/Nexus2.5/SDK/Win32'
NEXUS_SDK_PYTHON = 'C:/Program Files (x86)/Vicon/Nexus2.5/SDK/Python'
PYTHON_NEXUS = 'C:\\Program Files (x86)\\Vicon\\Nexus2.5\\Python'


APPDATA_FOLDER = os.getenv("PROGRAMDATA")
PYCGM2_APPDATA_PATH = APPDATA_FOLDER+"\\pyCGM2\\" 

if not os.path.exists(PYCGM2_APPDATA_PATH[:-1]):
    os.makedirs(PYCGM2_APPDATA_PATH[:-1])


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
generateSettings.generateCGM1_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM1_1_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM2_1_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM2_2_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM2_2_Expert_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM2_3_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM2_3_Expert_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM2_4_Settings(PYCGM2_APPDATA_PATH)
generateSettings.generateCGM2_4_Expert_Settings(PYCGM2_APPDATA_PATH)

# pipeline generation
# cgm1
generatePipeline.pipeline_pyCGM2_CGM1_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM1_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm1.1
generatePipeline.pipeline_pyCGM2_CGM1_1_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM1_1_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm2.1
generatePipeline.pipeline_pyCGM2_CGM2_1_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM2_1_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm2.2
generatePipeline.pipeline_pyCGM2_CGM2_2_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM2_2_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm2.2-Expert
generatePipeline.pipeline_pyCGM2_CGM2_2_Expert_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM2_2_Expert_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm2.3
generatePipeline.pipeline_pyCGM2_CGM2_3_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM2_3_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm2.3-Expert
generatePipeline.pipeline_pyCGM2_CGM2_3_Expert_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM2_3_Expert_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm2.3-Sara
generatePipeline.pipeline_pyCGM2_CGM2_3_SARA_kneeCalibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH)
# cgm2.3
generatePipeline.pipeline_pyCGM2_CGM2_4_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM2_4_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
# cgm2.4-Expert
generatePipeline.pipeline_pyCGM2_CGM2_4_Expert_Calibration(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 
generatePipeline.pipeline_pyCGM2_CGM2_4_Expert_Fitting(MAIN_PYCGM2_APPS_PATH,PYCGM2_APPDATA_PATH) 

# ------------------- METHODS ------------------------------------------------

def setLoggingLevel(level):
    logging.basicConfig(format = "[pyCGM2-%(levelname)s]-%(module)s-%(funcName)s : %(message)s",level = level) 

    
def addNexusPythonSdk():
    
    if NEXUS_SDK_WIN32  not in sys.path:
        sys.path.append( NEXUS_SDK_WIN32)
        print NEXUS_SDK_WIN32 + " added to the python path"
    if NEXUS_SDK_PYTHON  not in sys.path:
        sys.path.append( NEXUS_SDK_PYTHON)
        print NEXUS_SDK_WIN32 + " added to the python path"

    NEXUS_PYTHON_USE = True if PYTHON_NEXUS in sys.path else False
    if NEXUS_PYTHON_USE:
        raise Exception("untick Use nexus Python in your python pipeline operation. pyCGA apps recommand anaconda Packages ")
        

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


def checkConfig():

    for it in sys.path: 
        if "pyCGM2" in it: 
            print "OK"    
    try:
        import pyCGM2
        import pyCGM2.CONFIG
        pyCGM2.CONFIG.setLoggingLevel(logging.DEBUG)
        logging.info("pyCGM2 ---> OK")
    except ImportError:
        raise Exception ("[pyCGM2] : pyCGM2 module not in your python path")
    
    
    # vicon nexus
    pyCGM2.CONFIG.addNexusPythonSdk()
    try:
        import ViconNexus
        logging.info("vicon API ---> OK" )
    except ImportError:
        logging.error ("[pyCGM2] : viconNexus is not in your python path. Check CONFIG")
    
    
    # openMA
    pyCGM2.CONFIG.addOpenma()
    try:
        import ma.io
        import ma.body
        logging.info("openMA API ---> OK" )        
    except ImportError:
        logging.error ("[pyCGM2] : openma is not in your python path. Check CONFIG")
        
        
    # btk
    pyCGM2.CONFIG.addBtk()
    try:
        import btk
        logging.info("btk API ---> OK" )        
    except ImportError:
        logging.error ("[pyCGM2] : btk is not in your python path. Check CONFIG")  
   