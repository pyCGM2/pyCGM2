# -*- coding: utf-8 -*-
from setuptools import setup,find_packages
import os,sys
import string
import logging

#logging.basicConfig(filename = "installer.log", level=logging.DEBUG)

import shutil
import site

developMode = False
if len(sys.argv) >= 2:
    if sys.argv[1] == "develop": developMode = True
if developMode:
    logging.warning("You have sleected a developer model ( local install)")


if sys.maxsize > 2**32:
    raise Exception ("64-bit python version detected. PyCGM2 requires a 32 bits python version")

VERSION ="3.4.1"


for it in site.getsitepackages():
    if "site-packages" in it:
        SITE_PACKAGE_PATH = it +"\\"

NAME_IN_SITEPACKAGE = "pyCGM2-"+VERSION+"-py2.7.egg"


MAIN_PYCGM2_PATH = os.getcwd() + "\\"


PYCGM2_SETTINGS_FOLDER = MAIN_PYCGM2_PATH+"PyCGM2\\Settings\\"
NEXUS_PYCGM2_VST_PATH = MAIN_PYCGM2_PATH + "PyCGM2\\Install\\vst\\"
NEXUS_PIPELINE_TEMPLATE_PATH = MAIN_PYCGM2_PATH + "PyCGM2\\Install\\pipelineTemplate\\"

PATH_TO_PYTHON_SCRIPTS = os.path.dirname(sys.executable)+"\\Scripts\\"

# do not serve anymore since all apps are now in Scripts ( i still keep it)
if not developMode:
    PATH_IN_SITEPACKAGE = SITE_PACKAGE_PATH+NAME_IN_SITEPACKAGE+"\\"
else:
    PATH_IN_SITEPACKAGE = MAIN_PYCGM2_PATH

user_folder =  os.getenv("PUBLIC")
NEXUS_PUBLIC_PATH = user_folder+"\\Documents\\Vicon\\Nexus2.x\\"
NEXUS_PUBLIC_DOCUMENT_VST_PATH = NEXUS_PUBLIC_PATH + "ModelTemplates\\"
NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH = NEXUS_PUBLIC_PATH+"Configurations\\Pipelines\\"

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = parse_requirements("requirements.txt")

def scanViconTemplatePipeline(sourcePath,desPath,pyCGM2nexusAppsPath):

    toreplace= "[TOREPLACE]"

    sourcePath = sourcePath[:-1] if sourcePath[-1:]=="\\" else sourcePath
    desPath = desPath[:-1] if desPath[-1:]=="\\" else desPath
    pyCGM2nexusAppsPath = pyCGM2nexusAppsPath[:-1] if pyCGM2nexusAppsPath[-1:]=="\\" else pyCGM2nexusAppsPath

    pyCGM2nexusAppsPath_antislash = string.replace(pyCGM2nexusAppsPath, '\\', '/')

    for file in os.listdir(sourcePath):
        with open(sourcePath+"\\"+file, 'r') as f:
            file_contents = f.read()

        content = string.replace(file_contents, toreplace,pyCGM2nexusAppsPath_antislash)


        if not os.path.isfile( desPath +"\\"+ file):
            with open(desPath + "\\"+file, "w") as text_file:
                text_file.write(content)


def gen_data_files(*dirs):
    results = []
    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            results.append((root, map(lambda f:root + "/" + f, files)))
    return results

def gen_data_files_forScripts(*dirs):
    results = []
    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            for file in files:
                if file[-3:] ==".py":
                    results.append(os.path.join(root, file))
    return results




def getSubDirectories(dir):
    subdirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    return subdirs

def getFiles(dir):
    return  [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

#------------------------- UNINSTALL--------------------------------------------

# remove pyCGM2 folder or egg-link
dirSitepackage = getSubDirectories(SITE_PACKAGE_PATH[:-1])
for folder in  dirSitepackage:
    if "pyCGM2" in folder:
        shutil.rmtree(SITE_PACKAGE_PATH+folder)
        logging.info("package pyCGM2 (%s) removed"%(folder))



if "pyCGM2.egg-link" in os.listdir(SITE_PACKAGE_PATH[:-1]):
    os.remove(SITE_PACKAGE_PATH+"pyCGM2.egg-link")

# remove Build/dist/egg info in the downloaded folder
localDirPath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
localDirPathDirs = getSubDirectories(localDirPath)
if "build" in  localDirPathDirs:    shutil.rmtree(localDirPath+"\\build")
if "dist" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\dist")
if "pyCGM2.egg-info" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\pyCGM2.egg-info")


# delete everything in programData
if os.getenv("PROGRAMDATA") is not None:
    pd = os.getenv("PROGRAMDATA")
    pddirs = getSubDirectories(pd)
    if "pyCGM2" in  pddirs:
        shutil.rmtree(pd+"\\pyCGM2")
        logging.info("pprogramData/pyCGM2---> remove")

if os.path.isdir(NEXUS_PUBLIC_PATH):
    # delete all previous vst and pipelines in Nexus Public Documents
    files = getFiles(NEXUS_PUBLIC_DOCUMENT_VST_PATH)
    for file in files:
        if "pyCGM2" in file[0:6]: # check 6 first letters
            os.remove(os.path.join(NEXUS_PUBLIC_DOCUMENT_VST_PATH,file))

    files = getFiles(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH)
    for file in files:
        if "pyCGM2" in file[0:6]:
            os.remove(os.path.join(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH,file))


# dirs = getSubDirectories(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH)
# if "pyCGM2" in dirs:
#     shutil.rmtree(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH+"pyCGM2")
#------------------------------------------------------------------


#------------------------- PRE INSTALL---------------------------------------

#--- management of the folder ProgramData/pyCGM2----
if not developMode:
    if os.getenv("PROGRAMDATA"):
        PYCGM2_APPDATA_PATH = os.getenv("PROGRAMDATA")+"\\pyCGM2"
        shutil.copytree(PYCGM2_SETTINGS_FOLDER[:-1], PYCGM2_APPDATA_PATH)

#--- management of nexus-related files ( vst+pipelines)-----
if os.path.isdir(NEXUS_PUBLIC_PATH):
    # vst
    content = os.listdir(NEXUS_PYCGM2_VST_PATH[:-1])
    for item in content:
        full_filename = os.path.join(NEXUS_PYCGM2_VST_PATH, item)
        shutil.copyfile(full_filename,  os.path.join(NEXUS_PUBLIC_DOCUMENT_VST_PATH,item))


    scanViconTemplatePipeline(NEXUS_PIPELINE_TEMPLATE_PATH,
                                                NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH,
                                                PATH_TO_PYTHON_SCRIPTS)

else:
    logging.error("[pyCGM2] - Nexus folder not detected - No generation of VST and pipelines")

#------------------------- INSTALL--------------------------------------------
# numpy 1.11 scipy==1.2.1'matplotlib<3.0.0' pandas ==0.19.1'enum34>=1.1.2 configparser>=3.5.0' beautifulsoup4>=3.5.0' 'pyyaml>=3.13.0 yamlordereddictloader>=0.4.0 'xlrd >=0.9.0
setup(name = 'pyCGM2',
    version = VERSION,
    author = 'Fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    description = "Conventional Gait models and Gait analysis",
    long_description= "A python implementation of conventional gait models and methods for processing motion capture data",
    url = 'https://pycgm2.github.io',
    keywords = 'python CGM Vicon PluginGait',
    packages=find_packages(),
	include_package_data=True,
    license='CC-BY-SA',
	install_requires = reqs,
    #'qtmWebGaitReport>=0.0.1'],
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Operating System :: Microsoft :: Windows',
                 'Natural Language :: English'],
    #scripts=gen_data_files_forScripts("Apps/ViconApps")
    entry_points={
          'console_scripts': [
                # NEXUS
                'Nexus_CGM1_Calibration  =  pyCGM2.Apps.ViconApps.CGM1.CGM1_Calibration:main',
                'Nexus_CGM1_Fitting      =  pyCGM2.Apps.ViconApps.CGM1.CGM1_Fitting:main',
                'Nexus_CGM11_Calibration =  pyCGM2.Apps.ViconApps.CGM1_1.CGM1_1_Calibration:main',
                'Nexus_CGM11_Fitting     =  pyCGM2.Apps.ViconApps.CGM1_1.CGM1_1_Fitting:main',
                'Nexus_CGM21_Calibration =  pyCGM2.Apps.ViconApps.CGM2_1.CGM2_1_Calibration:main',
                'Nexus_CGM21_Fitting     =  pyCGM2.Apps.ViconApps.CGM2_1.CGM2_1_Fitting:main',
                'Nexus_CGM22_Calibration =  pyCGM2.Apps.ViconApps.CGM2_2.CGM2_2_Calibration:main',
                'Nexus_CGM22_Fitting     =  pyCGM2.Apps.ViconApps.CGM2_2.CGM2_2_Fitting:main',
                'Nexus_CGM23_Calibration =  pyCGM2.Apps.ViconApps.CGM2_3.CGM2_3_Calibration:main',
                'Nexus_CGM23_Fitting     =  pyCGM2.Apps.ViconApps.CGM2_3.CGM2_3_Fitting:main',
                'Nexus_CGM24_Calibration =  pyCGM2.Apps.ViconApps.CGM2_4.CGM2_4_Calibration:main',
                'Nexus_CGM24_Fitting     =  pyCGM2.Apps.ViconApps.CGM2_4.CGM2_4_Fitting:main',
                'Nexus_CGM25_Calibration =  pyCGM2.Apps.ViconApps.CGM2_5.CGM2_5_Calibration:main',
                'Nexus_CGM25_Fitting     =  pyCGM2.Apps.ViconApps.CGM2_5.CGM2_5_Fitting:main',
                'Nexus_CGM26_2DOF =  pyCGM2.Apps.ViconApps.CGM2_6.CGM_Knee2DofCalibration:main',
                'Nexus_CGM26_SARA     =  pyCGM2.Apps.ViconApps.CGM2_6.CGM_Knee2DofCalibration:main',

                'Nexus_plotMAP                      =  pyCGM2.Apps.ViconApps.DataProcessing.plotMAP:main',
                'Nexus_plotNormalizedKinematics     =  pyCGM2.Apps.ViconApps.DataProcessing.plotNormalizedKinematics:main',
                'Nexus_plotNormalizedKinetics       =  pyCGM2.Apps.ViconApps.DataProcessing.plotNormalizedKinetics:main',
                'Nexus_plotSpatioTemporalParameters =  pyCGM2.Apps.ViconApps.DataProcessing.plotSpatioTemporalParameters:main',
                'Nexus_plotTemporalKinematics       =  pyCGM2.Apps.ViconApps.DataProcessing.plotTemporalKinematics:main',
                'Nexus_plotTemporalKinetics         =  pyCGM2.Apps.ViconApps.DataProcessing.plotTemporalKinetics:main',

                'Nexus_plotNormalizedEmg = pyCGM2.Apps.ViconApps.EMG.plotNormalizedEmg:main',
                'Nexus_plotTemporalEmg   = pyCGM2.Apps.ViconApps.EMG.plotTemporalEmg:main',

                'Nexus_zeniDetector     =  pyCGM2.Apps.ViconApps.Events.zeniDetector:main',
                'Nexus_KalmanGapFilling =  pyCGM2.Apps.ViconApps.MoGapFill.KalmanGapFilling:main',

                'Nexus_resetProgramData =  pyCGM2.Apps.ViconApps.Miscellaneous.pyCGM2_resetProgramData:main',
                'Nexus_check_inputArgs  =  pyCGM2.Apps.ViconApps.Miscellaneous.check_inputArgs:main',

                # QTM
                'QTM_CGM1_workflow  =  pyCGM2.Apps.QtmApps.CGMi.CGM1_workflow:command',
                'QTM_CGM11_workflow  =  pyCGM2.Apps.QtmApps.CGMi.CGM11_workflow:command',
                'QTM_CGM21_workflow  =  pyCGM2.Apps.QtmApps.CGMi.CGM21_workflow:command',
                'QTM_CGM22_workflow  =  pyCGM2.Apps.QtmApps.CGMi.CGM22_workflow:command',
                'QTM_CGM23_workflow  =  pyCGM2.Apps.QtmApps.CGMi.CGM23_workflow:command',
                'QTM_CGM24_workflow  =  pyCGM2.Apps.QtmApps.CGMi.CGM24_workflow:command',

          ]
      },
    )
