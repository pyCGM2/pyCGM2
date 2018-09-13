# -*- coding: utf-8 -*-
from setuptools import setup,find_packages
import os,sys
import string
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers
#logging.basicConfig(filename = "installer.log", level=logging.DEBUG)

import shutil
import site


if "64-bit" in sys.version:
    raise Exception ("64-bit python version detected. PyCGM2 requires a 32 bits python version")

VERSION ="3.0.0"



for it in site.getsitepackages():
    if "site-packages" in it:
        SITE_PACKAGE_PATH = it +"\\"

NAME_IN_SITEPACKAGE = "pyCGM2-"+VERSION+"-py2.7.egg"
PATH_IN_SITEPACKAGE = SITE_PACKAGE_PATH+NAME_IN_SITEPACKAGE+"\\"

if "develop" in sys.argv:
    MAIN_PYCGM2_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))+"\\pyCGM2\\"
else:
    MAIN_PYCGM2_PATH = PATH_IN_SITEPACKAGE

PYCGM2_SESSION_SETTINGS_FOLDER = MAIN_PYCGM2_PATH+"SessionSettings\\"
NEXUS_PYCGM2_VST_PATH = MAIN_PYCGM2_PATH + "Extern\\vicon\\vst\\non-official\\"
NEXUS_PIPELINE_TEMPLATE_PATH = MAIN_PYCGM2_PATH + "installData\\pipelineTemplate\\"

NEXUS_PUBLIC_DOCUMENT_VST_PATH = "C:/Users/Public/Documents/Vicon/Nexus2.x/ModelTemplates/"
NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH = "C:/Users/Public/Documents/Vicon/Nexus2.x/Configurations/Pipelines/"



def scanViconTemplatePipeline(sourcePath,desPath,pyCGM2nexusAppsPath):

    toreplace= "C:/Users/HLS501/Documents/Programming/API/pyCGM2/pyCGM2/Apps"

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

def getSubDirectories(dir):
    subdirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    return subdirs


#------------------------- UNINSTALL--------------------------------------------

# remove pyCGM2 folder or egg-link
dirSitepackage = getSubDirectories(SITE_PACKAGE_PATH[:-1])
if NAME_IN_SITEPACKAGE in dirSitepackage:
    shutil.rmtree(PATH_IN_SITEPACKAGE[:-1])

if "pyCGM2.egg-link" in os.listdir(SITE_PACKAGE_PATH[:-1]):
    os.remove(SITE_PACKAGE_PATH+"pyCGM2.egg-link")

# remove Build/dist/egg info in the downloaded folder
localDirPath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
localDirPathDirs = getSubDirectories(localDirPath+"\\pyCGM2")
if "build" in  localDirPathDirs:    shutil.rmtree(localDirPath+"\\pyCGM2\\build")
if "dist" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\pyCGM2\\dist")
if "pyCGM2.egg-info" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\pyCGM2\\pyCGM2.egg-info")


# delete everything in programData
pd = os.getenv("PROGRAMDATA")
pddirs = getSubDirectories(pd)
if "pyCGM2" in  pddirs:
    shutil.rmtree(pd+"\\pyCGM2")
    logging.info("pprogramData/pyCGM2---> remove")

# delete all previous vst and pipelines in Nexus Public Documents
dirs = getSubDirectories(NEXUS_PUBLIC_DOCUMENT_VST_PATH)
if "pyCGM2" in dirs:
    shutil.rmtree(NEXUS_PUBLIC_DOCUMENT_VST_PATH+"pyCGM2")

dirs = getSubDirectories(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH)
if "pyCGM2" in dirs:
    shutil.rmtree(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH+"pyCGM2")
#------------------------------------------------------------------


#------------------------- INSTALL--------------------------------------------
setup(name = 'pyCGM2',
    version = VERSION,
    author = 'fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    keywords = 'python CGM Vicon PluginGait',
    packages=find_packages(),
    data_files = gen_data_files("Apps","Data","Extern","SessionSettings","installData"),
	include_package_data=True,
    license='CC-BY-SA',
	install_requires = ['numpy>=1.11.0',
                        'scipy>=0.17.0',
                        'matplotlib>=1.5.3',
                        'pandas >=0.19.1',
                        'enum34>=1.1.2',
                        'configparser>=3.5.0',
                        'beautifulsoup4>=3.5.0'],
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Operating System :: Windows OS',
                 'Natural Language :: English-French',
                 'Topic :: Clinical Gait Analysis']
    )
#------------------------------------------------------------------------------

#------------------------- POST INSTALL---------------------------------------


#--- management of the folder ProgramData/pyCGM2----
PYCGM2_APPDATA_PATH = os.getenv("PROGRAMDATA")+"\\pyCGM2\\"
os.makedirs(PYCGM2_APPDATA_PATH[:-1])

# CGM i settings
settings = ["CGM1-pyCGM2.settings", "CGM1_1-pyCGM2.settings","CGM2_1-pyCGM2.settings", "CGM2_2-pyCGM2.settings","CGM2_3-pyCGM2.settings","CGM2_4-pyCGM2.settings"]
for setting in settings:
    shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+"GlobalSettings\\"+setting, PYCGM2_APPDATA_PATH + setting)

# translators
src_files = os.listdir(PYCGM2_SESSION_SETTINGS_FOLDER+"translators")
os.makedirs(PYCGM2_APPDATA_PATH +"translators")
for filename in src_files:
    full_filename = os.path.join(PYCGM2_SESSION_SETTINGS_FOLDER+"translators", filename)
    shutil.copyfile(full_filename, PYCGM2_APPDATA_PATH +"translators\\"+filename)

# IkWeightSets
src_files = os.listdir(PYCGM2_SESSION_SETTINGS_FOLDER+"IkWeightSets")
os.makedirs(PYCGM2_APPDATA_PATH +"IkWeightSets")
for filename in src_files:
    full_filename = os.path.join(PYCGM2_SESSION_SETTINGS_FOLDER+"IkWeightSets", filename)
    shutil.copyfile(full_filename, PYCGM2_APPDATA_PATH +"IkWeightSets\\"+filename)


#--- management of nexus-related files ( vst+pipelines)-----



# vst
os.makedirs(NEXUS_PUBLIC_DOCUMENT_VST_PATH+"pyCGM2")
src_files = os.listdir(NEXUS_PYCGM2_VST_PATH[:-1])
for filename in src_files:
    full_filename = os.path.join(NEXUS_PYCGM2_VST_PATH, filename)
    shutil.copyfile(full_filename, NEXUS_PUBLIC_DOCUMENT_VST_PATH +"pyCGM2\\"+filename)




os.makedirs(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH+"pyCGM2")
scanViconTemplatePipeline(NEXUS_PIPELINE_TEMPLATE_PATH,
                                            NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH+"pyCGM2",
                                            MAIN_PYCGM2_PATH)
