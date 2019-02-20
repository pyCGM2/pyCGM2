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

developMode = False
if len(sys.argv) >= 2:
    if sys.argv[1] == "develop": developMode = True
if developMode:
    logging.warning("You have sleected a developer model ( local install)")


if sys.maxsize > 2**32:
    raise Exception ("64-bit python version detected. PyCGM2 requires a 32 bits python version")

VERSION ="3.1.0"


for it in site.getsitepackages():
    if "site-packages" in it:
        SITE_PACKAGE_PATH = it +"\\"

NAME_IN_SITEPACKAGE = "pyCGM2-"+VERSION+"-py2.7.egg"


MAIN_PYCGM2_PATH = os.getcwd() + "\\"


PYCGM2_SESSION_SETTINGS_FOLDER = MAIN_PYCGM2_PATH+"SessionSettings\\"
NEXUS_PYCGM2_VST_PATH = MAIN_PYCGM2_PATH + "Extern\\vicon\\vst\\"
NEXUS_PIPELINE_TEMPLATE_PATH = MAIN_PYCGM2_PATH + "installData\\pipelineTemplate\\"

if not developMode:
    PATH_IN_SITEPACKAGE = SITE_PACKAGE_PATH+NAME_IN_SITEPACKAGE+"\\"
else:
    PATH_IN_SITEPACKAGE = MAIN_PYCGM2_PATH

user_folder = pd = os.getenv("PUBLIC")
NEXUS_PUBLIC_DOCUMENT_VST_PATH = user_folder+"\\Documents\\Vicon\\Nexus2.x\\ModelTemplates\\"
NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH = user_folder+"\\Documents\\Vicon/Nexus2.x\\Configurations\\Pipelines\\"



def scanViconTemplatePipeline(sourcePath,desPath,pyCGM2nexusAppsPath):

    toreplace= "C:/Users/HLS501/Documents/Programming/API/pyCGM2/pyCGM2"

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
        print src_dir
        for root,dirs,files in os.walk(src_dir):
            for file in files:
                if file[-3:] ==".py":
                    results.append(os.path.join(root, file))
    print results
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


# if NAME_IN_SITEPACKAGE in dirSitepackage:
#     shutil.rmtree(PATH_IN_SITEPACKAGE[:-1])

if "pyCGM2.egg-link" in os.listdir(SITE_PACKAGE_PATH[:-1]):
    os.remove(SITE_PACKAGE_PATH+"pyCGM2.egg-link")

# remove Build/dist/egg info in the downloaded folder
localDirPath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
localDirPathDirs = getSubDirectories(localDirPath)
if "build" in  localDirPathDirs:    shutil.rmtree(localDirPath+"\\build")
if "dist" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\dist")
if "pyCGM2.egg-info" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\pyCGM2.egg-info")


# delete everything in programData
if not developMode:
    pd = os.getenv("PROGRAMDATA")
    pddirs = getSubDirectories(pd)
    if "pyCGM2" in  pddirs:
        shutil.rmtree(pd+"\\pyCGM2")
        logging.info("pprogramData/pyCGM2---> remove")

# delete all previous vst and pipelines in Nexus Public Documents
files = getFiles(NEXUS_PUBLIC_DOCUMENT_VST_PATH)
for file in files:
    if "pyCGM2" in file[0:6]: # check 6 first letters
        os.remove(os.path.join(NEXUS_PUBLIC_DOCUMENT_VST_PATH,file))

files = getFiles(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH)
for file in files:
    if "pyCGM2" in file[0:6]:
        os.remove(os.path.join(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH,file))




#
# dirs = getSubDirectories(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH)
# if "pyCGM2" in dirs:
#     shutil.rmtree(NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH+"pyCGM2")
#------------------------------------------------------------------


#------------------------- INSTALL--------------------------------------------
setup(name = 'pyCGM2',
    version = VERSION,
    author = 'Fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    keywords = 'python CGM Vicon PluginGait',
    packages=find_packages(),
    data_files = gen_data_files("Apps","Data","Extern","SessionSettings","installData"),
	include_package_data=True,
    license='CC-BY-SA',
	install_requires = ['numpy>=1.11.0',
                        'scipy>=0.17.0',
                        'matplotlib<3.0.0',
                        'pandas >=0.19.1',
                        'enum34>=1.1.2',
                        'configparser>=3.5.0',
                        'beautifulsoup4>=3.5.0',
                        'pyyaml>=3.13.0',
                        'yamlordereddictloader>=0.4.0',
                        'xlrd >=0.9.0'],
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Operating System :: Windows OS',
                 'Natural Language :: English-French',
                 'Topic :: Clinical Gait Analysis'],
    scripts=gen_data_files_forScripts("Apps")
    )

#------------------------------------------------------------------------------

#------------------------- POST INSTALL---------------------------------------


#--- management of the folder ProgramData/pyCGM2----
if not developMode:
    PYCGM2_APPDATA_PATH = os.getenv("PROGRAMDATA")+"\\pyCGM2\\"
    os.makedirs(PYCGM2_APPDATA_PATH[:-1])

    # CGM i settings
    settings = ["CGM1-pyCGM2.settings", "CGM1_1-pyCGM2.settings",
                "CGM2_1-pyCGM2.settings", "CGM2_2-pyCGM2.settings",
                "CGM2_3-pyCGM2.settings","CGM2_4-pyCGM2.settings",
                "emg.settings"]
    for setting in settings:
        shutil.copyfile(PYCGM2_SESSION_SETTINGS_FOLDER+setting, PYCGM2_APPDATA_PATH + setting)

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
else:
    open(MAIN_PYCGM2_PATH + 'localMode', 'a').close()

#--- management of nexus-related files ( vst+pipelines)-----



# vst

content = os.listdir(NEXUS_PYCGM2_VST_PATH[:-1])
for item in content:
    full_filename = os.path.join(NEXUS_PYCGM2_VST_PATH, item)
    shutil.copyfile(full_filename,  os.path.join(NEXUS_PUBLIC_DOCUMENT_VST_PATH,item))


scanViconTemplatePipeline(NEXUS_PIPELINE_TEMPLATE_PATH,
                                            NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH,
                                            PATH_IN_SITEPACKAGE)

# if not developMode:
#     if os.getcwd().lower in sys.path or os.getcwd() in sys.path:
#         sys.path.remove (os.getcwd().lower())
#         sys.path.remove (os.getcwd())
#         import ipdb; pdb.set_trace()
#         print "===========================REMOVE============================"
