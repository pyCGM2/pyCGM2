from setuptools import setup,find_packages
import os,sys
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers
#logging.basicConfig(filename = "installer.log", level=logging.DEBUG)

import registry
import shutil
from setuptools import Command
import json

def getSubDirectories(dir):
    subdirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    return subdirs

def gen_data_files(*dirs):
    results = []
    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            results.append((root, map(lambda f:root + "/" + f, files)))
    return results


class uninstallCommand(Command):
    """ Run my command.
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        # remove previous pyCGM2 located in site-package
        logging.info("******* Remove previous pyCGM2 in site-package ******")
        dir =os.path.dirname(os.__file__) + '/site-packages'
        dirs = getSubDirectories(dir)
        dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        for it in dirs:
            if "pyCGM2" in it:
                shutil.rmtree(dir+"/"+it)
                logging.warning("sitePackage/pyCGM2---> remove")

        # remove pyCGM2 link in site-package if previous setup done in develop mode
        if "pyCGM2.egg-link" in os.listdir(dir):
            os.remove(dir+"\\pyCGM2.egg-link")
            logging.warning("sitePackage/pyCGM2 develop link---> remove")

        # remove aumatically installed folder in your local pyCGM2 folder
        localDirPath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
        localDirPathDirs = getSubDirectories(localDirPath)
        if "Build" in  localDirPathDirs:    shutil.rmtree(localDirPath+"\\Build")
        if "Dist" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\Dist")
        if "pyCGM2.egg-info" in  localDirPathDirs:     shutil.rmtree(localDirPath+"\\pyCGM2.egg-info")

        # remove pycgm2 folder in programData
        logging.info("******* Remove pyCGM2 in programData ******")
        pd = os.getenv("PROGRAMDATA")
        pddirs = getSubDirectories(pd)
        if "pyCGM2" in  pddirs:
            shutil.rmtree(pd+"\\pyCGM2")
            logging.info("pprogramData/pyCGM2---> remove")

class checkCommand(Command):
    """ Run my command.
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # issue Logging.info doesnt display any message. print instead

        logging.info( " *******Check your pyCGM2 installation******")
        try:
            import pyCGM2
            print ("pyCGM2 ---> OK")
        except ImportError:
            raise Exception ("[pyCGM2] : pyCGM2 module not imported")

        # vicon nexus
        try:
            import ViconNexus
            print("vicon API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : viconNexus is not in your python path. Check and edit Nexus paths in file CONFIG.py")

        # openMA
        try:
            import ma.io
            import ma.body
            print("openMA API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : openma not imported")

        # btk
        try:
            import btk
            print("btk API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : btk not imported")

        # opensim
        try:
            import opensim
            print("opensim API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : Opensim API not imported. Can t run CGM version superior to Cgm2.1")

if "install" in sys.argv or "develop" in sys.argv:
    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    format = logging.Formatter("%(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)
    fh = handlers.RotatingFileHandler("pyCGM2-installer.log", maxBytes=(1048576*5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)

    # programData Folder
    pd = os.getenv("PROGRAMDATA")
    dirname = pd+"\\pyCGM2"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    dirname = pd+"\\pyCGM2\\viconPipelines"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    dirname = pd+"\\pyCGM2\\translators"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    dirname = pd+"\\pyCGM2\\IkWeightSets"
    if not os.path.exists(dirname):
        os.makedirs(dirname)


    # thirdparty questions
    logging.info( " *******Third party package installation******")

    btkFlag = raw_input("Do you want to use your own Btk package (y/N) ")
    PYCGM2_BTK = True if  btkFlag in ["y", "Y"] else False
    if PYCGM2_BTK:
        logging.info("Btk installed from pyCGM2")
    else:
        logging.info("Custom Btk package")

    openmaFlag = raw_input("Do you want to use your own openMA package (y/N) ")
    PYCGM2_OPENMA = True if   openmaFlag in ["y", "Y"] else False
    if PYCGM2_OPENMA:
        logging.info("OpenMA installed from pyCGM2")
    else:
        logging.info("Custom OpenMA package")

    opensimFlag = raw_input("Do you want to use your own opensim package (y/N) ")
    PYCGM2_OPENSIM = True if   opensimFlag in ["y", "Y"] else False
    if PYCGM2_OPENSIM:
        logging.info("OpenSim installed from pyCGM2")
    else:
        logging.info("Custom OpenSim package")



if "install" in sys.argv or "develop" in sys.argv:

    # check python.exe command
    runPythonExe = sys.executable
    if "pythonw" in runPythonExe:
        runPythonExe =runPythonExe.replace("pythonw", "python")
    _PYTHONEXE = runPythonExe
    _COMPATIBLENEXUSKEY = "\""+ _PYTHONEXE+"\"  \"%1\" %*" # HERE IS the COMPATIBLE NEXUS python executable command


    reg_key = registry.getPythonExeRegisterKey()
    logging.info("******* Alteration of your python registry key fo Nexus ******")
    if reg_key != _COMPATIBLENEXUSKEY:
        logging.warning( "Python registry key ---> incompatible")
        registry.setPythonExeRegisterKey(_COMPATIBLENEXUSKEY)
        logging.warning( "Python registry key ---> altered")
    else:
        logging.info("Python registry key ---> compatible")

    # check if 64 bits
    logging.info( " ******* Detection of your python version ******")
    if "64-bit" in sys.version:
        raise Exception ("64-bit python version detected. PyCGM2 requires a 32 bits python version")
    else:
        logging.info ("python core ----->2.7 (OK)")

    # write nexus path in programData
    nexusPaths_content = """
        {
            "NEXUS_SDK_WIN32": "C:/Program Files (x86)/Vicon/Nexus2.6/SDK/Win32",
            "NEXUS_SDK_PYTHON" : "C:/Program Files (x86)/Vicon/Nexus2.6/SDK/Python",
            "PYTHON_NEXUS" : "C:/Program Files (x86)/Vicon/Nexus2.6/Python"
        }
        """
    nexusPaths = json.loads(nexusPaths_content)

    with open(str(pd+"\\pyCGM2\\nexusPaths"), 'w') as npf:
        json.dump(nexusPaths, npf,indent=4, separators=(',', ': '))
    logging.info( "nexusPaths file in programData ---> Edited")

    # write thirdParty
    thirdParty={"Btk": PYCGM2_BTK,
                "OpenMA": PYCGM2_OPENMA,
                "OpenSim": PYCGM2_OPENSIM}

    with open(str(pd+"\\pyCGM2\\thirdPartyLibraries"), 'w') as tpf:
        json.dump(thirdParty, tpf, indent=4, separators=(',', ': '))
    logging.info( "thirdPartyLibraries file in programData ---> Edited")

setup(name = 'pyCGM2',
    version = '1.1.0',
    author = 'fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    keywords = 'python Conventional Gait Model 2',
    packages=find_packages(),
    data_files = gen_data_files("Apps","Data","Extern","NoViconApps","SessionSettings","thirdParty"),
	include_package_data=True,
	install_requires = ['numpy>=1.11.0',
                        'scipy>=0.17.0',
                        'matplotlib>=1.5.3',
                        'pandas >=0.19.1',
                        'enum34>=1.1.2',
                        'configparser>=3.5.0',
                        'beautifulsoup4>=3.5.0'],
    cmdclass={'check': checkCommand,
            'uninstall': uninstallCommand,    },
    )
