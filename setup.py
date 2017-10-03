from setuptools import setup,find_packages
import os,sys
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers
#logging.basicConfig(filename = "installer.log", level=logging.DEBUG)

import registry
import shutil
from setuptools import Command

log = logging.getLogger('')
log.setLevel(logging.DEBUG)
format = logging.Formatter("%(levelname)s - %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)
fh = handlers.RotatingFileHandler("pyCGM2-installer.log", maxBytes=(1048576*5), backupCount=7)
fh.setFormatter(format)
log.addHandler(fh)

def getSubDirectories(dir):
    subdirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    return subdirs

def gen_data_files(*dirs):
    results = []
    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            results.append((root, map(lambda f:root + "/" + f, files)))
    return results

class checkCommand(Command):
    """ Run my command.
    """
    user_options = []

    def initialize_options(self):
        pass


    def finalize_options(self):
        pass

    def run(self):

        logging.info( " *******Check your pyCGM2 installation******")
        try:
            import pyCGM2
            logging.info("pyCGM2 ---> OK")
            print pyCGM2.__path__
        except ImportError:
            raise Exception ("[pyCGM2] : pyCGM2 module not imported")

        # vicon nexus
        pyCGM2.CONFIG.addNexusPythonSdk()
        try:
            import ViconNexus
            logging.info("vicon API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : viconNexus is not in your python path. Check and edit Nexus paths in file CONFIG.py")

        # openMA
        pyCGM2.CONFIG.addOpenma()
        try:
            import ma.io
            import ma.body
            logging.info("openMA API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : openma not imported")

        # btk
        pyCGM2.CONFIG.addBtk()
        try:
            import btk
            logging.info("btk API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : btk not imported")

        # opensim
        try:
            import opensim
            logging.info("opensim API ---> OK" )
        except ImportError:
            logging.error ("[pyCGM2] : Opensim API not imported. Can t run CGM version superior to Cgm2.1")


if "install" in sys.argv or "develop" in sys.argv:

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
        raise Exception ("64-bit python version. PyCGM2 requires a 32 bits python version")
    else:
        logging.info ("python core ----->2.7 (OK)")

setup(name = 'pyCGM2',
    version = '1.1.0',
    author = 'fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    keywords = 'python Conventional Gait Model',
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
    cmdclass={'check': checkCommand,},
    )


# check configuratiuon
if "install" in sys.argv or "develop" in sys.argv:
    logging.info( " *******Check your pyCGM2 installation******")
    try:
        import pyCGM2
        logging.info("pyCGM2 ---> OK")
    except ImportError:
        raise Exception ("[pyCGM2] : pyCGM2 module not imported")

    # vicon nexus
    pyCGM2.CONFIG.addNexusPythonSdk()
    try:
        import ViconNexus
        logging.info("vicon API ---> OK" )
    except ImportError:
        logging.error ("[pyCGM2] : viconNexus is not in your python path. Check and edit Nexus paths in file pyCGM2/CONFIG.py. Then rerun setup")

    # openMA
    pyCGM2.CONFIG.addOpenma()
    try:
        import ma.io
        import ma.body
        logging.info("openMA API ---> OK" )
    except ImportError:
        logging.error ("[pyCGM2] : openma not imported")

    # btk
    pyCGM2.CONFIG.addBtk()
    try:
        import btk
        logging.info("btk API ---> OK" )
    except ImportError:
        logging.error ("[pyCGM2] : btk not imported")

    # opensim
    try:
        import opensim
        logging.info("opensim API ---> OK" )
    except ImportError:
        logging.error ("[pyCGM2] : Opensim API not imported. Can t run CGM version superior to Cgm2.1")
