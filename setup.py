from setuptools import setup,find_packages
import os,sys
import logging
logging.basicConfig(level=logging.DEBUG)
import registry

def gen_data_files(*dirs):
    results = []
    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            results.append((root, map(lambda f:root + "/" + f, files)))
    return results


# check python.exe command
runPythonExe = sys.executable
if "pythonw" in runPythonExe:
    runPythonExe =runPythonExe.replace("pythonw", "python")
_PYTHONEXE = runPythonExe
_COMPATIBLENEXUSKEY = "\""+ _PYTHONEXE+"\"  \"%1\" %*" # HERE IS the COMPATIBLE NEXUS python executable command


reg_key = registry.getPythonExeRegisterKey()
print " ******* Alteration of your python registry key fo Nexus ******"
if reg_key != _COMPATIBLENEXUSKEY:
    logging.warning( "register key of python.exe modified to be compatible with Nexus")
    registry.setPythonExeRegisterKey(_COMPATIBLENEXUSKEY)
else:
    logging.info("Python registry key compatible")

print " ******* End alteration ******"

# check if 64 bits
if "64-bit" in sys.version:
    raise Exception ("64-bit python version. PyCGM2 requires a 32 bits python version")

setup(name = 'pyCGM2',
    version = '1.0.3-beta',
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
                        'enum34>=1.1.2'],
    )

print " *******CHECK CONFIG******"
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

# opensim
try:
    import opensim
    logging.info("opensim API ---> OK" )
except ImportError:
    logging.error ("[pyCGM2] : Opensim API is not in your python path. Check CONFIG")
