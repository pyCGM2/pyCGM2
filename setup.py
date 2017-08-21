from setuptools import setup,find_packages
import os,sys
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
    print ( "register key of python.exe modified to be compatible with Nexus")
    registry.setPythonExeRegisterKey(_COMPATIBLENEXUSKEY)
print " ******* End alteration ******"


setup(name = 'pyCGM2',
    version = '1.0.0-beta',
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
