from setuptools import setup,find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

import os,sys


import registry

# check python.exe command
runPythonExe = sys.executable
if "pythonw" in runPythonExe:          
    runPythonExe =runPythonExe.replace("pythonw", "python")                
_PYTHONEXE = runPythonExe
_COMPATIBLENEXUSKEY = "\""+ _PYTHONEXE+"\"  \"%1\" %*" # HERE IS the COMPATIBLE NEXUS python executable command


# - overloading of install and develop command
class CustomDevelopCommand(develop):
    """overload of the develop command"""
    def run(self):
        reg_key = registry.getPythonExeRegisterKey() 
        if reg_key != _COMPATIBLENEXUSKEY:
            print ( "register key of python.exe modified to be compatible with Nexus")
            registry.setPythonExeRegisterKey(_COMPATIBLENEXUSKEY)
        
        develop.run(self)
        
class CustomInstallCommand(install):
    """overload of the develop command"""
    def run(self):
        reg_key = registry.getPythonExeRegisterKey() 
        if reg_key != _COMPATIBLENEXUSKEY:
            print ( "register key of python.exe modified to be compatible with Nexus")
            registry.setPythonExeRegisterKey(_COMPATIBLENEXUSKEY)
        install.run(self)



def gen_data_files(*dirs):
    results = []

    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            results.append((root, map(lambda f:root + "/" + f, files)))
    return results



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
   cmdclass={
        'develop': CustomDevelopCommand,
        'install': CustomInstallCommand}
     )




