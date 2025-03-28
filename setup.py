# -*- coding: utf-8 -*-
from setuptools import setup,find_packages
import os,sys
import string
import logging

import shutil
import site

VERSION ="4.4.0-rc1" # to change in __init__ also

MAIN_PYCGM2_PATH = os.getcwd() + "\\"

pyversion = str(sys.version_info.major) + "."+ str(sys.version_info.minor)
logging.info("python version used : " + pyversion)

if pyversion not in ["3.9","3.10","3.11"]:
     raise Exception ("pycgm2 not compatible with your python version")

if sys.maxsize < 2**32:
    raise Exception ("32-bit python version detected. PyCGM2-python3 requires a 64 bits python version")


SITE_PACKAGE_PATH = site.getsitepackages()[0] + "\\"
NAME_IN_SITEPACKAGE = "pyCGM2-"+VERSION+"-py"+pyversion+".egg"


def parse_requirements(requirements):
    try:
        with open(requirements) as f:
            return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]
    except:
        return []

reqs = [] #parse_requirements("requirements.txt")


#------------------------- INSTALL--------------------------------------------
setup(name = 'pyCGM2',
    version = VERSION,
    author = 'Fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    description = "Conventional Gait models and Gait analysis",
    long_description= "A python implementation of the conventional gait models and methods for processing gait motion capture data",
    url = 'https://github.com/pyCGM2/pyCGM2',
    keywords = 'python CGM Vicon PluginGait CGM Gait',
    packages=find_packages(),
	include_package_data=True,
    license='CC-BY-SA',
	install_requires = reqs,
    classifiers=['Programming Language :: Python',
                  'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
                 'Operating System :: Microsoft :: Windows',
                 'Natural Language :: English'],
    entry_points={
          'console_scripts': [
                #RULE THEM ALL COMMANDS
                'pyCGM2  =  pyCGM2.Apps.Commands.rullThemAllCommands:main',
          ]
      },
    )
