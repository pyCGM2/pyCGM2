# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import sys
import os


# ----------------------------------------------------------------------
# Vérification de la version Python
pyversion = f"{sys.version_info.major}.{sys.version_info.minor}"
if pyversion not in ["3.9", "3.10", "3.11"]:
    raise Exception("pyCGM2 is not compatible with your Python version")

if sys.maxsize < 2**32:
    raise Exception("32-bit Python version detected. pyCGM2 requires 64-bit Python")

# ----------------------------------------------------------------------
# Lecture de la version depuis pyCGM2/__init__.py
def read_version():
    with open(os.path.join("pyCGM2", "__init__.py"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Cannot find version information")

VERSION = read_version()

# ----------------------------------------------------------------------
# Lecture optionnelle de requirements.txt
def parse_requirements(filename="requirements.txt"):
    try:
        with open(filename, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

reqs = parse_requirements()


# Setup principal
if __name__ == "__main__":
    setup(
        name="pyCGM2",
        version=VERSION,
        author="Fabien Leboeuf",
        author_email="fabien.leboeuf@gmail.com",
        description="Conventional Gait models and Gait analysis",
        long_description=(
            "A Python implementation of the Conventional Gait Models (CGM) "
            "and methods for processing motion capture gait data"
        ),
        url="https://github.com/pyCGM2/pyCGM2",
        keywords="python CGM Vicon PluginGait CGM Gait biomechanics",
        packages=find_packages(),
        include_package_data=True,
        license="CC-BY-SA",
        install_requires=reqs,
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Operating System :: Microsoft :: Windows",
            "Natural Language :: English",
        ],
        entry_points={
            "console_scripts": [
                "pyCGM2 = pyCGM2.Apps.Commands.rullThemAllCommands:main",
                "pyCGM2-setup_NexusPackages = pyCGM2.Apps.Commands.install.setup_pyCGM2_Nexus:main_installViconPackages",
                "pyCGM2-generate_pyCGM2_Nexus = pyCGM2.Apps.Commands.install.setup_pyCGM2_Nexus:main_install_pyCGM2_NexusFiles",
            ]
        },
    )

