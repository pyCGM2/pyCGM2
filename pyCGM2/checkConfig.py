import sys
import os
import logging
logging.basicConfig(level=logging.DEBUG)

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
    logging.error ("[pyCGM2] : viconNexus is not in your python path. Check and edit Nexus paths in file pyCGM2/CONFIG.py. Then rerun setup")

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
    logging.error ("[pyCGM2] : Opensim API not imported. Can t run CGM version superior to Cgm2.1")
