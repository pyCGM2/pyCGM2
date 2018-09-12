# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:40:17 2016

@author: fabien Leboeuf ( Salford Univ)
"""

import pdb
import logging


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon
import ViconNexus




# local lib
import lib.functions  as libf





if __name__ == "__main__":
    print "======== [pyCGM2-Virtual KAD] ========="

#    pyNEXUS = ViconNexus.ViconNexus()
#    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

    NEXUS_PYTHON_CONNECTED = True

    if NEXUS_PYTHON_CONNECTED:

        # ---- INPUTS ----
        vertical_global_axis = "Z"
        left_medial_knee_marker = "LMEPI"
        right_medial_knee_marker ="RMEPI"
        updateC3d = False

        # ---- DATA ----

        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\virtualKad\\"
        filename =  "MRI-US-01, 2008-08-08, 3DGA 02.c3d"


        # ---- PROCESSING ----
        fullFilenameVirtualKad = libf.virtualKAD (filename,DATA_PATH,
                    vertical_global_axis = vertical_global_axis,
                    left_medial_knee_marker = left_medial_knee_marker,
                    right_medial_knee_marker = right_medial_knee_marker,
                    updateC3d = updateC3d)

        if updateC3d :
            logging.warning ("[pyCGM2-virtual Kad] Static file updated")
        else:
            logging.info ("[pyCGM2-virtual Kad] New static file created")
    else:
        logging.error("[[pyCGM2-virtual Kad]] : Nexus Not Connected")
