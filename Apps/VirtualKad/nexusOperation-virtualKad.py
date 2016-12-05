# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:40:17 2016

@author: fabien Leboeuf ( Salford Univ)
"""

import sys
import pdb
import logging

# pyCGM2 settings
import pyCGM2 
pyCGM2.pyCGM2_CONFIG.setLoggingLevel(logging.INFO)

# vicon
pyCGM2.pyCGM2_CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.pyCGM2_CONFIG.addOpenma()
import ma.io
import ma.body


#local lib
import lib.functions  as libf

    


    
if __name__ == "__main__":
    
    print "======== [pyCGM2-Virtual KAD] ========="    
    
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()
    
     
    if NEXUS_PYTHON_CONNECTED: 

        # ---- INPUTS ----
        vertical_global_axis = sys.argv[1] #"Z" 
        left_medial_knee_marker =  sys.argv[2] #"LMEPI"
        right_medial_knee_marker =  sys.argv[3]# "RMEPI"            
        updateC3d = bool(int(sys.argv[4])) #False

        # ---- DATA ----
        DATA_PATH, filenameNoExt = pyNEXUS.GetTrialName()
        filename = filenameNoExt+".c3d"
        print "Name: "+ filename
        print "Path: "+ DATA_PATH
        

        
        # ---- PROCESSING ----
        fullFilenameVirtualKad = libf.virtualKAD (filename,DATA_PATH, 
                    vertical_global_axis = vertical_global_axis, 
                    left_medial_knee_marker = left_medial_knee_marker, 
                    right_medial_knee_marker = right_medial_knee_marker, 
                    updateC3d = updateC3d)

    else:
        logging.error("[[pyCGM2-virtual Kad]] : Nexus Not Connected")
        

        
         
