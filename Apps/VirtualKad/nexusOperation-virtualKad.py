# -*- coding: utf-8 -*-
import sys
import pdb
import logging
import  argparse

# pyCGM2 settings
import pyCGM2 
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon
pyCGM2.CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body


#local lib
import lib.functions  as libf

    


    
if __name__ == "__main__":
    
    print "======== [pyCGM2-Virtual KAD] ========="    

    # ---- INPUTS ----
    parser = argparse.ArgumentParser(description='CGM')
    parser.add_argument('-o','--overwrite', action='store_true', help='overwrite' )
    parser.add_argument('--verticalGlobalAxis', default="Z", type=str)
    parser.add_argument('--leftMedialKneeMarker', default="LMEPI", type=str)
    parser.add_argument('--rightMedialKneeMarker', default="RMEPI", type=str)
     
    args = parser.parse_args()
    logging.info(args)
    
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()
    
     
    if NEXUS_PYTHON_CONNECTED: 

        # --- INPUTS -----
        vertical_global_axis = args.verticalGlobalAxis 
        left_medial_knee_marker = args.leftMedialKneeMarker 
        right_medial_knee_marker = args.rightMedialKneeMarker      
        updateC3d = args.overwrite

        print left_medial_knee_marker
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
        

        
         
