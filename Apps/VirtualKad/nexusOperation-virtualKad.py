# -*- coding: utf-8 -*-
"""
Usage:
    file.py
    file.py -h | --help
    file.py --version
    file.py  <vertical_global_axis> [ --left_medial_knee_marker=<lmm> --right_medial_knee_marker=<rmm> -u ]  

Arguments:

 
Options:
    -h --help   Show help message
    -u          update c3d
    --left_medial_knee_marker=<lmm>  suffix associated with classic vicon output label  [default: LMEPI].
    --right_medial_knee_marker=<rmm>  suffix associated with classic vicon output label  [default: RMEPI].


"""

import sys
import pdb
import logging
from docopt import docopt

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
    
    args = docopt(__doc__, version='0.1')
    
    print args
    print "======== [pyCGM2-Virtual KAD] ========="    
    
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()
    
     
    if NEXUS_PYTHON_CONNECTED: 

        # ---- INPUTS ----
        vertical_global_axis = args["<vertical_global_axis>"] #sys.argv[1] #"Z" 
        left_medial_knee_marker = args["--left_medial_knee_marker"]# sys.argv[2] #"LMEPI"
        
        right_medial_knee_marker = args["--right_medial_knee_marker"]# sys.argv[3]# "RMEPI"            
        updateC3d = args["-u"]

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
        

        
         
