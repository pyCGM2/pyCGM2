# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 15:14:18 2016

@author: aaa34169
"""

import logging
import matplotlib.pyplot as plt

try:
    import pyCGM2.pyCGM2_CONFIG 
except ImportError:
    logging.error("[pyCGM2] : pyCGM2 module not in your python path")



# pyCGM2
from pyCGM2.Core.Tools import  btkTools
from pyCGM2.Core.Model.CGM2 import forceplates




class test_matchedFootPlatForm(): 

    @classmethod
    def twoPF(cls):

        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1 
        gaitFilename="walking_oppositeX_2pf.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        forceplates.appendForcePlateCornerAsMarker(acqGait)       
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)

        if mappedForcePlate!="LR":
            raise Exception ("uncorrected force plate matching")

        # --- Motion 2 
        gaitFilename="walking_X_2pf.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        forceplates.appendForcePlateCornerAsMarker(acqGait)       
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)

        if mappedForcePlate!="RL":
            raise Exception ("uncorrected force plate matching")

    @classmethod
    def threePF(cls):

        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1 
        gaitFilename="walking_Y_3pf.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        forceplates.appendForcePlateCornerAsMarker(acqGait)       
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        
        
        if mappedForcePlate!="RLR":
            raise Exception ("uncorrected force plate matching")

    @classmethod
    def threePF_patho(cls):

        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1 
        gaitFilename="walking_pathoY_onlyRight.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        forceplates.appendForcePlateCornerAsMarker(acqGait)       
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        
        
        if mappedForcePlate!="RRR":
            raise Exception ("uncorrected force plate matching")

    @classmethod
    def fourPF(cls):

        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1 
        gaitFilename="walking-X-4pf.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        forceplates.appendForcePlateCornerAsMarker(acqGait)       
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        

        if mappedForcePlate!="LRLR":
            raise Exception ("uncorrected force plate matching")
        
if __name__ == "__main__":
    plt.close("all")
        
    test_matchedFootPlatForm.twoPF()
    test_matchedFootPlatForm.threePF()    
    test_matchedFootPlatForm.threePF_patho()    
    test_matchedFootPlatForm.fourPF()