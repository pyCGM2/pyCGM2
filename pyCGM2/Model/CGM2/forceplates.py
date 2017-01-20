# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import btk
import numpy as np
import matplotlib.pyplot as plt
import pdb
import logging

from pyCGM2.Tools import  btkTools





def appendForcePlateCornerAsMarker (btkAcq):
    """
        Add a marker at each force plate corners
        
        :Parameters:
           - `btkAcq` (btkAcquisition) : Btk acquisition instance from a c3d        
        
    """


    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()
    
    
    for i in range(0,pfc.GetItemNumber()):
        val_corner0 = pfc.GetItem(i).GetCorner(0).T * np.ones((btkAcq.GetPointFrameNumber(),3))      
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner0",val_corner0, desc="forcePlate") 
        
        val_corner1 = pfc.GetItem(i).GetCorner(1).T * np.ones((btkAcq.GetPointFrameNumber(),3))      
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner1",val_corner1, desc="forcePlate") 
        
        val_corner2 = pfc.GetItem(i).GetCorner(2).T * np.ones((btkAcq.GetPointFrameNumber(),3))      
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner2", val_corner2, desc="forcePlate") 
    
        val_corner3 = pfc.GetItem(i).GetCorner(3).T * np.ones((btkAcq.GetPointFrameNumber(),3))      
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner3",val_corner3, desc="forcePlate") 







def matchingFootSideOnForceplate (btkAcq, left_markerLabelToe ="LTOE", left_markerLabelHeel ="LHEE", 
                 right_markerLabelToe ="RTOE", right_markerLabelHeel ="RHEE",  display = False):
    """
        Convenient function detecting foot in contact with a force plate
        
        :Parameters:
           - `btkAcq` (btkAcquisition) - Btk acquisition instance from a c3d        
           - `left_markerLabelToe` (str) - label of the left toe marker  
           - `left_markerLabelHeel` (str) - label of the left heel marker 
           - `right_markerLabelToe` (str) - label of the right toe marker
           - `right_markerLabelHeel` (str) - label of the right heel marker 
           - `display` (bool) - display n figures ( n depend on force plate number) presenting relative distance between mid foot and the orgin of the force plate 
        
    """
     
    ff=btkAcq.GetFirstFrame()
    lf=btkAcq.GetLastFrame()
    appf=btkAcq.GetNumberAnalogSamplePerFrame()
    
    
    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    grwf = btk.btkGroundReactionWrenchFilter()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    grwf.SetInput(pfc)
    grwc = grwf.GetOutput()
    grwc.Update()
                
    midfoot_L=(btkAcq.GetPoint(left_markerLabelToe).GetValues() + btkAcq.GetPoint(left_markerLabelHeel).GetValues())/2.0 
    midfoot_R=(btkAcq.GetPoint(right_markerLabelToe).GetValues() + btkAcq.GetPoint(right_markerLabelHeel).GetValues())/2.0           
    
    suffix=str()

    
    for i in range(0,grwc.GetItemNumber()):
        pos= grwc.GetItem(i).GetPosition().GetValues()
        pos_downsample = pos[0:(lf-ff+1)*appf:appf]   # downsample 
      
        diffL = np.linalg.norm( midfoot_L-pos_downsample,axis =1)
        diffR = np.linalg.norm( midfoot_R-pos_downsample,axis =1)      
        
        if display:
            plt.figure()
            ax = plt.subplot(1,1,1)
            plt.title("Force plate " + str(i+1))
            ax.plot(diffL,'-r')
            ax.plot(diffR,'-b')

        if np.min(diffL)<np.min(diffR):
            logging.debug(" Force plate " + str(i) + " : left foot")
            suffix = suffix +  "L"
        else:
            logging.debug(" Force plate " + str(i) + " : right foot")
            suffix = suffix +  "R"

    logging.info("Matched Force plate ===> %s", (suffix))
    return suffix
