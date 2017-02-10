# -*- coding: utf-8 -*-

import numpy as np
import logging
import pdb

import btk


def appendModelledMarkerFromAcq(nexusHandle,vskName,label, acq):

    lst = nexusHandle.GetModelOutputNames(vskName)
    if label in lst:
        nexusHandle.GetModelOutput(vskName, label) 
    else:
        nexusHandle.CreateModeledMarker(vskName, label)

    values = acq.GetPoint(label).GetValues()
        
    ff,lf = nexusHandle.GetTrialRange()
        
    j=0
    for i in range(ff,lf+1):
        nexusHandle.SetModelOutputAtFrame(vskName, label, i, [values[j,0], values[j,1], values[j,2]], True )
        j+=1

def appendAngleFromAcq(nexusHandle,vskName,label, acq):
    lst = nexusHandle.GetModelOutputNames(vskName)
    if label in lst:
        nexusHandle.GetModelOutput(vskName, label) 
    else:
        nexusHandle.CreateModelOutput( vskName, label, "Angles", ["X","Y","Z"], ["Angle","Angle","Angle"])
        
    values = acq.GetPoint(label).GetValues()
        
    ff,lf = nexusHandle.GetTrialRange()
        
    j=0
    for i in range(ff,lf+1):
        nexusHandle.SetModelOutputAtFrame(vskName, label, i, [values[j,0], values[j,1], values[j,2]], True )
        j+=1
        
def appendBones(nexusHandle,vskName,label,segment,OriginValues=None,manualScale=None):
    
    lst = nexusHandle.GetModelOutputNames(vskName)
    if label in lst:
        nexusHandle.GetModelOutput(vskName, label)
    else:
        nexusHandle.CreateModelOutput( vskName, label, 'Plug-in Gait Bones', ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'SX', 'SY', 'SZ'], ['Angle', 'Angle', 'Angle', 'Length', 'Length', 'Length', 'Length', 'Length', 'Length'])
    
    ff,lf = nexusHandle.GetTrialRange()
    
   
    j=0
    for i in range(ff,lf+1):
        if OriginValues is None:
            T= segment.anatomicalFrame.motion[j].getTranslation()
        else:
            T = OriginValues[j,:]            

        R= segment.anatomicalFrame.motion[j].getAngleAxis()
        
        if manualScale is None: 
            S = segment.m_bsp["length"]
        else:
            S = manualScale

        nexusHandle.SetModelOutputAtFrame(vskName, label, i, [R[0], R[1], R[2] , T[0], T[1], T[2], S, S, S  ], True )
        j+=1