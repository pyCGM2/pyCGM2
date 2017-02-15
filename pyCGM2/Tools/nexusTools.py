# -*- coding: utf-8 -*-

import numpy as np
import logging
import pdb



def appendModelledMarkerFromAcq(NEXUS,vskName,label, acq):

    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        logging.debug( "marker (%s) already exist" %(label))
    else:
        NEXUS.CreateModeledMarker(vskName, label)

    values = acq.GetPoint(label).GetValues()
        
    ff,lf = NEXUS.GetTrialRange()
    framecount = NEXUS.GetFrameCount()


    data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
    exists = [False]*framecount
   
    j=0
    for i in range(ff-1,lf):
        exists[i] = True
        data[0][i] = values[j,0]
        data[1][i] = values[j,1]
        data[2][i] = values[j,2]
        j+=1

    NEXUS.SetModelOutput( vskName, label, data, exists )  




def appendAngleFromAcq(NEXUS,vskName,label, acq):

    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        logging.debug( "angle (%s) already exist" %(label))
    else:
        NEXUS.CreateModelOutput( vskName, label, "Angles", ["X","Y","Z"], ["Angle","Angle","Angle"])

    values = acq.GetPoint(label).GetValues()
        
    ff,lf = NEXUS.GetTrialRange()
    framecount = NEXUS.GetFrameCount()


    data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
    exists = [False]*framecount
   
    j=0
    for i in range(ff-1,lf):
        exists[i] = True
        data[0][i] = values[j,0]
        data[1][i] = values[j,1]
        data[2][i] = values[j,2]
        j+=1

    NEXUS.SetModelOutput( vskName, label, data, exists )  



def appendForceFromAcq(NEXUS,vskName,label, acq):

    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        logging.debug( "force (%s) already exist" %(label))
    else:
        NEXUS.CreateModelOutput( vskName, label, "Forces", ["X","Y","Z"], ["Force","Force","Force"])

    values = acq.GetPoint(label).GetValues()
        
    ff,lf = NEXUS.GetTrialRange()
    framecount = NEXUS.GetFrameCount()


    data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
    exists = [False]*framecount
   
    j=0
    for i in range(ff-1,lf):
        exists[i] = True
        data[0][i] = values[j,0]
        data[1][i] = values[j,1]
        data[2][i] = values[j,2]
        j+=1

    NEXUS.SetModelOutput( vskName, label, data, exists )  



def appendMomentFromAcq(NEXUS,vskName,label, acq):

    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        logging.debug( "moment (%s) already exist" %(label))
    else:
        NEXUS.CreateModelOutput( vskName, label, "Moments", ["X","Y","Z"], ["Torque","Torque","Torque"])

    values = acq.GetPoint(label).GetValues()
        
    ff,lf = NEXUS.GetTrialRange()
    framecount = NEXUS.GetFrameCount()


    data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
    exists = [False]*framecount
   
    j=0
    for i in range(ff-1,lf):
        exists[i] = True
        data[0][i] = values[j,0]
        data[1][i] = values[j,1]
        data[2][i] = values[j,2]
        j+=1

    NEXUS.SetModelOutput( vskName, label, data, exists )  

def appendPowerFromAcq(NEXUS,vskName,label, acq):

    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        logging.debug( "power (%s) already exist" %(label))
    else:
        NEXUS.CreateModelOutput( vskName, label, "Powers", ["X","Y","Z"], ["Power","Power","Power"])

    values = acq.GetPoint(label).GetValues()
        
    ff,lf = NEXUS.GetTrialRange()
    framecount = NEXUS.GetFrameCount()


    data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
    exists = [False]*framecount
   
    j=0
    for i in range(ff-1,lf):
        exists[i] = True
        data[0][i] = values[j,0]
        data[1][i] = values[j,1]
        data[2][i] = values[j,2]
        j+=1

    NEXUS.SetModelOutput( vskName, label, data, exists )    
        
def appendBones(NEXUS,vskName,label,segment,OriginValues=None,manualScale=None):
    
    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
    else:
        NEXUS.CreateModelOutput( vskName, label, 'Plug-in Gait Bones', ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'SX', 'SY', 'SZ'], ['Angle', 'Angle', 'Angle', 'Length', 'Length', 'Length', 'Length', 'Length', 'Length'])
    
    ff,lf = NEXUS.GetTrialRange()
    framecount = NEXUS.GetFrameCount()
    
    
    data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount))),
           list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount))),
           list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
    exists = [False]*framecount
   
    j=0
    for i in range(ff-1,lf):
        if OriginValues is None:
            T= segment.anatomicalFrame.motion[j].getTranslation()
        else:
            T = OriginValues[j,:]            

        R= segment.anatomicalFrame.motion[j].getAngleAxis()
        
        if manualScale is None: 
            S = segment.m_bsp["length"]
        else:
            S = manualScale
            
        exists[i] = True
        data[0][i] = R[0]
        data[1][i] = R[1]
        data[2][i] = R[2]
        data[3][i] = T[0]
        data[4][i] = T[1]
        data[5][i] = T[2]
        data[6][i] = S
        data[7][i] = S
        data[8][i] = S

        j+=1

    NEXUS.SetModelOutput( vskName, label, data, exists )      
    
   
