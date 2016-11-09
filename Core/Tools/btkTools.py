# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:36:32 2016

@author: Fabien Leboeuf ( Salford Univ, UK)


TODO : findProgression Axis should be in another folder. (?) 
"""


import btk
import numpy as np
import logging

def smartReader(filename):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename)
    reader.Update()
    acq=reader.GetOutput()
    return acq 

def smartWriter(acq, filename):
    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(filename)
    writer.Update()


def isPointExist(acq,label):
    #TODO : replace by btkIterate
    i = acq.GetPoints().Begin()
    while i != acq.GetPoints().End(): 
        if i.value().GetLabel()==label:
            flagPoint= True
            break
        else:
            i.incr() 
            flagPoint= False
                            
    if flagPoint:
        return True
    else: 
        return False

def isPointsExist(acq,labels):
    for label in labels:
        if not isPointExist(acq,label):
            raise Exception("[pyCGM2] markers (%s) doesn't exist"% label )


def smartAppendPoint(acq,label,values, PointType=btk.btkPoint.Marker,desc=""):
    logging.debug("new point (%s) added to the c3d" % label)

    # TODO : si value = 1 lignes alors il faudrait dupliquer la lignes pour les n franes
    # valueProj *np.ones((aquiStatic.GetPointFrameNumber(),3))
    
    
    
    if isPointExist(acq,label):
        acq.GetPoint(label).SetValues(values)
        acq.GetPoint(label).SetDescription(desc)
        acq.GetPoint(label).SetType(PointType)
        
    else: 
        new_btkPoint = btk.btkPoint(label,acq.GetPointFrameNumber())        
        new_btkPoint.SetValues(values)
        new_btkPoint.SetDescription(desc)
        new_btkPoint.SetType(PointType)
        acq.AppendPoint(new_btkPoint)

def clearPoints(acq, pointlabelList):
    
    i = acq.GetPoints().Begin()
    while i != acq.GetPoints().End():
        label =  i.value().GetLabel()
        if label not in pointlabelList:
            i = acq.RemovePoint(i)
            logging.debug( label + " removed")
        else:
            i.incr()
            logging.debug( label + " found")



    return acq

def findProgressionFromPoints(acq,originPointLabel, longitudinal_extremityPointLabel,lateral_extremityPointLabel):
    if not isPointExist(acq,originPointLabel):
        raise Exception( "[pyCGM2] : origin point doesnt exist")

    if not isPointExist(acq,longitudinal_extremityPointLabel):
        raise Exception( "[pyCGM2] : longitudinal point  doesnt exist")

    if not isPointExist(acq,lateral_extremityPointLabel):
        raise Exception( "[pyCGM2] : lateral point  doesnt exist")

    originValues = acq.GetPoint(originPointLabel).GetValues()[0,:]
    longitudinal_extremityValues = acq.GetPoint(longitudinal_extremityPointLabel).GetValues()[0,:]
    lateral_extremityValues = acq.GetPoint(lateral_extremityPointLabel).GetValues()[0,:]

    a1=(longitudinal_extremityValues-originValues)
    a1=a1/np.linalg.norm(a1)

    a2=(lateral_extremityValues-originValues)
    a2=a2/np.linalg.norm(a2)

    globalAxes = {"X" : np.array([1,0,0]), "Y" : np.array([0,1,0]), "Z" : np.array([0,0,1])}

    # longitudinal axis    
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a1,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    longitudinalAxis =  globalAxes.keys()[maxIndex]
    forwardProgression = True if tmp[maxIndex]>0 else False
    
    # lateral axis
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a2,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    lateralAxis =  globalAxes.keys()[maxIndex]    
    

    # global frame
    if "X" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"X")
    if "Y" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Y")        
    if "Z" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Z")        
           
    return   longitudinalAxis,forwardProgression,globalFrame  


def findProgressionFromVectors(a1_long,a2_lat):

    a1=a1_long/np.linalg.norm(a1_long)

    a2=a2_lat/np.linalg.norm(a2_lat)

    globalAxes = {"X" : np.array([1,0,0]), "Y" : np.array([0,1,0]), "Z" : np.array([0,0,1])}

    # longitudinal axis    
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a1,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    longitudinalAxis =  globalAxes.keys()[maxIndex]
    forwardProgression = True if tmp[maxIndex]>0 else False
    
    # lateral axis
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a2,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    lateralAxis =  globalAxes.keys()[maxIndex]    
    

    # global frame
    if "X" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"X")
    if "Y" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Y")        
    if "Z" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Z")        
           
    return   longitudinalAxis,forwardProgression,globalFrame
    
    
def checkMarkers( acq, markerList):
    """
    checkMarkers(acqGait, ["LASI", "RASI","LPSI", "RPSI", "RTHIAP", "RTHIAD", "RTHI", "RKNE", "RSHN","RTIAP", "RTIB", "RANK", "RHEE", "RTOE","RCUN","RD1M","RD5M" ])                   
    """
    for m in markerList:
        if not isPointExist(acq, m):
            raise Exception("[pyCGM2] markers %s not found" % m )

def checkFirstAndLastFrame (acq, markerLabel):

    if acq.GetPoint(markerLabel).GetValues()[0,0] == 0:
        raise Exception ("[pyCGM2] no marker on first frame")
    
    if acq.GetPoint(markerLabel).GetValues()[-1,0] == 0:
        raise Exception ("[pyCGM2] no marker on last frame")
        
def isGap(acq, markerList):
    for m in markerList:
         residualValues = acq.GetPoint(m).GetResiduals()
         if any(residualValues== -1.0):
             raise Exception("[pyCGM2] gap founded for markers %s " % m )    
             
def modifyEventSubject(acq,newSubjectlabel):
    # events
    nEvents = acq.GetEventNumber()
    if nEvents>=1: 
        for i in range(0, nEvents):
            acq.GetEvent(i).SetSubject(newSubjectlabel)
    return acq

def modifySubject(acq,newSubjectlabel):
    acq.GetMetaData().FindChild("SUBJECTS").value().FindChild("NAMES").value().GetInfo().SetValue(0,str(newSubjectlabel))