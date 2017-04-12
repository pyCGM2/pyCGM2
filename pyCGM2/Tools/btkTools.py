# -*- coding: utf-8 -*-

import numpy as np
import logging
import pdb

import btk

# --- acquisition -----
def smartReader(filename):
    """
        Convenient function to read a c3d with Btk    

        :Parameters:
            - `filename` (str) - path and filename of the c3d
    """
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename)
    reader.Update()
    acq=reader.GetOutput()
    return acq 

def smartWriter(acq, filename):
    """
        Convenient function to write a c3d with Btk    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `filename` (str) - path and filename of the c3d
    """
    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(filename)
    writer.Update()


def isPointExist(acq,label):
    """
        Check if a point label exists inside an acquisition    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `label` (str) - point label
    """
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
    """
        Check if point labels exist inside an acquisition    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `labels` (list of str) - point labels
    """
    for label in labels:
        if not isPointExist(acq,label):
            logging.debug("[pyCGM2] markers (%s) doesn't exist"% label )
            return False
    return True

def smartAppendPoint(acq,label,values, PointType=btk.btkPoint.Marker,desc=""):
    """
        Append/Update a point inside an acquisition    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `label` (str) - point label
            - `values` (numpy.array(n,3)) - point label
            - `PointType` (enums of btk.btkPoint) - type of Point            
    """    
    
    logging.debug("new point (%s) added to the c3d" % label)

    # TODO : si value = 1 lignes alors il faudrait dupliquer la lignes pour les n franes
    # valueProj *np.ones((aquiStatic.GetPointFrameNumber(),3))

    values = np.nan_to_num(values)
    
    
    
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
    """
        Clear points    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `lapointlabelListel` (list of str) - point labels

    """    
    
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

def checkFirstAndLastFrame (acq, markerLabel):
    """
        Check if extremity frames are correct    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerLabel` (str) - marker label
    """

    if acq.GetPoint(markerLabel).GetValues()[0,0] == 0:
        raise Exception ("[pyCGM2] no marker on first frame")
    
    if acq.GetPoint(markerLabel).GetValues()[-1,0] == 0:
        raise Exception ("[pyCGM2] no marker on last frame")
        
def isGap(acq, markerList):
    """
        Check if there is a gap    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerList` (list of str) - marker labels
    """
    for m in markerList:
         residualValues = acq.GetPoint(m).GetResiduals()
         if any(residualValues== -1.0):
             raise Exception("[pyCGM2] gap founded for markers %s " % m )  

def findValidFrames(acq,markerLabels):

    flag = list()
    for i in range(0,acq.GetPointFrameNumber()):
        pointFlag=list()
        for marker in markerLabels:
            if acq.GetPoint(marker).GetResidual(i) == 0 :
                pointFlag.append(1)
            else:
                pointFlag.append(0)
        
        if all(pointFlag)==1:
            flag.append(1)
        else:
            flag.append(0)
        
    firstValidFrame = flag.index(1) 
    lastValidFrame = len(flag) - flag[::-1].index(1) - 1

    return flag,firstValidFrame,lastValidFrame


def applyValidFramesOnOutput(acq,validFrames):

    validFrames = np.asarray(validFrames)
    
    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() in [btk.btkPoint.Angle, btk.btkPoint.Force, btk.btkPoint.Moment,btk.btkPoint.Power]:
            values = it.GetValues()
            for i in range(0,3):
                values[:,i] =  values[:,i] * validFrames           
            it.SetValues(values)
            


             
# --- Model -----

def applyTranslators(acq, translators):
    """
    :note: Difficulty is to prevent c3d with duplicated point label.  

    """
    acqClone = btk.btkAcquisition.Clone(acq) 
        
    markerList = translators.keys()
    
    # gather all labels    
    for it in translators.keys():
        if it not in markerList:
            markerList.append(translators[it])

    # remove all points in clone if exist within the former list                     
    for point in  btk.Iterate(acq.GetPoints()):
        if point.GetType() == btk.btkPoint.Marker:
            label = point.GetLabel()
            if label in markerList:
                acqClone.RemovePoint(label)

    # add renamed point from translator                    
    for it in translators.keys():
       logging.debug("Initial point (%s) renamed (%s)  added into the c3d" %(str(translators[it]), str(it)))
       smartAppendPoint(acqClone,str(it),acq.GetPoint(str(translators[it])).GetValues(),PointType=btk.btkPoint.Marker) 



    # add initial point  
    newlist =  list()
    for itP in btk.Iterate(acqClone.GetPoints()):
        if itP.GetType() == btk.btkPoint.Marker:
            newlist.append(itP.GetLabel())
            
    logging.debug("------------------------------------------------------------------------")            
    for it in translators.keys():
        if translators[it] not in newlist: 
            logging.debug("Initial point (%s) added to the new c3d" %(str(translators[it])))  
            smartAppendPoint(acqClone,str(translators[it] ),acq.GetPoint(str(translators[it])).GetValues(),PointType=btk.btkPoint.Marker) 


    return acqClone    
       



def findProgressionAxisFromPelvicMarkers(acq,markers):

    if not isPointsExist(acq,markers):
        raise Exception( "[pyCGM2] : one of pelvic marker doesn't exist")

    # default
    longitudinalAxis="X"
    globalFrame="XYZ"
    forwardProgression = True

    # find valid frames and get the first one
    flag,vff,vlf = findValidFrames(acq,markers)
    index = vff

    originValues = (acq.GetPoint("LPSI").GetValues()[index,:] + acq.GetPoint("RPSI").GetValues()[index,:])/2.0 
    longitudinal_extremityValues = (acq.GetPoint("LASI").GetValues()[index,:] + acq.GetPoint("RASI").GetValues()[index,:])/2.0 
    lateral_extremityValues = acq.GetPoint("LPSI").GetValues()[index,:]


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

    logging.info("Longitudinal axis : %s"%(longitudinalAxis))
    logging.info("forwardProgression : %s"%(str(forwardProgression)))
    logging.info("globalFrame : %s"%(str(globalFrame)))
           
    return   longitudinalAxis,forwardProgression,globalFrame  



def findProgressionFromPoints(acq,originPointLabel, longitudinal_extremityPointLabel,lateral_extremityPointLabel):
    """
        Find progression from 3 markers    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `originPointLabel` (str) - origin marker label 
            - `longitudinal_extremityPointLabel` (str) - forward marker label
            - `lateral_extremityPointLabel` (str) - lateral marker label

    """ 
    if not isPointExist(acq,originPointLabel):
        raise Exception( "[pyCGM2] : origin point doesnt exist")

    if not isPointExist(acq,longitudinal_extremityPointLabel):
        raise Exception( "[pyCGM2] : longitudinal point  doesnt exist")

    if not isPointExist(acq,lateral_extremityPointLabel):
        raise Exception( "[pyCGM2] : lateral point  doesnt exist")


    index = findFirstValidFrame(acq,[originPointLabel, longitudinal_extremityPointLabel,lateral_extremityPointLabel])

    originValues = acq.GetPoint(originPointLabel).GetValues()[index,:]
    longitudinal_extremityValues = acq.GetPoint(longitudinal_extremityPointLabel).GetValues()[index,:]
    lateral_extremityValues = acq.GetPoint(lateral_extremityPointLabel).GetValues()[index,:]

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

    logging.info("Longitudinal axis : %s"%(longitudinalAxis))
    logging.info("forwardProgression : %s"%(str(forwardProgression)))
    logging.info("globalFrame : %s"%(str(globalFrame)))
           
    return   longitudinalAxis,forwardProgression,globalFrame  


def findProgressionFromVectors(a1_long,a2_lat):
    """
        Find progression from 2 vectors    

        :Parameters:
            - `a1_long` (numpy.array(3,)) - forward vector 
            - `a1_long` (numpy.array(3,)) - lateral vector

    """ 

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

    logging.info("Longitudinal axis : %s"%(longitudinalAxis))
    logging.info("forwardProgression : %s"%(str(forwardProgression)))
    logging.info("globalFrame : %s"%(str(globalFrame)))
           
    return   longitudinalAxis,forwardProgression,globalFrame
    
    
def checkMarkers( acq, markerList):
    """
        Check if marker labels exist inside an acquisition    

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerList` (list of str) - marker labels
    """
    for m in markerList:
        if not isPointExist(acq, m):
            raise Exception("[pyCGM2] markers %s not found" % m )




  
# --- events -----             
def modifyEventSubject(acq,newSubjectlabel):
    """
        update the subject name of all events   

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `newSubjectlabel` (str) - desired subject name
    """
    
    # events
    nEvents = acq.GetEventNumber()
    if nEvents>=1: 
        for i in range(0, nEvents):
            acq.GetEvent(i).SetSubject(newSubjectlabel)
    return acq

def modifySubject(acq,newSubjectlabel):
    """
        update the subject name inside c3d metadata   

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `newSubjectlabel` (str) - desired subject name
    """
    acq.GetMetaData().FindChild("SUBJECTS").value().FindChild("NAMES").value().GetInfo().SetValue(0,str(newSubjectlabel))
    

def getNumberOfModelOutputs(acq):
    n_angles=0                
    n_forces=0                
    n_moments=0                
    n_powers=0                

 
    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Angle:
            n_angles+=1

        if it.GetType() == btk.btkPoint.Force:
            n_forces+=1

        if it.GetType() == btk.btkPoint.Moment:
            n_moments+=1

        if it.GetType() == btk.btkPoint.Power:
            n_powers+=1

    return  n_angles,n_forces ,n_moments,n_powers
    
# --- metadata -----
def hasChild(md,mdLabel):
    """
        Check if a label is within metadata 

        .. note:: 
        
            btk has a HasChildren method. HasChild doesn t exist, you have to use MetadataIterator to loop metadata   
        
        :Parameters:
            - `md` (btkMetadata) - a btk metadata instance
            - `mdLabel` (str) - label of the metadata you want to check
    """

    outMd = None
    for itMd in btk.Iterate(md):
        if itMd.GetLabel() == mdLabel:
            outMd = itMd
            break
    return outMd





