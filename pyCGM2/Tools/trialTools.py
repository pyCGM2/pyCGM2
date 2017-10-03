# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import logging

# openMA
import ma.io
import ma.body




def isTimeSequenceExist(trial,label):
    """
        Check if a Time sequence exists inside a trial    

        :Parameters:
            - `trial` (openma.trial) - an openma trial instance
            - `label` (str) - label of the time sequence
    """
    try:
        trial.findChild(ma.T_TimeSequence,label)
        return True
    except ValueError:
        return False

            
    
def sortedEvents(trial):

    """
        Sort out events of a trial    

        :Parameters:
            - `trial` (openma.trial) - an openma trial instance

    """

    evs = trial.findChildren(ma.T_Event)
   
    contextLst=[] # recuperation de tous les contextes   
    for it in evs:
        if it.context() not in contextLst:
            contextLst.append(it.context())


    valueTime=[] # recuperation de toutes les frames SANS doublons
    for it in evs:
        if it.time() not in valueTime:
            valueTime.append(it.time())
    valueTime.sort() # trie


    events = ma.Node("SortedEvents")     #events =list()
    for contextLst_it in contextLst:  
        for timeSort in valueTime:     
            for it in evs:
                if it.time()==timeSort and it.context()==contextLst_it:
                    ev = ma.Event(it.name(),
                                  it.time(),
                                  contextLst_it,
                                  "")
                    ev.addParent(events)

        
    events.addParent(trial.child(0))
         

def addTimeSequencesToTrial(trial,nodeToAdd):
    """
        Add a Time sequence into a trial    

        :Parameters:
            - `trial` (openma.trial) - an openma trial instance
            - `nodeToAdd` (openma.TimeSequence) - time sequence
    """    
    trialTimeSequences = trial.timeSequences() 
    tss = nodeToAdd.child(0).findChildren(ma.T_TimeSequence)
    for ts in tss:
        #print ts.name()
        ts.addParent(trialTimeSequences)
 


def isKineticFlag(trial):
    """
        Flag up if correct kinetics available       

        :Parameters:
            - `trial` (openma.trial) - an openma trial instance
        
        :Return:
            - `` (bool) - flag if kinetic available
            - `kineticEvent_times` (lst) - time of maximal Normal reaction Forces for both context
            - `kineticEvent_times_left` (lst) - time of maximal Normal reaction Forces for the Left context            
            - `kineticEvent_times_right` (lst) - time of maximal Normal reaction Forces for the Right context
    """


    kineticEvent_times=[]
    kineticEvent_times_left=[]
    kineticEvent_times_right=[]
    
    evsn = trial.findChild(ma.T_Node,"SortedEvents")
    for ev in evsn.findChildren(ma.T_Event):
        if ev.context() == "General":
            if ev.name() in ['Left-FP','Right-FP']:
                kineticEvent_times.append(ev.time())
            if ev.name() in ['Left-FP']:
                kineticEvent_times_left.append(ev.time())
            if ev.name() in ['Right-FP']:
                kineticEvent_times_right.append(ev.time())

    if kineticEvent_times==[]:
        return False,0,0,0
    else:
        return True,kineticEvent_times,kineticEvent_times_left,kineticEvent_times_right


def automaticKineticDetection(dataPath,filenames):
    """
        convenient method for detecting correct kinetic in a filename set    

        :Parameters:
            - `dataPath` (str) - folder path 
            - `filenames` (list of str) - filename of the different acquisitions
    """
    kineticTrials=[]
    kineticFilenames=[]
    for filename in filenames:
        if filename in kineticFilenames:
            logging.warning("[pyCGM2] : filename %s duplicated in the input list" %(filename))
        else:     
            fileNode = ma.io.read(str(dataPath + filename))
            trial = fileNode.findChild(ma.T_Trial)
            sortedEvents(trial)
        
            flag_kinetics,times, times_l, times_r = isKineticFlag(trial)
            
            if flag_kinetics: 
                kineticFilenames.append(filename)
                kineticTrials.append(trial)
    
    kineticTrials = None if kineticTrials ==[] else kineticTrials    
    
    return kineticTrials,kineticFilenames,flag_kinetics  


def findProgression(trial,pointLabel):

    if not isTimeSequenceExist(trial,pointLabel):
        raise Exception( "[pyCGM2] : origin point doesnt exist")
                     
    values = trial.findChild(ma.T_TimeSequence,pointLabel).data()[:,0:3]
    
    MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
    absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

    ind = np.argmax(absMaxValues)
    diff = MaxValues[ind]
    
    if ind ==0 :
        progressionAxis = "X"
        lateralAxis = "Y"
    else:
        progressionAxis = "Y"
        lateralAxis = "X"

    forwardProgression = True if diff>0 else False

    globalFrame = str(progressionAxis+lateralAxis+"Z")        

    logging.info("Progression axis : %s"%(progressionAxis))
    logging.info("forwardProgression : %s"%(str(forwardProgression)))
    logging.info("globalFrame : %s"%(str(globalFrame)))
        
    return   progressionAxis,forwardProgression,globalFrame


def findProgressionFromPelvicMarkers(trial,LASI="LASI",RASI="RASI", LPSI="LPSI", RPSI="RPSI", SACR=None):
    """
 

        :Parameters:
            - `trial` (openMA trial) - an openma trial instance
            - `originPointLabel` (str) - origin marker label 
            - `longitudinal_extremityPointLabel` (str) - forward marker label
            - `lateral_extremityPointLabel` (str) - lateral marker label

    """    
    
    if not isTimeSequenceExist(trial,LASI):
        raise Exception( "[pyCGM2] : LPSI doesnt exist")

    if not isTimeSequenceExist(trial,RASI):
        raise Exception( "[pyCGM2] : RASI  doesnt exist")


    LASIvalues = trial.findChild(ma.T_TimeSequence,LASI).data()[0,0:3]
    RASIvalues = trial.findChild(ma.T_TimeSequence,RASI).data()[0,0:3]    

    midASISvalues =   (LASIvalues+RASIvalues)/2.0  

    if SACR is not None:
        if not isTimeSequenceExist(trial,LPSI):
            raise Exception( "[pyCGM2] : LPSI  doesnt exist")
    
        if not isTimeSequenceExist(trial,RPSI):
            raise Exception( "[pyCGM2] : RPSI  doesnt exist")


        LPSIvalues = trial.findChild(ma.T_TimeSequence,LPSI).data()[0,0:3]        
        RPSIvalues = trial.findChild(ma.T_TimeSequence,LPSI).data()[0,0:3]
        midPSISvalues =   (LPSIvalues+RPSIvalues)/2.0
    else:
        if not isTimeSequenceExist(trial,SACR):
            raise Exception( "[pyCGM2] : SACR  doesnt exist")

        midPSISvalues = trial.findChild(ma.T_TimeSequence,SACR).data()[0,0:3]
    
    a1=(midASISvalues-midPSISvalues)
    a1=a1/np.linalg.norm(a1)

    a2=(RPSIvalues-midPSISvalues)
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
    
def renameOpenMAtoVicon(analysis, suffix=""):
    """
        Convenient function renaming openma standard point label to vicon CGM point label     
        
        :Parameters:
            - `analysis` (openMA node) - openma node containing time sequences of model outputs
            - `suffix` (str) - point label suffix 
                
    """
    tss = analysis.child(0).findChildren(ma.T_TimeSequence)
    for ts in tss:
        name = ts.name()
        
        if "Angle" in name:
            newName = name.replace(".", "")
            newName= newName + "s" + suffix
            if ("Pelvis" in name):
                newName = newName.replace("Progress", "") + suffix
        if "Force" in name:
            newName = name.replace(".", "")
            newName = newName[0: newName.rfind("Force")+5] + suffix
        if "Moment" in name:
            newName = name.replace(".", "")
            newName = newName[0: newName.rfind("Moment")+6] + suffix
        if "Power" in name:
            newName = name.replace(".", "")
            newName = newName[0: newName.rfind("Power")+5] + suffix
        ts.setName(newName)    
        
def buildTrials(dataPath,trialfilenames):
    """
        Get trial list from filenames   

        :Parameters:
            - `dataPath` (str) - folder path 
            - `trialfilenames` (list of str) - filename of the different acquisitions
    """
    
    trials=[]
    filenames =[]
    for filename in trialfilenames:
        fileNode = ma.io.read(str(dataPath + filename))
        trial = fileNode.findChild(ma.T_Trial)
        sortedEvents(trial)

        trials.append(trial)
        filenames.append(filename)
    
    return trials,filenames
    
        
def smartTrialReader(dataPath,trialfilename):
    fileNode = ma.io.read(str(dataPath + trialfilename))
    trial = fileNode.findChild(ma.T_Trial)
    sortedEvents(trial)

    return trial        