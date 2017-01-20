# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 16:17:08 2016

@author: fabien Leboeuf
"""
import numpy as np
import pdb

# openMA
import ma.io
import ma.body
import logging


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
    """


    kineticEvent_times=[]
    evsn = trial.findChild(ma.T_Node,"SortedEvents")
    for ev in evsn.findChildren(ma.T_Event):
        if ev.context() == "General":
            if ev.name() in ["","Event"]:
                kineticEvent_times.append(ev.time())
         

    if kineticEvent_times==[]:
        return False,0
    else:
        return True,kineticEvent_times

            
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
        
            flag_kinetics,times = isKineticFlag(trial)
            
            if flag_kinetics: 
                kineticFilenames.append(filename)
                kineticTrials.append(trial)
    
    kineticTrials = None if kineticTrials ==[] else kineticTrials    
    
    return kineticTrials,kineticFilenames,flag_kinetics            


def findProgressionFromPoints(trial,originPointLabel, longitudinal_extremityPointLabel,lateral_extremityPointLabel):
    """
        Find progression from 3 markers    

        :Parameters:
            - `trial` (openMA trial) - an openma trial instance
            - `originPointLabel` (str) - origin marker label 
            - `longitudinal_extremityPointLabel` (str) - forward marker label
            - `lateral_extremityPointLabel` (str) - lateral marker label

    """    
    
    if not isTimeSequenceExist(trial,originPointLabel):
        raise Exception( "[pyCGM2] : origin point doesnt exist")

    if not isTimeSequenceExist(trial,longitudinal_extremityPointLabel):
        raise Exception( "[pyCGM2] : longitudinal point  doesnt exist")

    if not isTimeSequenceExist(trial,lateral_extremityPointLabel):
        raise Exception( "[pyCGM2] : lateral point  doesnt exist")

    originValues = trial.findChild(ma.T_TimeSequence,originPointLabel).data()[0,0:3]
    longitudinal_extremityValues = trial.findChild(ma.T_TimeSequence,longitudinal_extremityPointLabel).data()[0,0:3]
    lateral_extremityValues = trial.findChild(ma.T_TimeSequence,lateral_extremityPointLabel).data()[0,0:3]#[ acq.GetPoint(originPointLabel).GetValues()[0,:]


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