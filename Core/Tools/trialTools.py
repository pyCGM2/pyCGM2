# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 16:17:08 2016

@author: fabien Leboeuf
"""

# openMA
import ma.io
import ma.body
import logging


def isTimeSequenceExist(trial,label):
    try:
        trial.findChild(ma.T_TimeSequence,label)
        return True
    except ValueError:
        return False

            
    
def sortedEvents(standardTrial):
    """

    """
    evs = standardTrial.findChildren(ma.T_Event)
   
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

        
    events.addParent(standardTrial.child(0))
         

def addTimeSequencesToTrial(trial,nodeToAdd):
        trialTimeSequences = trial.timeSequences() 
        tss = nodeToAdd.child(0).findChildren(ma.T_TimeSequence)
        for ts in tss:
            #print ts.name()
            ts.addParent(trialTimeSequences)
     

def isKineticFlag(trial):


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
    kineticTrials=[]
    kineticFilenames=[]
    for filename in filenames:
        if filename in kineticFilenames:
            logging.warning("[pyCGM2] : filename %s duplicated in the input list" %(filename))
        else:     
            trial = ma.io.read(str(dataPath + filename))
            sortedEvents(trial)
        
            flag_kinetics,times = isKineticFlag(trial)
            
            if flag_kinetics: 
                kineticFilenames.append(filename)
                kineticTrials.append(trial)
    
    kineticTrials = None if kineticTrials ==[] else kineticTrials    
    
    return kineticTrials,kineticFilenames,flag_kinetics            
