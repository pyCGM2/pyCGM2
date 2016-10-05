# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 16:17:08 2016

@author: fabien Leboeuf
"""

# openMA
import ma.io
import ma.body


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
         


