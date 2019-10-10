# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import logging

import pyCGM2
# openMA
from pyCGM2 import ma
from pyCGM2.ma import io
from pyCGM2.ma import body
from pyCGM2 import btk



def isTimeSequenceExist(trial,label):
    """
        Check if a Time sequence exists inside a trial

        :Parameters:
            - `trial` (openma.trial) - an openma trial instance
            - `label` (str) - label of the time sequence
    """
    try:
        ts = trial.findChild(ma.T_TimeSequence,label)
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


def automaticKineticDetection(dataPath,filenames,trials=None,encoder=pyCGM2.ENCODER):
    """
        convenient method for detecting correct kinetic in a filename set

        :Parameters:
            - `dataPath` (str) - folder path
            - `filenames` (list of str) - filename of the different acquisitions
    """
    kineticTrials=[]
    kineticFilenames=[]

    i=0
    for filename in filenames:
        if filename in kineticFilenames:
            logging.debug("[pyCGM2] : filename %s duplicated in the input list" %(filename))
        else:
            if trials is None:
                fileNode = ma.io.read((dataPath + filename).decode("utf-8").encode(encoder))
                trial = fileNode.findChild(ma.T_Trial)

            else:
                trial = trials[i]

            sortedEvents(trial)
            flag_kinetics,times, times_l, times_r = isKineticFlag(trial)

            if flag_kinetics:
                kineticFilenames.append(filename)
                kineticTrials.append(trial)
    i+=1

    kineticTrials = None if kineticTrials ==[] else kineticTrials
    flag_kinetics = False if kineticTrials ==[] else True


    return kineticTrials,kineticFilenames,flag_kinetics


def findProgression(trial,pointLabel):

    if not isTimeSequenceExist(trial,pointLabel):
        raise Exception( "[pyCGM2] : origin point doesnt exist")

    f,ff,lf = findValidFrames(trial,[pointLabel])

    values = trial.findChild(ma.T_TimeSequence,pointLabel).data()[ff:lf,0:3]

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

    logging.debug("Progression axis : %s"%(progressionAxis))
    logging.debug("forwardProgression : %s"%(str(forwardProgression)))
    logging.debug("globalFrame : %s"%(str(globalFrame)))

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

    f,ff,lf = findValidFrames(trial,[LASI,RASI,LPSI,RPSI])

    LASIvalues = trial.findChild(ma.T_TimeSequence,LASI).data()[ff,0:3]
    RASIvalues = trial.findChild(ma.T_TimeSequence,RASI).data()[ff,0:3]

    midASISvalues =   (LASIvalues+RASIvalues)/2.0

    if SACR is not None:
        if not isTimeSequenceExist(trial,LPSI):
            raise Exception( "[pyCGM2] : LPSI  doesnt exist")

        if not isTimeSequenceExist(trial,RPSI):
            raise Exception( "[pyCGM2] : RPSI  doesnt exist")


        LPSIvalues = trial.findChild(ma.T_TimeSequence,LPSI).data()[ff,0:3]
        RPSIvalues = trial.findChild(ma.T_TimeSequence,LPSI).data()[ff,0:3]
        midPSISvalues =   (LPSIvalues+RPSIvalues)/2.0
    else:
        if not isTimeSequenceExist(trial,SACR):
            raise Exception( "[pyCGM2] : SACR  doesnt exist")

        midPSISvalues = trial.findChild(ma.T_TimeSequence,SACR).data()[ff,0:3]

    a1=(midASISvalues-midPSISvalues)
    a1=a1/np.linalg.norm(a1)

    a2=(RASIvalues-midASISvalues)
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

    logging.debug("Longitudinal axis : %s"%(longitudinalAxis))
    logging.debug("forwardProgression : %s"%(str(forwardProgression)))
    logging.debug("globalFrame : %s"%(str(globalFrame)))

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

def buildTrials(dataPath,trialfilenames,encoder=pyCGM2.ENCODER):
    """
        Get trial list from filenames

        :Parameters:
            - `dataPath` (str) - folder path
            - `trialfilenames` (list of str) - filename of the different acquisitions
    """

    trials=[]
    filenames =[]
    for filename in trialfilenames:
        logging.debug( dataPath)
        logging.debug( filename)
        logging.debug( "------------------")

        fileNode = ma.io.read((dataPath + filename).decode("utf-8").encode(encoder))
        trial = fileNode.findChild(ma.T_Trial)
        sortedEvents(trial)

        trials.append(trial)
        filenames.append(filename)

    return trials,filenames


def smartTrialReader(dataPath,trialfilename,encoder=pyCGM2.ENCODER):
    if dataPath is None:
        fileNode = ma.io.read((trialfilename).decode("utf-8").encode(encoder))
    else:
        fileNode = ma.io.read((dataPath + trialfilename).decode("utf-8").encode(encoder))

    trial = fileNode.findChild(ma.T_Trial)
    sortedEvents(trial)

    return trial


def findValidFrames(trial,markerLabels):

    flag = list()
    pfn = trial.findChild(ma.T_TimeSequence,"",[["type",ma.TimeSequence.Type_Marker]]).samples()
    for i in range(0,pfn):
        pointFlag=list()
        for marker in markerLabels:
            residue = trial.findChild(ma.T_TimeSequence,marker).data()[i,3]
            if residue >= 0 :
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


def convertBtkAcquisition(acq, returnType = "Trial"):


    root = ma.Node('root')
    trial = ma.Trial("Trial",root)

    framerate = acq.GetPointFrequency()
    firstFrame = acq.GetFirstFrame()

    numberAnalogSamplePerFrame = acq.GetNumberAnalogSamplePerFrame()
    analogFramerate = acq.GetAnalogFrequency()

    if firstFrame ==1:
        time_init = 0.0
    else:
        time_init = firstFrame/framerate


    for it in btk.Iterate(acq.GetPoints()):

        label = it.GetLabel()
        values = it.GetValues()
        residuals = it.GetResiduals()
        desc = it.GetDescription()

        data = np.zeros((values.shape[0],4))
        data[:,0:3] = values
        data[:,3] = residuals[:,0]

        if it.GetType() == btk.btkPoint.Marker:
            ts = ma.TimeSequence(str(label),4,data.shape[0],framerate,time_init,ma.TimeSequence.Type_Marker,"mm", trial.timeSequences())

        elif it.GetType() == btk.btkPoint.Angle:
            ts = ma.TimeSequence(str(label),4,data.shape[0],framerate,time_init,ma.TimeSequence.Type_Angle,"Deg", trial.timeSequences())

        elif it.GetType() == btk.btkPoint.Force:
            ts = ma.TimeSequence(str(label),4,data.shape[0],framerate,time_init,ma.TimeSequence.Type_Force,"N.Kg-1", trial.timeSequences())

        elif it.GetType() == btk.btkPoint.Moment:
            ts = ma.TimeSequence(str(label),4,data.shape[0],framerate,time_init,ma.TimeSequence.Type_Moment,"Nmm.Kg-1", trial.timeSequences())


        elif it.GetType() == btk.btkPoint.Power:
            ts = ma.TimeSequence(str(label),4,data.shape[0],framerate,time_init,ma.TimeSequence.Type_Power,"Watt.Kg-1", trial.timeSequences())

        else:
            logging.warning("[pyCGM2] point [%s] not copied into openma trial"%(label))

        ts.setData(data)
        ts.setDescription(desc)


    for it in btk.Iterate(acq.GetAnalogs()):

        label = it.GetLabel()
        values = it.GetValues()
        desc = it.GetDescription()

        data = values

        ts = ma.TimeSequence(str(label),1,data.shape[0],analogFramerate,time_init,ma.TimeSequence.Type_Analog,"V", 1.0,0.0,[-10.0,10.0], trial.timeSequences())
        ts.setData(data)
        ts.setDescription(desc)




    for it in btk.Iterate(acq.GetEvents()):

        label = it.GetLabel()
        time = it.GetTime()
        context = it.GetContext()
        subject = it.GetSubject()

        ev = ma.Event(label,time,context,str(subject),trial.events())

    sortedEvents(trial)

    if returnType == "Trial":
        return trial
    else:
        return root
