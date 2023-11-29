"""
Convenient functions for working with nexus API
"""

import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools
import btk

try:
    from viconnexusapi import ViconNexus

except ImportError as e:
    LOGGER.logger.error(f"viconnexusapi not installed: {e}")

from typing import List, Tuple, Dict, Optional,Union

def _setPointData(ftr, framecount, ff, values):
    """
    Sets point data for a given frame range in Nexus.
    Args:
        ftr: The first frame of the trial range.
        framecount: Total number of frames in the trial.
        ff: The first frame of the data to be set.
        values: The values to be set.
    Returns:
        A tuple containing data and a list of boolean flags indicating the existence of data at each frame.
    """
    beg = ff - ftr

    data = [list(np.zeros((framecount))), list(
        np.zeros((framecount))), list(np.zeros((framecount)))]
    exists = [False]*framecount

    i = beg
    for val in values:
        data[0][i] = val[0]
        data[1][i] = val[1]
        data[2][i] = val[2]
        exists[i] = False if val[0] == 0 and val[1] == 0 and val[2] == 0 else True
        i += 1

    return data, exists


def getActiveSubject(NEXUS:ViconNexus.ViconNexus):
    """
    Retrieves the active subject from Nexus.
    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
    Returns:
        The name of the active subject.
    """

    names, templates, active = NEXUS.GetSubjectInfo()
    if active.count(True) > 1:
        raise Exception(
            "[pyCGM2] : two subjects are activated. Select one only")

    for i in range(0, len(names)):
        if active[i]:
            return names[i]

    return names, templates, active


def checkActivatedSubject(NEXUS, subjectNames):
    """
    Checks and retrieves the activated subject in Nexus (Obsolete, use `getActiveSubject` instead).
    Args:
        NEXUS, subjectNames.
    Returns:
        The name of the activated subject.
    """
    LOGGER.logger.warning(
        "This method is deprecated. prefer getActiveSubject now")

    subjectMarkerWithTraj = {}
    for subject in subjectNames:
        markers = NEXUS.GetMarkerNames(subject)
        marker = None
        for mark in markers:
            if NEXUS.GetTrajectory(subject, mark) != ([], [], [], []):
                marker = mark
                LOGGER.logger.debug(
                    "Subject : %s ( marker (%s) with trajectory )" % (subject, marker))
                subjectMarkerWithTraj[subject] = marker
                break

    flags = []
    for value in subjectMarkerWithTraj.itervalues():
        if value is not None:
            flags.append(True)
        else:
            flags.append(False)

    if flags.count(True) > 1:
        raise Exception(
            "[pyCGM2] : two subjects are activated. Select one ony")
    else:
        index = flags.index(True)
        LOGGER.logger.debug("Active subject is %s" %
                            (subjectMarkerWithTraj.keys()[index]))

    return subjectMarkerWithTraj.keys()[index]


def setTrajectoryFromArray(NEXUS:ViconNexus.ViconNexus, 
                           vskName:str, label:str, array:np.ndarray, firstFrame:int=0):
    """Sets a trajectory in Nexus using data from an array.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The name of the VSK file.
        label (str): The label of the trajectory.
        array (np.ndarray): The data array.
        firstFrame (int): The first frame of the trajectory in the acquisition.
    """

    trialRange_init = NEXUS.GetTrialRange()[0]
    # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    framecount = NEXUS.GetFrameCount()
    data, exists = _setPointData(
        trialRange_init, framecount, firstFrame, array)

    NEXUS.SetTrajectory(vskName, label, data[0], data[1], data[2], exists)


def setTrajectoryFromAcq(NEXUS:ViconNexus.ViconNexus, vskName:str, label:str, acq:btk.btkAcquisition):
    """Sets a trajectory in Nexus using data from a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The name of the VSK file.
        label (str): The label of the trajectory.
        acq (btk.btkAcquisition): The BTK acquisition object.
    """

    markers = NEXUS.GetMarkerNames(vskName)
    if label not in markers:
        raise Exception(
            "[pyCGM2] - trajectory of marker (%s) not found. update of trajectory impossible " % (label))

    values = acq.GetPoint(label).GetValues()

    #ff,lf = NEXUS.GetTrialRange()
    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()
    pfn = acq.GetPointFrameNumber()

    trialRange_init = NEXUS.GetTrialRange()[0]
    # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    framecount = NEXUS.GetFrameCount()
    data, exists = _setPointData(trialRange_init, framecount, ff, values)

    NEXUS.SetTrajectory(vskName, label, data[0], data[1], data[2], exists)


def appendModelledMarkerFromAcq(NEXUS:ViconNexus.ViconNexus, 
                                vskName:str, label:str, acq:btk.btkAcquisition, suffix:str=""):
    """Appends a modeled marker to Nexus from a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The name of the VSK file.
        label (str): The label of the marker.
        acq (btk.btkAcquisition): The BTK acquisition object.
        suffix (str): Suffix to be added to the model output.
    """

    lst = NEXUS.GetModelOutputNames(vskName)
    output_label = label+suffix
    if output_label in lst:
        LOGGER.logger.debug("marker (%s) already exist" % (output_label))
    else:
        NEXUS.CreateModeledMarker(vskName, output_label)

    values = acq.GetPoint(label).GetValues()

    #ff,lf = NEXUS.GetTrialRange()
    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()
    pfn = acq.GetPointFrameNumber()

    trialRange_init = NEXUS.GetTrialRange()[0]
    # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    framecount = NEXUS.GetFrameCount()
    data, exists = _setPointData(trialRange_init, framecount, ff, values)

    NEXUS.SetModelOutput(vskName, output_label, data, exists)


def appendAngleFromAcq(NEXUS:ViconNexus.ViconNexus, 
                    vskName:str, label:str, acq:btk.btkAcquisition):
    
    """Appends an angle data to Nexus from a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The name of the VSK file.
        label (str): The label of the angle data.
        acq (btk.btkAcquisition): The BTK acquisition object.
    """


    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        LOGGER.logger.debug( "angle (%s) already exist" %(label))
    else:
        NEXUS.CreateModelOutput( vskName, label, "Angles", ["X","Y","Z"], ["Angle","Angle","Angle"])

    values = acq.GetPoint(label).GetValues()

    #ff,lf = NEXUS.GetTrialRange()
    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()


    pfn = acq.GetPointFrameNumber()
    trialRange_init = NEXUS.GetTrialRange()[0]
    framecount = NEXUS.GetFrameCount()
    data,exists = _setPointData(trialRange_init,framecount,ff,values)

    NEXUS.SetModelOutput( vskName, label, data, exists )



def appendForceFromAcq(NEXUS:ViconNexus.ViconNexus,
                       vskName:str,label:str, acq:btk.btkAcquisition,
                       normalizedData:bool=True):
    """Appends force data to Nexus from a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The name of the VSK file.
        label (str): The label of the force data.
        acq (btk.btkAcquisition): The BTK acquisition object.
        normalizedData (bool): Indicates if the data is normalized.
    """


    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        LOGGER.logger.debug( "force (%s) already exist" %(label))
    else:
        if normalizedData:
            NEXUS.CreateModelOutput( vskName, label, "Forces", ["X","Y","Z"], ["ForceNormalized","ForceNormalized","ForceNormalized"])
        else:
            NEXUS.CreateModelOutput( vskName, label, "Forces", ["X","Y","Z"], ["Force","Force","Force"])

    values = acq.GetPoint(label).GetValues()

    #ff,lf = NEXUS.GetTrialRange()
    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()
    pfn = acq.GetPointFrameNumber()

    trialRange_init = NEXUS.GetTrialRange()[0]
    framecount = NEXUS.GetFrameCount() # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    data,exists = _setPointData(trialRange_init,framecount,ff,values)


    NEXUS.SetModelOutput( vskName, label, data, exists )



def appendMomentFromAcq(NEXUS:ViconNexus.ViconNexus,vskName:str,label:str, acq:btk.btkAcquisition,
                        normalizedData:bool=True):
    """Appends moment data to Nexus from a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The name of the VSK file.
        label (str): The label of the moment data.
        acq (btk.btkAcquisition): The BTK acquisition object.
        normalizedData (bool): Indicates if values are normalized in amplitude.
    """

    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        LOGGER.logger.debug( "moment (%s) already exist" %(label))
    else:
        if normalizedData:
            NEXUS.CreateModelOutput( vskName, label, "Moments", ["X","Y","Z"], ["TorqueNormalized","TorqueNormalized","TorqueNormalized"])#
        else:
            NEXUS.CreateModelOutput( vskName, label, "Moments", ["X","Y","Z"], ["Torque","Torque","Torque"])


    values = acq.GetPoint(label).GetValues()

    #ff,lf = NEXUS.GetTrialRange()
    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()
    pfn = acq.GetPointFrameNumber()


    trialRange_init = NEXUS.GetTrialRange()[0]
    framecount = NEXUS.GetFrameCount() # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    data,exists = _setPointData(trialRange_init,framecount,ff,values)



    NEXUS.SetModelOutput( vskName, label, data, exists )

def appendPowerFromAcq(NEXUS:ViconNexus.ViconNexus,vskName:str,label:str, acq:btk.btkAcquisition,
                       normalizedData:bool=True):
    """Appends power data to Nexus from a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The name of the VSK file.
        label (str): The label of the power data.
        acq (btk.btkAcquisition): The BTK acquisition object.
        normalizedData (bool): Indicates if values are normalized in amplitude.
    """
    lst = NEXUS.GetModelOutputNames(vskName)
    if label in lst:
        NEXUS.GetModelOutput(vskName, label)
        LOGGER.logger.debug( "power (%s) already exist" %(label))
    else:
        if normalizedData:
            NEXUS.CreateModelOutput( vskName, label, "Powers", ["X","Y","Z"], ["PowerNormalized","PowerNormalized","PowerNormalized"])
        else:
            NEXUS.CreateModelOutput( vskName, label, "Powers", ["X","Y","Z"], ["Power","Power","Power"])


    values = acq.GetPoint(label).GetValues()

    #ff,lf = NEXUS.GetTrialRange()
    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()
    pfn = acq.GetPointFrameNumber()
    trialRange_init = NEXUS.GetTrialRange()[0]
    framecount = NEXUS.GetFrameCount() # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    data,exists = _setPointData(trialRange_init,framecount,ff,values)

    NEXUS.SetModelOutput( vskName, label, data, exists )

def appendBones(NEXUS:ViconNexus.ViconNexus,
                vskName:str,acq:btk.btkAcquisition,label:str,segment:str,
                OriginValues:Optional[np.ndarray]=None,
                manualScale:Optional[np.ndarray]=None,
                suffix:str="",
                existFromPoint:Optional[str] = None):
    """Appends a Vicon bone from a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        vskName (str): The VSK file name.
        acq (btk.btkAcquisition): The BTK acquisition instance.
        label (str): The bone label.
        segment (str): The segment to append.
        OriginValues (Optional[np.ndarray]): Manual assignment of the bone origin.
        manualScale (Optional[np.ndarray]): Manual scale values.
        suffix (str): Suffix added to bone outputs.
        existFromPoint (Optional[str]): BTK point label conditioning the presence or absence of the bone.
    """

    if any(segment.getExistFrames()):

        residuals = None
        if existFromPoint is not None:
            try:
                residuals = np.invert(acq.GetPoint(existFromPoint).GetResiduals().astype(bool))
            except RuntimeError:
                pass


        output_label = label+suffix
        lst = NEXUS.GetModelOutputNames(vskName)
        if output_label not in lst:
            NEXUS.CreateModelOutput( vskName,output_label, 'Plug-in Gait Bones', ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'SX', 'SY', 'SZ'], ['Angle', 'Angle', 'Angle', 'Length', 'Length', 'Length', 'Length', 'Length', 'Length'])

        #ff,lf = NEXUS.GetTrialRange()
        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()
        pfn = acq.GetPointFrameNumber()
        framecount = NEXUS.GetFrameCount()
        trialRange_init = NEXUS.GetTrialRange()[0]

        beg = ff-trialRange_init
        end = lf-trialRange_init+1


        data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount))),
               list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount))),
               list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
        exists = [False]*framecount


        j=0
        for i in range(beg,end):

            if residuals is not None:
                existFlag = residuals[j]
            else:
                existFlag = segment.getExistFrames()[j]



            if OriginValues is None:
                T= segment.anatomicalFrame.motion[j].getTranslation()
            else:
                T = OriginValues[j,:]

            R= np.rad2deg(segment.anatomicalFrame.motion[j].getAngleAxis())

            if manualScale is None:
                S = segment.m_bsp["length"]
            else:
                S = manualScale

            exists[i] = True if existFlag else False
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

        NEXUS.SetModelOutput( vskName, output_label, data, exists )

def appendBtkScalarFromAcq(NEXUS:ViconNexus.ViconNexus,vskName:str,
                           groupName:str,label:str,nexusTypes:Union[str,list],acq:btk.btkAcquisition):
    """Appends BTK scalar to the acquisition instance.

    Args:
        NEXUS (ViconNexus.ViconNexus): NEXUS handle.
        vskName (str): Subject name.
        groupName (str): Data group name.
        label (str): Scalar label.
        nexusTypes (Union[str, list]): Nexus data type.
        acq (btk.btkAcquisition): A BTK acquisition instance.
    """

    if isinstance(nexusTypes, str):
        nexusTypes  = [nexusTypes,"None","None"]
    elif isinstance(nexusTypes, list):
        if  len(nexusTypes) != 3:
            raise Exception("[pyCGM2] - nexusType is a string or a list of 3 variables")
         
    exist= True
    try:
        values = acq.GetPoint(label+"["+groupName+"]").GetValues()
    except RuntimeError:
        try:
            values = acq.GetPoint(label).GetValues()
        except RuntimeError:
            exist = False
            LOGGER.logger.error(f"[pyCGM2] push to nexus failed  - the scalar [{label}]is not found. ")
    
    if exist:

        lst = NEXUS.GetModelOutputNames(vskName)
        if label in lst:
            NEXUS.GetModelOutput(vskName, label)
            LOGGER.logger.debug( "parameter (%s) already exist" %(label))
        else:
            NEXUS.CreateModelOutput( vskName, label, groupName, ["X","Y","Z"],  nexusTypes)

        #ff,lf = NEXUS.GetTrialRange()
        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()

        pfn = acq.GetPointFrameNumber()
        trialRange_init = NEXUS.GetTrialRange()[0]
        framecount = NEXUS.GetFrameCount()
        data,exists = _setPointData(trialRange_init,framecount,ff,values)

        NEXUS.SetModelOutput( vskName, label, data, exists )

def createGeneralEvents(NEXUS:ViconNexus.ViconNexus,subject:str,acq:btk.btkAcquisition,labels:List):
    """Appends general events from a BTK acquisition to Nexus.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        subject (str): The VSK file name.
        acq (btk.btkAcquisition): The BTK acquisition instance.
        labels (List): List of general event labels.
    """

    freq = acq.GetPointFrequency()
    events= acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetLabel() in labels:
            NEXUS.CreateAnEvent( subject, "General", ev.GetLabel(), int(ev.GetTime()*freq), 0.0 )

def createEvents(NEXUS:ViconNexus.ViconNexus,subject:str,acq:btk.btkAcquisition,labels:List):
    """Appends specific events from a BTK acquisition to Nexus.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        subject (str): Subject-VSK name.
        acq (btk.btkAcquisition): The BTK acquisition instance.
        labels (List): List of event labels.
    """

    freq = acq.GetPointFrequency()
    events= acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetLabel() in labels:
            NEXUS.CreateAnEvent( subject, ev.GetContext(), ev.GetLabel(), int(ev.GetTime()*freq), 0.0 )

def getForcePlateAssignment(NEXUS:ViconNexus.ViconNexus):
    """Retrieves force plate assignments from Nexus.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.

    Returns:
        A dictionary with force plate IDs as keys and their assignments as values.
    """
    out = {}
    for id in NEXUS.GetDeviceIDs():
        name, type, rate, output_ids, forceplate, eyetracker = NEXUS.GetDeviceDetails(id)
        if type == 'ForcePlate':
            if forceplate.Context=="Invalid":
                out[str(id)]="X"
            if forceplate.Context=="Left":
                out[str(id)]="L"
            if forceplate.Context=="Right":
                out[str(id)]="R"
            if forceplate.Context=="Auto":
                out[str(id)]="A"
    return out

def appendAnalysisParameters(NEXUS:ViconNexus.ViconNexus, acq:btk.btkAcquisition):
    """Appends analysis parameters to a BTK acquisition.

    Args:
        NEXUS (ViconNexus.ViconNexus): The Nexus handle.
        acq (btk.btkAcquisition): The BTK acquisition instance.
    """
   
    parameters = btkTools.getAllParamAnalysis(acq)

    for parameter in parameters:
        NEXUS.CreateAnalysisParam(parameter["subject"],parameter["name"],parameter["value"], parameter["unit"])
