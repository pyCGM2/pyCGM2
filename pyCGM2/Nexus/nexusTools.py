# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Nexus
#APIDOC["Draft"]=False
#--end--

"""
Convenient functions for working with nexus API
"""

import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")


def _setPointData(ftr, framecount, ff, values):

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


def getActiveSubject(NEXUS):
    """return the active subject

    Args:
        NEXUS (): vicon nexus handle

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
    **Obsolete**  prefer the function getActiveSubject instead
    """
    LOGGER.logger.warning(
        "This method is deprecated. prefer getActiveSubject now")

    subjectMarkerWithTraj = dict()
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

    flags = list()
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


def setTrajectoryFromArray(NEXUS, vskName, label, array, firstFrame=0):
    """Set a trajectory (ie marker) from an array

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): trajectory label ( eq. marker label)
        array (np.array(n,3)): array
        firstFrame (int,Optional[0]): first frame of the acquisition.


    """

    trialRange_init = NEXUS.GetTrialRange()[0]
    # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    framecount = NEXUS.GetFrameCount()
    data, exists = _setPointData(
        trialRange_init, framecount, firstFrame, array)

    NEXUS.SetTrajectory(vskName, label, data[0], data[1], data[2], exists)


def setTrajectoryFromAcq(NEXUS, vskName, label, acq):
    """Set a trajectory ( eq. marker) from an btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): trajectory label ( eq. marker label)
        acq (btk.acquisition): a btk.acquisition instance

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


def appendModelledMarkerFromAcq(NEXUS, vskName, label, acq, suffix=""):
    """append a modelled marker ( eg HJC) from a btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): modelled marker label ( eq. marker label)
        acq (btk.acquisition): a btk.acquisition instance
        suffix (str,Optional[""]): suffix added to the model outputs

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


def appendAngleFromAcq(NEXUS, vskName, label, acq):
    """append a angle from a btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): trajectory label ( eq. marker label)
        acq (btk.acquisition): a btk.acquisition instance

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



def appendForceFromAcq(NEXUS,vskName,label, acq,normalizedData=True):
    """append a Force  from an btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): force label
        acq (btk.acquisition): a btk.acquisition instance
        normalizedData (bool,Optional[True]): indicate if values are normalized in amplitude.

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



def appendMomentFromAcq(NEXUS,vskName,label, acq,normalizedData=True):
    """append a Moment  from a btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): moment label
        acq (btk.acquisition): a btk.acquisition instance
        normalizedData (bool,Optional[True]): indicate if values are normalized in amplitude.

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

def appendPowerFromAcq(NEXUS,vskName,label, acq,normalizedData=True):
    """append a power from a btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): power label
        acq (btk.acquisition): a btk.acquisition instance
        normalizedData (bool,Optional[True]): indicate if values are normalized in amplitude.

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

def appendBones(NEXUS,vskName,acq,label,segment,OriginValues=None,manualScale=None,suffix="",existFromPoint = None):
    """append a vicon bone  from a btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        label (str): bone label
        acq (btk.acquisition): a btk.acquisition instance
        OriginValues (np.array(n,3),Optional[None]): manual assignement of the bone origin.
        manualScale (np.array(1,3),Optional[None]): manual scale.
        suffix (str,optional("")): suffix added to bone outputs
        existFromPoint (str,Optional[None]): btk point label conditioning presence or absence of the bone.

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

            R= segment.anatomicalFrame.motion[j].getAngleAxis()

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


def createGeneralEvents(NEXUS,subject,acq,labels):
    """append general events from an btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        labels (list): general event labels
        acq (btk.acquisition): a btk.acquisition instance

    """

    freq = acq.GetPointFrequency()
    events= acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetLabel() in labels:
            NEXUS.CreateAnEvent( subject, "General", ev.GetLabel(), int(ev.GetTime()*freq), 0.0 )

def createEvents(NEXUS,subject,acq,labels):
    """append events from an btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        vskName (str): vsk name.
        labels (list): general event labels
        acq (btk.acquisition): a btk.acquisition instance

    """

    freq = acq.GetPointFrequency()
    events= acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetLabel() in labels:
            NEXUS.CreateAnEvent( subject, ev.GetContext(), ev.GetLabel(), int(ev.GetTime()*freq), 0.0 )

def getForcePlateAssignment(NEXUS):
    """get Force plate assignement

    Args:
        NEXUS (): vicon nexus handle.

    """
    out = dict()
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

def appendAnalysisParameters(NEXUS, acq):
    """append analysis parameter to a btk.acquisition

    Args:
        NEXUS (): vicon nexus handle.
        acq (btk.acquisition): a btk.acquisition instance

    """
    parameters = btkTools.getAllParamAnalysis(acq)

    for parameter in parameters:
        NEXUS.CreateAnalysisParam(parameter["subject"],parameter["name"],parameter["value"], parameter["unit"])
