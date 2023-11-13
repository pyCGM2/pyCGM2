"""
Convenient functions for working with nexus API
"""
from typing import Optional,Union
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools
try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

try:
    from viconnexusapi import ViconNexus

except ImportError as e:
    LOGGER.logger.error(f"viconnexusapi not installed: {e}")


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


def getActiveSubject(NEXUS:ViconNexus.ViconNexus):
    """return the active subject

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle

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


def setTrajectoryFromArray(NEXUS:ViconNexus.ViconNexus, 
                           vskName:str, label:str, array:np.ndarray, firstFrame:int=0):
    """Set a trajectory (ie marker) from an array

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): trajectory label ( eq. marker label)
        array (np.ndarray): array (shape(n,3))
        firstFrame (int,optional): first frame of the acquisition. default set to 0


    """

    trialRange_init = NEXUS.GetTrialRange()[0]
    # instead of GetFrameCount ( nexus7 API differed from nexus 2.6 API)
    framecount = NEXUS.GetFrameCount()
    data, exists = _setPointData(
        trialRange_init, framecount, firstFrame, array)

    NEXUS.SetTrajectory(vskName, label, data[0], data[1], data[2], exists)


def setTrajectoryFromAcq(NEXUS:ViconNexus.ViconNexus, vskName:str, label:str, acq:btk.btkAcquisition):
    """Set a trajectory ( eq. marker) from an btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): trajectory label ( eq. marker label)
        acq (btk.btkAcquisition): a btk.acquisition instance

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
    """append a modelled marker ( eg HJC) from a btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): modelled marker label ( eq. marker label)
        acq (btk.btkAcquisition): a btk.acquisition instance
        suffix (str,optional): suffix added to the model outputs, defalut set to ""

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
    
    """append a angle from a btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): modelled marker label ( eq. marker label)
        acq (btk.btkAcquisition): a btk.acquisition instance

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
    """append a Force  from an btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): force label
        acq (btk.btkAcquisition): a btk.acquisition instance
        normalizedData (bool,optional): indicate if values are normalized in amplitude.

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
    """append a Moment  from a btk.acquisition instance

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): moment label
        acq (btk.btkAcquisition): a btk.acquisition instance
        normalizedData (bool,optional): indicate if values are normalized in amplitude.

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
    """append a power from a btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): power label
        acq (btk.btkAcquisition): a btk.acquisition instance
        normalizedData (bool,optional): indicate if values are normalized in amplitude.

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
    """append a vicon bone  from a btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        vskName (str): vsk name.
        label (str): bone label
        acq (btk.btkAcquisition): a btk.acquisition instance
        OriginValues (np.ndarray(n,3),Optional[None]): manual assignement of the bone origin.
        manualScale (np.ndarray(1,3),Optional[None]): manual scale.
        suffix (str,optional): suffix added to bone outputs. Defalut set to ""
        existFromPoint (str,optional): btk point label conditioning presence or absence of the bone. Default set to None.

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
    """append btk scalar to the acquisition instance

    Args:
        NEXUS (ViconNexus.ViconNexus): NEXUS handling
        vskName (str): subject name
        groupName (str): data group name
        label (str): scalar label
        nexusTypes (Union[str,list]): nexus data type 
        acq (btk.btkAcquisition): a btk acquisition instance


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

def createGeneralEvents(NEXUS:ViconNexus.ViconNexus,subject:str,acq:btk.btkAcquisition,labels:list):
    """append general events from an btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        subject (str): vsk name.
        labels (list): general event labels
        acq (btk.btkAcquisition): a btk.acquisition instance

    """

    freq = acq.GetPointFrequency()
    events= acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetLabel() in labels:
            NEXUS.CreateAnEvent( subject, "General", ev.GetLabel(), int(ev.GetTime()*freq), 0.0 )

def createEvents(NEXUS:ViconNexus.ViconNexus,subject:str,acq:btk.btkAcquisition,labels:list):
    """append events from an btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        subject (str): subject-vsk name.
        labels (list): event labels
        acq (btk.btkAcquisition): a btk.acquisition instance

    """

    freq = acq.GetPointFrequency()
    events= acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetLabel() in labels:
            NEXUS.CreateAnEvent( subject, ev.GetContext(), ev.GetLabel(), int(ev.GetTime()*freq), 0.0 )

def getForcePlateAssignment(NEXUS:ViconNexus.ViconNexus):
    """get Force plate assignement

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.

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

def appendAnalysisParameters(NEXUS:ViconNexus.ViconNexus, acq:btk.btkAcquisition):
    """append analysis parameter to a btk.acquisition

    Args:
        NEXUS (ViconNexus.ViconNexus): vicon nexus handle.
        acq (btk.btkAcquisition): a btk.acquisition instance

    """
    parameters = btkTools.getAllParamAnalysis(acq)

    for parameter in parameters:
        NEXUS.CreateAnalysisParam(parameter["subject"],parameter["name"],parameter["value"], parameter["unit"])
