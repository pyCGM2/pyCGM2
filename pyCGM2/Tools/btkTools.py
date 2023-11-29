# -*- coding: utf-8 -*-
"""
This module contains convenient functions for working with btk

check out **test_btkTools** for examples

"""

import numpy as np
from scipy import spatial
from  pyCGM2.Math import geometry
import pyCGM2
LOGGER = pyCGM2.LOGGER

import btk

from pyCGM2.External.ktk.kineticstoolkit import timeseries

from typing import List, Tuple, Dict, Optional, Union, Callable

# --- acquisition -----
def smartReader(filename:str, translators:Optional[Dict]=None):
    """
    Read a C3D file using BTK, with optional marker translation.

    Args:
        filename (str): Filename with its path.
        translators (dict, optional): Dictionary for marker translation.

    Returns:
        btk.btkAcquisition: BTK acquisition instance read from the file.
    """
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()

    if translators is not None:
        acq = applyTranslators(acq, translators)

    try:
        # management force plate type 5
        if checkForcePlateExist(acq):
            if "5" in smartGetMetadata(acq, "FORCE_PLATFORM", "TYPE"):
                LOGGER.logger.warning(
                    "[pyCGM2] Type 5 Force plate detected. Due to a BTK known-issue,  type 5 force plate has been corrected as type 2")
                # inelegant code but avoir circular import !!
                from pyCGM2.ForcePlates import forceplates
                forceplates.correctForcePlateType5(acq)
    except:
        LOGGER.logger.error("[pyCGM2] import of Force plate FAILED. you may work with different force plate types")


    # sort events
    sortedEvents(acq)

    # ANALYSIS Section

    md = acq.GetMetaData()
    if "ANALYSIS" not in _getSectionFromMd(md):
        analysis = btk.btkMetaData('ANALYSIS')
        acq.GetMetaData().AppendChild(analysis)
        used = btk.btkMetaData('USED', 0)
        acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(used)

    # deal with user model ouputs made from Nexus
    detectUsermoFlag = False
    labels = ["USERMOS"]
    for it in btk.Iterate(acq.GetPoints()):
        if "USERMO" in it.GetLabel():
            detectUsermoFlag=True
            labels.append(it.GetLabel())
            #print ("user label [%s] found"%(it.GetLabel()))
            description = it.GetDescription()
            labelFromDescription = description[:description.find("[")]
            groupname = smartGetMetadata(acq, "POINT", it.GetLabel())[0]
            groupname = groupname[:groupname.find(":")]
            smartAppendPoint(acq, labelFromDescription +"[" +groupname+"]", it.GetValues(),
                PointType="Scalar", desc= groupname)

            acq.RemovePoint(it.GetLabel())

            # md.FindChild("POINT").value().RemoveChild(it.GetLabel())
            #print ("Processed")

    if detectUsermoFlag:
        for label in labels:
            acq.GetMetaData().FindChild("POINT").value().RemoveChild(label)

    return acq


def smartWriter(acq:btk.btkAcquisition, filename:str, extension:Optional[str]=None):
    """
    Write a BTK acquisition instance to a C3D or other specified format.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance to write.
        filename (str): Filename with its path.
        extension (str, optional): File format extension (default: C3D).

    Returns:
        str: Filename of the written file.
    """
    if extension == "trc":
        filename = filename+ ".trc"

    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(filename)
    writer.Update()

    return filename




def GetMarkerNames(acq:btk.btkAcquisition):
    """
    Retrieves marker names from a BTK acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        List[str]: List of marker names.
    """

    markerNames = []
    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Marker and it.GetLabel()[0] != "*":
            markerNames.append(it.GetLabel())
    return markerNames


def GetAnalogNames(acq:btk.btkAcquisition):
    """
    Retrieves analog signal names from a BTK acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        List[str]: List of analog signal names.
    """

    analogNames = []
    for it in btk.Iterate(acq.GetAnalogs()):
        analogNames.append(it.GetLabel())
    return analogNames


def isGap(acq:btk.btkAcquisition, markerLabel:List[str]):
    """
    Checks if there is a gap (missing data) in the specified marker.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markerLabel (list): Marker labels to check for gaps.

    Returns:
        bool: True if there is a gap in the marker data, False otherwise.
    """
    residualValues = acq.GetPoint(markerLabel).GetResiduals()
    if any(residualValues == -1.0):
        LOGGER.logger.warning(
            "[pyCGM2] gap found for marker (%s)" % (markerLabel))
        return True
    else:
        return False


def findMarkerGap(acq:btk.btkAcquisition):
    """
    Identifies markers with detected gaps in an acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        List[str]: List of markers with gaps.
    """
    gaps = []
    markerNames = GetMarkerNames(acq)
    for marker in markerNames:
        if isGap(acq, marker):
            gaps.append(marker)

    return gaps


def isPointExist(acq:btk.btkAcquisition, label:str, ignorePhantom:bool=True):
    """
    Checks if a point (marker or model output) exists in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Point label to check.
        ignorePhantom (bool, optional): Whether to ignore zero markers. Default is True.

    Returns:
        bool: True if the point exists, False otherwise.
    """

    try:
        acq.GetPoint(label)
    except RuntimeError:
        return False
    else:

        frameNumber = acq.GetPointFrameNumber()
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetLabel() == label:
                LOGGER.logger.debug(
                    "[pyCGM2] marker (%s) is a phantom " % label)
                values = it.GetValues()
                residuals = it.GetResiduals()
                if not ignorePhantom:
                    if all(residuals == -1):
                        return False
                    else:
                        return True
                else:
                    return True


def isPointsExist(acq:btk.btkAcquisition, labels:List[str], ignorePhantom:bool=True):
    """
    Checks if a list of points exist in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        labels (list): List of point labels to check.
        ignorePhantom (bool, optional): Whether to ignore zero markers. Default is True.

    Returns:
        bool: True if all points exist, False otherwise.
    """
    for label in labels:
        if not isPointExist(acq, label, ignorePhantom=ignorePhantom):
            LOGGER.logger.debug("[pyCGM2] markers (%s) doesn't exist" % label)
            return False
    return True


def smartAppendPoint(acq:btk.btkAcquisition, label:str, values:np.ndarray, PointType:str="Marker", desc:str="", residuals:Optional[np.ndarray]=None):
    """
    Appends or updates a point in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Label of the point to append or update.
        values (numpy.ndarray): Array of point values.
        PointType (str, optional): Type of the point (e.g., 'Marker', 'Angle', 'Force'). Default is 'Marker'.
        desc (str, optional): Description of the point. Default is an empty string.
        residuals (numpy.ndarray, optional): Array of point residual values.

    """

    if PointType == "Marker":
        PointType = btk.btkPoint.Marker
    elif PointType == "Angle":
        PointType = btk.btkPoint.Angle
    elif PointType == "Force":
        PointType = btk.btkPoint.Force
    elif PointType == "Moment":
        PointType = btk.btkPoint.Moment
    elif PointType == "Power":
        PointType = btk.btkPoint.Power
    elif PointType == "Scalar":
        PointType = btk.btkPoint.Scalar
    elif PointType == "Reaction":
        PointType = btk.btkPoint.Reaction
    else:
        raise Exception ("[pyCGM2] point type unknown. ")



    LOGGER.logger.debug("new point (%s) added to the c3d" % label)

    # TODO : deal with values containing only on line

    values = np.nan_to_num(values)

    if residuals is None:
        residuals = np.zeros((values.shape[0]))
        for i in range(0, values.shape[0]):
            if np.all(values[i, :] == np.zeros((3))):
                residuals[i] = -1
            else:
                residuals[i] = 0

    if isPointExist(acq, label):
        acq.GetPoint(label).SetValues(values)
        acq.GetPoint(label).SetDescription(desc)
        acq.GetPoint(label).SetType(PointType)
        acq.GetPoint(label).SetResiduals(residuals)

    else:
        new_btkPoint = btk.btkPoint(label, acq.GetPointFrameNumber())
        new_btkPoint.SetValues(values)
        new_btkPoint.SetDescription(desc)
        new_btkPoint.SetType(PointType)
        new_btkPoint.SetResiduals(residuals)
        acq.AppendPoint(new_btkPoint)


def clearPoints(acq:btk.btkAcquisition, pointlabelList:List[str]):
    """
    Removes specified points from the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        pointlabelList (list): List of point labels to remove.

    """

    i = acq.GetPoints().Begin()
    while i != acq.GetPoints().End():
        label = i.value().GetLabel()
        if label not in pointlabelList:
            i = acq.RemovePoint(i)
            LOGGER.logger.debug(label + " removed")
        else:
            i.incr()
            LOGGER.logger.debug(label + " found")

    return acq


def keepAndDeleteOtherPoints(acq:btk.btkAcquisition, pointToKeep:List[str]):
    """
    Removes all points from the acquisition except the specified ones.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        pointToKeep (list): List of point labels to keep.

    """

    for it in btk.Iterate(acq.GetPoints()):
        if it.GetLabel() not in pointToKeep:
            acq.RemovePoint(it.GetLabel())


def isPhantom(acq:btk.btkAcquisition, label:str):
    """
    Checks if a point is a phantom (i.e., zero point).

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Label of the point to check.

    Returns:
        bool: True if the point is a phantom, False otherwise.
    """
    residuals = acq.GetPoint(label).GetResiduals()
    return False if all(residuals == -1) else True


def getValidFrames(acq:btk.btkAcquisition, markerLabels:List[str], frameBounds:Optional[Tuple[int,int]]=None):
    """
    Gets valid frames of markers within the specified frame boundaries.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markerLabels (list): List of marker labels to check.
        frameBounds (tuple, optional): Frame boundaries as a tuple (start, end).

    Returns:
        List[bool]: List indicating valid frames (True) or invalid frames (False).
    """

    ff = acq.GetFirstFrame()

    flag = []
    for i in range(0, acq.GetPointFrameNumber()):
        flag_index = True
        for marker in markerLabels:
            if not acq.GetPoint(marker).GetResidual(i) >= 0:
                flag_index = False

        flag.append(flag_index)

    if frameBounds is not None:
        begin = frameBounds[0] - ff
        end = frameBounds[1] - ff
        flag[0:begin] = [False]*len(flag[0:begin])
        flag[end:] = [False]*len(flag[end:])
        flag[begin:end+1] = [True]*len(flag[begin:end+1])

    return flag


def getFrameBoundaries(acq:btk.btkAcquisition, markerLabels:List[str]):
    """
    Gets frame boundaries from a list of markers.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markerLabels (list): List of marker labels.

    Returns:
        Tuple[int, int]: Tuple of first and last valid frame numbers.
    """

    flag = getValidFrames(acq, markerLabels)

    ff = acq.GetFirstFrame()

    firstValidFrame = flag.index(True)+ff
    lastValidFrame = ff+len(flag) - flag[::-1].index(True) - 1

    return firstValidFrame, lastValidFrame


def checkGap(acq:btk.btkAcquisition, markerList:List[str], frameBounds:Optional[Tuple[int,int]]=None):
    """
    Checks for gaps in marker data within the specified frame bounds.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markerList (list): List of marker labels.
        frameBounds (tuple, optional): Frame boundaries as a tuple (start, end).

    Returns:
        bool: True if there are gaps, False otherwise.
    """
    ff = acq.GetFirstFrame()
    flag = False
    for m in markerList:
        residualValues = acq.GetPoint(m).GetResiduals()
        if frameBounds is not None:
            begin = frameBounds[0] - ff
            end = frameBounds[1] - ff
            if any(residualValues[begin:end] == -1.0):
                LOGGER.logger.warning(
                    "[pyCGM2] gap found for marker (%s)" % (m))
                flag = True
        else:
            if any(residualValues == -1.0):
                flag = True

    return flag


def applyOnValidFrames(acq:btk.btkAcquisition, validFrames:List[int]):
    """
    Sets all points to zero if the frame is not valid.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        validFrames (list): List of frames with 1 (valid) or 0 (invalid).
    """

    frameNumber = acq.GetPointFrameNumber()
    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() in [btk.btkPoint.Angle, btk.btkPoint.Force, btk.btkPoint.Moment, btk.btkPoint.Power]:
            for i in range(0, frameNumber):
                if not validFrames[i]:
                    it.SetResidual(i, -1)
                    it.SetValue(i, 0, 0)
                    it.SetValue(i, 1, 0)
                    it.SetValue(i, 2, 0)


def findValidFrames(acq:btk.btkAcquisition, markerLabels:List[str]):
    """
    Finds valid frames to process based on marker data.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markerLabels (list): List of marker labels.

    Returns:
        Tuple[List[int], int, int]: Tuple containing a list of valid frames, first valid frame, and last valid frame.
    """

    flag = []
    for i in range(0, acq.GetPointFrameNumber()):
        pointFlag = []
        for marker in markerLabels:
            if acq.GetPoint(marker).GetResidual(i) >= 0:
                pointFlag.append(1)
            else:
                pointFlag.append(0)

        if all(pointFlag) == 1:
            flag.append(1)
        else:
            flag.append(0)

    firstValidFrame = flag.index(1)
    lastValidFrame = len(flag) - flag[::-1].index(1) - 1

    return flag, firstValidFrame, lastValidFrame


def applyValidFramesOnOutput(acq:btk.btkAcquisition, validFrames:List[int]):
    """
    Sets model outputs to zero if not a valid frame.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        validFrames (list): List of valid frames.
    """

    validFrames = np.asarray(validFrames)

    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() in [btk.btkPoint.Angle, btk.btkPoint.Force, btk.btkPoint.Moment, btk.btkPoint.Power]:
            values = it.GetValues()
            for i in range(0, 3):
                values[:, i] = values[:, i] * validFrames
            it.SetValues(values)


def checkMultipleSubject(acq:btk.btkAcquisition):
    """
    Checks if multiple subjects are detected in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Raises:
        Exception: If the acquisition contains data from multiple subjects.
    """
    if acq.GetPoint(0).GetLabel().count(":"):
        raise Exception(
            "[pyCGM2] Your input static c3d was saved with two activate subject. Re-save it with only one before pyCGM2 calculation")


# --- Model -----

def applyTranslators(acq:btk.btkAcquisition, translators:Dict):
    """
    Renames markers based on the provided translators.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        translators (dict): Dictionary mapping original marker names to new names.

    Returns:
        btk.btkAcquisition: Acquisition instance with renamed markers.
    """
    acqClone = btk.btkAcquisition.Clone(acq)

    if translators is not None:
        modifiedMarkerList = []

        # gather all labels
        for it in translators.items():
            wantedLabel, initialLabel = it[0], it[1]
            if wantedLabel != initialLabel and initialLabel != "None":
                modifiedMarkerList.append(it[0])
                modifiedMarkerList.append(it[1])

        # Remove Modified Markers from Clone
        for point in btk.Iterate(acq.GetPoints()):
            if point.GetType() == btk.btkPoint.Marker:
                label = point.GetLabel()
                if label in modifiedMarkerList:
                    acqClone.RemovePoint(label)
                    LOGGER.logger.debug(
                        "point (%s) remove in the clone acq  " % ((label)))

        # Add Modify markers to clone
        for it in translators.items():
            wantedLabel, initialLabel = it[0], it[1]
            if initialLabel != "None":
                if isPointExist(acq, wantedLabel):
                    smartAppendPoint(acqClone, (wantedLabel+"_origin"), acq.GetPoint(
                        wantedLabel).GetValues(), PointType="Marker")  # modified marker
                    LOGGER.logger.info(
                        "wantedLabel (%s)_origin created" % ((wantedLabel)))
                if isPointExist(acq, initialLabel):
                    if initialLabel in translators.keys():
                        if translators[initialLabel] == "None":
                            LOGGER.logger.info("Initial point (%s)and (%s) point to similar values" % (
                                (initialLabel), (wantedLabel)))
                            smartAppendPoint(acqClone, (wantedLabel), acq.GetPoint(
                                initialLabel).GetValues(), PointType="Marker")
                            smartAppendPoint(acqClone, (initialLabel), acq.GetPoint(
                                initialLabel).GetValues(), PointType="Marker")  # keep initial marker
                        elif translators[initialLabel] == wantedLabel:
                            LOGGER.logger.info("Initial point (%s) swaped with (%s)" % (
                                (initialLabel), (wantedLabel)))
                            initialValue = acq.GetPoint(
                                initialLabel).GetValues()
                            wantedlValue = acq.GetPoint(
                                wantedLabel).GetValues()
                            smartAppendPoint(
                                acqClone, (wantedLabel), initialValue, PointType="Marker")
                            smartAppendPoint(
                                acqClone, ("TMP"), wantedlValue, PointType="Marker")
                            acqClone.GetPoint("TMP").SetLabel(initialLabel)
                            acqClone.RemovePoint(wantedLabel+"_origin")
                            acqClone.RemovePoint(initialLabel+"_origin")
                    else:
                        LOGGER.logger.info("Initial point (%s) renamed (%s)  added into the c3d" % (
                            (initialLabel), (wantedLabel)))
                        smartAppendPoint(acqClone, (wantedLabel), acq.GetPoint(
                            initialLabel).GetValues(), PointType="Marker")
                        smartAppendPoint(acqClone, (initialLabel), acq.GetPoint(
                            initialLabel).GetValues(), PointType="Marker")

                else:
                    LOGGER.logger.info(
                        "initialLabel (%s) doesn t exist  " % ((initialLabel)))
                    #raise Exception ("your translators are badly configured")

    return acqClone


def checkMarkers(acq:btk.btkAcquisition, markerList:List[str]):
    """
    Checks the presence of specified markers in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markerList (list): List of marker names to check.

    Raises:
        Exception: If any of the specified markers are not found.
    """
    for m in markerList:
        if not isPointExist(acq, m):
            raise Exception("[pyCGM2] markers %s not found" % m)


def clearEvents(acq:btk.btkAcquisition, labels:List[str]):
    """
    Removes events based on their labels.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        labels (list): List of event labels to be removed.
    """

    events = acq.GetEvents()
    newEvents = btk.btkEventCollection()
    for ev in btk.Iterate(events):
        if ev.GetLabel() not in labels:
            newEvents.InsertItem(ev)

    acq.ClearEvents()
    acq.SetEvents(newEvents)


def deleteContextEvents(acq:btk.btkAcquisition, context:str):
    """
    Removes events with a specified context (e.g., Left, Right, General).

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        context (str): Context of the events to be removed.
    """
    events = acq.GetEvents()
    newEvents = btk.btkEventCollection()
    for ev in btk.Iterate(events):
        if ev.GetContext() != context:
            newEvents.InsertItem(ev)

    acq.ClearEvents()
    acq.SetEvents(newEvents)


def modifyEventSubject(acq:btk.btkAcquisition, newSubjectlabel:str):
    """
    Updates the subject name for all events in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        newSubjectlabel (str): New subject name to be applied.
    """

    # events
    nEvents = acq.GetEventNumber()
    if nEvents >= 1:
        for i in range(0, nEvents):
            acq.GetEvent(i).SetSubject(newSubjectlabel)
    return acq


def modifySubject(acq:btk.btkAcquisition, newSubjectlabel:str):
    """
    Updates the subject name in the acquisition's metadata.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        newSubjectlabel (str): New subject name to be set.
    """
    acq.GetMetaData().FindChild("SUBJECTS").value().FindChild(
        "NAMES").value().GetInfo().SetValue(0, (newSubjectlabel))


def getNumberOfModelOutputs(acq:btk.btkAcquisition):
    """
    Returns the count of different types of model outputs in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        tuple: Counts of angles, forces, moments, and powers.
    """
    n_angles = 0
    n_forces = 0
    n_moments = 0
    n_powers = 0

    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Angle:
            n_angles += 1

        if it.GetType() == btk.btkPoint.Force:
            n_forces += 1

        if it.GetType() == btk.btkPoint.Moment:
            n_moments += 1

        if it.GetType() == btk.btkPoint.Power:
            n_powers += 1

    return n_angles, n_forces, n_moments, n_powers

# --- metadata -----


def hasChild(md:btk.btkMetaData, mdLabel:str):
    """
    Checks if the specified metadata child exists.

    Args:
        md (btk.btkMetaData): Metadata instance.
        mdLabel (str): Label of the metadata child to check.

    Returns:
        btk.btkMetaData: Child metadata if exists, else None.
    """
    outMd = None
    for itMd in btk.Iterate(md):
        if itMd.GetLabel() == mdLabel:
            outMd = itMd
            break
    return outMd


def getVisibleMarkersAtFrame(acq:btk.btkAcquisition, markers:List[str], index:int):
    """
    Returns markers that are visible at a specific frame.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markers (list): List of marker labels.
        index (int): Frame index.

    Returns:
        list: List of visible markers at the specified frame.
    """
    visibleMarkers = []
    for marker in markers:
        if acq.GetPoint(marker).GetResidual(index) != -1:
            visibleMarkers.append(marker)
    return visibleMarkers


def isAnalogExist(acq:btk.btkAcquisition, label:str):
    """
    Checks if an analog label exists in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Analog label to check.

    Returns:
        bool: True if the analog label exists, False otherwise.
    """
    #TODO : replace by btkIterate
    i = acq.GetAnalogs().Begin()
    while i != acq.GetAnalogs().End():
        if i.value().GetLabel() == label:
            flag = True
            break
        else:
            i.incr()
            flag = False

    if flag:
        return True
    else:
        return False


def smartAppendAnalog(acq:btk.btkAcquisition, label:str, values:np.ndarray, desc:str=""):
    """
    Appends or updates an analog output in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Analog label.
        values (np.ndarray): Analog values to append or update.
        desc (str, optional): Description of the analog. Defaults to an empty string.
    """

    if isAnalogExist(acq, label):
        acq.GetAnalog(label).SetValues(values)
        acq.GetAnalog(label).SetDescription(desc)
        #acq.GetAnalog(label).SetType(PointType)

    else:
        newAnalog = btk.btkAnalog(acq.GetAnalogFrameNumber())
        newAnalog.SetValues(values)
        newAnalog.SetLabel(label)
        acq.AppendAnalog(newAnalog)


def markerUnitConverter(acq:btk.btkAcquisition, unitOffset:float):
    """
    Applies an offset to convert marker units.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        unitOffset (float): Offset value for unit conversion.
    """
    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Marker:
            values = it.GetValues()
            it.SetValues(values*unitOffset)


def constructMarker(acq:btk.btkAcquisition, label:str, markers:List[str], numpyMethod:callable=np.mean, desc:str=""):
    """
    Constructs a new marker from existing markers using a specified numpy method.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Label for the new marker.
        markers (list): List of existing marker labels to be used in construction.
        numpyMethod (callable, optional): Numpy function to handle marker data. Defaults to np.mean.
        desc (str, optional): Description of the new marker. Defaults to an empty string.
    """
    nFrames = acq.GetPointFrameNumber()

    x = np.zeros((nFrames, len(markers)))
    y = np.zeros((nFrames, len(markers)))
    z = np.zeros((nFrames, len(markers)))

    i = 0
    for marker in markers:
        x[:, i] = acq.GetPoint(marker).GetValues()[:, 0]
        y[:, i] = acq.GetPoint(marker).GetValues()[:, 1]
        z[:, i] = acq.GetPoint(marker).GetValues()[:, 2]
        i += 1

    values = np.zeros((nFrames, 3))
    values[:, 0] = numpyMethod(x, axis=1)
    values[:, 1] = numpyMethod(y, axis=1)
    values[:, 2] = numpyMethod(z, axis=1)
    smartAppendPoint(acq, label, values, desc=desc)


def constructPhantom(acq:btk.btkAcquisition, label:str, desc:str=""):
    """
    Constructs a phantom marker (marker with zero values).

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Label for the phantom marker.
        desc (str, optional): Description of the phantom marker. Defaults to an empty string.
    """
    nFrames = acq.GetPointFrameNumber()
    values = np.zeros((nFrames, 3))
    smartAppendPoint(acq, label, values, desc=desc,
                     residuals=np.ones((nFrames, 1))*-1.0)


def createPhantoms(acq:btk.btkAcquisition, markerLabels:List[str]):
    """
    Constructs phantom markers for a list of specified labels.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markerLabels (list): List of marker labels for which phantoms will be created.

    Returns:
        tuple: A tuple containing lists of actual and phantom markers.
    """
    phantom_markers = []
    actual_markers = []
    for label in markerLabels:
        if not isPointExist(acq, label):
            constructPhantom(acq, label, desc="phantom")
            phantom_markers.append(label)
        else:
            actual_markers.append(label)
    return actual_markers, phantom_markers


def getNumberOfForcePlate(acq:btk.btkAcquisition):
    """
    Returns the number of force plates in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        int: The number of force plates.
    """
    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(acq)
    pfc = pfe.GetOutput()
    pfc.Update()

    return pfc.GetItemNumber()


def getStartEndEvents(acq:btk.btkAcquisition, context:str, startLabel:str="Start", endLabel:str="End"):
    """
    Retrieves the frame numbers of start and end events based on their context and labels.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        context (str): Context of the events.
        startLabel (str, optional): Label of the start event. Defaults to "Start".
        endLabel (str, optional): Label of the end event. Defaults to "End".

    Returns:
        tuple: Frame numbers of the start and end events, if they exist.
    """
    events = acq.GetEvents()

    start = []
    end = []
    for ev in btk.Iterate(events):
        if ev.GetContext() == context and ev.GetLabel() == startLabel:
            start.append(ev.GetFrame())
        if ev.GetContext() == context and ev.GetLabel() == endLabel:
            end.append(ev.GetFrame())
    if start == [] or end == []:
        return None, None

    if len(start) > 1 or len(end) > 1:
        raise("[pyCGM2]: You can t have multiple Start and End events")
    elif end < start:
        raise("[pyCGM2]: wrong order ( start<end)")
    else:
        return start[0], end[0]


def _getSectionFromMd(md:btk.btkMetaData):
    """
    Retrieves a list of child sections from metadata.

    Args:
        md (btk.btkMetaData): BTK metadata instance.

    Returns:
        list: List of child section labels in the metadata.
    """
    md_sections = []
    for i in range(0, md.GetChildNumber()):
        md_sections.append(md.GetChild(i).GetLabel())
    return md_sections


def changeSubjectName(acq:btk.btkAcquisition, subjectName:str):
    """
    Changes the subject name in the acquisition's metadata.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        subjectName (str): New name for the subject.
    """

    # change subject name in the metadata
    md = acq.GetMetaData()

    if "SUBJECTS" in _getSectionFromMd(md):
        subjectMd = acq.GetMetaData().FindChild("SUBJECTS").value()
        if "NAMES" in _getSectionFromMd(subjectMd):
            subjectMd.FindChild("NAMES").value(
            ).GetInfo().SetValue(0, subjectName)

        if "USES_PREFIXES" not in _getSectionFromMd(subjectMd):
            btk.btkMetaDataCreateChild(subjectMd, "USES_PREFIXES", 0)

    if "ANALYSIS" in _getSectionFromMd(md):
        analysisMd = acq.GetMetaData().FindChild("ANALYSIS").value()
        if "SUBJECTS" in _getSectionFromMd(analysisMd):
            anaSubMdi = analysisMd.FindChild("SUBJECTS").value().GetInfo()
            for i in range(0, anaSubMdi.GetDimension(1)):
                anaSubMdi.SetValue(i, subjectName)

    events = acq.GetEvents()
    for ev in btk.Iterate(events):
        ev.SetSubject(subjectName)

    # Do not work
    # eventMd =  btkAcq.GetMetaData().FindChild("EVENT").value()
    # eventMdi = eventMd.FindChild("SUBJECTS").value().GetInfo()
    # for i in range(0,eventMdi.GetDimension(1) ):
    #     eventMdi.SetValue(i,"TEST")

    return acq


def smartGetMetadata(acq:btk.btkAcquisition, firstLevel:str, secondLevel:str, returnType:str="String"):
    """
    Retrieves metadata information based on specified levels.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        firstLevel (str): First level metadata label.
        secondLevel (str): Second level metadata label.
        returnType (str, optional): Type of the returned metadata (String, Integer, Double). Defaults to "String".

    Returns:
        str or int or float: Metadata information based on the returnType.
    """
    md = acq.GetMetaData()
    if secondLevel is not None:
        if firstLevel in _getSectionFromMd(md):
            firstMd = acq.GetMetaData().FindChild(firstLevel).value()
            if secondLevel in _getSectionFromMd(firstMd):
                info = firstMd.FindChild(secondLevel).value().GetInfo()
                if returnType == "String":
                    return info.ToString()
                elif returnType == "Integer":
                    return info.ToInt()
                elif returnType == "Double":
                    return info.ToDouble()
    else:
        if firstLevel in _getSectionFromMd(md):
            info = md.FindChild(firstLevel).value().GetInfo()
            if returnType == "String":
                return info.ToString()
            elif returnType == "Integer":
                return info.ToInt()
            elif returnType == "Double":
                return info.ToDouble()


def smartSetMetadata(acq:btk.btkAcquisition, firstLevel:str, secondLevel:str, index:int, value:str):
    """
    Sets a value in the acquisition's metadata.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        firstLevel (str): First level metadata label.
        secondLevel (str): Second level metadata label.
        index (int): Index at which to set the value.
        value (str): Value to set in the metadata.
    """
    md = acq.GetMetaData()
    if firstLevel in _getSectionFromMd(md):
        firstMd = acq.GetMetaData().FindChild(firstLevel).value()
        if secondLevel in _getSectionFromMd(firstMd):
            return firstMd.FindChild(secondLevel).value().GetInfo().SetValue(index, value)


def checkMetadata(acq:btk.btkAcquisition, firstLevel:str, secondLevel:str):
    """
    Checks the existence of specified metadata.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        firstLevel (str): First level metadata label.
        secondLevel (str): Second level metadata label.

    Returns:
        bool: True if the metadata exists, False otherwise.
    """
    md = acq.GetMetaData()
    flag = False
    if secondLevel is not None:
        if firstLevel in _getSectionFromMd(md):
            firstMd = acq.GetMetaData().FindChild(firstLevel).value()
            if secondLevel in _getSectionFromMd(firstMd):
                flag = True
    else:
        if firstLevel in _getSectionFromMd(md):
            flag = True

    return flag


def checkForcePlateExist(acq:btk.btkAcquisition):
    """
    Checks if force plates exist in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        bool: True if force plates exist, False otherwise.
    """
    if checkMetadata(acq, "FORCE_PLATFORM", "USED"):
        if smartGetMetadata(acq, "FORCE_PLATFORM", "USED")[0] != "0":
            return True
        else:
            return False
    else:
        return False


def sortedEvents(acq:btk.btkAcquisition):
    """
    Sorts the events in the acquisition based on their time of occurrence.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
    """
    evs = acq.GetEvents()

    contextLst = []  # recuperation de tous les contextes
    for it in btk.Iterate(evs):
        if it.GetContext() not in contextLst:
            contextLst.append(it.GetContext())

    valueFrame = []  # recuperation de toutes les frames SANS doublons
    for it in btk.Iterate(evs):
        if it.GetFrame() not in valueFrame:
            valueFrame.append(it.GetFrame())
    valueFrame.sort()  # trie

    events = []
    for frame in valueFrame:
        for it in btk.Iterate(evs):
            if it.GetFrame() == frame:
                events.append(it)

    newEvents = btk.btkEventCollection()
    for ev in events:
        newEvents.InsertItem(ev)

    acq.ClearEvents()
    acq.SetEvents(newEvents)


def buildTrials(dataPath:str, filenames:List[str]):
    """
    Constructs BTK acquisitions from a list of filenames located in a specified directory.

    Args:
        dataPath (str): Path to the directory containing the C3D files.
        filenames (List[str]): List of filenames of the C3D files to be processed.

    Returns:
        List[btk.btkAcquisition]: List of BTK acquisition objects created from the specified files.
    """

    acqs = []
    acqFilenames = []
    for filename in filenames:
        LOGGER.logger.debug(dataPath)
        LOGGER.logger.debug(filename)
        acq = smartReader(dataPath + filename)
        sortedEvents(acq)

        acqs.append(acq)
        acqFilenames.append(filename)

    return acqs, acqFilenames


def isKineticFlag(acq:btk.btkAcquisition):
    """
    Checks if kinetic data (force plate events) are present in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        bool: True if kinetic data are present, False otherwise.
    """

    kineticEvent_frames = []
    kineticEvent_frames_left = []
    kineticEvent_frames_right = []

    events = acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetContext() == "General":
            if ev.GetLabel() in ["Left-FP", "Right-FP"]:
                kineticEvent_frames.append(ev.GetFrame())
            if ev.GetLabel() in ["Left-FP"]:
                kineticEvent_frames_left.append(ev.GetFrame())
            if ev.GetLabel() in ["Right-FP"]:
                kineticEvent_frames_right.append(ev.GetFrame())

    if kineticEvent_frames == []:
        return False, 0, 0, 0
    else:
        return True, kineticEvent_frames, kineticEvent_frames_left, kineticEvent_frames_right


def automaticKineticDetection(dataPath:str, filenames:List[str], acqs:Optional[List[btk.btkAcquisition]]=None):
    """
    Automatically detects and processes acquisitions with kinetic data.

    Args:
        dataPath (str): Path to the data directory.
        filenames (List[str]): List of filenames to check for kinetic data.
        acqs (Optional[List[btk.btkAcquisition]], optional): List of preloaded acquisitions. Defaults to None.

    Returns:
        Tuple[List[btk.btkAcquisition], List[str], bool]: Tuple containing a list of acquisitions with kinetic data,
        their filenames, and a flag indicating the presence of kinetic data.
    """

    kineticAcqs = []
    kineticFilenames = []

    i = 0
    for filename in filenames:
        if filename in kineticFilenames:
            LOGGER.logger.debug(
                "[pyCGM2] : filename %s duplicated in the input list" % (filename))
        else:
            if acqs is None:
                acq = smartReader(dataPath + filename)

            else:
                acq = acqs[i]

            sortedEvents(acq)
            flag_kinetics, times, times_l, times_r = isKineticFlag(acq)

            if flag_kinetics:
                kineticFilenames.append(filename)
                kineticAcqs.append(acq)
    i += 1

    kineticAcqs = None if kineticAcqs == [] else kineticAcqs
    flag_kinetics = False if kineticAcqs == [] else True

    return kineticAcqs, kineticFilenames, flag_kinetics


def getForcePlateWrench(acq:btk.btkAcquisition, fpIndex:int=None):
    """
    Retrieves the ground reaction wrenches for force plates in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        fpIndex (int, optional): Index of the specific force plate. If None, all force plates are considered. Defaults to None.

    Returns:
        btk.btkWrenchCollection: Collection of ground reaction wrenches for the force plates.
    """
    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    grwf = btk.btkGroundReactionWrenchFilter()
    pfe.SetInput(acq)
    pfc = pfe.GetOutput()
    grwf.SetInput(pfc)
    grwc = grwf.GetOutput()
    grwc.Update()

    if fpIndex is not None:
        return grwc.GetItem(fpIndex-1)
    else:
        return grwc


def applyRotation(acq:btk.btkAcquisition, markers:List[str], globalFrameOrientation:str, forwardProgression:bool):
    """
    Applies a rotation to the specified markers in the acquisition based on the global frame orientation.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markers (List[str]): List of marker names to apply rotation to.
        globalFrameOrientation (str): Orientation of the global frame (e.g., 'XYZ' for X:longitudinal, Y:transversal, Z:vertical).
        forwardProgression (bool): Indicates if progression is along the positive axis of the frame.

    Returns:
        None: The function directly modifies the acquisition object.
    """
    if globalFrameOrientation == "XYZ":
        rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif globalFrameOrientation == "YXZ":
        rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    else:
        raise Exception("[pyCGM2] code cannot work with Z as non-normal axis")

    for marker in markers:
        values = acq.GetPoint(marker).GetValues()

        valuesRot = np.zeros((acq.GetPointFrameNumber(), 3))
        for i in range(0, acq.GetPointFrameNumber()):
            valuesRot[i, :] = np.dot(rot, values[i, :])
        if not forwardProgression:
            valuesRot[i, :] = np.dot(
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), valuesRot[i, :])

        acq.GetPoint(marker).SetValues(valuesRot)


def smartGetEvents(acq:btk.btkAcquisition, label:str, context:str):
    """
    Retrieves events from the acquisition based on label and context.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Label of the events to retrieve.
        context (str): Context of the events to retrieve.

    Returns:
        List[int]: List of frames where the specified events occur.
    """
    evs = acq.GetEvents()

    out = []
    for it in btk.Iterate(evs):
        if it.GetContext() == context and it.GetLabel() == label:
            out.append(it.GetFrame())

    return out

def isEventExist(acq:btk.btkAcquisition, label:str, context:str):
    """
    Checks if an event with the specified label and context exists in the acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): The label of the event to check for.
        context (str): The context of the event to check for.

    Returns:
        bool: True if the event exists, False otherwise.
    """
    evs = acq.GetEvents()
    out = False
    for it in btk.Iterate(evs):
        if it.GetContext() == context and it.GetLabel() == label:
            out=True
    return out

def renameEvent(acq:btk.btkAcquisition, label:str, context:str, newlabel:str, newcontext:str):
    """
    Renames an existing event in a BTK acquisition.

    Args:
        acq (btk.btkAcquisition): The BTK acquisition instance containing the event.
        label (str): The current label of the event to be renamed.
        context (str): The current context of the event to be renamed.
        newlabel (str): The new label for the event.
        newcontext (str): The new context for the event.

    This function searches for an event in the given acquisition that matches the specified label and context.
    If such an event is found, its label and context are updated to the new values provided.
    """
    evs = acq.GetEvents()
    for it in btk.Iterate(evs):
        if it.GetContext() == context and it.GetLabel() == label:
            it.SetLabel(newlabel)
            it.SetContext(newcontext)
    

def cleanAcq(acq:btk.btkAcquisition):
    """
    Cleans a BTK acquisition by removing points with zero values across all frames.

    Args:
        acq (btk.btkAcquisition): The BTK acquisition instance to be cleaned.

    This function iterates through all points in the given BTK acquisition. If a point (such as a marker, angle, 
    force, moment, or power) has zero values for all frames, it is removed from the acquisition. This is useful for 
    tidying up motion capture data, especially when certain points are not relevant or are placeholders with no actual data.
    """

    nframes = acq.GetPointFrameNumber()

    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() in [btk.btkPoint.Marker,  
                            btk.btkPoint.Angle, 
                            btk.btkPoint.Force, 
                            btk.btkPoint.Moment, 
                            btk.btkPoint.Power]:
            values = it.GetValues()

            if np.all(values == np.zeros(3)) or np.all(values == np.array([180, 0, 180])):
                LOGGER.logger.debug(
                    "point %s remove from acquisition" % (it.GetLabel()))
                acq.RemovePoint(it.GetLabel())


def smartCreateEvent(acq:btk.btkAcquisition, label:str, context:str, frame:int, type:str="Automatic", subject:str="", desc:str=""):
    """
    Creates a new event in the BTK acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Label of the event.
        context (str): Context of the event (e.g., 'Left', 'Right').
        frame (int): Frame number where the event occurs.
        type (str, optional): Type of the event (e.g., 'Automatic', 'Manual'). Defaults to 'Automatic'.
        subject (str, optional): Name of the subject associated with the event. Defaults to an empty string.
        desc (str, optional): Description of the event. Defaults to an empty string.

    Returns:
        None: The event is added directly to the acquisition object.
    """

    if type=="Automatic":
        type = btk.btkEvent.Automatic
    if type=="Manual":
        type = btk.btkEvent.Manual
    if type=="Unknown":
        type = btk.btkEvent.Unknown
    if type=="FromForcePlatform":
        type = btk.btkEvent.FromForcePlatform


    time = frame / acq.GetPointFrequency()
    ev = btk.btkEvent(label, time, context, type, subject, desc)
    ev.SetFrame(int(frame))
    acq.AppendEvent(ev)


def smartAppendParamAnalysis(acq:btk.btkAcquisition, name:str, eventcontext:str, value:float, description:str="", subject:str="", unit:str=""):
    """
    Appends a new analysis parameter to the BTK acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        name (str): Name of the analysis parameter.
        eventcontext (str): Context of the event related to the analysis parameter.
        value (float): Value of the analysis parameter.
        description (str, optional): Description of the analysis parameter. Defaults to an empty string.
        subject (str, optional): Subject associated with the analysis parameter. Defaults to an empty string.
        unit (str, optional): Unit of the analysis parameter. Defaults to an empty string.

    Returns:
        None: The analysis parameter is appended directly to the acquisition object.
    """

    used = smartGetMetadata(acq, "ANALYSIS", "USED", returnType="Integer")

    index = None
    # check if param exist in the current parameters
    if used[0] != 0:
        itemNumber = len(smartGetMetadata(acq, "ANALYSIS", "NAMES"))

        names2 = [it.strip()
                  for it in smartGetMetadata(acq, "ANALYSIS", "NAMES")]
        contexts2 = [it.strip()
                     for it in smartGetMetadata(acq, "ANALYSIS", "CONTEXTS")]
        subjects2 = [it.strip()
                     for it in smartGetMetadata(acq, "ANALYSIS", "SUBJECTS")]

        for i in range(0, itemNumber):
            if name == names2[i] and eventcontext == contexts2[i] and subject == subjects2[i]:
                LOGGER.logger.debug(
                    "analysis parameter detected in the current parameters")
                index = i
                break

    if index is not None:
        # parameter detected  = amend from index
        smartSetMetadata(acq, "ANALYSIS", "VALUES", index, value)
        smartSetMetadata(acq, "ANALYSIS",
                         "DESCRIPTIONS", index, description)
        smartSetMetadata(acq, "ANALYSIS", "UNITS", index, unit)

    else:
        # parameter not detected
        if checkMetadata(acq, "ANALYSIS", "NAMES"):

            names = [it.strip() for it in smartGetMetadata(
                acq, "ANALYSIS", "NAMES")]  # remove space
            names.append(name)

            acq.GetMetaData().FindChild("ANALYSIS").value().RemoveChild("NAMES")
            newMd = btk.btkMetaData('NAMES', btk.btkStringArray(names))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)
        else:
            names = [name]
            newMd = btk.btkMetaData('NAMES', btk.btkStringArray(names))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)

        if checkMetadata(acq, "ANALYSIS", "DESCRIPTIONS"):
            descriptions = [it for it in smartGetMetadata(
                acq, "ANALYSIS", "DESCRIPTIONS")]

            descriptions.append(description)

            acq.GetMetaData().FindChild("ANALYSIS").value().RemoveChild("DESCRIPTIONS")
            newMd = btk.btkMetaData(
                'DESCRIPTIONS', btk.btkStringArray(descriptions))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)

        else:
            descriptions = [description]
            newMd = btk.btkMetaData(
                'DESCRIPTIONS', btk.btkStringArray(descriptions))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)

        if checkMetadata(acq, "ANALYSIS", "SUBJECTS"):
            subjects = [it for it in smartGetMetadata(
                acq, "ANALYSIS", "SUBJECTS")]

            subjects.append(subject)

            acq.GetMetaData().FindChild("ANALYSIS").value().RemoveChild("SUBJECTS")
            newMd = btk.btkMetaData('SUBJECTS', btk.btkStringArray(subjects))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)
        else:
            subjects = [subject]
            newMd = btk.btkMetaData('SUBJECTS', btk.btkStringArray(subjects))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)

        if checkMetadata(acq, "ANALYSIS", "CONTEXTS"):
            contexts = [it for it in smartGetMetadata(
                acq, "ANALYSIS", "CONTEXTS")]

            contexts.append(eventcontext)

            acq.GetMetaData().FindChild("ANALYSIS").value().RemoveChild("CONTEXTS")
            newMd = btk.btkMetaData('CONTEXTS', btk.btkStringArray(contexts))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)
        else:
            contexts = [eventcontext]
            newMd = btk.btkMetaData('CONTEXTS', btk.btkStringArray(contexts))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)

        if checkMetadata(acq, "ANALYSIS", "UNITS"):
            units = [it for it in smartGetMetadata(
                acq, "ANALYSIS", "UNITS")]

            units.append(unit)

            acq.GetMetaData().FindChild("ANALYSIS").value().RemoveChild("UNITS")
            newMd = btk.btkMetaData('UNITS', btk.btkStringArray(units))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)
        else:
            units = [unit]
            newMd = btk.btkMetaData('UNITS', btk.btkStringArray(units))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)

        if checkMetadata(acq, "ANALYSIS", "VALUES"):
            values = [it for it in smartGetMetadata(
                acq, "ANALYSIS", "VALUES", returnType="Double")]

            values.append(value)

            acq.GetMetaData().FindChild("ANALYSIS").value().RemoveChild("VALUES")
            newMd = btk.btkMetaData('VALUES', btk.btkDoubleArray(values))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)
        else:
            values = [value]
            newMd = btk.btkMetaData('VALUES',  btk.btkDoubleArray(values))
            acq.GetMetaData().FindChild("ANALYSIS").value().AppendChild(newMd)

        smartSetMetadata(acq, "ANALYSIS", "USED", 0, used[0]+1)


def getAllParamAnalysis(acq:btk.btkAcquisition):
    """
    Retrieves all analysis parameters from the BTK acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        List[Dict]: A list of dictionaries, each containing information about an analysis parameter.
    """

    names = [it.strip()
             for it in smartGetMetadata(acq, "ANALYSIS", "NAMES")]
    contexts = [it.strip()
                for it in smartGetMetadata(acq, "ANALYSIS", "CONTEXTS")]
    subjects = [it.strip()
                for it in smartGetMetadata(acq, "ANALYSIS", "SUBJECTS")]
    descriptions = [it.strip() for it in smartGetMetadata(
        acq, "ANALYSIS", "DESCRIPTIONS")]
    units = [it.strip()
             for it in smartGetMetadata(acq, "ANALYSIS", "UNITS")]
    values = [it for it in smartGetMetadata(
        acq, "ANALYSIS", "VALUES", returnType="Double")]

    itemNumber = len(names)

    items = []
    for i in range(0, itemNumber):
        item = {"name": names[i], "context": contexts[i], "subject": subjects[i],
                "description": descriptions[i], "unit": units[i], "value": values[i]}
        items.append(item)

    return items


def getParamAnalysis(btkAcq:btk.btkAcquisition, name:str, context:str, subject:str):
    """
    Retrieves a specific analysis parameter from the BTK acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        name (str): Name of the analysis parameter.
        context (str): Context of the event related to the analysis parameter.
        subject (str): Subject associated with the analysis parameter.

    Returns:
        Dict: Dictionary containing information about the specified analysis parameter.
    """

    names = [it.strip()
             for it in smartGetMetadata(btkAcq, "ANALYSIS", "NAMES")]
    contexts = [it.strip()
                for it in smartGetMetadata(btkAcq, "ANALYSIS", "CONTEXTS")]
    subjects = [it.strip()
                for it in smartGetMetadata(btkAcq, "ANALYSIS", "SUBJECTS")]
    descriptions = [it.strip() for it in smartGetMetadata(
        btkAcq, "ANALYSIS", "DESCRIPTIONS")]
    units = [it.strip()
             for it in smartGetMetadata(btkAcq, "ANALYSIS", "UNITS")]
    values = [it for it in smartGetMetadata(
        btkAcq, "ANALYSIS", "VALUES", returnType="Double")]

    itemNumber = len(names)

    for i in range(0, itemNumber):
        if name.strip() == names[i] and context.strip() == contexts[i] and subject.strip() == subjects[i]:
            item = {"name": names[i], "context": contexts[i], "subject": subjects[i],
                    "description": descriptions[i], "unit": units[i], "value": values[i]}
            return item
    LOGGER.logger.error("no analysis parameter found with specification [%s,%s,%s]" % (
        name, context, subject))

def getLabelsFromScalar(acq:btk.btkAcquisition, description:Optional[str]=None):
    """
    Retrieves labels of scalar points from a BTK acquisition based on their description.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        description (str, optional): Description to filter scalar points. If None, retrieves all scalar points. Defaults to None.

    Returns:
        List[str]: List of labels of the scalar points that match the given description.
    """
    out = []

    if description is not None:
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetType() == btk.btkPoint.Scalar and description in it.GetDescription():
                index = it.GetLabel().find("[")
                out.append(it.GetLabel()[:index])
    else:
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetType() == btk.btkPoint.Scalar:
                out.append(it.GetLabel())

    return out

def getScalar(acq:btk.btkAcquisition,label:str):
    """
    Retrieves a scalar point from a BTK acquisition by its label.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        label (str): Label of the scalar point to retrieve.

    Returns:
        btk.btkPoint: The requested scalar point, if found.
    """

    out = None

    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Scalar and it.GetLabel() == label:
            out = it
            break
    
    return out



def btkPointToKtkTimeseries(acq:btk.btkAcquisition, type:btk.btkPoint=btk.btkPoint.Marker):
    """
    Converts BTK points of a specified type from a BTK acquisition to a Kinetics Toolkit timeseries.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        type (btk.btkPoint, optional): Type of BTK points to convert (e.g., Marker, Angle). Defaults to btk.btkPoint.Marker.

    Returns:
        ktk.kineticstoolkit.timeseries.TimeSeries: A timeseries object containing the converted data.
    """
     
    freq = acq.GetPointFrequency()
    frames = np.arange(0, acq.GetPointFrameNumber())

    ts = timeseries.TimeSeries()
    ts.time = frames*1/freq
    for point in btk.Iterate(acq.GetPoints()):
        if point.GetType() == type:
            ts.data[point.GetLabel()] = point.GetValues()
    
    return ts


def btkAnalogToKtkTimeseries(acq:btk.btkAcquisition):
    """
    Converts all BTK analog data from a BTK acquisition to a Kinetics Toolkit timeseries.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.

    Returns:
        ktk.kineticstoolkit.timeseries.TimeSeries: A timeseries object containing the converted analog data.
    """     
    freq = acq.GetAnalogFrequency()
    frames = np.arange(0, acq.GetAnalogFrameNumber())

    ts = timeseries.TimeSeries()
    ts.time = frames*1/freq
    for analog in btk.Iterate(acq.GetAnalogs()):
        ts.data[analog.GetLabel()] = analog.GetValues()
    
    return ts


def calculateAngleFrom3points( acq:btk.btkAcquisition,pt1:str,pt2:str,pt3:str):
    """
    Calculates the angle formed by three points at each frame in a BTK acquisition.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        pt1 (str): Label of the first point.
        pt2 (str): Label of the second point.
        pt3 (str): Label of the third point.

    Returns:
        np.ndarray: An array of angles calculated at each frame.
    """
    nrow = acq.GetPointFrameNumber()
    out = np.zeros((nrow,3))

    for i in range (0, nrow):
        pt1r = acq.GetPoint(pt1).GetResiduals()[i,0]
        pt2r = acq.GetPoint(pt2).GetResiduals()[i,0]
        pt3r = acq.GetPoint(pt3).GetResiduals()[i,0]

        if pt1r == -1.0 or pt2r == -1.0 or pt3r == -1.0:
            out[i,0] = 0
            LOGGER.logger.warning("there are gap at frame %i - value set to 0"%(i))
        else:
            u1 = acq.GetPoint(pt2).GetValues()[i,:] -acq.GetPoint(pt1).GetValues()[i,:]
            v1 = acq.GetPoint(pt3).GetValues()[i,:] -acq.GetPoint(pt1).GetValues()[i,:]

            theta = geometry.computeAngle(u1,v1)

            out[i,0] = theta

    return out

def markersToArray(acq:btk.btkAcquisition,markers:Optional[List[str]]=None):
    """
    Converts marker position data from a BTK acquisition to a numpy array.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance.
        markers (List[str], optional): List of marker labels to include. If None, includes all markers. Defaults to None.

    Returns:
        np.ndarray: Array of marker trajectories with dimensions [n_frames, n_markers * 3].
    """

    markerNames  = GetMarkerNames(acq)
    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()
    pfn = acq.GetPointFrameNumber()

    if markers is not None:
        btkmarkers = markers
    else:
        btkmarkers =[]
        for ml in markerNames:
            if isPointExist(acq,ml) :
                btkmarkers.append(ml)
    # --------
    array = np.zeros((pfn,len(btkmarkers)*3))
    for i in range(0,len(btkmarkers)):
        values = acq.GetPoint(btkmarkers[i]).GetValues()
        residualValues = acq.GetPoint(btkmarkers[i]).GetResiduals()
        array[:,3*i-3] = values[:,0]
        array[:,3*i-2] = values[:,1]
        array[:,3*i-1] = values[:,2]
        E = residualValues[:,0]
        array[np.asarray(E)==-1,3*i-3] = np.nan
        array[np.asarray(E)==-1,3*i-2] = np.nan
        array[np.asarray(E)==-1,3*i-1] = np.nan
    return array
