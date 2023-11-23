# -*- coding: utf-8 -*-
from pyCGM2.Tools import btkTools
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters, progressionFrameProcedures 
import pyCGM2
LOGGER = pyCGM2.LOGGER
import btk


def detectProgressionFrame(acq:btk.btkAcquisition, staticFlag:bool=False):
    """
    High-level function to detect  the progression axis 

    Args:
        acq (btk.btkAcquisition): static or motion acquisition

    Keyword Arguments:
        staticFlag (bool): enable if you deal with a static file

    Returns:
        progressionAxis (str): the label of the progression frame ( eg "X") 
        forwardProgression (bool): flag to indicate if the subject progresses along the progression axis or in the opposite direction 
        globalFrame (str): label of the global frame (eg "YXZ" : Y:forward, X:lateral, Z:Vertical)

    """


    progressionFlag = False

    if staticFlag:
        if btkTools.isPointsExist(acq, ['LASI', 'RASI', 'RPSI', 'LPSI'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Pelvic markers ")
            pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
            pff = progressionFrameFilters.ProgressionFrameFilter(acq, pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]

            progressionFlag = True
        elif btkTools.isPointsExist(acq, ['C7', 'T10', 'CLAV', 'STRN'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Thoracic markers ")
            pfp = progressionFrameProcedures.ThoraxProgressionFrameProcedure()
            pff = progressionFrameFilters.ProgressionFrameFilter(acq, pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]

        else:
            globalFrame = "XYZ"
            progressionAxis = "X"
            forwardProgression = True
            LOGGER.logger.error(
                "[pyCGM2] - impossible to detect progression axis - neither pelvic nor thoracic markers are present. Progression set to +X by default ")
    else:
        if btkTools.isPointsExist(acq, ['LASI', 'RASI', 'RPSI', 'LPSI'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Pelvic markers ")
            pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
            pff = progressionFrameFilters.ProgressionFrameFilter(acq, pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]

            progressionFlag = True
        elif btkTools.isPointsExist(acq, ['C7', 'T10', 'CLAV', 'STRN'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Thoracic markers ")
            pfp = progressionFrameProcedures.ThoraxProgressionFrameProcedure()
            pff = progressionFrameFilters.ProgressionFrameFilter(acq, pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]

        else:
            globalFrame = "XYZ"
            progressionAxis = "X"
            forwardProgression = True
            LOGGER.logger.error(
                "[pyCGM2] - impossible to detect progression axis - neither pelvic nor thoracic markers are present. Progression set to +X by default ")
    
    return progressionAxis, forwardProgression, globalFrame
