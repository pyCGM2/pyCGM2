# -*- coding: utf-8 -*-
from pyCGM2.Tools import btkTools
from pyCGM2.Processing import progressionFrame
import pyCGM2
LOGGER = pyCGM2.LOGGER


def detectProgressionFrame(acq, staticFlag=False):

    progressionFlag = False
    if staticFlag:
        if btkTools.isPointsExist(acq, ['LASI', 'RASI', 'RPSI', 'LPSI'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Pelvic markers ")
            pfp = progressionFrame.PelvisProgressionFrameProcedure()
            pff = progressionFrame.ProgressionFrameFilter(acq, pfp)
            pff.compute()
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]

            progressionFlag = True
        elif btkTools.isPointsExist(acq, ['C7', 'T10', 'CLAV', 'STRN'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Thoracic markers ")
            pfp = progressionFrame.ThoraxProgressionFrameProcedure()
            pff = progressionFrame.ProgressionFrameFilter(acq, pfp)
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
        if btkTools.isPointExist(acq, 'LHEE', ignorePhantom=False) or btkTools.isPointExist(acq, 'RHEE', ignorePhantom=False):

            pfp = progressionFrame.PointProgressionFrameProcedure(marker="LHEE") \
                if btkTools.isPointExist(acq, 'LHEE', ignorePhantom=False) \
                else progressionFrame.PointProgressionFrameProcedure(marker="RHEE")

            pff = progressionFrame.ProgressionFrameFilter(acq, pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]
            progressionFlag = True

        elif btkTools.isPointsExist(acq, ['LASI', 'RASI', 'RPSI', 'LPSI'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Pelvic markers ")
            pfp = progressionFrame.PelvisProgressionFrameProcedure()
            pff = progressionFrame.ProgressionFrameFilter(acq, pfp)
            pff.compute()
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]

            progressionFlag = True
        elif btkTools.isPointsExist(acq, ['C7', 'T10', 'CLAV', 'STRN'], ignorePhantom=False) and not progressionFlag:
            LOGGER.logger.info(
                "[pyCGM2] - progression axis detected from Thoracic markers ")
            pfp = progressionFrame.ThoraxProgressionFrameProcedure()
            pff = progressionFrame.ProgressionFrameFilter(acq, pfp)
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
