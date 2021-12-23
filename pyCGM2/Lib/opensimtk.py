# -*- coding: utf-8 -*-
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Lib import processing
from pyCGM2.Tools import btkTools
from pyCGM2.Tools import opensimTools
import pyCGM2
LOGGER = pyCGM2.LOGGER

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk


def prepareC3dFiles(DATA_PATH, staticFilename=None, dynamicFilenames=None, grfFlag=True):

    if staticFilename is not None:
        acq = btkTools.smartReader(DATA_PATH+staticFilename)
        opensimTools.smartTrcExport(acq, DATA_PATH + staticFilename[:-4])

    if dynamicFilenames is not None:
        if isinstance(dynamicFilenames, str):
            dynamicFilenames = [dynamicFilenames]

        for dynamicFilename in dynamicFilenames:
            acq = btkTools.smartReader(DATA_PATH+dynamicFilename)
            acqClone = btk.btkAcquisition.Clone(acq)

            progressionAxis, forwardProgression, globalFrame = processing.detectProgressionFrame(
                acqClone)

            # affect rotation Lab/Osim
            R_LAB_OSIM = opensimTools.setGlobalTransormation_lab_osim(
                progressionAxis, forwardProgression)
            opensimTools.globalTransformationLabToOsim(acqClone, R_LAB_OSIM)

            # export trc
            opensimTools.smartTrcExport(
                acqClone, DATA_PATH + dynamicFilename[:-4])

            # generate grf
            if grfFlag:
                opensimTools.footReactionMotFile(
                    acq, DATA_PATH+dynamicFilename[:-4]+"_grf.mot",
                    progressionAxis, forwardProgression, mfpa=None)
