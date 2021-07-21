# -*- coding: utf-8 -*-
import os

import pyCGM2; LOGGER = pyCGM2.LOGGER


try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim


def createGroundReactionForceMOT_file(DATA_PATH,c3dFile):

    c3dFileAdapter = opensim.C3DFileAdapter()
    c3dFileAdapter.setLocationForForceExpression(opensim.C3DFileAdapter.ForceLocation_PointOfWrenchApplication); #ForceLocation_OriginOfForcePlate , ForceLocation_CenterOfPressure
    tables = c3dFileAdapter.read(DATA_PATH + c3dFile)

    forces = c3dFileAdapter.getForcesTable(tables)
    forcesFlat = forces.flatten()

    forcesFilename = DATA_PATH+c3dFile[:-4] + '_GRF.mot'
    stoAdapter = opensim.STOFileAdapter()
    stoAdapter.write(forcesFlat, forcesFilename)
