# -*- coding: utf-8 -*-
import ipdb
import logging
import numpy as np

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

from pyCGM2.Model import modelFilters,model,modelDecorator,anthropometricMeasurement
from pyCGM2.Model.RCM import rcm

from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files



if __name__ == "__main__":

    DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "RCM\\Qualisys\\subject297\\"
    staticFilename = "subject_297_static_0001.c3d"
    translators = files.openTranslators(DATA_PATH,"RCM.translators")

    acqStatic0 = btkTools.smartReader(str(DATA_PATH +  staticFilename))
    acqStatic =  btkTools.applyTranslators(acqStatic0,translators)

    dynamicFilename = "subject_297_run_0001.c3d"
    acqDynamic0 = btkTools.smartReader(str(DATA_PATH +  dynamicFilename))
    acqDynamic =  btkTools.applyTranslators(acqDynamic0,translators)

    # ---- Model configuration ----
    bioMechModel = rcm.RCM()
    bioMechModel.configure()

    mp={
    'LeftKneeWidth' : anthropometricMeasurement.measureNorm(acqStatic,"LKNE","LKNM",markerDiameter =14),
    'RightKneeWidth' : anthropometricMeasurement.measureNorm(acqStatic,"RKNE","RKNM",markerDiameter =14),
    'LeftAnkleWidth' : anthropometricMeasurement.measureNorm(acqStatic,"LANK","LMED",markerDiameter =14),
    'RightAnkleWidth' : anthropometricMeasurement.measureNorm(acqStatic,"RANK","RMED",markerDiameter =14),
    }
    bioMechModel.addAnthropoInputParameters(mp)


    scp=modelFilters.StaticCalibrationProcedure(bioMechModel)

    smf = modelFilters.ModelCalibrationFilter(scp,acqStatic,bioMechModel)
    smf.compute()

    #---Motion Filter---
    mmf=modelFilters.ModelMotionFilter(scp,acqDynamic,bioMechModel,enums.motionMethod.Sodervisk)
    mmf.compute()




    btkTools.smartWriter(acqStatic, "rcmStatic.c3d")
