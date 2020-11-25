# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_CGM23.py::Test_CGM23::test_lowLevel

import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm2
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2 import enums
from pyCGM2.Model.Opensim import opensimFilters


class Test_CGM23:
    def test_lowLevel(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM2.3\\Hannibal-medial\\"

        staticFilename = "static.c3d"
        gaitFilename= "gait1.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(DATA_PATH +  staticFilename))
        translators = files.getTranslators(DATA_PATH,"CGM2_3.translators")
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        model=cgm2.CGM2_3()()
        model.configure()

        model.addAnthropoInputParameters(mp)

        # ---- Calibration ----

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           markerDiameter=markerDiameter).compute()


        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(DATA_PATH +  gaitFilename))
        acqGait =  btkTools.applyTranslators(acqGait,translators)


        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        import ipdb; ipdb.set_trace()

        # ------- OPENSIM IK --------------------------------------
        # --- osim builder ---
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml"

        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                DATA_PATH)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build(exportOsim=False)

        # --- fitting ---
        #procedure
        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)

        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml"

        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          DATA_PATH )


        acqIK = osrf.run(acqGait,str(DATA_PATH + gaitFilename ),exportSetUp=False)


        # -------- NEW MOTION FILTER ON IK MARKERS ------------------

        modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion_ik.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.setFilterBool(False)
        finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#

        btkTools.smartWriter(acqIK,"cgm23_fullIK_Motion.c3d")
