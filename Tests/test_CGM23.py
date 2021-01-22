# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_CGM23.py::Test_CGM23::test_lowLevel

import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2 import enums
from pyCGM2.Model.Opensim import opensimFilters

from pyCGM2.Lib.CGM import cgm2_3

class Test_CGM23:
    def test_lowLevel(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM2.3\\Hannibal-medial\\"

        staticFilename = "static.c3d"
        gaitFilename= "gait1.c3d"

        markerDiameter=14
        required_mp={
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
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        # --- Calibration ---
        # ---check marker set used----
        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model=cgm2.CGM2_3()
        model.configure(acq=acqStatic,detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)

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
        acqGait = btkTools.smartReader(DATA_PATH +  gaitFilename)


        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()



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
                                                          DATA_PATH,
                                                          acqGait )


        acqIK = osrf.run(str(DATA_PATH + gaitFilename ),exportSetUp=False)

        # -------- NEW MOTION FILTER ON IK MARKERS ------------------

        modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion_ik.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#

        btkTools.smartWriter(acqIK,"cgm23_fullIK_Motion.c3d")

    def test_highLevel(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM2.3\\Hannibal-medial\\"

        staticFilename = "static.c3d"
        reconstructFilenameLabelled= "gait1.c3d"

        markerDiameter=14
        required_mp={
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
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        model,finalAcqStatic = cgm2_3.calibrate(DATA_PATH,
            staticFilename,
            settings["Translators"],
            settings["Fitting"]["Weight"],
            required_mp,
            optional_mp,
            True,
            True,
            True,
            True,
            14,
            settings["Calibration"]["HJC"],
            None,
            displayCoordinateSystem=True,
            noKinematicsCalculation=False)


        acqGait = cgm2_3.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            settings["Translators"],
            settings,
            True,14.0,
            "acc2",
            "XX",
            momentProjection =  enums.MomentProjection.JCS)


        outFilename = reconstructFilenameLabelled
        btkTools.smartWriter(acqGait, str(DATA_PATH + outFilename))
