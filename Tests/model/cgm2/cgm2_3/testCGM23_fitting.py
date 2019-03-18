# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

import pyCGM2
# pyCGM2
from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm2
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2 import enums
from pyCGM2.Math import numeric
from pyCGM2.Model.Opensim import opensimFilters
import json
from collections import OrderedDict

class CGM2_3_Tests():

    @classmethod
    def noIK_determinist(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\cgm2.3\\fullBody\\"
        staticFilename = "PN01OP01S01STAT.c3d"
        gaitFilename= "PN01OP01S01SS01.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 83.0,
        'LeftLegLength' : 874.0,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        translators = files.getTranslators(MAIN_PATH,"CGM2_3.translators")
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        model=cgm2.CGM2_3()()
        model.configure()

        model.addAnthropoInputParameters(mp)

        # ---- Calibration ----

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        print "----"
        print model.getSegment("Left Shank").getReferential("TF").relativeMatrixAnatomic
        print "----"

        # # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")
        #
        # # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                            markerDiameter=markerDiameter).compute()


        
        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        acqGait =  btkTools.applyTranslators(acqGait,translators)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait).display()

        btkTools.smartWriter(acqGait,"cgm23_noIKdeterminist_Motion.c3d")

    @classmethod
    def noIK_6dof(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\cgm2.3\\fullBody\\"
        staticFilename = "PN01OP01S01STAT.c3d"
        gaitFilename= "PN01OP01S01SS01.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 83.0,
        'LeftLegLength' : 874.0,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        translators = files.getTranslators(MAIN_PATH,"CGM2_3.translators")
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        model=cgm2.CGM2_3()()
        model.configure()

        model.addAnthropoInputParameters(mp)

        # ---- Calibration ----

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        print "----"
        print model.getSegment("Left Shank").getReferential("TF").relativeMatrixAnatomic
        print "----"

        # # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")
        #
        # # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                            markerDiameter=markerDiameter).compute()


        
        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        acqGait =  btkTools.applyTranslators(acqGait,translators)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait).display()

        btkTools.smartWriter(acqGait,"cgm23_noIK6dof_Motion.c3d")


    @classmethod
    def full_IK(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\cgm2.3\\fullBody\\"
        staticFilename = "PN01OP01S01STAT.c3d"
        gaitFilename= "PN01OP01S01SS01.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 83.0,
        'LeftLegLength' : 874.0,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))
        translators = files.getTranslators(MAIN_PATH,"CGM2_3.translators")
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
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        acqGait =  btkTools.applyTranslators(acqGait,translators)


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
                                                MAIN_PATH)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build(exportOsim=False)

        # --- fitting ---
        #procedure
        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)

        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml"

        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          MAIN_PATH )


        acqIK = osrf.run(acqGait,str(MAIN_PATH + gaitFilename ),exportSetUp=False)


        # -------- NEW MOTION FILTER ON IK MARKERS ------------------

        modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion_ik.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.setFilterBool(False)
        finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#

        btkTools.smartWriter(acqIK,"cgm23_fullIK_Motion.c3d")

if __name__ == "__main__":

    #CGM2_3_Tests.noIK_determinist()
    CGM2_3_Tests.noIK_6dof()
    #CGM2_3_Tests.full_IK()
