# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

import pyCGM2
# btk
pyCGM2.CONFIG.addBtk()

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


class CGM2_4_calibration():


    @classmethod
    def calibration_noFlatFoot(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\medial\\"
        staticFilename = "static.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 69.0,
        'LeftLegLength' : 930.0,
        'RightLegLength' : 930.0 ,
        'LeftKneeWidth' : 94.0,
        'RightKneeWidth' : 64.0,
        'LeftAnkleWidth' : 67.0,
        'RightAnkleWidth' : 62.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        "LeftToeOffset" : 0,
        "RightToeOffset" : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_4LowerLimbs()
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

        # display CS
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqStatic)
        csf.setStatic(True)
        csf.display()


        btkTools.smartWriter(acqStatic,"cgm2.4_noFlatFoot.c3d")

    @classmethod
    def calibration_FlatFoot(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\medial\\"
        staticFilename = "static.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 69.0,
        'LeftLegLength' : 930.0,
        'RightLegLength' : 930.0 ,
        'LeftKneeWidth' : 94.0,
        'RightKneeWidth' : 64.0,
        'LeftAnkleWidth' : 67.0,
        'RightAnkleWidth' : 62.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        "LeftToeOffset" : 0,
        "RightToeOffset" : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_4LowerLimbs()
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
                           markerDiameter=markerDiameter,
                           leftFlatFoot = True, rightFlatFoot = True).compute()


        # display CS
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqStatic)
        csf.setStatic(True)
        csf.display()

        btkTools.smartWriter(acqStatic,"cgm2.4_FlatFoot.c3d")


    @classmethod
    def calibration_GarchesFlatFoot(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "Datasets Tests\\didier\\08_02_18_Vincent Pere\\"
        staticFilename = "08_02_18_Vincent_Pere_Statique_000_MOKKA.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 70.0,
        'LeftLegLength' : 890.0,
        'RightLegLength' : 890.0 ,
        'LeftKneeWidth' : 150.0,
        'RightKneeWidth' : 150.0,
        'LeftAnkleWidth' : 88.0,
        'RightAnkleWidth' : 99.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        "LeftToeOffset" : 0,
        "RightToeOffset" : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_4LowerLimbs()
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
                           markerDiameter=markerDiameter,
                           leftFlatFoot = True, rightFlatFoot = True).compute()


        # display CS
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqStatic)
        csf.setStatic(True)
        csf.display()

        btkTools.smartWriter(acqStatic,"cgm2.4_GarchesFlatFoot.c3d")



class CGM2_4_staticFitting():

    @classmethod
    def noIK_determinist(cls):
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\fullBody\\"
        staticFilename = "PN01OP01S01STAT.c3d"
        gaitFilename= "PN01OP01S01STAT.c3d"

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

        translators = files.getTranslators(MAIN_PATH,"CGM2_4.translators")
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        model=cgm2.CGM2_4LowerLimbs()
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


        #import ipdb; ipdb.set_trace()
        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        acqGait =  btkTools.applyTranslators(acqGait,translators)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        btkTools.smartWriter(acqGait,"cgm24_noIKdeterminist_staticMotion.c3d")

    @classmethod
    def noIK_6dof(cls):
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\fullBody\\"
        staticFilename = "PN01OP01S01STAT.c3d"
        gaitFilename= "PN01OP01S01STAT.c3d"

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

        translators = files.getTranslators(MAIN_PATH,"CGM2_4.translators")
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        model=cgm2.CGM2_4LowerLimbs()
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


        #import ipdb; ipdb.set_trace()
        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        acqGait =  btkTools.applyTranslators(acqGait,translators)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        btkTools.smartWriter(acqGait,"cgm24_noIK6dof_staticMotion.c3d")


    @classmethod
    def full_IK(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\fullBody\\"
        staticFilename = "PN01OP01S01STAT.c3d"
        gaitFilename= "PN01OP01S01STAT.c3d"

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
        translators = files.getTranslators(MAIN_PATH,"CGM2_4.translators")
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        model=cgm2.CGM2_4LowerLimbs()
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
        markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml"

        osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                MAIN_PATH)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build(exportOsim=False)

        # --- fitting ---
        #procedure
        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)

        iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml"

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

        btkTools.smartWriter(acqIK,"cgm24_fullIK_staticMotion.c3d")



    @classmethod
    def noIK_determinist_Garches(cls):
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "Datasets Tests\\didier\\08_02_18_Vincent Pere\\"
        staticFilename = "08_02_18_Vincent_Pere_Statique_000_MOKKA.c3d"
        gaitFilename= "08_02_18_Vincent_Pere_Statique_000_MOKKA.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 70.0,
        'LeftLegLength' : 890.0,
        'RightLegLength' : 890.0 ,
        'LeftKneeWidth' : 150.0,
        'RightKneeWidth' : 150.0,
        'LeftAnkleWidth' : 88.0,
        'RightAnkleWidth' : 99.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        model=cgm2.CGM2_4LowerLimbs()
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


        #import ipdb; ipdb.set_trace()
        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        btkTools.smartWriter(acqGait,"cgm24_noIKDeter_staticMotion_Garches.c3d")

    @classmethod
    def noIK_6dof_Garches(cls):
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "Datasets Tests\\didier\\08_02_18_Vincent Pere\\"
        staticFilename = "08_02_18_Vincent_Pere_Statique_000_MOKKA.c3d"
        gaitFilename= "08_02_18_Vincent_Pere_Statique_000_MOKKA.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 70.0,
        'LeftLegLength' : 890.0,
        'RightLegLength' : 890.0 ,
        'LeftKneeWidth' : 150.0,
        'RightKneeWidth' : 150.0,
        'LeftAnkleWidth' : 88.0,
        'RightAnkleWidth' : 99.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        model=cgm2.CGM2_4LowerLimbs()
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


        #import ipdb; ipdb.set_trace()
        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        btkTools.smartWriter(acqGait,"cgm24_noIK6dof_staticMotion_Garches.c3d")


if __name__ == "__main__":

    # CGM2_4_calibration.calibration_noFlatFoot()
    # CGM2_4_calibration.calibration_FlatFoot()

    #CGM2_4_staticFitting.noIK_determinist()
    # CGM2_4_staticFitting.noIK_6dof()
    #CGM2_4_staticFitting.full_IK()

    # tests on Garches Data
    CGM2_4_calibration.calibration_GarchesFlatFoot()
    CGM2_4_staticFitting.noIK_6dof_Garches()
    CGM2_4_staticFitting.noIK_determinist_Garches()
