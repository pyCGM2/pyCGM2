# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:46:40 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import scipy as sp

import pdb
import logging
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame, bodySegmentParameters
from pyCGM2 import enums



class CGM1_com():



    @classmethod
    def CGM1_fullbody_static(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT.c3d"

        # CALIBRATION ###############################
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        markerDiameter=14

        # Lower Limb
        mp={
        'Bodymass'   : 83,
        'LeftLegLength' : 874,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30}

        model=cgm.CGM1()
        model.configure(bodyPart=enums.BodyPart.FullBody)
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = True,
                                            rightFlatFoot = True,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True
                                            ).compute()
        csp = modelFilters.ModelCoordinateSystemProcedure(model)


        btkTools.smartWriter(acqStatic,"stat-COM.c3d")
        # --- motion ----
        gaitFilename="PN01NORMSTAT-COM.c3d" # "PN01NORMSS01-COM.c3d"#"PN01NORMSTAT-COM.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        # csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        # csdf.setStatic(False)
        # csdf.display()


        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        btkTools.smartAppendPoint(acqGait,"pelvisCOM_py",model.getSegment("Pelvis").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"headCOM_py",model.getSegment("Head").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"ThoraxCOM_py",model.getSegment("Thorax").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhumCOM_py",model.getSegment("Left UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LforeCom_py",model.getSegment("Left ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhandCom_py",model.getSegment("Left Hand").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhumCOM_py",model.getSegment("Right UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RforeCom_py",model.getSegment("Right ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhandCom_py",model.getSegment("Right Hand").getComTrajectory())

        modelFilters.CentreOfMassFilter(model,acqGait).compute(pointLabelSuffix="py2")

        #np.testing.assert_almost_equal(model.getSegment("Head").getComTrajectory(),acqGait.GetPoint("HeadCOM").GetValues() , decimal = 0)
        #np.testing.assert_almost_equal(model.getSegment("Thorax").getComTrajectory(),acqGait.GetPoint("ThoraxCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Left UpperArm").getComTrajectory(),acqGait.GetPoint("LeftHumerusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Left ForeArm").getComTrajectory(),acqGait.GetPoint("LeftRadiusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Left Hand").getComTrajectory(),acqGait.GetPoint("LeftHandCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Right UpperArm").getComTrajectory(),acqGait.GetPoint("RightHumerusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Right ForeArm").getComTrajectory(),acqGait.GetPoint("RightRadiusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Right Hand").getComTrajectory(),acqGait.GetPoint("RightHandCOM").GetValues() , decimal = 3)



        #tl5 =  model.getSegment("Thorax").getReferential("TF").getNodeTrajectory("TL5")
        #btkTools.smartAppendPoint(acqGait,"TL5_tho",tl5)
        #
        # c7o2 =  model.getSegment("Thorax").getReferential("TF").getNodeTrajectory("C7o")
        # btkTools.smartAppendPoint(acqGait,"C7O2",c7o2)
        #
        #tl5_2 =  model.getSegment("Pelvis").getReferential("TF").getNodeTrajectory("TL5")
        #btkTools.smartAppendPoint(acqGait,"TL5_pelvis",tl5_2)

        btkTools.smartWriter(acqGait,"PN01NORMSTAT-COM_VERIF.c3d")

    @classmethod
    def CGM1_fullbody_gait(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT.c3d"

        # CALIBRATION ###############################
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        markerDiameter=14

        # Lower Limb
        mp={
        'Bodymass'   : 83,
        'LeftLegLength' : 874,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30}

        model=cgm.CGM1()
        model.configure(bodyPart=enums.BodyPart.FullBody)
        model.addAnthropoInputParameters(mp)


        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = True,
                                            rightFlatFoot = True,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True
                                            ).compute()
        csp = modelFilters.ModelCoordinateSystemProcedure(model)

        btkTools.smartWriter(acqStatic,"stat-COM.c3d")
        # --- motion ----
        gaitFilename="PN01NORMSS01-COM.c3d" # "PN01NORMSS01-COM.c3d"#"PN01NORMSTAT-COM.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        # csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        # csdf.setStatic(False)
        # csdf.display()

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()


        btkTools.smartAppendPoint(acqGait,"headCOM_py",model.getSegment("Head").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"ThoraxCOM_py",model.getSegment("Thorax").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhumCOM_py",model.getSegment("Left UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LforeCom_py",model.getSegment("Left ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhandCom_py",model.getSegment("Left Hand").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhumCOM_py",model.getSegment("Right UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RforeCom_py",model.getSegment("Right ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhandCom_py",model.getSegment("Right Hand").getComTrajectory())

        modelFilters.CentreOfMassFilter(model,acqGait).compute(pointLabelSuffix="py2")

        # np.testing.assert_almost_equal(model.getSegment("Head").getComTrajectory(),acqGait.GetPoint("HeadCOM").GetValues() , decimal = 2)
        # np.testing.assert_almost_equal(model.getSegment("Thorax").getComTrajectory(),acqGait.GetPoint("ThoraxCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Left UpperArm").getComTrajectory(),acqGait.GetPoint("LeftHumerusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Left ForeArm").getComTrajectory(),acqGait.GetPoint("LeftRadiusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Left Hand").getComTrajectory(),acqGait.GetPoint("LeftHandCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Right UpperArm").getComTrajectory(),acqGait.GetPoint("RightHumerusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Right ForeArm").getComTrajectory(),acqGait.GetPoint("RightRadiusCOM").GetValues() , decimal = 3)
        # np.testing.assert_almost_equal(model.getSegment("Right Hand").getComTrajectory(),acqGait.GetPoint("RightHandCOM").GetValues() , decimal = 3)



        #tl5 =  model.getSegment("Thorax").getReferential("TF").getNodeTrajectory("TL5")
        #btkTools.smartAppendPoint(acqGait,"TL5_tho",tl5)
        #
        # c7o2 =  model.getSegment("Thorax").getReferential("TF").getNodeTrajectory("C7o")
        # btkTools.smartAppendPoint(acqGait,"C7O2",c7o2)
        #
        #tl5_2 =  model.getSegment("Pelvis").getReferential("TF").getNodeTrajectory("TL5")
        #btkTools.smartAppendPoint(acqGait,"TL5_pelvis",tl5_2)

        btkTools.smartWriter(acqGait,"PN01NORMSS01-COM_VERIF.c3d")

    @classmethod
    def CGM1_fullBody_L5C7_STATIC(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\Full PIG - StephenL5_C7\\"
        staticFilename = "PN01NORMSTAT_stephen.c3d"

        # CALIBRATION ###############################
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        markerDiameter=14

        # Lower Limb
        mp={
        'Bodymass'   : 83,
        'LeftLegLength' : 874,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30}

        model=cgm.CGM1()
        model.configure(bodyPart=enums.BodyPart.FullBody)
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = True,
                                            rightFlatFoot = True,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True
                                            ).compute()


        # MOTION ###############################
        gaitFilename= staticFilename
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
                                      markerDiameter=14,
                                      viconCGM1compatible=False)
        modMotion.compute()


        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        btkTools.smartAppendPoint(acqGait,"pelvisCOM_py",model.getSegment("Pelvis").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"headCOM_py",model.getSegment("Head").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"ThoraxCOM_py",model.getSegment("Thorax").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhumCOM_py",model.getSegment("Left UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LforeCom_py",model.getSegment("Left ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhandCom_py",model.getSegment("Left Hand").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhumCOM_py",model.getSegment("Right UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RforeCom_py",model.getSegment("Right ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhandCom_py",model.getSegment("Right Hand").getComTrajectory())

        modelFilters.CentreOfMassFilter(model,acqGait).compute(pointLabelSuffix="py2")



        TL5_pelvis = model.getSegment("Pelvis").anatomicalFrame.getNodeTrajectory("TL5")
        TL5_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("TL5")
        TL5motion_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("T5motion")
        C7_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7")
        C7_head = model.getSegment("Head").anatomicalFrame.getNodeTrajectory("C7")
        C7motion_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7motion")
        btkTools.smartAppendPoint(acqGait,"TL5motion_thorax",TL5motion_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7motion_thorax",C7motion_thorax, desc="")


        btkTools.smartWriter(acqGait,"PN01NORMSTAT_stephen_VERIF.c3d")


    @classmethod
    def CGM1_fullBody_L5C7_GAIT(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\Full PIG - StephenL5_C7\\"
        staticFilename = "PN01NORMSTAT_stephen.c3d"

        # CALIBRATION ###############################
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        markerDiameter=14

        # Lower Limb
        mp={
        'Bodymass'   : 83,
        'LeftLegLength' : 874,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30}

        model=cgm.CGM1()
        model.configure(bodyPart=enums.BodyPart.FullBody)
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = True,
                                            rightFlatFoot = True,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True
                                            ).compute()


        # MOTION ###############################
        gaitFilename="PN01NORMSS01_stephen.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
                                      markerDiameter=14,
                                      viconCGM1compatible=False)
        modMotion.compute()


        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        btkTools.smartAppendPoint(acqGait,"pelvisCOM_py",model.getSegment("Pelvis").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"headCOM_py",model.getSegment("Head").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"ThoraxCOM_py",model.getSegment("Thorax").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhumCOM_py",model.getSegment("Left UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LforeCom_py",model.getSegment("Left ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhandCom_py",model.getSegment("Left Hand").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhumCOM_py",model.getSegment("Right UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RforeCom_py",model.getSegment("Right ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhandCom_py",model.getSegment("Right Hand").getComTrajectory())

        modelFilters.CentreOfMassFilter(model,acqGait).compute(pointLabelSuffix="py2")



        TL5_pelvis = model.getSegment("Pelvis").anatomicalFrame.getNodeTrajectory("TL5")
        TL5_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("TL5")
        TL5motion_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("T5motion")
        C7_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7")
        C7_head = model.getSegment("Head").anatomicalFrame.getNodeTrajectory("C7")
        C7motion_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7motion")
        btkTools.smartAppendPoint(acqGait,"TL5motion_thorax",TL5motion_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7motion_thorax",C7motion_thorax, desc="")




        btkTools.smartWriter(acqGait,"PN01NORMSS01_stephen_VERIF.c3d")



if __name__ == "__main__":



    # CGM1_com.CGM1_fullbody_static()
    # CGM1_com.CGM1_fullbody_gait()
    CGM1_com.CGM1_fullBody_L5C7_STATIC()
    CGM1_com.CGM1_fullBody_L5C7_GAIT()
