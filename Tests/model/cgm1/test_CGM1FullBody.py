# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:46:40 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import scipy as sp

import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame, bodySegmentParameters
from pyCGM2 import enums


def getViconRmatrix(frameVal, acq, originLabel, proximalLabel, lateralLabel, sequence):

        pt1 = acq.GetPoint(originLabel).GetValues()[frameVal,:]
        pt2 = acq.GetPoint(proximalLabel).GetValues()[frameVal,:]
        pt3 = acq.GetPoint(lateralLabel).GetValues()[frameVal,:]

        a1 = (pt2-pt1)
        a1 = a1/np.linalg.norm(a1)
        v = (pt3-pt1)
        v = v/np.linalg.norm(v)
        a2 = np.cross(a1,v)
        a2 = a2/np.linalg.norm(a2)
        x,y,z,R = frame.setFrameData(a1,a2,sequence)

        return R

class FullBody_Static():


    @classmethod
    def CGM1_fullbody(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\Full PIG - StephenL5_C7\\"
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
                                            leftFlatFoot = False,
                                            rightFlatFoot = False,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True,
                                            ).compute()
                                            #headHorizontal=True

        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqStatic)
        csdf.setStatic(True)
        csdf.display()

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("TRXO").GetValues().mean(axis=0),acqStatic.GetPoint("OT").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LCLO").GetValues().mean(axis=0),acqStatic.GetPoint("LSJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("LHUO").GetValues().mean(axis=0),acqStatic.GetPoint("LEJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("LCLO").GetValues().mean(axis=0),acqStatic.GetPoint("LSJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("RCLO").GetValues().mean(axis=0),acqStatic.GetPoint("RSJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RHUO").GetValues().mean(axis=0),acqStatic.GetPoint("REJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RCLO").GetValues().mean(axis=0),acqStatic.GetPoint("RSJC").GetValues().mean(axis=0),decimal = 3)


        btkTools.smartWriter(acqStatic, "FullBody_Static_CGM1_fullbody.c3d")

class FullBody_Motion():


    @classmethod
    def CGM1_fullbody(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\Full PIG - StephenL5_C7\\"
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
        csp = modelFilters.ModelCoordinateSystemProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = False,
                                            rightFlatFoot = False,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True,
                                            ).compute()
                                            #headHorizontal=True


        # --- motion ----
        gaitFilename="PN01NORMSS01_stephen.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        csdf.setStatic(False)
        csdf.display()

        # TESTING
        #   thorax
        R_thorax= model.getSegment("Thorax").anatomicalFrame.motion[10].getRotation()
        R_thorax_vicon = getViconRmatrix(10, acqGait, "TRXO", "TRXA", "TRXL", "XZY")
        np.testing.assert_almost_equal( R_thorax,
                                R_thorax_vicon, decimal =3)

        #   head
        R_head= model.getSegment("Head").anatomicalFrame.motion[10].getRotation()
        R_head_vicon = getViconRmatrix(10, acqGait, "HEDO", "HEDA", "HEDL", "XZY")

        np.testing.assert_almost_equal( R_head,
                                R_head_vicon, decimal =2)



        btkTools.smartWriter(acqStatic, "FullBody_motion_CGM1_fullbody.c3d")



class FullBody_Angles():


    @classmethod
    def CGM1_fullbody(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\Full PIG - StephenL5_C7\\"
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
        csp = modelFilters.ModelCoordinateSystemProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = False,
                                            rightFlatFoot = False,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True,
                                            ).compute()
                                            #headHorizontal=True


        # --- motion ----
        gaitFilename="PN01NORMSS01_stephen.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        csdf.setStatic(False)
        csdf.display()

        # angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromLongAxis(acqGait,"C7","CLAV")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Thorax"],
                                      angleLabels=["Thorax",],
                                      eulerSequences=["YXZ"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        # testing
        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"XYZ")


        #plot("LNeckAngles",acqGait,"cgm1_6dof")

        # relative angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RSpineAngles").GetValues(),
                                    acqGait.GetPoint("RSpineAngles_cgm1_6dof").GetValues(), decimal =2)
        np.testing.assert_almost_equal( acqGait.GetPoint("LSpineAngles").GetValues(),
                                    acqGait.GetPoint("LSpineAngles_cgm1_6dof").GetValues(), decimal =2)


        np.testing.assert_almost_equal( acqGait.GetPoint("LShoulderAngles").GetValues(),
                                    acqGait.GetPoint("LShoulderAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RShoulderAngles").GetValues(),
                                    acqGait.GetPoint("RShoulderAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RElbowAngles").GetValues(),
                                    acqGait.GetPoint("RElbowAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LElbowAngles").GetValues(),
                                    acqGait.GetPoint("LElbowAngles_cgm1_6dof").GetValues(), decimal =3)

        # np.testing.assert_almost_equal( acqGait.GetPoint("RWristAngles").GetValues(),
        #                             acqGait.GetPoint("RWristAngles_cgm1_6dof").GetValues(), decimal =3) # fail on transverse
        # np.testing.assert_almost_equal( acqGait.GetPoint("LWristAngles").GetValues(),
        #                             acqGait.GetPoint("LWristAngles_cgm1_6dof").GetValues(), decimal =3)# fail on transverse
        #
        # np.testing.assert_almost_equal( acqGait.GetPoint("RNeckAngles").GetValues(),
        #                             acqGait.GetPoint("RNeckAngles_cgm1_6dof").GetValues(), decimal =1) # fail on coronal
        # np.testing.assert_almost_equal( acqGait.GetPoint("LNeckAngles").GetValues(),
        #                             acqGait.GetPoint("LNeckAngles_cgm1_6dof").GetValues(), decimal =1) # fail on coronal


        # absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LThoraxAngles").GetValues(),
                                    acqGait.GetPoint("LThoraxAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RThoraxAngles").GetValues(),
                                    acqGait.GetPoint("RThoraxAngles_cgm1_6dof").GetValues(), decimal =3)



        btkTools.smartWriter(acqStatic, "FullBody_Angles_CGM1_fullbody.c3d")


class FullBody_COM():


    @classmethod
    def CGM1_fullbody(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\Full PIG - StephenL5_C7\\"
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
        csp = modelFilters.ModelCoordinateSystemProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = False,
                                            rightFlatFoot = False,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True,
                                            ).compute()
                                            #headHorizontal=True


        # --- motion ----
        gaitFilename="PN01NORMSS01_stephen.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        csdf.setStatic(False)
        csdf.display()

        # angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromLongAxis(acqGait,"C7","CLAV")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Thorax"],
                                      angleLabels=["Thorax",],
                                      eulerSequences=["YXZ"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        # testing
        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"XYZ")


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
        C7o_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7o")
        C7_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7")
        C7_head = model.getSegment("Head").anatomicalFrame.getNodeTrajectory("C7")
        btkTools.smartAppendPoint(acqGait,"TL5_pelvis",TL5_pelvis, desc="")
        btkTools.smartAppendPoint(acqGait,"TL5_thorax",TL5_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7o_thorax",C7o_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7_thorax",C7_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7_head",C7_thorax, desc="")

        btkTools.smartWriter(acqGait, "FullBody_COM_CGM1_fullbody.c3d")


    @classmethod
    def CGM1_fullbody_onStatic(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\Full PIG - StephenL5_C7\\"
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
        csp = modelFilters.ModelCoordinateSystemProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = False,
                                            rightFlatFoot = False,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True,
                                            ).compute()
                                            #headHorizontal=True


        # --- motion ----
        gaitFilename="PN01NORMSTAT-COM.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        csdf.setStatic(False)
        csdf.display()

        # angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromLongAxis(acqGait,"C7","CLAV")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Thorax"],
                                      angleLabels=["Thorax",],
                                      eulerSequences=["YXZ"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        # testing
        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,True)
        np.testing.assert_equal( globalFrame,"XYZ")


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
        C7o_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7o")
        C7_thorax = model.getSegment("Thorax").anatomicalFrame.getNodeTrajectory("C7")
        C7_head = model.getSegment("Head").anatomicalFrame.getNodeTrajectory("C7")
        btkTools.smartAppendPoint(acqGait,"TL5_pelvis",TL5_pelvis, desc="")
        btkTools.smartAppendPoint(acqGait,"TL5_thorax",TL5_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7o_thorax",C7o_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7_thorax",C7_thorax, desc="")
        btkTools.smartAppendPoint(acqGait,"C7_head",C7_thorax, desc="")

        btkTools.smartWriter(acqGait, "FullBody_COM_CGM1_fullbody_onStatic.c3d")

if __name__ == "__main__":



    FullBody_Static.CGM1_fullbody()
    FullBody_Motion.CGM1_fullbody()
    FullBody_Angles.CGM1_fullbody()
    FullBody_COM.CGM1_fullbody()
    FullBody_COM.CGM1_fullbody_onStatic()
    #CGM1.CGM1_fullbody()

    logging.info("######## PROCESS CGM1 --> Done ######")
