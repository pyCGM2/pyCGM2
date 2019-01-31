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
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2 import enums


def plot(angleLabel,acqGait,mysuffix):


    f, (ax1, ax2,ax3) = plt.subplots(1, 3)
    ax1.plot(acqGait.GetPoint(angleLabel).GetValues()[:,0],"-ob")
    ax1.plot(acqGait.GetPoint(angleLabel+"_"+mysuffix).GetValues()[:,0],"-r")

    ax2.plot(acqGait.GetPoint(angleLabel).GetValues()[:,1],"-ob")
    ax2.plot(acqGait.GetPoint(angleLabel+"_"+mysuffix).GetValues()[:,1],"-r")

    ax3.plot(acqGait.GetPoint(angleLabel).GetValues()[:,2],"-ob")
    ax3.plot(acqGait.GetPoint(angleLabel+"_"+mysuffix).GetValues()[:,2],"-r")

    plt.show()


class CGM1_angleTest():

    @classmethod
    def CGM1_upperLimb(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1()
        model.configure(bodyPart=enums.BodyPart.UpperLimb)


        markerDiameter=14
        mp={
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30         }
        model.addAnthropoInputParameters(mp)

         # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,headHorizontal=False).compute()

        # --- motion ----
        gaitFilename="PN01NORMSS01.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        csdf.setStatic(False)
        csdf.display()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #plot("LNeckAngles",acqGait,"cgm1_6dof")

        np.testing.assert_almost_equal( acqGait.GetPoint("LShoulderAngles").GetValues(),
                                    acqGait.GetPoint("LShoulderAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RShoulderAngles").GetValues(),
                                    acqGait.GetPoint("RShoulderAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RElbowAngles").GetValues(),
                                    acqGait.GetPoint("RElbowAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LElbowAngles").GetValues(),
                                    acqGait.GetPoint("LElbowAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RWristAngles").GetValues(),
                                    acqGait.GetPoint("RWristAngles_cgm1_6dof").GetValues(), decimal =3) # fail on transverse
        np.testing.assert_almost_equal( acqGait.GetPoint("LWristAngles").GetValues(),
                                    acqGait.GetPoint("LWristAngles_cgm1_6dof").GetValues(), decimal =3)# fail on transverse

        np.testing.assert_almost_equal( acqGait.GetPoint("RNeckAngles").GetValues(),
                                    acqGait.GetPoint("RNeckAngles_cgm1_6dof").GetValues(), decimal =1) # fail on coronal
        np.testing.assert_almost_equal( acqGait.GetPoint("LNeckAngles").GetValues(),
                                    acqGait.GetPoint("LNeckAngles_cgm1_6dof").GetValues(), decimal =1) # fail on coronal




        btkTools.smartWriter(acqGait,"upperLimb_angle.c3d")


        #plot("LNeckAngles",acqGait,"cgm1_6dof")

    @classmethod
    def CGM1_upperLimb_absoluteAngles(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1()
        model.configure(bodyPart=enums.BodyPart.UpperLimb)


        markerDiameter=14
        mp={
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30         }
        model.addAnthropoInputParameters(mp)

         # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,headHorizontal=False).compute()



        # --- motion ----
        gaitFilename="PN01NORMSS01.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromLongAxis(acqGait,"C7","CLAV")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Thorax"],
                                      angleLabels=["Thorax",],
                                      eulerSequences=["YXZ"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")


        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"XYZ")

        # plot("LHeadAngles",acqGait,"cgm1_6dof")
        # plot("RHeadAngles",acqGait,"cgm1_6dof")
        # plot("RThoraxAngles",acqGait,"cgm1_6dof")
        # plot("LThoraxAngles",acqGait,"cgm1_6dof")

        # np.testing.assert_almost_equal( acqGait.GetPoint("LHeadAngles").GetValues(),
        #                             acqGait.GetPoint("LHeadAngles_cgm1_6dof").GetValues(), decimal =3)
        # np.testing.assert_almost_equal( acqGait.GetPoint("RHeadAngles").GetValues(),
        #                             acqGait.GetPoint("RHeadAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LThoraxAngles").GetValues(),
                                    acqGait.GetPoint("LThoraxAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RThoraxAngles").GetValues(),
                                    acqGait.GetPoint("RThoraxAngles_cgm1_6dof").GetValues(), decimal =3)



    @classmethod
    def CGM1_fullbody(cls):

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


        # MOTION ###############################
        gaitFilename="PN01NORMSS01.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
                                      markerDiameter=14,
                                      viconCGM1compatible=False)
        modMotion.compute()


        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait).display()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        plot("RSpineAngles",acqGait,"cgm1_6dof")
        plot("LSpineAngles",acqGait,"cgm1_6dof")


        np.testing.assert_almost_equal( acqGait.GetPoint("RSpineAngles").GetValues(),
                                    acqGait.GetPoint("RSpineAngles_cgm1_6dof").GetValues(), decimal =2)
        np.testing.assert_almost_equal( acqGait.GetPoint("LSpineAngles").GetValues(),
                                    acqGait.GetPoint("LSpineAngles_cgm1_6dof").GetValues(), decimal =2)

        btkTools.smartWriter(acqGait,"fullbody.c3d")


    @classmethod
    def CGM1_upperLimb_absoluteAngles_static(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1()
        model.configure(bodyPart=enums.BodyPart.UpperLimb)


        markerDiameter=14
        mp={
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30         }
        model.addAnthropoInputParameters(mp)

         # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,headHorizontal=False).compute()



        # --- motion ----
        gaitFilename="PN01NORMSTAT.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromLongAxis(acqGait,"C7","CLAV")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Thorax"],
                                      angleLabels=["Thorax",],
                                      eulerSequences=["YXZ"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")


        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,True)
        np.testing.assert_equal( globalFrame,"XYZ")

        # plot("LHeadAngles",acqGait,"cgm1_6dof")
        # plot("RHeadAngles",acqGait,"cgm1_6dof")
        plot("RThoraxAngles",acqGait,"cgm1_6dof")
        plot("LThoraxAngles",acqGait,"cgm1_6dof")

        # np.testing.assert_almost_equal( acqGait.GetPoint("LHeadAngles").GetValues(),
        #                             acqGait.GetPoint("LHeadAngles_cgm1_6dof").GetValues(), decimal =3)
        # np.testing.assert_almost_equal( acqGait.GetPoint("RHeadAngles").GetValues(),
        #                             acqGait.GetPoint("RHeadAngles_cgm1_6dof").GetValues(), decimal =3)

        # np.testing.assert_almost_equal( acqGait.GetPoint("LThoraxAngles").GetValues(),
        #                             acqGait.GetPoint("LThoraxAngles_cgm1_6dof").GetValues(), decimal =3)
        # np.testing.assert_almost_equal( acqGait.GetPoint("RThoraxAngles").GetValues(),
        #                             acqGait.GetPoint("RThoraxAngles_cgm1_6dof").GetValues(), decimal =3)


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


        # MOTION ###############################
        gaitFilename="PN01NORMSTAT.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
                                      markerDiameter=14,
                                      viconCGM1compatible=False)
        modMotion.compute()


        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait).display()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        plot("RSpineAngles",acqGait,"cgm1_6dof")
        plot("LSpineAngles",acqGait,"cgm1_6dof")


        np.testing.assert_almost_equal( acqGait.GetPoint("RSpineAngles").GetValues(),
                                    acqGait.GetPoint("RSpineAngles_cgm1_6dof").GetValues(), decimal =2)
        np.testing.assert_almost_equal( acqGait.GetPoint("LSpineAngles").GetValues(),
                                    acqGait.GetPoint("LSpineAngles_cgm1_6dof").GetValues(), decimal =2)

        btkTools.smartWriter(acqGait,"fullbody.c3d")



if __name__ == "__main__":



    #CGM1_angleTest.CGM1_upperLimb()
    #CGM1_angleTest.CGM1_upperLimb_absoluteAngles()
    #CGM1_angleTest.CGM1_fullbody()
    #CGM1_angleTest.CGM1_upperLimb_absoluteAngles_static()
    CGM1_angleTest.CGM1_fullbody_static()


    logging.info("######## PROCESS CGM1 --> Done ######")
