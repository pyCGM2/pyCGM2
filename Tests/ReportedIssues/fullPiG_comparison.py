# -*- coding: utf-8 -*-
import ipdb
import os
import argparse
import traceback
import logging

import pyCGM2
from pyCGM2 import enums
from pyCGM2.Utils import files
from pyCGM2.Configurator import ModelManager
from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Tools import btkTools
from pyCGM2 import log; log.setLogger()
from pyCGM2.Eclipse import vskTools

from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Processing import progressionFrame
import numpy as np
from pyCGM2.Lib.CGM import  cgm1
import matplotlib.pyplot as plt

def main():


        # FullBody.pig_flat()
        FullBody.pig_noflat()






class FullBody():

    @classmethod
    def pig_flat(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Datatests from Vicon\\issues\\pig_flatOptions"+"\\"

        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(DATA_PATH +  staticFilename))


        markerDiameter=14
        mp={
        'Bodymass'   : 75.0,
        'LeftLegLength' : 940.0,
        'RightLegLength' : 940.0 ,
        'LeftKneeWidth' : 105.0,
        'RightKneeWidth' : 105.0,
        'LeftAnkleWidth' : 70.0,
        'RightAnkleWidth' : 70.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 40,
        'LeftElbowWidth' : 74,
        'LeftWristWidth' : 55 ,
        'LeftHandThickness' : 34 ,
        'RightShoulderOffset'   : 40,
        'RightElbowWidth' : 74,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 34}

        optional_mp={
        'InterAsisDistance'   :  0,#0,
        'LeftAsisTrocanterDistance' :  0,#0,
        'LeftTibialTorsion' :  0,#0,
        'LeftThighRotation' :  0,#0,
        'LeftShankRotation' : 0,#0,
        'RightAsisTrocanterDistance' : 0,#0,
        'RightTibialTorsion' :  0,#0,
        'RightThighRotation' :  0,#0,
        'RightShankRotation' : 0}



        logging.info("=============Calibration=============")
        model,finalAcqStatic = cgm1.calibrate(DATA_PATH,
            staticFilename,
            None,
            mp,
            optional_mp,
            True,
            True,
            True,
            14,
            "test",
            displayCoordinateSystem=True)



        btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        logging.info("Static Calibration -----> Done")

        gaitFilename="Walking 9.c3d"
        logging.info("=============Fitting=============")

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            14,
            "test",
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)


        btkTools.smartWriter(acqGait, str(reconstructFilenameLabelled[:-4]+"-pyCGM2modelled.c3d"))
        logging.info("---->dynamic trial (%s) processed" %(reconstructFilenameLabelled))

        angleLabel ="LHeadAngles"

        fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)

        ax1.plot(acqGait.GetPoint(angleLabel).GetValues()[:,0],"-r")
        ax1.plot(acqGait.GetPoint(angleLabel+"_test").GetValues()[:,0],"-b")


        ax2.plot(acqGait.GetPoint(angleLabel).GetValues()[:,1],"-r")
        ax2.plot(acqGait.GetPoint(angleLabel+"_test").GetValues()[:,1],"-b")

        ax3.plot(acqGait.GetPoint(angleLabel).GetValues()[:,2],"-r")
        ax3.plot(acqGait.GetPoint(angleLabel+"_test").GetValues()[:,2],"-b")


        plt.show()

    @classmethod
    def pig_noflat(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Datatests from Vicon\\issues\\pig_noOptions"+"\\"

        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(DATA_PATH +  staticFilename))


        markerDiameter=14
        mp={
        'Bodymass'   : 75.0,
        'LeftLegLength' : 940.0,
        'RightLegLength' : 940.0 ,
        'LeftKneeWidth' : 105.0,
        'RightKneeWidth' : 105.0,
        'LeftAnkleWidth' : 70.0,
        'RightAnkleWidth' : 70.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 40,
        'LeftElbowWidth' : 74,
        'LeftWristWidth' : 55 ,
        'LeftHandThickness' : 34 ,
        'RightShoulderOffset'   : 40,
        'RightElbowWidth' : 74,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 34}

        optional_mp={
        'InterAsisDistance'   :  0,#0,
        'LeftAsisTrocanterDistance' :  0,#0,
        'LeftTibialTorsion' :  0,#0,
        'LeftThighRotation' :  0,#0,
        'LeftShankRotation' : 0,#0,
        'RightAsisTrocanterDistance' : 0,#0,
        'RightTibialTorsion' :  0,#0,
        'RightThighRotation' :  0,#0,
        'RightShankRotation' : 0}



        logging.info("=============Calibration=============")
        model,finalAcqStatic = cgm1.calibrate(DATA_PATH,
            staticFilename,
            None,
            mp,
            optional_mp,
            False,
            False,
            False,
            14,
            "test",
            displayCoordinateSystem=True)



        btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        logging.info("Static Calibration -----> Done")

        gaitFilename="Walking 9.c3d"
        logging.info("=============Fitting=============")

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            14,
            "test",
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)


        btkTools.smartWriter(acqGait, str(reconstructFilenameLabelled[:-4]+"-pyCGM2modelled.c3d"))
        logging.info("---->dynamic trial (%s) processed" %(reconstructFilenameLabelled))




        angleLabel ="LHeadAngles"

        fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)

        ax1.plot(acqGait.GetPoint(angleLabel).GetValues()[:,0],"-r")
        ax1.plot(acqGait.GetPoint(angleLabel+"_test").GetValues()[:,0],"-b")


        ax2.plot(acqGait.GetPoint(angleLabel).GetValues()[:,1],"-r")
        ax2.plot(acqGait.GetPoint(angleLabel+"_test").GetValues()[:,1],"-b")

        ax3.plot(acqGait.GetPoint(angleLabel).GetValues()[:,2],"-r")
        ax3.plot(acqGait.GetPoint(angleLabel+"_test").GetValues()[:,2],"-b")


        plt.show()

        # np.testing.assert_almost_equal( acqGait.GetPoint("LShoulderAngles").GetValues(),
        #                                acqGait.GetPoint("LShoulderAngles_cgm1_6dof").GetValues(), decimal =3)

if __name__ == "__main__":

    plt.close("all")
    main()


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # ---definition---
# model=cgm.CGM1()
# model.configure(bodyPart = enums.BodyPart.LowerLimb)
#
#
#
# markerDiameter=14
# mp={
# 'Bodymass'   : 75.0,
# 'LeftLegLength' : 940.0,
# 'RightLegLength' : 940.0 ,
# 'LeftKneeWidth' : 105.0,
# 'RightKneeWidth' : 105.0,
# 'LeftAnkleWidth' : 70.0,
# 'RightAnkleWidth' : 70.0,
# 'LeftSoleDelta' : 0,
# 'RightSoleDelta' : 0,
# }
# model.addAnthropoInputParameters(mp)
#
# # -----------CGM STATIC CALIBRATION--------------------
# scp=modelFilters.StaticCalibrationProcedure(model)
#
# modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
#                                     leftFlatFoot = True,
#                                     rightFlatFoot = True,
#                                     markerDiameter = 14.0,
#                                     headFlat= True,
#                                     viconCGM1compatible=True
#                                     ).compute()
#
#
#
# # -------- CGM FITTING -------------------------------------------------
# # --- motion ----
# gaitFilename="Walking 9.c3d"
# acqGait = btkTools.smartReader(str(DATA_PATH +  gaitFilename))
#
#
# modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
#                                           markerDiameter=14.0,
#                                           viconCGM1compatible=True)
# modMotion.compute()
#
#
# # relative angles
# modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
#
#
# pfp = progressionFrame.PelvisProgressionFrameProcedure()
# pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
# pff.compute()
# globalFrame = pff.outputs["globalFrame"]
# forwardProgression = pff.outputs["forwardProgression"]
#
# modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
#     segmentLabels=["Left Foot","Right Foot","Pelvis"],
#     angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
#     eulerSequences=["TOR","TOR", "TOR"],
#     globalFrameOrientation = globalFrame,
#     forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")
#
#
# btkTools.smartWriter(acqGait, "tests.c3d")
