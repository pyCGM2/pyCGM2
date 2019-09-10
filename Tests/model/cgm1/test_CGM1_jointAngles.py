# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 13:59:48 2016

@author: aaa34169


I prefer numpy.testing than unitest.
easy to debug and assert method better suits.

"""

import matplotlib.pyplot as plt
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

import pyCGM2

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric
from pyCGM2.Processing import progressionFrame




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

class CGM1_motionJCSTest():

    @classmethod
    def basicCGM1(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def basicCGM1_flatFoot(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic FlatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   leftFlatFoot = True, rightFlatFoot = True).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

    @classmethod
    def advancedCGM1_kad_noOptions(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()



        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

    @classmethod
    def advancedCGM1_kad_flatFoot(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()



        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   leftFlatFoot = True, rightFlatFoot = True).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


class CGM1_motionAbsoluteAnglesTest():

    @classmethod
    def basicCGM1_absoluteAngles_lowerLimb(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]

        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                              segmentLabels=["Left Foot","Right Foot","Pelvis"],
                              angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                              eulerSequences=["TOR","TOR", "TOR"],
                              globalFrameOrientation = globalFrame,
                              forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqGait, "verifX.c3d")
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]

        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")



        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def basicCGM1_absoluteAngles_lowerLimb_AxisY(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-Y axis\\"
        staticFilename = "sujet 1 Cal 01.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 75.0,
        'LeftLegLength' : 820.0,
        'RightLegLength' : 820.0 ,
        'LeftKneeWidth' : 103.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 71.0,
        'RightAnkleWidth' : 71.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe Y -------
        gaitFilename="marchePIG04.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]

        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                              segmentLabels=["Left Foot","Right Foot","Pelvis"],
                              angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                              eulerSequences=["TOR","TOR", "TOR"],
                              globalFrameOrientation = globalFrame,
                              forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "verifY.c3d")
        #---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


        # ------ Test 2 Motion Axe -Y -------
        gaitFilename="marchePIG04.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)




class CGM1_motionFullAnglesTest():
    @classmethod
    def basicCGM1(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

         # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def basicCGM1_manualTibialTorsion(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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

        optional_mp={
        'LeftTibialTorsion' : -30.0,
        'RightTibialTorsion' : -30.0
        }
        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,viconCGM1compatible=True)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,viconCGM1compatible=True)
        modMotion.compute()

         # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def advancedCGM1_kad_noOptions(cls):
        """

        """

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

         # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def advancedCGM1_kad_flatFoot(cls):
        """

        """

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   leftFlatFoot = True, rightFlatFoot = True).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

         # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)



    @classmethod
    def advancedCGM1_kad_midMaleolus(cls):
        """

        """

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-Med\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

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
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                        viconCGM1compatible = True)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")


        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "advancedCGM1_kad_midMaleolus_viconComaptible-14.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =2)


        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 viconCGM1compatible = True)
        modMotion.compute()

         # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "test.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def advancedCGM1_kad_manualTibialTorsion(cls):
        """

        """

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-manualTibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()


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

        optional_mp={
        'LeftTibialTorsion' : -30,
        'RightTibialTorsion' : -30
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 viconCGM1compatible = True)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")


        # absolute angles
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]

        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqGait, "advancedCGM1_kad_manualTibialTorsion_viconCompatible-14.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =2)


        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =2)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =2)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =2)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =2)



if __name__ == "__main__":


    plt.close("all")

    logging.info("######## PROCESS CGM1 - JCSK ######")
    CGM1_motionJCSTest.basicCGM1()
    CGM1_motionJCSTest.basicCGM1_flatFoot()
    CGM1_motionJCSTest.advancedCGM1_kad_noOptions()
    CGM1_motionJCSTest.advancedCGM1_kad_flatFoot()
    logging.info("######## PROCESS CGM1 - JCSK --> Done ######")

    logging.info("######## PROCESS CGM1 - Absolute ######")
    CGM1_motionAbsoluteAnglesTest.basicCGM1_absoluteAngles_lowerLimb()
    CGM1_motionAbsoluteAnglesTest.basicCGM1_absoluteAngles_lowerLimb_AxisY()
    logging.info("######## PROCESS CGM1 - Absolute ---> Done ######")

    logging.info("######## PROCESS CGM1 - Full angles ######")
    CGM1_motionFullAnglesTest.basicCGM1()
    CGM1_motionFullAnglesTest.basicCGM1_manualTibialTorsion()  # reproduce vicon error
    CGM1_motionFullAnglesTest.advancedCGM1_kad_noOptions()
    CGM1_motionFullAnglesTest.advancedCGM1_kad_flatFoot()
    CGM1_motionFullAnglesTest.advancedCGM1_kad_midMaleolus() # reproduce vicon error
    CGM1_motionFullAnglesTest.advancedCGM1_kad_manualTibialTorsion()
