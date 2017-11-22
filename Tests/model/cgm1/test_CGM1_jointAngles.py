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
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

import pyCGM2
# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric




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
    def basicCGM1(cls):  #def basicCGM1(self):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
    def basicCGM1_flatFoot(cls):  #def basicCGM1(self):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic FlatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
    def advancedCGM1_kad_flatFoot(cls):  #def basicCGM1(self):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
    def basicCGM1_absoluteAngles_lowerLimb(cls):  #def basicCGM1(self):
        """

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])

        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                              segmentLabels=["Left Foot","Right Foot","Pelvis"],
                              angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                              eulerSequences=["TOR","TOR", "TOR"],
                              globalFrameOrientation = globalFrame,
                              forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "verifX.c3d")
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

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic-Y axis\\"
        staticFilename = "sujet 1 Cal 01.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])

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

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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
    def basicCGM1_absoluteAngles_pelikin(cls):
        """

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\Pelikin\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])

        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                              segmentLabels=["Left Foot","Right Foot","Pelvis"],
                              angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                              eulerSequences=["TOR","TOR", "ROT"],
                              globalFrameOrientation = globalFrame,
                              forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait, "verifX.c3d")
        # ---   tests on angles

        np.testing.assert_almost_equal( acqGait.GetPoint("RNewPelAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LNewPelAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "ROT"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RNewPelAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LNewPelAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


class CGM1_motionFullAnglesTest():
    @classmethod
    def basicCGM1(cls):
        """

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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

        # tibial torsion
        ltt_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LTibialTorsion").value().GetInfo().ToDouble()[0])
        rtt_vicon =np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RTibialTorsion").value().GetInfo().ToDouble()[0])


        logging.info(" LTibialTorsion : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(ltt_vicon,model.mp_computed["LeftTibialTorsionOffset"]))
        logging.info(" RTibialTorsion : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rtt_vicon,model.mp_computed["RightTibialTorsionOffset"]))

        # foot offsets
        spf_l,sro_l = model.getViconFootOffset("Left")
        spf_r,sro_r = model.getViconFootOffset("Right")
        vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
        vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])


        logging.info(" LStaticPlantFlex : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(spf_l,vicon_spf_l))
        logging.info(" RStaticPlantFlex : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(spf_r,vicon_spf_r))
        logging.info(" LStaticRotOff : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(sro_l,vicon_sro_l))
        logging.info(" RStaticRotOff : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(sro_r,vicon_sro_r))

         # thigh and shank Offsets
        lto = model.getViconThighOffset("Left")
        lso = model.getViconShankOffset("Left")
        rto = model.getViconThighOffset("Right")
        rso = model.getViconShankOffset("Right")

        lto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0])
        lso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0])

        rto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0])
        rso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0])



        logging.info(" LThighRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(lto_vicon,lto))
        logging.info(" LShankRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(lso_vicon,lso))
        logging.info(" RThighRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rto_vicon,rto))
        logging.info(" RShankRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rso_vicon,rso))


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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqGait, "advancedCGM1_kad_midMaleolus-14.c3d")

        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)


        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


#        # tests on angles influence by Vicon error
#        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)
#        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
#                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
#        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
#                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)


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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      globalFrameOrientation = globalFrame,
                                      eulerSequences=["TOR","TOR", "TOR"],
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


        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)


#        # tests on angles influence by Vicon error
#        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)
#        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
#                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
#        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
#                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def advancedCGM1_kad_midMaleolus_viconCompatible(cls):
        """

        """

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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
    def advancedCGM1_kad_midMaleolus_viconCompatible_tibialTorsionManually(cls):
        """

        """

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
        'LeftTibialTorsion' : -12.0031,
        'RightTibialTorsion' : -17.7351
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
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



#


class CGM1_motionFullAnglesTest_customApproach():
    @classmethod
    def basicCGM1_bodyBuilderFoot(cls):
        """
        goal : know  effet on Foot kinematics of a foot referential built according ta sequence metionned in some bodybuilder code:
        LFoot = [LTOE,LAJC-LTOE,LAJC-LKJC,zyx]

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            useBodyBuilderFoot=True).compute()

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
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","LPSI","RASI","RPSI"])
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqGait, "testuseBodyBuilderFoot.c3d")





#



if __name__ == "__main__":


#    plt.close("all")

#    logging.info("######## PROCESS CGM1 - JCSK ######")
    CGM1_motionJCSTest.basicCGM1()
    CGM1_motionJCSTest.basicCGM1_flatFoot()
    CGM1_motionJCSTest.advancedCGM1_kad_noOptions()
    CGM1_motionJCSTest.advancedCGM1_kad_flatFoot()
#    logging.info("######## PROCESS CGM1 - JCSK --> Done ######")

#    #logging.info("######## PROCESS CGM1 - Absolute ######")
    CGM1_motionAbsoluteAnglesTest.basicCGM1_absoluteAngles_lowerLimb()
    CGM1_motionAbsoluteAnglesTest.basicCGM1_absoluteAngles_lowerLimb_AxisY()
    CGM1_motionAbsoluteAnglesTest.basicCGM1_absoluteAngles_pelikin()
#    logging.info("######## PROCESS CGM1 - Absolute ---> Done ######")

#    logging.info("######## PROCESS CGM1 - Full angles ######")
    CGM1_motionFullAnglesTest.basicCGM1()
    CGM1_motionFullAnglesTest.advancedCGM1_kad_noOptions()
    CGM1_motionFullAnglesTest.advancedCGM1_kad_flatFoot()
    CGM1_motionFullAnglesTest.advancedCGM1_kad_midMaleolus_viconCompatible() # reproduce vicon error
    CGM1_motionFullAnglesTest.advancedCGM1_kad_midMaleolus() # with fixed vicon error  -->  look inside some tests fail for ankle angle and foot progress

#
#    logging.info("######## PROCESS CGM1 - Full angles - CUSTOM APPROACHES ######")
#    CGM1_motionFullAnglesTest_customApproach.basicCGM1_bodyBuilderFoot()   # not really a test
#    logging.info("######## PROCESS CGM1 - Full angles ---> Done ######")
