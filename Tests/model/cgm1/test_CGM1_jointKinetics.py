# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools

from pyCGM2.ForcePlates import  forceplates
from pyCGM2.Model import  modelFilters,modelDecorator, frame,bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm

import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric

plt.close("all")

def plotMoment(acqGait,label1,label2,title=None):
    plt.figure()
    if title is not None : plt.suptitle(title)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
    ax1.plot(acqGait.GetPoint(label1).GetValues()[:,0])
    ax1.plot(acqGait.GetPoint(label2).GetValues()[:,0],"-r")

    ax2.plot(acqGait.GetPoint(label1).GetValues()[:,1])
    ax2.plot(acqGait.GetPoint(label2).GetValues()[:,1],"-r")

    ax3.plot(acqGait.GetPoint(label1).GetValues()[:,2])
    ax3.plot(acqGait.GetPoint(label2).GetValues()[:,2],"-r")

def compareKinetics(acqGait, init, end, forceThreshold, momentThreshold, powerThreshold ):

    forceArrayThreshold = np.array([forceThreshold, forceThreshold, forceThreshold])
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LHipForce").GetValues()[init:end,:]-acqGait.GetPoint("LHipForce_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 forceArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LKneeForce").GetValues()[init:end,:]-acqGait.GetPoint("LKneeForce_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 forceArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LAnkleForce").GetValues()[init:end,:]-acqGait.GetPoint("LAnkleForce_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 forceArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RHipForce").GetValues()[init:end,:]-acqGait.GetPoint("RHipForce_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 forceArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RKneeForce").GetValues()[init:end,:]-acqGait.GetPoint("RKneeForce_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 forceArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RAnkleForce").GetValues()[init:end,:]-acqGait.GetPoint("RAnkleForce_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 forceArrayThreshold)


    momentArrayThreshold = np.array([momentThreshold, momentThreshold, momentThreshold])

    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RHipMoment").GetValues()[init:end,:]-acqGait.GetPoint("RHipMoment_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 momentArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RKneeMoment").GetValues()[init:end,:]-acqGait.GetPoint("RKneeMoment_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 momentArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RAnkleMoment").GetValues()[init:end,:]-acqGait.GetPoint("RAnkleMoment_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 momentArrayThreshold)

    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LHipMoment").GetValues()[init:end,:]-acqGait.GetPoint("LHipMoment_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 momentArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LKneeMoment").GetValues()[init:end,:]-acqGait.GetPoint("LKneeMoment_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 momentArrayThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LAnkleMoment").GetValues()[init:end,:]-acqGait.GetPoint("LAnkleMoment_cgm1_6dof").GetValues()[init:end,:]), axis = 0),
                                 momentArrayThreshold)

    powerThreshold = powerThreshold
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LHipPower").GetValues()[init:end,:]-acqGait.GetPoint("LHipPower_cgm1_6dof").GetValues()[init:end,:]), axis = 0)[2],
                                 powerThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LKneePower").GetValues()[init:end,:]-acqGait.GetPoint("LKneePower_cgm1_6dof").GetValues()[init:end,:]), axis = 0)[2],
                                 powerThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("LAnklePower").GetValues()[init:end,:]-acqGait.GetPoint("LAnklePower_cgm1_6dof").GetValues()[init:end,:]), axis = 0)[2],
                                 powerThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RHipPower").GetValues()[init:end,:]-acqGait.GetPoint("RHipPower_cgm1_6dof").GetValues()[init:end,:]), axis = 0)[2],
                                 powerThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RKneePower").GetValues()[init:end,:]-acqGait.GetPoint("RKneePower_cgm1_6dof").GetValues()[init:end,:]), axis = 0)[2],
                                 powerThreshold)
    np.testing.assert_array_less(numeric.rms((acqGait.GetPoint("RAnklePower").GetValues()[init:end,:]-acqGait.GetPoint("RAnklePower_cgm1_6dof").GetValues()[init:end,:]), axis = 0)[2],
                                 powerThreshold)


class CGM1_motionInverseDynamicsTest():

    @classmethod
    def basicCGM1_distal(cls,plotFlag=False):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-filtered\\"
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
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 13.distal.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,"LR",
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()


        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Distal,
                             viconCGM1compatible=True
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        # writer
        #btkTools.smartWriter(acqGait,"testInvDyn.c3d")

        if plotFlag:
            plotMoment(acqGait,"LHipMoment","LHipMoment_cgm1_6dof")
            plotMoment(acqGait,"LKneeMoment","LKneeMoment_cgm1_6dof")
            plotMoment(acqGait,"LAnkleMoment","LAnkleMoment_cgm1_6dof")

            plotMoment(acqGait,"RHipMoment","RHipMoment_cgm1_6dof")
            plotMoment(acqGait,"RKneeMoment","RKneeMoment_cgm1_6dof")
            plotMoment(acqGait,"RAnkleMoment","RAnkleMoment_cgm1_6dof")

            plt.show()

        # TEST ------
        #compareKinetics(acqGait, 5, -5, 0.2, 50.0, 0.2 )


    @classmethod
    def basicCGM1_proximal(cls,plotFlag=False):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"CGM1\\CGM1-TESTS\\basic-filtered\\"
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
        }
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.proximal.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,"RL",
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()


        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Proximal,
                             viconCGM1compatible=True
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait,"testInvDyn.c3d")
        if plotFlag:
            plotMoment(acqGait,"LHipMoment","LHipMoment_cgm1_6dof")
            plotMoment(acqGait,"LKneeMoment","LKneeMoment_cgm1_6dof")
            plotMoment(acqGait,"LAnkleMoment","LAnkleMoment_cgm1_6dof")

            plotMoment(acqGait,"RHipMoment","RHipMoment_cgm1_6dof")
            plotMoment(acqGait,"RKneeMoment","RKneeMoment_cgm1_6dof")
            plotMoment(acqGait,"RAnkleMoment","RAnkleMoment_cgm1_6dof")
            plt.show()

        # TEST ------
        #compareKinetics(acqGait, 5, -5, 0.2, 40.0, 0.1 )


    @classmethod
    def basicCGM1_global(cls,plotFlag=False):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"CGM1\\CGM1-TESTS\\basic-filtered\\"
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
        }
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.global.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,"RL",
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()


        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Global
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        #btkTools.smartWriter(acqGait,"testInvDyn.c3d")

        if plotFlag:
            plotMoment(acqGait,"LHipMoment","LHipMoment_cgm1_6dof")
            plotMoment(acqGait,"LKneeMoment","LKneeMoment_cgm1_6dof")
            plotMoment(acqGait,"LAnkleMoment","LAnkleMoment_cgm1_6dof")

            plotMoment(acqGait,"RHipMoment","RHipMoment_cgm1_6dof")
            plotMoment(acqGait,"RKneeMoment","RKneeMoment_cgm1_6dof")
            plotMoment(acqGait,"RAnkleMoment","RAnkleMoment_cgm1_6dof")
            plt.show()

        # TEST ------
        compareKinetics(acqGait, 5, -5, 0.2, 40.0, 0.1 )

    @classmethod
    def kadMedCGM1_distal(cls,plotFlag=False):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
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
                                                 viconCGM1compatible=True)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,"RL",
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()


        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Distal,
                             viconCGM1compatible=True
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        # writer
        btkTools.smartWriter(acqGait,"testInvDyn_kadMed.c3d")
        if plotFlag:
            plotMoment(acqGait,"LAnkleMoment","LAnkleMoment_cgm1_6dof","kadMedCGM1_distal-LAnkleMoment")
            plotMoment(acqGait,"RAnkleMoment","RAnkleMoment_cgm1_6dof","kadMedCGM1_distal-RAnkleMoment")
            plt.show()

        # TEST ------
        #compareKinetics(acqGait, 5, -5, 0.2, 60.0, 0.2 )



    @classmethod
    def kadMedCGM1_proximal(cls,plotFlag=False):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
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

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.Proximal.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 viconCGM1compatible=False)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,"RL",
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()


        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Proximal,
                             viconCGM1compatible=True
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        # writer
        btkTools.smartWriter(acqGait,"testInvDyn_kadMed.c3d")

        if plotFlag:
            plotMoment(acqGait,"LAnkleMoment","LAnkleMoment_cgm1_6dof","kadMedCGM1_proximal-LAnkleMoment")
            plotMoment(acqGait,"RAnkleMoment","RAnkleMoment_cgm1_6dof","kadMedCGM1_proximal-RAnkleMoment")
            plt.show()
#


        # TEST ------
        #compareKinetics(acqGait, 5, -5, 0.2, 50.0, 0.2 )



class CGM1_motionInverseDynamics_pathologicalSubjectTest():

    @classmethod
    def basicCGM1_distal(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"CGM1\\CGM1-TESTS\\basic_pathologicalSubject\\"
        staticFilename = "BOVE Vincent Cal 01.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 72.0,
        'LeftLegLength' : 840.0,
        'RightLegLength' : 850.0 ,
        'LeftKneeWidth' : 105.0,
        'RightKneeWidth' : 110.4,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 74.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Travelling Axis Y -------
        gaitFilename="20120213_BV-PRE-S-NNNN-I-dyn 04.distal.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()

        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Distal
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqGait,"testInvDynPathoYdist.c3d")

        # TEST ------
        compareKinetics(acqGait, 5, -5, 0.2, 40.0, 0.1 )

    @classmethod
    def basicCGM1_proximal(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"CGM1\\CGM1-TESTS\\basic_pathologicalSubject\\"
        staticFilename = "BOVE Vincent Cal 01.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 72.0,
        'LeftLegLength' : 840.0,
        'RightLegLength' : 850.0 ,
        'LeftKneeWidth' : 105.0,
        'RightKneeWidth' : 110.4,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 74.0,
        }
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Travelling Axis Y -------
        gaitFilename="20120213_BV-PRE-S-NNNN-I-dyn 04.proximal.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()

        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Proximal
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqGait,"testInvDynPathoYprox.c3d")

        # TEST ------
        compareKinetics(acqGait, 5, -5, 0.2, 40.0, 0.1 )

    @classmethod
    def basicCGM1_global(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"CGM1\\CGM1-TESTS\\basic_pathologicalSubject\\"
        staticFilename = "BOVE Vincent Cal 01.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 72.0,
        'LeftLegLength' : 840.0,
        'RightLegLength' : 850.0 ,
        'LeftKneeWidth' : 105.0,
        'RightKneeWidth' : 110.4,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 74.0,
        }
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Travelling Axis Y -------
        gaitFilename="20120213_BV-PRE-S-NNNN-I-dyn 04.global2.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied
        forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()

        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Global
                             ).compute(pointLabelSuffix="cgm1_6dof")


        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqGait,"testInvDynPatho.c3d")

        # TEST ------
        compareKinetics(acqGait, 5, -5, 0.2, 40.0, 0.1 )




if __name__ == "__main__":

    logging.info("######## PROCESS CGM1 - InverseDynamics ######")
    CGM1_motionInverseDynamicsTest.basicCGM1_distal(plotFlag=False)
    CGM1_motionInverseDynamicsTest.basicCGM1_proximal(plotFlag=False)
    CGM1_motionInverseDynamicsTest.basicCGM1_global(plotFlag=False)
    CGM1_motionInverseDynamicsTest.kadMedCGM1_distal(plotFlag=True) # no tests
    CGM1_motionInverseDynamicsTest.kadMedCGM1_proximal(plotFlag=True) # no tests

    CGM1_motionInverseDynamics_pathologicalSubjectTest.basicCGM1_distal()
    CGM1_motionInverseDynamics_pathologicalSubjectTest.basicCGM1_proximal()
    #CGM1_motionInverseDynamics_pathologicalSubjectTest.basicCGM1_global() # No success -  TODO : with Y as traveling axis, i got inversion on X and Y Force Components


    logging.info("######## PROCESS CGM1 - InverseDynamics ----> Done ######")
