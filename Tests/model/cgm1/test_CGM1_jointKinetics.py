# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import pdb
import logging

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()  

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator, frame, bodySegmentParameters, forceplates
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric

plt.close("all")

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
    def basicCGM1_distal(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic-filtered\\"
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
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
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

        # writer        
        btkTools.smartWriter(acqGait,"testInvDyn.c3d")


        # TEST ------
        compareKinetics(acqGait, 5, -5, 0.2, 50.0, 0.2 )


    @classmethod
    def basicCGM1_proximal(cls):

        MAIN_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGA-Data\\CGM1\\PIG standard\\basic-filtered\\"
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
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
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

        #btkTools.smartWriter(acqGait,"testInvDyn.c3d")

        # TEST ------
        compareKinetics(acqGait, 5, -5, 0.2, 40.0, 0.1 )


    @classmethod
    def basicCGM1_global(cls):

        MAIN_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGA-Data\\CGM1\\PIG standard\\basic-filtered\\"
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
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
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

        #btkTools.smartWriter(acqGait,"testInvDyn.c3d")

        # TEST ------
        compareKinetics(acqGait, 5, -5, 0.2, 40.0, 0.1 )


class CGM1_motionInverseDynamics_pathologicalSubjectTest(): 

    @classmethod
    def basicCGM1_distal(cls):

        MAIN_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGA-Data\\CGM1\\PIG standard\\basic_pathologicalSubject\\"
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
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
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

        MAIN_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGA-Data\\CGM1\\PIG standard\\basic_pathologicalSubject\\"
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
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
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

        MAIN_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGA-Data\\CGM1\\PIG standard\\basic_pathologicalSubject\\"
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
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
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


class CGM1_motionInverseDynamics_batchprocessing_Test(): 

    @classmethod
    def basicCGM1_distal(cls):

        MAIN_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGA-Data\\CGM1\\PIG standard\\basic-filtered\\"
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


        for gaitFilename in ["MRI-US-01, 2008-08-08, 3DGA 14.distal.c3d" , "MRI-US-01, 2008-08-08, 3DGA 13.distal.c3d"]:

            acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
    
            
            # Motion FILTER 
            # optimisation segmentaire et calibration fonctionnel
            modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
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
    
            btkTools.smartWriter(acqGait, str(gaitFilename[:-4]+"_testInvDyn.c3d"))    
    
            # TEST ------
            compareKinetics(acqGait, 5, -5, 0.2, 50.0, 0.2 )

if __name__ == "__main__":
    
    logging.info("######## PROCESS CGM1 - InverseDynamics ######")    
    CGM1_motionInverseDynamicsTest.basicCGM1_distal() 
    CGM1_motionInverseDynamicsTest.basicCGM1_proximal()
    CGM1_motionInverseDynamicsTest.basicCGM1_global()
    
    CGM1_motionInverseDynamics_pathologicalSubjectTest.basicCGM1_distal()
    CGM1_motionInverseDynamics_pathologicalSubjectTest.basicCGM1_proximal()
    #CGM1_motionInverseDynamics_pathologicalSubjectTest.basicCGM1_global() # No success -  TODO : with Y as traveling axis, i got inversion on X and Y Force Components
    
    
    CGM1_motionInverseDynamics_batchprocessing_Test.basicCGM1_distal()
    logging.info("######## PROCESS CGM1 - InverseDynamics ----> Done ######")    