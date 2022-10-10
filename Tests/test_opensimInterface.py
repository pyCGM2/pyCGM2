# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensimInterface.py::Test_GenericXmlprocedures::test_cgm23_complete
# pytest -s --disable-pytest-warnings  test_opensimInterface.py::Test_CGM_Xmlprocedures::test_cgm23_scaling_ik_muscle
from pickle import NONE
import ipdb
import os
import matplotlib.pyplot as plt


import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2
from pyCGM2 import opensim4 as opensim

from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools
from pyCGM2.Tools import  opensimTools
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2 import enums
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Lib.Processing import progression
from pyCGM2.ForcePlates import forceplates

from pyCGM2.Model.Opensim import opensimFilters

from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures.scaling import opensimScalingInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.inverseKinematics import opensimInverseKinematicsInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.inverseDynamics import opensimInverseDynamicsInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.analysisReport import opensimAnalysesInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.staticOptimisation import opensimStaticOptimizationInterfaceProcedure



class Test_GenericXmlprocedures:
    def test_cgm23_complete(self):

        data_path = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\CGM23-progressionX-test\\"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        staticFilename = "static.c3d" 
        gaitFilename = "gait1.c3d"

        modelVersion = "CGM2.3"

        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]


        markerDiameter=14
        required_mp={
        'Bodymass'   : 71.0,
        'Height'   : 1780.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset' : 0,
        'RightShoulderOffset' : 0,
        'LeftElbowWidth' : 0,
        'LeftWristWidth' : 0,
        'LeftHandThickness' : 0,
        'RightElbowWidth' : 0,
        'RightWristWidth' : 0,
        'RightHandThickness' : 0
        }
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        # --- Calibration ---
        acqStatic = btkTools.smartReader(data_path +  staticFilename)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model =cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)

        # ---- Calibration ----
        scp = modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        markerDiameter=markerDiameter).compute()

        

        # ------- FITTING 0--------------------------------------

        acqGait = btkTools.smartReader(data_path +  gaitFilename)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        # # ------- OLD OPENSIM --------------------------------------

        # # --- osim builder ---
        # cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
        # markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml"
        # osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"

        # oscf = opensimFilters.opensimCalibrationFilter(osimfile,
        #                                         model,
        #                                         cgmCalibrationprocedure,
        #                                         DATA_PATH)
        # oscf.addMarkerSet(markersetFile)
        # scalingOsim = oscf.build(exportOsim=False)

        # # --- IK ---
        # cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
        # iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml"

        # osrf = opensimFilters.opensimFittingFilter(iksetupFile,
        #                                                   scalingOsim,
        #                                                   cgmFittingProcedure,
        #                                                   DATA_PATH,
        #                                                   acqGait )
        # acqIK_old = osrf.run(str(DATA_PATH + gaitFilename ),exportSetUp=False)

        # # -------- FITTING 1 ------------------

        # modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK_old,model,enums.motionMethod.Sodervisk,
        #                                             useForMotionTest=True)
        # modMotion_ik.compute()

        # finalJcs =modelFilters.ModelJCSFilter(model,acqIK_old)
        # finalJcs.compute(description="old", pointLabelSuffix = "old")


        # ------- NEW OPENSIM --------------------------------------

        # --- osim builder ---
        
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"interface\\CGM23\\OsimToCGM.settings")

        # scaling
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\pycgm2-gait2354_simbody.osim"
        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.ScalingXmlProcedure(data_path)
        proc.setSetupFiles(osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.setStaticTrial( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        proc.prepareXml()
        
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()
        


        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-ikSetUp_template.xml"

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait)

        procIK = opensimInverseKinematicsInterfaceProcedure.InverseKinematicXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.setSetupFile(ikTemplateFullFile)
        procIK.prepareDynamicTrial(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-8)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        procIK.prepareXml()
        

        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.pushFittedMarkersIntoAcquisition()
        oiikf.pushMotToAcq(osimConverterSettings)
        acqIK =oiikf.getAcq()

        



        # ----- compute angles
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = "new")#

        #correct the ankle angles in the mot files
        motDataframe = opensimIO.OpensimDataFrame(data_path+"musculoskeletal_modelling\\",gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,2]
        motDataframe.save()


        # fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        # ax1 = plt.subplot(1,3,1)
        # ax2 = plt.subplot(1,3,2)
        # ax3 = plt.subplot(1,3,3)

        
        # ax1.plot(acqIK_old.GetPoint("LKneeAngles_old").GetValues()[:,0],"-r")
        # ax1.plot(acqIK.GetPoint("LKneeAngles_new").GetValues()[:,0],"-ob")

        # ax2.plot(acqIK_old.GetPoint("LKneeAngles_old").GetValues()[:,1],"-r")
        # ax2.plot(acqIK.GetPoint("LKneeAngles_new").GetValues()[:,1],"-ob")

        # ax3.plot(acqIK_old.GetPoint("LKneeAngles_old").GetValues()[:,2],"-r")
        # ax3.plot(acqIK.GetPoint("LKneeAngles_new").GetValues()[:,2],"-ob")

        # fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        # ax1 = plt.subplot(1,3,1)
        # ax2 = plt.subplot(1,3,2)
        # ax3 = plt.subplot(1,3,3)

        # ax1.plot(acqIK_old.GetPoint("LAnkleAngles_old").GetValues()[:,0],"-r")
        # ax1.plot(acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,0],"-ob")

        # ax2.plot(acqIK_old.GetPoint("LAnkleAngles_old").GetValues()[:,1],"-r")
        # ax2.plot(acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,1],"-ob")

        # ax3.plot(acqIK_old.GetPoint("LAnkleAngles_old").GetValues()[:,2],"-r")
        # ax3.plot(acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,2],"-ob")

        plt.show()
        

        # --- ID ------
        idTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-idToolSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"


        procID = opensimInverseDynamicsInterfaceProcedure.InverseDynamicsXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procID.setProgression(progressionAxis,forwardProgression)
        procID.prepareDynamicTrial(acqIK,gaitFilename[:-4],None)
        procID.setSetupFiles(idTemplateFullFile,externalLoadTemplateFullFile)
        procID.setTimeRange()
        procID.prepareXml()
    
        oiidf = opensimInterfaceFilters.opensimInterfaceInverseDynamicsFilter(procID)
        oiidf.run()
        oiidf.pushStoToAcq(model.mp["Bodymass"], osimConverterSettings)        
       
        # --- ----
        # find foot  in contact
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqIK)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqIK,mappedForcePlate,
                                    leftSegmentLabel="Left Foot",
                                    rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)

        #---- Joint kinetics----
        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                                acqIK,
                                procedure = idp,
                                projection = enums.MomentProjection.JCS,
                                globalFrameOrientation = globalFrame,
                                forwardProgression = forwardProgression
                                ).compute(pointLabelSuffix=None)
        # #---- Joint energetics----
        modelFilters.JointPowerFilter(model,acqIK).compute(pointLabelSuffix="cgm")




       # --- opensimStaticOptimizationInterfaceProcedure ------

        soTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-soSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"
 
        procSO = opensimStaticOptimizationInterfaceProcedure.StaticOptimisationXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procSO.setProgression(progressionAxis,forwardProgression)
        procSO.prepareDynamicTrial(acqIK,gaitFilename[:-4],None)
        procSO.setSetupFiles(soTemplateFullFile,externalLoadTemplateFullFile)
        procSO.setTimeRange()
        procSO.prepareXml()

        oiamf = opensimInterfaceFilters.opensimInterfaceStaticOptimizationFilter(procSO)
        oiamf.run()


        # --- Analyses ------

        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-muscleAnalysisSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.AnalysesXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procAna.setSetupFiles(anaTemplateFullFile,externalLoadTemplateFullFile)
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.prepareDynamicTrial(acqIK,gaitFilename[:-4],None)
        procAna.setTimeRange()
        procAna.prepareXml()

        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.pushStoToAcq()
        
        
        btkTools.smartWriter(acqIK,"Opensim-check.c3d")


    def test_cgm23_scaling_ik_muscle(self):

        data_path = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\CGM23-progressionX-test\\"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        staticFilename = "static.c3d" 
        gaitFilename = "gait1.c3d"

        modelVersion = "CGM2.3"

        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]


        markerDiameter=14
        required_mp={
        'Bodymass'   : 71.0,
        'Height'   : 1780.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset' : 0,
        'RightShoulderOffset' : 0,
        'LeftElbowWidth' : 0,
        'LeftWristWidth' : 0,
        'LeftHandThickness' : 0,
        'RightElbowWidth' : 0,
        'RightWristWidth' : 0,
        'RightHandThickness' : 0
        }
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        # --- Calibration ---
        acqStatic = btkTools.smartReader(data_path +  staticFilename)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model =cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)

        # ---- Calibration ----
        scp = modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        markerDiameter=markerDiameter).compute()

        

        # ------- FITTING 0--------------------------------------

        acqGait = btkTools.smartReader(data_path +  gaitFilename)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        
        # ------- NEW OPENSIM --------------------------------------

        # --- osim builder ---
        
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"interface\\CGM23\\OsimToCGM.settings")

        # scaling
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\pycgm2-gait2354_simbody.osim"
        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.ScalingXmlProcedure(data_path)
        proc.setSetupFiles(osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.setStaticTrial( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        proc.prepareXml()
        
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()
        


        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-ikSetUp_template.xml"

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait)

        procIK = opensimInverseKinematicsInterfaceProcedure.InverseKinematicXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.setSetupFile(ikTemplateFullFile)
        procIK.prepareDynamicTrial(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-8)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        procIK.prepareXml()
        

        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.pushFittedMarkersIntoAcquisition()
        oiikf.pushMotToAcq(osimConverterSettings)
        acqIK =oiikf.getAcq()

        
        # ----- compute angles
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = "new")#

        #correct the ankle angles in the mot files
        motDataframe = opensimIO.OpensimDataFrame(data_path+"musculoskeletal_modelling\\",gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,2]
        motDataframe.save()

        # --- Analyses ------

        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-muscleAnalysisSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.AnalysesXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procAna.setSetupFiles(anaTemplateFullFile,externalLoadTemplateFullFile)
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.prepareDynamicTrial(acqIK,gaitFilename[:-4],None)
        procAna.setTimeRange()
        procAna.prepareXml()

        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.pushStoToAcq()
        
        
        btkTools.smartWriter(acqIK,"Opensim-check.c3d")


    def test_cgm23_scaling_ik_muscle_noLoad(self):

        data_path = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\CGM23-progressionX-test\\"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        staticFilename = "static.c3d" 
        gaitFilename = "gait1.c3d"

        modelVersion = "CGM2.3"

        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]


        markerDiameter=14
        required_mp={
        'Bodymass'   : 71.0,
        'Height'   : 1780.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset' : 0,
        'RightShoulderOffset' : 0,
        'LeftElbowWidth' : 0,
        'LeftWristWidth' : 0,
        'LeftHandThickness' : 0,
        'RightElbowWidth' : 0,
        'RightWristWidth' : 0,
        'RightHandThickness' : 0
        }
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        # --- Calibration ---
        acqStatic = btkTools.smartReader(data_path +  staticFilename)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model =cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)

        # ---- Calibration ----
        scp = modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        markerDiameter=markerDiameter).compute()

        

        # ------- FITTING 0--------------------------------------

        acqGait = btkTools.smartReader(data_path +  gaitFilename)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        
        # ------- NEW OPENSIM --------------------------------------

        # --- osim builder ---
        
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"interface\\CGM23\\OsimToCGM.settings")

        # scaling
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\pycgm2-gait2354_simbody.osim"
        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.ScalingXmlProcedure(data_path)
        proc.setSetupFiles(osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.setStaticTrial( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        proc.prepareXml()
        
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()
        


        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-ikSetUp_template.xml"

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait)

        procIK = opensimInverseKinematicsInterfaceProcedure.InverseKinematicXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.setSetupFile(ikTemplateFullFile)
        procIK.prepareDynamicTrial(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-8)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        procIK.prepareXml()
        

        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.pushFittedMarkersIntoAcquisition()
        oiikf.pushMotToAcq(osimConverterSettings)
        acqIK =oiikf.getAcq()

        



        # ----- compute angles
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = "new")#

        #correct the ankle angles in the mot files
        motDataframe = opensimIO.OpensimDataFrame(data_path+"musculoskeletal_modelling\\",gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,2]
        motDataframe.save()

        # --- Analyses ------

        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-muscleAnalysisSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.AnalysesXmlProcedure(data_path,scaledOsimName,"musculoskeletal_modelling")
        procAna.setSetupFiles(anaTemplateFullFile,None)
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.prepareDynamicTrial(acqIK,gaitFilename[:-4],None)
        procAna.setTimeRange()
        procAna.prepareXml()

        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.pushStoToAcq()
        
        btkTools.smartWriter(acqIK,"Opensim-check.c3d")


class Test_CGM_XmlProcedures:

    def test_cgm23_scaling_ik_muscle(self):

        data_path = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\CGM23-progressionX-test\\"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        staticFilename = "static.c3d" 
        gaitFilename = "gait1.c3d"

        modelVersion = "CGM2.3"

        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]


        markerDiameter=14
        required_mp={
        'Bodymass'   : 71.0,
        'Height'   : 1780.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset' : 0,
        'RightShoulderOffset' : 0,
        'LeftElbowWidth' : 0,
        'LeftWristWidth' : 0,
        'LeftHandThickness' : 0,
        'RightElbowWidth' : 0,
        'RightWristWidth' : 0,
        'RightHandThickness' : 0
        }
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        # --- Calibration ---
        acqStatic = btkTools.smartReader(data_path +  staticFilename)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model =cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)

        # ---- Calibration ----
        scp = modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        markerDiameter=markerDiameter).compute()

        

        # ------- FITTING 0--------------------------------------

        acqGait = btkTools.smartReader(data_path +  gaitFilename)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        
        # ------- NEW OPENSIM --------------------------------------

        # --- osim builder ---
        
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"interface\\CGM23\\OsimToCGM.settings")

        # scaling
        proc = opensimScalingInterfaceProcedure.ScalingXmlCgmProcedure(data_path,"CGM2.3")
        proc.setStaticTrial( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        proc.prepareXml()
        
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()
        


        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait)

        procIK = opensimInverseKinematicsInterfaceProcedure.InverseKinematicXmlCgmProcedure(data_path,scaledOsimName,"musculoskeletal_modelling","CGM2.3")
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.prepareDynamicTrial(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-8)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        procIK.prepareXml()
        

        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.pushFittedMarkersIntoAcquisition()
        oiikf.pushMotToAcq(osimConverterSettings)
        acqIK =oiikf.getAcq()

        
        # ----- compute angles
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = "new")#

        #correct the ankle angles in the mot files
        motDataframe = opensimIO.OpensimDataFrame(data_path+"musculoskeletal_modelling\\",gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles_new").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles_new").GetValues()[:,2]
        motDataframe.save()

        # --- Analyses ------
        procAna = opensimAnalysesInterfaceProcedure.AnalysesXmlCgmProcedure(data_path,scaledOsimName,"musculoskeletal_modelling","CGM2.3")
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.prepareDynamicTrial(acqIK,gaitFilename[:-4],None)
        procAna.setTimeRange()
        procAna.prepareXml()

        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.pushStoToAcq()
        
        

class Test_Generic_DrivenPose:

    def test_staticMuscleLength(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\CGM23-progressionX-test\\"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        staticFilename = "static.c3d" 
        gaitFilename = "gait1.c3d"

        modelVersion = "CGM2.3"

        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]


        markerDiameter=14
        required_mp={
        'Bodymass'   : 71.0,
        'Height'   : 1780.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset' : 0,
        'RightShoulderOffset' : 0,
        'LeftElbowWidth' : 0,
        'LeftWristWidth' : 0,
        'LeftHandThickness' : 0,
        'RightElbowWidth' : 0,
        'RightWristWidth' : 0,
        'RightHandThickness' : 0
        }
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        # --- Calibration ---
        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model =cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)

        # ---- Calibration ----
        scp = modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        markerDiameter=markerDiameter).compute()

        

        # ------- FITTING 0--------------------------------------

        acqGait = btkTools.smartReader(DATA_PATH +  gaitFilename)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)

        # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
        modMotion.compute()

        # ------- NEW OPENSIM --------------------------------------

        # --- osim builder ---
        
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"interface\\CGM23\\OsimToCGM.settings")

        # scaling
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\pycgm2-gait2354_simbody.osim"
        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.ScalingXmlProcedure(DATA_PATH,"CGM2.3")
        proc.setSetupFiles(osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.setStaticTrial( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        proc.prepareXml()
        
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()
        

        motDf = opensimIO.OpensimDataFrame(DATA_PATH, "referencePose.mot")
        motDf.getDataFrame()["pelvis_ty"] = 0.95
        motDf.save(filename="referencePose2.mot")



        # --- Analyses ------
        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-muscleAnalysisSetup_template.xml"
        externalLoadTemplateFullFile = None#pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.AnalysesXmldrivenModelProcedure(DATA_PATH,scaledOsimName,"musculoskeletal_modelling/driven_standstill","CGM2.3")
        procAna.setSetupFiles(anaTemplateFullFile,externalLoadTemplateFullFile)
        procAna.setReferencePose("referencePose.mot")
        procAna.updateReferencePos("hipKneeFlex.mot")
        procAna.prepareXml()

        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()

        muscleLengths = opensimIO.OpensimDataFrame(DATA_PATH, "musculoskeletal_modelling/driven_standstill/Driven-CGM23-analyses_MuscleAnalysis_Length.sto")
        print(muscleLengths.getDataFrame()["rect_fem_r"])
        
        
        import numpy as np
        coordinates = scaledOsim.getCoordinateSet()
        coordinates.get("hip_flexion_r").set_default_value(np.deg2rad(90))
        coordinates.get("knee_flexion_r").set_default_value(np.deg2rad(-90))

        states = scaledOsim.initSystem()
        muscles = scaledOsim.getMuscles()    
        rf = muscles.get("rect_fem_r")
        print(rf.getLength(states))
        ipdb.set_trace()
        
        

