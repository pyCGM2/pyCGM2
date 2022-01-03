# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensimInterface.py::Test_CGM23::test_cgm23_progressX_allOpensimSteps_local
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
from pyCGM2.Model.Opensim import opensimFilters,opensimInterfaceFilters
from pyCGM2.Model.Opensim import opensimScalingInterfaceProcedure
from pyCGM2.Model.Opensim import opensimInverseKinematicsInterfaceProcedure
from pyCGM2.Model.Opensim import opensimInverseDynamicsInterfaceProcedure

from pyCGM2.Model.Opensim import opensimAnalysesInterfaceProcedure
from pyCGM2.Model.Opensim import opensimStaticOptimizationInterfaceProcedure
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Lib import processing
from pyCGM2.ForcePlates import forceplates

def processCGM23(DATA_PATH,settings,staticFilename,gaitFilename):

    settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")
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
    # ---check marker set used----
    acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)
    acqStatic =  btkTools.applyTranslators(acqStatic,translators)
    trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
    actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)



    dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
    model=cgm2.CGM2_3()
    model.configure(detectedCalibrationMethods=dcm)
    model.addAnthropoInputParameters(required_mp,optional=optional_mp)
    model.setStaticTrackingMarkers(actual_trackingMarkers)


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


    # ------- Fitting --------------------------------------

    acqGait = btkTools.smartReader(DATA_PATH +  gaitFilename)
    trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
    actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)

    return model, acqStatic,acqGait,staticFilename,gaitFilename,scp

def process_oldIK(DATA_PATH,model,acqGait,gaitFilename,scp):
    # --- osim builder ---
    cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
    markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\cgm23-markerset.xml"
    osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\lowerLimb_ballsJoints.osim"

    # --- old implementation - No Scaling
    oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                            model,
                                            cgmCalibrationprocedure,
                                            DATA_PATH)
    oscf.addMarkerSet(markersetFile)
    scalingOsim = oscf.build(exportOsim=False)

    cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
    iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

    osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                      scalingOsim,
                                                      cgmFittingProcedure,
                                                      DATA_PATH,
                                                      acqGait )

    acqIK = osrf.run(str(DATA_PATH + gaitFilename ),exportSetUp=False)



    modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                useForMotionTest=True)
    modMotion.compute()

    finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
    finalJcs.compute(description="old", pointLabelSuffix = None)#

    return acqIK



class Test_CGM23:
    def test_cgm23_progressX_CHECKING(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\Hannibal-medial\\"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        #-----
        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings,"static.c3d","gait1.c3d")
        acqIK_old = process_oldIK(DATA_PATH,model,acqGait,gaitFilename,scp)

        modelVersion="CGM2.3"

        progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqGait)

        # --- osim builder ---
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\pycgm2-gait2354_simbody.osim"
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"setup\\CGM23\\OsimToC3dConverter.settings")

        # scaling
        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.highLevelScalingProcedure(DATA_PATH,modelVersion,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.preProcess( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()

        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

        procIK = opensimInverseKinematicsInterfaceProcedure.highLevelInverseKinematicsProcedure(DATA_PATH,scaledOsimName,modelVersion,ikTemplateFullFile)
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.preProcess(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-8)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.ikMarkerLocationToC3d()
        oiikf.motToC3d(osimConverterSettings)
        acqIK =oiikf.getAcq()


        # ----- compute angles
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = None)#


        fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)


        ax1.plot(acqIK_old.GetPoint("LKneeAngles").GetValues()[:,0],"-r")
        ax1.plot(acqIK.GetPoint("LKneeAngles").GetValues()[:,0],"-ob")

        ax2.plot(acqIK_old.GetPoint("LKneeAngles").GetValues()[:,1],"-r")
        ax2.plot(acqIK.GetPoint("LKneeAngles").GetValues()[:,1],"-ob")

        ax3.plot(acqIK_old.GetPoint("LKneeAngles").GetValues()[:,2],"-r")
        ax3.plot(acqIK.GetPoint("LKneeAngles").GetValues()[:,2],"-ob")

        fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)

        ax1.plot(acqIK_old.GetPoint("LAnkleAngles").GetValues()[:,0],"-r")
        ax1.plot(acqIK.GetPoint("LAnkleAngles").GetValues()[:,0],"-ob")

        ax2.plot(acqIK_old.GetPoint("LAnkleAngles").GetValues()[:,1],"-r")
        ax2.plot(acqIK.GetPoint("LAnkleAngles").GetValues()[:,1],"-ob")

        ax3.plot(acqIK_old.GetPoint("LAnkleAngles").GetValues()[:,2],"-r")
        ax3.plot(acqIK.GetPoint("LAnkleAngles").GetValues()[:,2],"-ob")

        plt.show()



    def test_cgm23_progressX_allOpensimSteps(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\Hannibal-medial\\"

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")



        #-----
        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings,"static.c3d","gait1.c3d")

        modelVersion="CGM2.3"

        progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqGait)


        # --- osim builder ---
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\pycgm2-gait2354_simbody.osim"
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"setup\\CGM23\\OsimToC3dConverter.settings")

        # scaling
        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.highLevelScalingProcedure(DATA_PATH,modelVersion,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.preProcess( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()

        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

        procIK = opensimInverseKinematicsInterfaceProcedure.highLevelInverseKinematicsProcedure(DATA_PATH,scaledOsimName,modelVersion,ikTemplateFullFile)
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.preProcess(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-5)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        # procIK.setResultsDirname("verif")
        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.ikMarkerLocationToC3d()
        oiikf.motToC3d(osimConverterSettings)
        acqIK =oiikf.getAcq()

        # ----- compute angles from rigid
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = None)#

        #correct the ankle angles
        motDataframe = opensimIO.OpensimDataFrame(DATA_PATH,gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,2]
        motDataframe.save()


        # --- ID ------
        idTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-idToolSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"

        procID = opensimInverseDynamicsInterfaceProcedure.highLevelInverseDynamicsProcedure(DATA_PATH,
            scaledOsimName,modelVersion,idTemplateFullFile,externalLoadTemplateFullFile)
        procID.setProgression(progressionAxis,forwardProgression)
        procID.preProcess(acqIK,gaitFilename[:-4])
        # procID.setResultsDirname("verif")
        procID.setTimeRange()
        oiidf = opensimInterfaceFilters.opensimInterfaceInverseDynamicsFilter(procID)
        oiidf.run()
        oiidf.stoToC3d(model.mp["Bodymass"], osimConverterSettings)

        # btkTools.smartWriter(acqIK, DATA_PATH+"verifOpensim.c3d")



        if btkTools.checkForcePlateExist(acqGait):

            # progression
            progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqIK)

            # --- force plate handling----
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
            # modelFilters.InverseDynamicFilter(model,
            #                      acqIK,
            #                      procedure = idp,
            #                      projection = enums.MomentProjection.Global,
            #                      globalFrameOrientation = globalFrame,
            #                      forwardProgression = forwardProgression
            #                      ).compute(pointLabelSuffix="Global")
            # modelFilters.InverseDynamicFilter(model,
            #                      acqIK,
            #                      procedure = idp,
            #                      projection = enums.MomentProjection.JCS_Dual,
            #                      globalFrameOrientation = globalFrame,
            #                      forwardProgression = forwardProgression
            #                      ).compute(pointLabelSuffix="Dual")
            # modelFilters.InverseDynamicFilter(model,
            #                      acqIK,
            #                      procedure = idp,
            #                      projection = enums.MomentProjection.Distal,
            #                      globalFrameOrientation = globalFrame,
            #                      forwardProgression = forwardProgression
            #                      ).compute(pointLabelSuffix="Distal")
            # modelFilters.InverseDynamicFilter(model,
            #                      acqIK,
            #                      procedure = idp,
            #                      projection = enums.MomentProjection.Proximal,
            #                      globalFrameOrientation = globalFrame,
            #                      forwardProgression = forwardProgression
            #                      ).compute(pointLabelSuffix="Proximal")
            #
            # #---- Joint energetics----
            modelFilters.JointPowerFilter(model,acqIK).compute(pointLabelSuffix=None)




        # --- opensimStaticOptimizationInterfaceProcedure ------

        soTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-soSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procSO = opensimStaticOptimizationInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,soTemplateFullFile,externalLoadTemplateFullFile)
        procSO.setProgression(progressionAxis,forwardProgression)
        procSO.preProcess(acqIK,gaitFilename[:-4])
        procSO.setTimeRange()
        oiamf = opensimInterfaceFilters.opensimInterfaceStaticOptimizationFilter(procSO)
        oiamf.run()


        # --- Analyses ------
        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile)
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.preProcess(acqIK,gaitFilename[:-4])
        # procAna.setResultsDirname("verif")
        procAna.setTimeRange()
        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.stoToC3d()

        btkTools.smartWriter(acqIK,DATA_PATH+ gaitFilename[:-4]+"-Muscles.c3d")
        import ipdb; ipdb.set_trace()

    def test_cgm23_progressX_allOpensimSteps_local(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\Hannibal-medial-local\\"

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")


        #-----
        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings,"static.c3d","gait1.c3d")

        modelVersion="CGM2.3"

        progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqGait)


        # --- osim builder ---
        markersetTemplateFullFile = "CGM23-markerset.xml"
        osimTemplateFullFile = "pycgm2-gait2354_simbody.osim" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\pycgm2-gait2354_simbody.osim"
        osimConverterSettings =  files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"setup\\CGM23\\OsimToC3dConverter.settings")

        # scaling
        scaleToolFullFile = "CGM23-ScaleTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.highLevelScalingProcedure(DATA_PATH,modelVersion,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile,local=True)
        proc.preProcess( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()


        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = "CGM23-IKTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

        procIK = opensimInverseKinematicsInterfaceProcedure.highLevelInverseKinematicsProcedure(DATA_PATH,scaledOsimName,modelVersion,ikTemplateFullFile,local=True)
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.preProcess(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-5)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        # procIK.setResultsDirname("verif")
        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.ikMarkerLocationToC3d()
        oiikf.motToC3d(osimConverterSettings)
        acqIK =oiikf.getAcq()


        # ----- compute angles from rigid
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = None)#

        #correct the ankle angles
        motDataframe = opensimIO.OpensimDataFrame(DATA_PATH,gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,2]
        motDataframe.save()


        # --- ID ------
        idTemplateFullFile =  "CGM23-idTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-idToolSetup_template.xml"
        externalLoadTemplateFullFile =  "CGM23-externalLoad.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"

        procID = opensimInverseDynamicsInterfaceProcedure.highLevelInverseDynamicsProcedure(DATA_PATH,
            scaledOsimName,modelVersion,idTemplateFullFile,externalLoadTemplateFullFile,local=True)
        procID.setProgression(progressionAxis,forwardProgression)
        procID.preProcess(acqIK,gaitFilename[:-4])
        # procID.setResultsDirname("verif")
        procID.setTimeRange()
        oiidf = opensimInterfaceFilters.opensimInterfaceInverseDynamicsFilter(procID)
        oiidf.run()
        oiidf.stoToC3d(model.mp["Bodymass"], osimConverterSettings)

        # btkTools.smartWriter(acqIK, DATA_PATH+"verifOpensim.c3d")
        # --- Analyses ------
        anaTemplateFullFile = "CGM23-analysesTool-setup.xml" # pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup_template.xml"
        externalLoadTemplateFullFile = "CGM23-externalLoad.xml" # pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile,local=True)
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.preProcess(acqIK,gaitFilename[:-4])
        # procAna.setResultsDirname("verif")
        procAna.setTimeRange()
        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.stoToC3d()

        btkTools.smartWriter(acqIK,DATA_PATH+ gaitFilename[:-4]+"-Muscles.c3d")


    def test_cgm23_progressX_allOpensimSteps_local_withXmlInteraction(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\Hannibal-medial-local\\"

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")


        #-----
        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings,"static.c3d","gait1.c3d")

        modelVersion="CGM2.3"

        progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqGait)


        # --- osim builder ---
        markersetTemplateFullFile = "CGM23-markerset.xml"
        osimTemplateFullFile = "pycgm2-gait2354_simbody.osim" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\pycgm2-gait2354_simbody.osim"
        osimConverterSettings =  files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"setup\\CGM23\\OsimToC3dConverter.settings")

        # scaling
        scaleToolFullFile = "CGM23-ScaleTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.highLevelScalingProcedure(DATA_PATH,modelVersion,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile,local=True)
        proc.setAutoXmlDefinition(False)
        proc.preProcess( acqStatic, staticFilename[:-4])


        #proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        # -- xml interaction ----
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        proc.xml.set_one("mass", str(model.mp["Bodymass"]))
        proc.xml.set_one("height", str(model.mp["Height"]))
        proc.xml.getSoup().find("ScaleTool").attrs["name"] = proc.m_modelVersion+"-Scale"
        proc.xml.set_one(["GenericModelMaker","model_file"],proc.m_osim)
        proc.xml.set_one(["GenericModelMaker","marker_set_file"],proc.m_markerset)
        proc._timeRangeFromStatic()
        proc.xml.set_many("marker_file", files.getFilename(proc._staticTrc))
        proc.xml.set_one(["MarkerPlacer","output_model_file"],proc.m_modelVersion+"-ScaledModel.osim")
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()


        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = "CGM23-IKTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

        procIK = opensimInverseKinematicsInterfaceProcedure.highLevelInverseKinematicsProcedure(DATA_PATH,scaledOsimName,modelVersion,ikTemplateFullFile,local=True)
        procIK.setAutoXmlDefinition(False)
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.preProcess(acqGait,gaitFilename[:-4])
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        procIK.setAccuracy(1e-5)
        # procIK.setResultsDirname("verif")
        # -- xml interaction ----
        procIK.xml.set_one("model_file", procIK.m_osimName)
        procIK.xml.set_one("marker_file", files.getFilename(procIK.m_markerFile))
        procIK.xml.set_one("output_motion_file", procIK.m_dynamicFile+".mot")
        for marker in procIK.m_weights.keys():
            procIK.xml.set_inList_fromAttr("IKMarkerTask","weight","name",marker,str(procIK.m_weights[marker]))

        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.ikMarkerLocationToC3d()
        oiikf.motToC3d(osimConverterSettings)
        acqIK =oiikf.getAcq()
        # btkTools.smartWriter(acqIK,DATA_PATH+ gaitFilename[:-4]+"-Muscles.c3d")



        # ----- compute angles from rigid
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = None)#

        #correct the ankle angles
        motDataframe = opensimIO.OpensimDataFrame(DATA_PATH,gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,2]
        motDataframe.save()


        # # --- ID ------
        # idTemplateFullFile =  "CGM23-idTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-idToolSetup_template.xml"
        # externalLoadTemplateFullFile =  "CGM23-externalLoad.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        #
        # procID = opensimInverseDynamicsInterfaceProcedure.highLevelInverseDynamicsProcedure(DATA_PATH,
        #     scaledOsimName,modelVersion,idTemplateFullFile,externalLoadTemplateFullFile,local=True)
        # procID.setAutoXmlDefinition(False)
        # procID.setProgression(progressionAxis,forwardProgression)
        # procID.preProcess(acqIK,gaitFilename[:-4])
        # # procID.setResultsDirname("verif")
        # #procID.setTimeRange()
        # oiidf = opensimInterfaceFilters.opensimInterfaceInverseDynamicsFilter(procID)
        # oiidf.run()
        # oiidf.stoToC3d(model.mp["Bodymass"], osimConverterSettings)
        #
        # # btkTools.smartWriter(acqIK, DATA_PATH+"verifOpensim.c3d")
        # # --- Analyses ------
        # anaTemplateFullFile = "CGM23-analysesTool-setup.xml" # pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup_template.xml"
        # externalLoadTemplateFullFile = "CGM23-externalLoad.xml" # pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        # procAna = opensimAnalysesInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile,local=True)
        # procAna.setAutoXmlDefinition(False)
        # procAna.setProgression(progressionAxis,forwardProgression)
        # procAna.preProcess(acqIK,gaitFilename[:-4])
        # # procAna.setResultsDirname("verif")
        # #procAna.setTimeRange()
        # oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        # oiamf.run()
        # oiamf.stoToC3d()
        #
        # btkTools.smartWriter(acqIK,DATA_PATH+ gaitFilename[:-4]+"-Muscles.c3d")


    def test_cgm23_progressX_allOpensimSteps_local_noXmlInteraction(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\Hannibal-medial-local\\"

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")


        #-----
        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings,"static.c3d","gait1.c3d")

        modelVersion="CGM2.3"

        progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqGait)


        # --- osim builder ---
        markersetTemplateFullFile = "CGM23-markerset.xml"
        osimTemplateFullFile = "pycgm2-gait2354_simbody.osim" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\pycgm2-gait2354_simbody.osim"
        osimConverterSettings =  files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"setup\\CGM23\\OsimToC3dConverter.settings")

        # scaling
        scaleToolFullFile = "CGM23-ScaleTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.highLevelScalingProcedure(DATA_PATH,modelVersion,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile,local=True)
        proc.setAutoXmlDefinition(False)
        proc.preProcess( acqStatic, staticFilename[:-4])
        #proc.setAnthropometry(model.mp["Bodymass"],model.mp["Height"])
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()


        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]
        ikTemplateFullFile = "CGM23-IKTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

        procIK = opensimInverseKinematicsInterfaceProcedure.highLevelInverseKinematicsProcedure(DATA_PATH,scaledOsimName,modelVersion,ikTemplateFullFile,local=True)
        procIK.setAutoXmlDefinition(False)
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.preProcess(acqGait,gaitFilename[:-4])
        # procIK.setWeights(ikWeights)
        # procIK.setTimeRange()
        #procIK.setAccuracy(1e-5)
        # procIK.setResultsDirname("verif")
        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.ikMarkerLocationToC3d()
        oiikf.motToC3d(osimConverterSettings)
        acqIK =oiikf.getAcq()
        # btkTools.smartWriter(acqIK,DATA_PATH+ gaitFilename[:-4]+"-Muscles.c3d")



        # ----- compute angles from rigid
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = None)#

        #correct the ankle angles
        motDataframe = opensimIO.OpensimDataFrame(DATA_PATH,gaitFilename[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = acqIK.GetPoint("RAnkleAngles").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = acqIK.GetPoint("LAnkleAngles").GetValues()[:,2]
        motDataframe.save()


        # --- ID ------
        idTemplateFullFile =  "CGM23-idTool-setup.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-idToolSetup_template.xml"
        externalLoadTemplateFullFile =  "CGM23-externalLoad.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"

        procID = opensimInverseDynamicsInterfaceProcedure.highLevelInverseDynamicsProcedure(DATA_PATH,
            scaledOsimName,modelVersion,idTemplateFullFile,externalLoadTemplateFullFile,local=True)
        procID.setAutoXmlDefinition(False)
        procID.setProgression(progressionAxis,forwardProgression)
        procID.preProcess(acqIK,gaitFilename[:-4])
        # procID.setResultsDirname("verif")
        #procID.setTimeRange()
        oiidf = opensimInterfaceFilters.opensimInterfaceInverseDynamicsFilter(procID)
        oiidf.run()
        oiidf.stoToC3d(model.mp["Bodymass"], osimConverterSettings)

        # btkTools.smartWriter(acqIK, DATA_PATH+"verifOpensim.c3d")
        # --- Analyses ------
        anaTemplateFullFile = "CGM23-analysesTool-setup.xml" # pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup_template.xml"
        externalLoadTemplateFullFile = "CGM23-externalLoad.xml" # pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile,local=True)
        procAna.setAutoXmlDefinition(False)
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.preProcess(acqIK,gaitFilename[:-4])
        # procAna.setResultsDirname("verif")
        #procAna.setTimeRange()
        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.stoToC3d()

        btkTools.smartWriter(acqIK,DATA_PATH+ gaitFilename[:-4]+"-Muscles.c3d")
