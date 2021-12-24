# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensimInterface.py::Test_CGM23::test_cgm23_progressX
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
from pyCGM2.Model.Opensim import opensimFilters,opensimInterfaceFilters,opensimScalingInterfaceProcedure,opensimInverseKinematicsInterfaceProcedure,opensimInverseDynamicsInterfaceProcedure
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


class Test_CGM23:
    def test_cgm23_progressX(self):

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
        proc.setAnthropometry(71.0,1780.0)
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
        oiikf.stoToC3d(osimConverterSettings)
        acqIK =oiikf.getAcq()

        # ----- compute angles
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

        # btkTools.smartWriter(acqIK, DATA_PATH+"verifOpensim.c3d")
        #import ipdb; ipdb.set_trace()

        # --- ID ------
        idTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-idTool-setup.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"



        procID = opensimInverseDynamicsInterfaceProcedure.highLevelInverseDynamicsProcedure(DATA_PATH,
            scaledOsimName,modelVersion,idTemplateFullFile,externalLoadTemplateFullFile)
        procID.setProgression(progressionAxis,forwardProgression)
        procID.preProcess(acqIK,gaitFilename[:-4])
        procID.setTimeRange()
        oiidf = opensimInterfaceFilters.opensimInterfaceInverseDynamicsFilter(procID)
        oiidf.run()
        oiidf.stoToC3d(model.mp["Bodymass"], osimConverterSettings)

        btkTools.smartWriter(acqIK, DATA_PATH+"verifOpensim.c3d")



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
            # modelFilters.JointPowerFilter(model,acqIK).compute(pointLabelSuffix=None)

            btkTools.smartWriter(acqIK, DATA_PATH+"verifDynOpensim.c3d")
            import ipdb; ipdb.set_trace()


        # # --- opensimStaticOptimizationInterfaceProcedure ------
        # opensimStaticOptimizationInterfaceProcedure
        # anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup-template.xml"
        # externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        # procAna = opensimStaticOptimizationInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile)
        # procAna.preProcess(acqIK,gaitFilename[:-4])
        # procAna.setTimeRange()
        # oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        # oiamf.run()


        # --- Analyses ------
        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup-template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile)
        procAna.preProcess(acqIK,gaitFilename[:-4])
        procAna.setTimeRange()
        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()

        btkTools.smartWriter(acqIK,DATA_PATH+ gaitFilename[:-4]+"-Muscles.c3d")
        import ipdb; ipdb.set_trace()
