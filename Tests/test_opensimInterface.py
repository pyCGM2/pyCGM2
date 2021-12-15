# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensimInterface.py::Test_gait2354::test_cgm23_highLevel
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

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
from pyCGM2.Model.Opensim import opensimInverseDynamicsInterfaceProcedure
from pyCGM2.Model.Opensim import opensimAnalysesInterfaceProcedure
from pyCGM2.Model.Opensim import osimProcessing



def processCGM23(DATA_PATH,settings):

    settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")
    translators = settings["Translators"]
    weights = settings["Fitting"]["Weight"]
    hjcMethod = settings["Calibration"]["HJC"]



    staticFilename = "static.c3d"
    gaitFilename= "gait1.c3d"

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


    # ------- OPENSIM IK --------------------------------------

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

class Test_Scaling:
    def test_cgm23_lowLevel(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\gait\\"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings)

        # --- osim builder ---

        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\cgm23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\lowerLimb_ballsJoints.osim"


        # --- NEW implementation - WITH Scaling
        osimProcessing.smartTrcExport(acqStatic,DATA_PATH +  staticFilename[:-4])

        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"


        proc = opensimScalingInterfaceProcedure.opensimInterfaceLowLevelScalingProcedure(DATA_PATH,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.preProcess( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(71.0,1780.0)
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()



    def test_cgm23_highLevel(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\gait\\"

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings)

        # --- osim builder ---
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\cgm23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\lowerLimb_ballsJoints.osim"

        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"


        proc = opensimScalingInterfaceProcedure.opensimInterfaceHighLevelScalingProcedure(DATA_PATH,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.preProcess( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(71.0,1780.0)
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()



class Test_InverseKinematics:
    def test_cgm23_lowLevel(self):
        pass
        # Not work

        # DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\gait\\"
        #
        # settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")
        #
        # model, acqStatic,acqGait,staticFilename,gaitFilename = CGM23_processing(DATA_PATH,settings)
        #
        # # --- osim builder ---
        #
        # OPENSIM_TEMPLATE_FOLDER = "C:/Users/fleboeuf/Documents/Programmation/pyCGM2/pyCGM2/Sandbox/opensim/setUpXmlFiles/"
        #
        # markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\cgm23-markerset.xml"
        # osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\lowerLimb_ballsJoints.osim"
        #
        #
        # # scaling
        # osimProcessing.smartTrcExport(acqStatic,DATA_PATH +  staticFilename[:-4])
        #
        # scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"
        #
        # statictrcFile = staticFilename[:-4]+".trc"
        #
        #
        # proc = opensimScalingInterfaceProcedure.opensimInterfaceLowLevelScalingProcedure(DATA_PATH,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile,statictrcFile)
        # proc.setAnthropometry(71.0,1780.0)
        # oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        # oisf.run()
        # scaledOsim = oisf.getOsim()
        #
        # # IK
        # ikWeights = settings["Fitting"]["Weight"]
        #
        # ikTemplateFullFile = OPENSIM_TEMPLATE_FOLDER + "cgm2_3-ikSetUp_template.xml"
        # procIK = opensimInverseKinematicsInterfaceProcedure.opensimInterfaceLowLevelInverseKinematicsProcedure(DATA_PATH,scaledOsim, ikTemplateFullFile,acqGait,gaitFilename)
        # procIK.setWeights(ikWeights)
        # oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        # oiikf.run()



    def test_cgm23_highLevel(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\gait\\"

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")


        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings)
        acqIK_old = process_oldIK(DATA_PATH,model,acqGait,gaitFilename,scp)

        # --- osim builder ---
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\cgm23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\lowerLimb_ballsJoints.osim"

        # scaling
        osimProcessing.smartTrcExport(acqStatic,DATA_PATH +  staticFilename[:-4])
        statictrcFile = staticFilename[:-4]+".trc"

        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"

        proc = opensimScalingInterfaceProcedure.opensimInterfaceHighLevelScalingProcedure(DATA_PATH,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.preProcess( acqStatic, staticFilename[:-4])
        proc.setAnthropometry(71.0,1780.0)
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()

        # --- IK ---
        ikWeights = settings["Fitting"]["Weight"]

        ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"
        procIK = opensimInverseKinematicsInterfaceProcedure.opensimInterfaceHighLevelInverseKinematicsProcedure(DATA_PATH,scaledOsim, ikTemplateFullFile)
        procIK.preProcess(acqGait,gaitFilename)
        procIK.setAccuracy(1e-8)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        acqIK =oiikf.getAcq()


        # --- CHECKING ---
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

class Test_gait2354:
    def test_cgm23_highLevel(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\gait-2354\\"

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        #-----
        model, acqStatic,acqGait,staticFilename,gaitFilename,scp = processCGM23(DATA_PATH,settings)

        modelVersion="CGM2.3"
        # --- osim builder ---
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\native\\gait2354_simbody.osim"

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
        procIK.preProcess(acqGait,gaitFilename[:-4])
        procIK.setAccuracy(1e-8)
        procIK.setWeights(ikWeights)
        procIK.setTimeRange()
        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        acqIK =oiikf.getAcq()

        # ----- compute angles
        modMotion=modelFilters.ModelMotionFilter(scp,acqIK,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.compute(description="new", pointLabelSuffix = None)#


        # --- ID ------
        idTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-idTool-setup.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procID = opensimInverseDynamicsInterfaceProcedure.highLevelInverseDynamicsProcedure(DATA_PATH,scaledOsimName,modelVersion,idTemplateFullFile,externalLoadTemplateFullFile)
        procID.preProcess(acqIK,gaitFilename[:-4])
        procIK.setTimeRange()
        oiidf = opensimInterfaceFilters.opensimInterfaceInverseDynamicsFilter(procID)
        oiidf.run()


        # --- Analyses ------
        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup-template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile)
        procAna.preProcess(acqIK,gaitFilename[:-4])
        procAna.setTimeRange()
        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()

        import ipdb; ipdb.set_trace()
