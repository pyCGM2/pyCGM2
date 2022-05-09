# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM2.6
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse

# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2 import enums
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools

from pyCGM2.Model import  modelFilters

from pyCGM2.Lib.CGM import  kneeCalibration

def main():


    parser = argparse.ArgumentParser(description='2Dof Knee Calibration')
    parser.add_argument('-s','--side', type=str, help="Side : Left or Right")
    parser.add_argument('-b','--beginFrame', type=int, help="begin frame")
    parser.add_argument('-e','--endFrame', type=int, help="end frame")



    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        args = parser.parse_args()

        DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.logger.info( "reconstructed file: "+ reconstructFilenameLabelled)

        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL - INIT ------------------------------
        model = files.loadModel(DATA_PATH,subject)
        LOGGER.logger.info("loaded model : %s" %(model.version ))


        if model.version == "CGM1.0":
            settings = files.loadModelSettings(DATA_PATH,"CGM1-pyCGM2.settings")
        elif model.version == "CGM1.1":
            settings = files.loadModelSettings(DATA_PATH,"CGM1_1-pyCGM2.settings")
        elif model.version == "CGM2.1":
            settings = files.loadModelSettings(DATA_PATH,"CGM2_1-pyCGM2.settings")
        elif model.version == "CGM2.2":
            settings = files.loadModelSettings(DATA_PATH,"CGM2_2-pyCGM2.settings")
        elif model.version == "CGM2.3":
            settings = files.loadModelSettings(DATA_PATH,"CGM2_3-pyCGM2.settings")
        elif model.version in  ["CGM2.4"]:
            settings = files.loadModelSettings(DATA_PATH,"CGM2_4-pyCGM2.settings")
        elif model.version in  ["CGM2.5"]:
            settings = files.loadModelSettings(DATA_PATH,"CGM2_5-pyCGM2.settings")

        else:
            raise Exception ("model version not found [contact admin]")

        # --------------------------SESSION INFOS ------------------------------------
        mpInfo,mpFilename = files.getMpFileContent(DATA_PATH,"mp.pyCGM2",subject)

        #  translators management
        if model.version in  ["CGM1.0"]:
            translators = files.getTranslators(DATA_PATH,"CGM1.translators")
        elif model.version in  ["CGM1.1"]:
            translators = files.getTranslators(DATA_PATH,"CGM1_1.translators")
        elif model.version in  ["CGM2.1"]:
            translators = files.getTranslators(DATA_PATH,"CGM2_1.translators")
        elif model.version in  ["CGM2.2"]:
            translators = files.getTranslators(DATA_PATH,"CGM2_2.translators")
        elif model.version in  ["CGM2.3"]:
            translators = files.getTranslators(DATA_PATH,"CGM2_3.translators")
        elif model.version in  ["CGM2.4"]:
            translators = files.getTranslators(DATA_PATH,"CGM2_4.translators")
        elif model.version in  ["CGM2.5"]:
            translators = files.getTranslators(DATA_PATH,"CGM2_5.translators")

        if not translators:
           translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructedFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------MODEL PROCESSING----------------------------
        model,acqFunc,side = kneeCalibration.calibration2Dof(model,
            DATA_PATH,reconstructFilenameLabelled,translators,
            args.side,args.beginFrame,args.endFrame,None,forceBtkAcq=acq)

        # ----------------------SAVE-------------------------------------------
        files.saveModel(model,DATA_PATH,subject)
        LOGGER.logger.warning("model updated with a  %s knee calibrated with 2Dof method" %(side))

        # save mp
        files.saveMp(mpInfo,model,DATA_PATH,mpFilename)

        # ----------------------VICON INTERFACE-------------------------------------------
        #--- update mp
        nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)

        if side == "Left":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"LFE0", model.getSegment("Left Thigh"),OriginValues = acqFunc.GetPoint("LKJC").GetValues() )
        elif side == "Right":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"RFE0", model.getSegment("Right Thigh"),OriginValues = acqFunc.GetPoint("RKJC").GetValues() )

        # --------------------------NEW MOTION FILTER - DISPLAY BONES---------
        scp=modelFilters.StaticCalibrationProcedure(model)
        if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2"]:
            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,enums.motionMethod.Determinist)
            modMotion.compute()

        elif model.version in  ["CGM2.3","CGM2.4"]:

            proximalSegmentLabel=str(side+" Thigh")
            distalSegmentLabel=str(side+" Shank")
            # Motion
            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,enums.motionMethod.Sodervisk)
            modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])


        # -- add nexus Bones
        if side == "Left":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"LFE1", model.getSegment("Left Thigh"),OriginValues = acqFunc.GetPoint("LKJC").GetValues() )
            LOGGER.logger.warning("offset %s" %(str(model.mp_computed["LeftKneeFuncCalibrationOffset"] )))
        elif side == "Right":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"RFE1", model.getSegment("Right Thigh"),OriginValues = acqFunc.GetPoint("RKJC").GetValues() )
            LOGGER.logger.warning("offset %s" %(str(model.mp_computed["RightKneeFuncCalibrationOffset"] )))

    else:
        return parser

if __name__ == "__main__":

    main()
