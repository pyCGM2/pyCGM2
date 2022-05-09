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
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools

from pyCGM2.Model import  modelFilters

from pyCGM2.Lib.CGM import  kneeCalibration



def main():

    parser = argparse.ArgumentParser(description='SARA Functional Knee Calibration')
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


        # --------------------------PATH + FILE ------------------------------------
        DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.logger.info( "reconstructed file: "+ reconstructFilenameLabelled)

       # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        LOGGER.logger.info("loaded model : %s" %(model.version ))
        # --------------------------CONFIG ------------------------------------

        # --------------------CHECKING ------------------------------
        if model.version in ["CGM1.0","CGM1.1","CGM2.1","CGM2.2"] :
            raise Exception ("Can t use SARA method with your model %s [minimal version : CGM2.3]"%(model.version))
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
        if model.version in  ["CGM2.3"]:
            translators = files.getTranslators(DATA_PATH,"CGM2-3.translators")
        elif model.version in  ["CGM2.4"]:
            translators = files.getTranslators(DATA_PATH,"CGM2-4.translators")
        elif model.version in  ["CGM2.5"]:
            translators = files.getTranslators(DATA_PATH,"CGM2-5.translators")
        if not translators:
           translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructedFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------MODEL PROCESSING----------------------------
        model,acqFunc,side = kneeCalibration.sara(model,
            DATA_PATH,reconstructFilenameLabelled,translators,
            args.side,args.beginFrame,args.endFrame,forceBtkAcq=acq)

        # ----------------------SAVE-------------------------------------------
        files.saveModel(model,DATA_PATH,subject)
        LOGGER.logger.warning("model updated with a  %s knee calibrated with SARA method" %(side))

        # save mp
        files.saveMp(mpInfo,model,DATA_PATH,mpFilename)
        # ----------------------VICON INTERFACE-------------------------------------------
        #--- update mp
        nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)


        # -- add nexus Bones
        if side == "Left":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"LFE0", model.getSegment("Left Thigh"),OriginValues = acqFunc.GetPoint("LKJC").GetValues() )
        elif side == "Right":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"RFE0", model.getSegment("Right Thigh"),OriginValues = acqFunc.GetPoint("RKJC").GetValues() )

        proximalSegmentLabel=str(side+" Thigh")
        distalSegmentLabel=str(side+" Shank")

        # add modelled markers
        meanOr_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_Sara")
        meanAxis_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_SaraAxis")
        btkTools.smartAppendPoint(acqFunc,side+"_KJC_Sara",meanOr_inThigh)
        btkTools.smartAppendPoint(acqFunc,side+"_KJC_SaraAxis",meanAxis_inThigh)

        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KJC_Sara", acqFunc)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KJC_SaraAxis", acqFunc)


        #---Second model motion filter

        # consider new anatomical frame
        scp=modelFilters.StaticCalibrationProcedure(model)
        modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,enums.motionMethod.Sodervisk)
        modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])


        # projection of the Sara axis in the transversale plane
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
