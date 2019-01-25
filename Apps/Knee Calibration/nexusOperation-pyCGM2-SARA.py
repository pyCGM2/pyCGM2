# -*- coding: utf-8 -*-
#import ipdb
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

from pyCGM2.Model import  modelFilters

from pyCGM2.Configurator import CgmArgsManager
from pyCGM2.Lib.CGM import  kneeCalibration



if __name__ == "__main__":

    plt.close("all")
    parser = argparse.ArgumentParser(description='SARA Functional Knee Calibration')
    parser.add_argument('-s','--side', type=str, help="Side : Left or Right")
    parser.add_argument('-b','--beginFrame', type=int, help="begin frame")
    parser.add_argument('-e','--endFrame', type=int, help="end frame")
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------PATH + FILE ------------------------------------
        DEBUG = False
        if DEBUG:
            # CGM2.3--
            DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\knee calibration\\CGM2.3-calibrationSara\\"
            reconstructedFilenameLabelledNoExt = "Right Knee"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructedFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "reconstructed file: "+ reconstructFilenameLabelled)

       # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        logging.info("loaded model : %s" %(model.version ))
        # --------------------------CONFIG ------------------------------------

        # --------------------CHECKING ------------------------------
        if model.version in ["CGM1.0","CGM1.1","CGM2.1","CGM2.2"] :
            raise Exception ("Can t use SARA method with your model %s [minimal version : CGM2.3]"%(model.version))
        elif model.version == "CGM2.3":
            settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.settings")
        elif model.version in  ["CGM2.4"]:
            settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")
        else:
            raise Exception ("model version not found [contact admin]")

        # --------------------------SESSION INFOS ------------------------------------
        mpInfo,mpFilename = files.getMpFileContent(DATA_PATH,"mp.pyCGM2",subject)

        #  translators management
        if model.version in  ["CGM2.3"]:
            translators = files.getTranslators(DATA_PATH,"CGM2-3.translators")
        elif model.version in  ["CGM2.4"]:
            translators = files.getTranslators(DATA_PATH,"CGM2-4.translators")
        if not translators:
           translators = settings["Translators"]

        # --------------------------MODEL PROCESSING----------------------------
        model,acqFunc,side = kneeCalibration.sara(model,
            DATA_PATH,reconstructFilenameLabelled,translators,
            args.side,args.beginFrame,args.endFrame)

        # ----------------------SAVE-------------------------------------------
        files.saveModel(model,DATA_PATH,subject)
        logging.warning("model updated with a  %s knee calibrated with SARA method" %(side))

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
            print model.mp_computed["LeftKneeFuncCalibrationOffset"]
            logging.warning("offset %s" %(str(model.mp_computed["LeftKneeFuncCalibrationOffset"] )))
        elif side == "Right":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"RFE1", model.getSegment("Right Thigh"),OriginValues = acqFunc.GetPoint("RKJC").GetValues() )
            logging.warning("offset %s" %(str(model.mp_computed["RightKneeFuncCalibrationOffset"] )))
            print model.mp_computed["RightKneeFuncCalibrationOffset"]


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
