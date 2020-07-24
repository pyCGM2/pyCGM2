# -*- coding: utf-8 -*-
"""Nexus Operation : **2DofCalibration**

Calibration of the knee with the Calibration2Dof method (dynaKad like method).

The script considers all frames of the c3d and detects the side autmaticallt from ANK marker trajectories

:param -s, --side [string]: lower limb side ( choice: Left or Right )
:param -b, --beginFrame [int]:  manual selection of the first  frame
:param -e, --endFrame [int]:   manual selection of the last  frame


Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -side=Left -b=50 -e=100
    (Left knee calibration between frames 50 and 100)


"""
import os
import traceback
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
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

from pyCGM2.Model import  modelFilters

from pyCGM2.Configurator import CgmArgsManager
from pyCGM2.Lib.CGM import  kneeCalibration

def main():

    parser = argparse.ArgumentParser(description='2Dof Knee Calibration')
    parser.add_argument('-s','--side', type=str, help="Side : Left or Right")
    parser.add_argument('-b','--beginFrame', type=int, help="begin frame")
    parser.add_argument('-e','--endFrame', type=int, help="end frame")

    args = parser.parse_args()
    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "reconstructed file: "+ reconstructFilenameLabelled)

        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL - INIT ------------------------------
        model = files.loadModel(DATA_PATH,subject)
        logging.info("loaded model : %s" %(model.version ))


        if model.version == "CGM1.0":
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM1-pyCGM2.settings"):
                settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")
            else:
                settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM1-pyCGM2.settings")

        elif model.version == "CGM1.1":
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM1_1-pyCGM2.settings"):
                settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")
            else:
                settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM1_1-pyCGM2.settings")

        elif model.version == "CGM2.1":
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_1-pyCGM2.settings"):
                settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")
            else:
                settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_1-pyCGM2.settings")

        elif model.version == "CGM2.2":
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_2-pyCGM2.settings"):
                settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_2-pyCGM2.settings")
            else:
                settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_2-pyCGM2.settings")
        elif model.version == "CGM2.3":
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_3-pyCGM2.settings"):
                settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.settings")
            else:
                settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")
        elif model.version in  ["CGM2.4"]:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_4-pyCGM2.settings"):
                settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")
            else:
                settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_4-pyCGM2.settings")

        elif model.version in  ["CGM2.5"]:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_5-pyCGM2.settings"):
                settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_5-pyCGM2.settings")
            else:
                settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_5-pyCGM2.settings")

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
        logging.warning("model updated with a  %s knee calibrated with 2Dof method" %(side))

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
            logging.warning("offset %s" %(str(model.mp_computed["LeftKneeFuncCalibrationOffset"] )))
        elif side == "Right":
            nexusTools.appendBones(NEXUS,subject,acqFunc,"RFE1", model.getSegment("Right Thigh"),OriginValues = acqFunc.GetPoint("RKJC").GetValues() )
            logging.warning("offset %s" %(str(model.mp_computed["RightKneeFuncCalibrationOffset"] )))

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    main()
