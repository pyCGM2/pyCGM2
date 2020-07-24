# -*- coding: utf-8 -*-
"""Nexus Operation : **CGM2.1 Calibration**

:param -l, --leftFlatFoot [int]: enable or disable the flat foot option on the left foot
:param -r, --rightFlatFoot [int]: enable or disable the flat foot option on the right foot
:param -hf, --headFlat [int]: enable or disable the head flat option
:param -md, --markerDiameter [int]: marker diameter
:param -ps, --pointSuffix [string]: suffix adds to the vicon nomenclature outputs
:param --check [bool]: add "cgm2.1" as point suffix
:param --resetMP [bool]: reset computation of optional parameters, like interAsisDistance, ShankOffsets...
:param --forceLHJC [array]: force the local position of the left hip joint centre in the pelvic coordinate system
:param --forceRHJC [array]: force the local position of the left hip joint centre in the pelvic coordinate system


Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>> -l=1 -r=0 -ps=py
    (if you want to add suffix py and enable the flat foot option on the left side only)
    >>> --leftFlatFoot=1 -r=0 --pointSuffix=py --resetMP
    (if you want to add suffix py, enable the flat foot option on the left side only and reset the computation of optional parameters, like interAsisDistance, ShankOffsets...)
    >>>  --forceLHJC 10 20 30  --forceRHJC 10 20 30
    (force the left and right hip joint centre positions (10 mm along the pelvic X-axis, 20 mm along the pelvic Y-axis, 30 mm along the pelvic Z-axis)

"""

#import ipdb
import os
import logging
import argparse

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


# vicon nexus
import ViconNexus


# pyCGM2 libraries
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

from pyCGM2.Configurator import CgmArgsManager
from pyCGM2.Lib.CGM import  cgm2_1

def main():

    parser = argparse.ArgumentParser(description='CGM2.1 Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--resetMP', action='store_true', help='reset optional mass parameters')
    parser.add_argument('--forceLHJC', nargs='+')
    parser.add_argument('--forceRHJC', nargs='+')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------GLOBAL SETTINGS ------------------------------------
        # global setting ( in user/AppData)
        if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_1-pyCGM2.settings"):
            settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")
        else:
            settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_1-pyCGM2.settings")

        settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        headFlat = argsManager.getHeadFlat()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.1")

        hjcMethod = settings["Calibration"]["HJC"]

        lhjc = argsManager.forceHjc("left")
        rhjc = argsManager.forceHjc("right")
        if  lhjc is not None:
            hjcMethod["Left"] = lhjc
        if  rhjc is not None:
            hjcMethod["Right"] = rhjc

        # --------------------------LOADING ------------------------------------
        DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)

        # --------------------------SUBJECT ------------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp,optional_mp = nexusUtils.getNexusSubjectMp(NEXUS,subject,resetFlag=args.resetMP)

        # -------------------------- INFOS ------------------------------------
        mpInfo,mpFilename = files.getMpFileContent(DATA_PATH,"mp.pyCGM2",subject)

        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM2_1.translators")
        if not translators:  translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,calibrateFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------MODELLING PROCESSING -----------------------
        model,acqStatic = cgm2_1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,
                      required_mp,optional_mp,
                      leftFlatFoot,rightFlatFoot,headFlat,
                      markerDiameter,
                      hjcMethod,
                      pointSuffix,forceBtkAcq=acq)

        # ----------------------SAVE-------------------------------------------
        #pyCGM2.model
        files.saveModel(model,DATA_PATH,subject)

        # save mp
        files.saveMp(mpInfo,model,DATA_PATH,mpFilename)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)
        nexusFilters.NexusModelFilter(NEXUS,
                                      model,acqStatic,subject,
                                      pointSuffix,
                                      staticProcessing=True).run()

        # ========END of the nexus OPERATION if run from Nexus  =========



    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":


    main()
