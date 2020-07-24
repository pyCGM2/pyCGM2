# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""Nexus Operation : **CGM2.5 Fitting**

:param --proj [string]: define in which coordinate system joint moment will be expressed (Choice : Distal, Proximal, Global)
:param -md, --markerDiameter [int]: marker diameter
:param -ps, --pointSuffix [string]: suffix adds to the vicon nomenclature outputs
:param --check [bool]: add "cgm2.5" as point suffix
:param --noIk [bool]: disable inverse kinematics

Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>> --proj=Global --noIk
    (means you disable the inverse kinematic solver and joint moments will be expressed into the Global Coordinate system, and )

"""
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
from pyCGM2.Nexus import nexusFilters,nexusTools

from pyCGM2.Configurator import CgmArgsManager
from pyCGM2.Lib.CGM import  cgm2_5


def main():

    parser = argparse.ArgumentParser(description='CGM2-5 Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------GLOBAL SETTINGS ------------------------------
        # global setting ( in user/AppData)
        if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_5-pyCGM2.settings"):
            settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_5-pyCGM2.settings")
        else:
            settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_5-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.5")
        momentProjection =  argsManager.getMomentProjection()
        ik_flag = argsManager.enableIKflag()


        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()


        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "reconstructed file: "+ reconstructFilenameLabelled)

        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        # check model
        logging.info("loaded model : %s" %(model.version))
        if model.version != "CGM2.5":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM2.4e calibration pipeline"%subject)

        # --------------------------SESSION INFOS -----------------------------
        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM2_5.translators")
        if not translators: translators = settings["Translators"]

        #  ikweight
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_4.ikw")
        if not ikWeight: ikWeight = settings["Fitting"]["Weight"]

        #force plate assignement from Nexus
        mfpa = nexusTools.getForcePlateAssignment(NEXUS)

        # btkAcquisition
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------MODELLING PROCESSING -----------------------
        finalAcqGait = cgm2_5.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,settings,
            ik_flag,markerDiameter,
            pointSuffix,
            mfpa,
            momentProjection,
            forceBtkAcq=acq)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(NEXUS,model,finalAcqGait,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,finalAcqGait,["Left-FP","Right-FP"])
        # ========END of the nexus OPERATION if run from Nexus  =========


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    main()
