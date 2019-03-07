# -*- coding: utf-8 -*-
"""Nexus Operation : **CGM1.1 Fitting**

:param --proj [string]: define in which coordinate system joint moment will be expressed (Choice : Distal, Proximal, Global)
:param -md, --markerDiameter [int]: marker diameter
:param -ps, --pointSuffix [string]: suffix adds to the vicon nomenclature outputs
:param --check [bool]: add "cgm1.1" as point suffix


Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>> --proj=Global
    (means joint moments will be expressed into the Global Coordinate system)

"""
#import ipdb
import os
import traceback
import logging
import argparse
import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Configurator import CgmArgsManager
from pyCGM2.Lib.CGM import  cgm1_1

from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

def main(args):

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation
        # --------------------------GLOBAL SETTINGS ------------------------------------
        # global setting ( in user/AppData)
        if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM1_1-pyCGM2.settings"):
            settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")
        else:
            settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM1_1-pyCGM2.settings")


        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1_1")
        momentProjection =  argsManager.getMomentProjection()

        # --------------------------LOADING ------------------------------------
        DEBUG= False
        if DEBUG:
            DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1\\native\\"
            reconstructFilenameLabelledNoExt = "gait Trial" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )
        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)

        # --------------------------SUBJECT ------------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        # --------------------------CHECKING -----------------------------------
        # check model is the CGM1
        logging.info("loaded model : %s" %(model.version ))
        if model.version != "CGM1.1":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM1.1 calibration pipeline"%model.version)

        # --------------------------SESSION INFOS ------------------------------------

        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM1_1.translators")
        if not translators:  translators = settings["Translators"]

        #force plate assignement from Nexus
        mfpa = nexusTools.getForcePlateAssignment(NEXUS)

        # --------------------------MODELLING PROCESSING -----------------------
        acqGait = cgm1_1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,
            markerDiameter,
            pointSuffix,
            mfpa,momentProjection)

        # ----------------------SAVE-------------------------------------------
        # Todo: pyCGM2 model :  cpickle doesn t work. Incompatibility with Swig. ( see about BTK wrench)


        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(NEXUS,model,acqGait,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,acqGait,["Left-FP","Right-FP"])


        # ========END of the nexus OPERATION if run from Nexus  =========
        if DEBUG:
            NEXUS.SaveTrial(30)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CGM1.1 Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')

    args = parser.parse_args()


    # ---- main script -----
    try:
        main(args)


    except Exception, errormsg:
        print "Error message: %s" % errormsg
        traceback.print_exc()
        raise
