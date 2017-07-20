
# -*- coding: utf-8 -*-

import os
import logging
import matplotlib.pyplot as plt

import pdb

import argparse


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# openMA
#import ma.io
#import ma.body

#btk
import btk
import configparser

# pyCGM2 libraries
from pyCGM2.Tools import btkTools,nexusTools

from pyCGM2.Model.CGM2 import  forceplates
#



if __name__ == "__main__":


    DEBUG = False

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='Flag Up Force plates')
    parser.add_argument('-mfpa',type=str,  help='manual assignment of force plates')
    args = parser.parse_args()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\Salford\\Alana MoCap data\\MRI-US-01 - myProcess\\PIG\\"
            reconstructFilenameLabelledNoExt = "MRI-US-01, 2008-08-08, 3DGA 16" #"static Cal 01-noKAD-noAnkleMed" #

            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )

        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)

        # --- btk acquisition ----
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))


        #   check if acq was saved with only one  activated subject
        if acqGait.GetPoint(0).GetLabel().count(":"):
            raise Exception("[pyCGM2] Your Trial c3d was saved with two activate subject. Re-save it with only one before pyCGM2 calculation")


        # --------------------------SUBJECT -----------------------------------

        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )


        # ---------- FORCE PLATE HANDLING -------------------------------------

        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        forceplates.addForcePlateGeneralEvents(acqGait,mappedForcePlate)
        logging.info("Force plate assignment : %s" %mappedForcePlate)

        if args.mfpa is not None:
            if len(args.mfpa) != len(mappedForcePlate):
                raise Exception("[pyCGM2] manual force plate assignment badly sets. Wrong force plate number. %s force plate require" %(str(len(mappedForcePlate))))
            else:
                mappedForcePlate = args.mfpa
                logging.warning("Force plates assign manually")
                forceplates.addForcePlateGeneralEvents(acqGait,mappedForcePlate)


#        # --------- ATTEMPT TO ALTER ENF FROM nexus operation -----------------
#
#        #  DONT WORk
#
#        # --------------------Modify ENF --------------------------------------
#        configEnf = configparser.ConfigParser()
#        configEnf.optionxform = str
#        enfFile = str(DATA_PATH+reconstructFilenameLabelledNoExt+".Trial.enf")
#        configEnf.read(enfFile)
#
#
#        indexFP=1
#        for letter in mappedForcePlate:
#
#            if letter =="L": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Left"
#            if letter =="R": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Right"
#            if letter =="X": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Invalid"
#
#            indexFP+=1
#
#        tmpFile =str(DATA_PATH+reconstructFilenameLabelledNoExt+".Trial.enf-tmp")
#        with open(tmpFile, 'w') as configfile:
#            configEnf.write(configfile)
#
#        os.remove(enfFile)
#        os.rename(tmpFile,enfFile)
#        logging.warning("Enf file updated with Force plate assignement")



        # ----------------------DISPLAY ON VICON-------------------------------

        nexusTools.createGeneralEvents(NEXUS,subject,acqGait,["Left-FP","Right-FP"])


        # ========END of the nexus OPERATION if run from Nexus  =========

        if DEBUG:

            NEXUS.SaveTrial(30)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
