# -*- coding: utf-8 -*-

import logging

import pyCGM2

import ViconNexus

import btk

from pyCGM2.Tools import btkTools

if __name__ == "__main__":

    DEBUG = False

    pyNEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

    print NEXUS_PYTHON_CONNECTED

    #NEXUS_PYTHON_CONNECTED = True

    if NEXUS_PYTHON_CONNECTED: # run Operation

        # inputs

        if DEBUG:
            DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\CGM1-Calibration\\"
            filenameNoExt = "static Cal 01-noKAD-noAnkleMed" 
            pyNEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

        else:
            DATA_PATH, filenameNoExt = pyNEXUS.GetTrialName()

            filename = filenameNoExt+".c3d"

            # add metadata
            acq= btkTools.smartReader(str(DATA_PATH + filename))
            md_Model = btk.btkMetaData('MODEL') # create main metadata
            btk.btkMetaDataCreateChild(md_Model, "NAME", "CGM1")
            btk.btkMetaDataCreateChild(md_Model, "PROCESSOR", "pyCGM2")
            acq.GetMetaData().AppendChild(md_Model)
    
            # save
            btkTools.smartWriter(acq,str(DATA_PATH + filenameNoExt + ".c3d"))
            logging.info( "[pyCGM2] : add Model Metdata in file ( %s) " % (filename))