# -*- coding: utf-8 -*-
import ipdb
import logging
import numpy as np
import matplotlib.pyplot as plt
# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2 import btk
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusFilters




class Tests():

    @classmethod
    def constructAcq(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\NexusAPI\\c3d_x2d\\"

        filenameNoExt = "gait_GAP"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]


        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.run()

        btkTools.smartWriter(acq,"constructAcq.c3d")




    @classmethod
    def constructAcq_wrenchOuputs(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\NexusAPI\\c3d_x2d\\"

        filenameNoExt = "gait_GAP"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]


        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.run()

        #btkTools.smartWriter(acq,"NEWC3D.c3d")

        # --- ground reaction force wrench ---
        pfe = btk.btkForcePlatformsExtractor()
        pfe.SetInput(acq)
        pfc = pfe.GetOutput()

        grwf = btk.btkGroundReactionWrenchFilter()
        grwf.SetInput(pfc)
        grwc = grwf.GetOutput()
        grwc.Update()


        # --- reference values ---
        acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))
        pfe0 = btk.btkForcePlatformsExtractor()
        pfe0.SetInput(acq0)
        pfc0 = pfe0.GetOutput()

        grwf0 = btk.btkGroundReactionWrenchFilter()
        grwf0.SetInput(pfc0)
        grwc0 = grwf0.GetOutput()
        grwc0.Update()

        aff = (acq0.GetFirstFrame()-1)*acq0.GetNumberAnalogSamplePerFrame()
        alf = (acq0.GetLastFrame())*acq0.GetNumberAnalogSamplePerFrame()


        plt.figure()
        plt.plot(grwc.GetItem(0).GetForce().GetValues()[aff:alf])
        plt.plot(grwc0.GetItem(0).GetForce().GetValues(),"o")


        plt.figure()
        plt.plot(grwc.GetItem(0).GetMoment().GetValues()[aff:alf])
        plt.plot(grwc0.GetItem(0).GetMoment().GetValues(),"o")


        plt.figure()
        plt.plot(grwc.GetItem(0).GetPosition().GetValues()[aff:alf])
        plt.plot(grwc0.GetItem(0).GetPosition().GetValues(),"o")
        plt.show()




if __name__ == "__main__":

    #Tests.constructAcq()
    Tests.constructAcq_wrenchOuputs()
