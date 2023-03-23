# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/EMG
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--

import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse

import pyCGM2

from pyCGM2.Lib import plot
from pyCGM2.Lib import emg
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from viconnexusapi import ViconNexus


def main():

    parser = argparse.ArgumentParser(description='plot temporal emg panel')
    parser.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser.add_argument('-r','--raw', action='store_true', help='non rectified data')
    parser.add_argument('-ina','--ignoreNormalActivity', action='store_true', help='do not display normal activity')

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        args = parser.parse_args()

        # --- acquisition file and path----
        DATA_PATH, inputFileNoExt = NEXUS.GetTrialName()
        inputFile = inputFileNoExt+".c3d"


        #--------------------------settings-------------------------------------
        emgManager = emg.loadEmg(DATA_PATH)

        # ----------------------INPUTS-------------------------------------------
        bandPassFilterFrequencies = emgManager.getProcessingSection()["BandpassFrequencies"]
        if args.BandpassFrequencies is not None:
            if len(args.BandpassFrequencies) != 2:
                raise Exception("[pyCGM2] - bad configuration of the bandpass frequencies ... set 2 frequencies only")
            else:
                bandPassFilterFrequencies = [float(args.BandpassFrequencies[0]),float(args.BandpassFrequencies[1])]
                LOGGER.logger.info("Band pass frequency set to %i - %i instead of 20-200Hz",bandPassFilterFrequencies[0],bandPassFilterFrequencies[1])

        envelopCutOffFrequency = emgManager.getProcessingSection()["EnvelopLowpassFrequency"]
        if args.EnvelopLowpassFrequency is not None:
            envelopCutOffFrequency =  args.EnvelopLowpassFrequency
            LOGGER.logger.info("Cut-off frequency set to %i instead of 6Hz ",envelopCutOffFrequency)

        rectifyBool = False if args.raw else True

        # --------------------------SUBJECT ------------------------------------
        subject = nexusTools.getActiveSubject(NEXUS)


        # btk Acquisition
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,inputFileNoExt,subject)
        acq = nacf.build()

        emgChannels = emgManager.getChannels()

        emg.processEMG_fromBtkAcq(acq, emgChannels,
            highPassFrequencies=bandPassFilterFrequencies,
            envelopFrequency=envelopCutOffFrequency) # high pass then low pass for all c3ds

        plot.plotTemporalEMG(DATA_PATH,inputFile,exportPdf=True,rectify=rectifyBool,
                            btkAcq=acq,ignoreNormalActivity= args.ignoreNormalActivity)

    else:
        return parser

if __name__ == "__main__":
    main()
