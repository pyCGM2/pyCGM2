# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/EMG
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--

import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse

import pyCGM2

from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Lib import emg

from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from pyCGM2.Nexus import eclipse

from viconnexusapi import ViconNexus


def main():

    parser = argparse.ArgumentParser(description='plot time normalized EMG panel from the current trial or marked trials of Vicon Eclipse ')
    parser.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser.add_argument('-c','--consistency', action='store_true', help='consistency plots')

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    ECLIPSE_MODE = False

    if not NEXUS_PYTHON_CONNECTED:
        return parser

    args = parser.parse_args()

    #--------------------------Data Location-------------------------------------
    if eclipse.getCurrentMarkedNodes() is not None:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        # --- acquisition file and path----
        DATA_PATH, inputFiles =eclipse.getCurrentMarkedNodes()
        ECLIPSE_MODE = True

    if not ECLIPSE_MODE:
        LOGGER.logger.info("[pyCGM2] - Script works with the loaded c3d in vicon Nexus")
        # --- acquisition file and path----
        DATA_PATH, inputFileNoExt = NEXUS.GetTrialName()
        inputFile = inputFileNoExt+".c3d"

    LOGGER.set_file_handler(DATA_PATH+"pyCGM2.log")
    #--------------------------settings-------------------------------------
    emgManager = emg.loadEmg(DATA_PATH)
    emgChannels = emgManager.getChannels()

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

    consistencyFlag = True if args.consistency else False

    # --------------emg Processing--------------

    if not ECLIPSE_MODE:
        # --------------------------SUBJECT ------------------------------------
        subject = nexusTools.getActiveSubject(NEXUS)

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,inputFileNoExt,subject)
        acq = nacf.build()



        emg.processEMG_fromBtkAcq(acq, emgChannels,
            highPassFrequencies=bandPassFilterFrequencies,
            envelopFrequency=envelopCutOffFrequency) # high pass then low pass for all c3ds

        # emgAnalysis = analysis.makeEmgAnalysis(DATA_PATH, [inputFile], EMG_LABELS,btkAcqs = [acq])

        emgAnalysis = analysis.makeAnalysis(DATA_PATH,
                            [inputFile],
                            type="Gait",
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = emgChannels,
                            pointLabelSuffix=None,
                            btkAcqs=[acq],
                            subjectInfo=None, experimentalInfo=None,modelInfo=None,
                            )

        outputName = inputFile
    else:

        emg.processEMG(DATA_PATH, inputFiles, emgChannels,
            highPassFrequencies=bandPassFilterFrequencies,
            envelopFrequency=envelopCutOffFrequency)

        emgAnalysis = analysis.makeAnalysis(DATA_PATH,
                            [inputFile],
                            type="Gait",
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = emgChannels,
                            pointLabelSuffix=None,
                            subjectInfo=None, experimentalInfo=None,modelInfo=None,
                            )

        outputName = "Eclipse - Time-NormalizedEMG"


    if not consistencyFlag:
        plot.plotDescriptiveEnvelopEMGpanel(DATA_PATH,emgAnalysis, normalized=False,exportPdf=True,outputName=outputName)
    else:
        plot.plotConsistencyEnvelopEMGpanel(DATA_PATH,emgAnalysis, normalized=False,exportPdf=True,outputName=outputName)




if __name__ == "__main__":

    main()
