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

from pyCGM2.Nexus import eclipse


from viconnexusapi import ViconNexus


def main():

    parser = argparse.ArgumentParser(description='Comparison plot panel of normalized emg from marked nodes of Nexus Eclipse')
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

    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is None:
        raise Exception("No nodes marked")
    else:
        args = parser.parse_args()
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        # --- acquisition file and path----
        DATA_PATH, inputFiles =eclipse.getCurrentMarkedNodes()
        if isinstance(DATA_PATH,list):
            LOGGER.logger.error("[pyCGM2] - comparison of EMG from two distinct sessions is not allowed")
            raise

        ECLIPSE_MODE = True
        if len(inputFiles)== 1:   raise Exception("Only one node marked")



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
    plotType = "Consistency" if consistencyFlag else "Descriptive"
    # --------------emg Processing--------------


    if  ECLIPSE_MODE:

        emg.processEMG(DATA_PATH, inputFiles, emgChannels, highPassFrequencies=bandPassFilterFrequencies,
                envelopFrequency=envelopCutOffFrequency)

        if len(inputFiles) == 2:
            analysisInstance1 = analysis.makeAnalysis(DATA_PATH,
                                [inputFiles[0]],
                                type="Gait",
                                kinematicLabelsDict=None,
                                kineticLabelsDict=None,
                                emgChannels = emgChannels,
                                pointLabelSuffix=None,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )

            emg.normalizedEMG(DATA_PATH,analysisInstance1,method="MeanMax", fromOtherAnalysis=None)

            analysisInstance2 = analysis.makeAnalysis(DATA_PATH,
                                [inputFiles[1]],
                                type="Gait",
                                kinematicLabelsDict=None,
                                kineticLabelsDict=None,
                                emgChannels = emgChannels,
                                pointLabelSuffix=None,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )
            emg.normalizedEMG(DATA_PATH,analysisInstance2,method="MeanMax", fromOtherAnalysis=analysisInstance1)

            # outputName = "Eclipse - CompareNormalizedKinematics"
        #
        analysesToCompare = [analysisInstance1, analysisInstance2]
        comparisonDetails =  inputFiles[0] + " Vs " + inputFiles[1]
        legends =[inputFiles[0],inputFiles[1]]

        plot.compareEmgEnvelops(DATA_PATH,analysesToCompare,
                                legends,
                              normalized=True,
                              plotType=plotType,show=True,
                              outputName=comparisonDetails,exportPng=False)




if __name__ == "__main__":

    main()
