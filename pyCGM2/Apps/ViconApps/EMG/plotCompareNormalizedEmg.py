# -*- coding: utf-8 -*-
"""Nexus Operation : **plotNormalizedEmg**

The script displays gait-normalized emg envelops

:param -bpf, --BandpassFrequencies [array]: bandpass frequencies
:param -ecf, --EnvelopLowpassFrequency [double]: cut-off low pass frequency for getting emg envelop
:param -c, --consistency [bool]: display consistency plot ( ie : all gait cycles) instead of a descriptive statistics view

Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -bpf 20 450 -ecf=8.9 --consistency
    (bandpass frequencies set to 20 and 450Hz and envelop made from a low-pass filter with a cutoff frequency of 8.9Hz,
    all gait cycles will be displayed)


"""
import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse

import pyCGM2

from pyCGM2.Utils import files
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Lib import emg

from pyCGM2.Nexus import nexusFilters,nexusTools
from pyCGM2.Eclipse import eclipse


from viconnexusapi import ViconNexus


def main():

    parser = argparse.ArgumentParser(description='EMG-plot_temporalEMG')
    parser.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser.add_argument('-c','--consistency', action='store_true', help='consistency plots')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    ECLIPSE_MODE = False

    if not NEXUS_PYTHON_CONNECTED:
        raise Exception("Vicon Nexus is not running")

    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is None:
        raise Exception("No nodes marked")
    else:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        # --- acquisition file and path----
        DATA_PATH, inputFiles =eclipse.getCurrentMarkedNodes()
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
