import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse

import pyCGM2

from pyCGM2.Lib import plot
from pyCGM2.Lib import emg
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from viconnexusapi import ViconNexus
from pyCGM2.Nexus import eclipse

from pyCGM2.Lib import analysis

def temporal(args):

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
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
        return 0

def normalized(args):

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    ECLIPSE_MODE = False

    if not NEXUS_PYTHON_CONNECTED:
        return 0

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


def normalizedComparison(args):

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    ECLIPSE_MODE = False
    if not NEXUS_PYTHON_CONNECTED:
        return 0

    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is None:
        raise Exception("No nodes marked")
    else:
        
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