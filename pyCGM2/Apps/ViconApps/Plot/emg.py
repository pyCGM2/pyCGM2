import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse
import matplotlib.pyplot as plt
import pyCGM2

from pyCGM2.Lib import plot
from pyCGM2.Lib import emg
from pyCGM2.Nexus import eclipse
from pyCGM2.Lib import analysis

def temporal(args):

    try:
        from viconnexusapi import ViconNexus
        from pyCGM2.Nexus import nexusFilters
        from pyCGM2.Nexus import nexusUtils
        from pyCGM2.Nexus import nexusTools
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        # --- acquisition file and path----
        DATA_PATH, inputFileNoExt = nexusTools.getTrialName(NEXUS)
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
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,inputFileNoExt,subject)
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
        from viconnexusapi import ViconNexus
        from pyCGM2.Nexus import nexusFilters
        from pyCGM2.Nexus import nexusUtils
        from pyCGM2.Nexus import nexusTools
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
        markedNodes = eclipse.getCurrentMarkedNodes()
        ECLIPSE_MODE = True

    if not ECLIPSE_MODE:
        LOGGER.logger.info("[pyCGM2] - Script works with the loaded c3d in vicon Nexus")
        # --- acquisition file and path----
        DATA_PATH, inputFileNoExt = nexusTools.getTrialName(NEXUS)
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
    eventType = "noEvents" if args.ignoreGaitEvent else "Gait"

    # --------------emg Processing--------------

    if not ECLIPSE_MODE:
        # --------------------------SUBJECT ------------------------------------
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info(  "Subject name : " + subject  )

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,inputFileNoExt,subject)
        acq = nacf.build()



        emg.processEMG_fromBtkAcq(acq, emgChannels,
            highPassFrequencies=bandPassFilterFrequencies,
            envelopFrequency=envelopCutOffFrequency) # high pass then low pass for all c3ds

        # emgAnalysis = analysis.makeEmgAnalysis(DATA_PATH, [inputFile], EMG_LABELS,btkAcqs = [acq])

        emgAnalysis = analysis.makeAnalysis(DATA_PATH,
                            [inputFile],
                            eventType=eventType,
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = emgChannels,
                            pointLabelSuffix=None,
                            btkAcqs=[acq],
                            subjectInfo=None, experimentalInfo=None,modelInfo=None,
                            )

        outputName = inputFile
    else:

        inputFiles = []
        count=0
        for node in markedNodes:
            inputFiles.append(node[1].replace(".Trial.enf", ".c3d"))
            if count ==0: 
                DATA_PATH=node[0]
            else:
                if node[0] != DATA_PATH:
                    raise  Exception("marked nodes must be from the same folder")

        emg.processEMG(DATA_PATH, inputFiles, emgChannels,
            highPassFrequencies=bandPassFilterFrequencies,
            envelopFrequency=envelopCutOffFrequency)

        emgAnalysis = analysis.makeAnalysis(DATA_PATH,
                            inputFiles,
                            eventType=eventType,
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

    plt.close("all")

    try:
        from viconnexusapi import ViconNexus
        from pyCGM2.Nexus import nexusTools
        from pyCGM2.Nexus import eclipse
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if not NEXUS_PYTHON_CONNECTED:
        return 0


    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is None:
        raise Exception("No nodes marked")
    else:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        DATA_PATH, calibrateFilenameLabelledNoExt = nexusTools.getTrialName(NEXUS) 

        # --- acquisition file and path----
        markedNodes = eclipse.getCurrentMarkedNodes()
        if len(markedNodes)== 1:   raise Exception("Only one node marked")






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
    eventType = "noEvents" if args.ignoreGaitEvent else "Gait"
    # --------------emg Processing--------------


    inputFiles = []
    for node in markedNodes:
        inputFiles.append( node[1].replace(".Trial.enf", ".c3d"))


    emg.processEMG(DATA_PATH, inputFiles, emgChannels, highPassFrequencies=bandPassFilterFrequencies,
                envelopFrequency=envelopCutOffFrequency)

    analysisInstances=[]
    legends =[]
    comparisonDetails =  "comparison plots"
    
    count=0
    for node in markedNodes:
        filename = node[1].replace(".Trial.enf", ".c3d")
        analysisInstances.append( analysis.makeAnalysis(node[0],
                                [filename],
                                eventType=eventType,
                                kinematicLabelsDict=None,
                                kineticLabelsDict=None,
                                emgChannels = emgChannels,
                                pointLabelSuffix=None,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                ))
        
        legends.append(filename)

        if count==0:
            emg.normalizedEMG(DATA_PATH,analysisInstances[0],method="MeanMax", fromOtherAnalysis=None)
        else:
            emg.normalizedEMG(DATA_PATH,analysisInstances[count],method="MeanMax", fromOtherAnalysis=analysisInstances[0])
        count+=1


        if len(inputFiles) == 2:
            analysisInstance1 = analysis.makeAnalysis(DATA_PATH,
                                [inputFiles[0]],
                                eventType=eventType,
                                kinematicLabelsDict=None,
                                kineticLabelsDict=None,
                                emgChannels = emgChannels,
                                pointLabelSuffix=None,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )

            emg.normalizedEMG(DATA_PATH,analysisInstance1,method="MeanMax", fromOtherAnalysis=None)

            analysisInstance2 = analysis.makeAnalysis(DATA_PATH,
                                [inputFiles[1]],
                                eventType=eventType,
                                kinematicLabelsDict=None,
                                kineticLabelsDict=None,
                                emgChannels = emgChannels,
                                pointLabelSuffix=None,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )
            emg.normalizedEMG(DATA_PATH,analysisInstance2,method="MeanMax", fromOtherAnalysis=analysisInstance1)

            # outputName = "Eclipse - CompareNormalizedKinematics"
        #

    plot.compareEmgEnvelops(DATA_PATH,analysisInstances,
                            legends,
                            eventType="other",
                            normalized=True,
                            plotType=plotType,show=True,
                            outputName=comparisonDetails,exportPng=False)