import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# pyCGM2 settings
import pyCGM2


# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets
from pyCGM2.ForcePlates import forceplates

def temporal(args):

    try:
        from viconnexusapi import ViconNexus
        from pyCGM2.Nexus import nexusFilters
        from pyCGM2.Nexus import nexusUtils
        from pyCGM2.Nexus import nexusTools
        from pyCGM2.Nexus import eclipse
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if NEXUS_PYTHON_CONNECTED:

        pointSuffix = args.pointSuffix
        # --------------------------INPUTS ------------------------------------
        DATA_PATH, modelledFilenameNoExt = nexusTools.getTrialName(NEXUS)

        modelledFilename = modelledFilenameNoExt+".c3d"

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.logger.info("file: " + modelledFilename)

        # ----- Subject -----
        # need subject to find input files
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("Subject name : " + subject)

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,
            DATA_PATH, modelledFilenameNoExt, subject)
        acq = nacf.build()

        # --------------------pyCGM2 MODEL ------------------------------
        plot.plotTemporalKinetic(DATA_PATH, modelledFilename, "LowerLimb",
                                 pointLabelSuffix=pointSuffix, exportPdf=True, btkAcq=acq)

    else:
        return 0



def normalized(args):


    plt.close("all")


    try:
        from viconnexusapi import ViconNexus
        from pyCGM2.Nexus import nexusFilters
        from pyCGM2.Nexus import nexusUtils
        from pyCGM2.Nexus import nexusTools
        from pyCGM2.Nexus import eclipse
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    ECLIPSE_MODE = False

    if not NEXUS_PYTHON_CONNECTED:
        return 0

    #-----------------------SETTINGS---------------------------------------
    pointSuffix = args.pointSuffix
    normativeData = {"Author" : args.normativeData, "Modality" : args.normativeDataModality}


    if normativeData["Author"] == "Schwartz2008":
        chosenModality = normativeData["Modality"]
    elif normativeData["Author"] == "Pinzone2014":
        chosenModality = normativeData["Modality"]
    nds = normativeDatasets.NormativeData(normativeData["Author"],chosenModality)


    consistencyFlag = True if args.consistency else False

    if eclipse.getCurrentMarkedNodes() is not None:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        # --- acquisition file and path----
        markedNodes = eclipse.getCurrentMarkedNodes()
        ECLIPSE_MODE = True

    if not ECLIPSE_MODE:
        LOGGER.logger.info("[pyCGM2] - Script works with the loaded c3d in vicon Nexus")
        # --- acquisition file and path----
        DATA_PATH, modelledFilenameNoExt = nexusTools.getTrialName(NEXUS)
        modelledFilename = modelledFilenameNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.logger.info( "file: "+ modelledFilename)



    if not ECLIPSE_MODE:
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info(  "Subject name : " + subject  )
        
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,modelledFilenameNoExt,subject)
        acq = nacf.build()

        mfpa = nexusTools.getForcePlateAssignment(NEXUS)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acq,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acq,mappedForcePlate,subject=subject)
        nexusTools.updateEvents(NEXUS,subject,acq)
        # btkTools.printEvents(acq)




        outputName = modelledFilename

        # --------------------------PROCESSING --------------------------------

        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            [modelledFilename],
                            eventType="Gait",
                            kinematicLabelsDict = None,
                            emgChannels = None,
                            pointLabelSuffix=pointSuffix,
                            btkAcqs=[acq],
                            subjectInfo=None, experimentalInfo=None,modelInfo=None)

    else:

        
        modelledFilenames = []
        paths=[]
        count=0
        for node in markedNodes:
            modelledFilenames.append(node[1].replace(".Trial.enf", ".c3d"))
            if count ==0: 
                DATA_PATH=node[0]
            else:
                if node[0] != DATA_PATH:
                    raise  Exception("marked nodes must be from the same folder")
                

        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            modelledFilenames,
                            eventType="Gait",
                            kinematicLabelsDict = None,
                            emgChannels = None,
                            pointLabelSuffix=pointSuffix,
                            subjectInfo=None, experimentalInfo=None,modelInfo=None)

        outputName = "Eclipse - NormalizedKinetics"

    if not consistencyFlag:
        plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)
    else:
        plot.plot_ConsistencyKinetic(DATA_PATH,analysisInstance,"LowerLimb",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)



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
        DATA_PATH, calibrateFilenameLabelledNoExt = nexusTools.getTrialName(NEXUS) 
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        markedNodes =eclipse.getCurrentMarkedNodes()
        if len(markedNodes)== 1:   raise Exception("Only one node marked")


    #-----------------------SETTINGS---------------------------------------
    normativeData = {"Author" : args.normativeData, "Modality" : args.normativeDataModality}

    if normativeData["Author"] == "Schwartz2008":
        chosenModality = normativeData["Modality"]
    elif normativeData["Author"] == "Pinzone2014":
        chosenModality = normativeData["Modality"]
    nds = normativeDatasets.NormativeData(normativeData["Author"],chosenModality)

    consistencyFlag = True if args.consistency else False
    plotType = "Consistency" if consistencyFlag else "Descriptive"

    pointSuffix = args.pointSuffix

    analysisInstances=[]
    legends =[]
    comparisonDetails =  "Kinetic comparison plots"
    for node in markedNodes:
        filename = node[1].replace(".Trial.enf", ".c3d")
        analysisInstances.append( analysis.makeAnalysis(node[0],
                                [filename],
                                eventType="Gait",
                                kinematicLabelsDict=None,
                                emgChannels = None,
                                pointLabelSuffix=pointSuffix,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None)
                                )
        
        legends.append(filename)
    
    plot.compareKinetic(DATA_PATH,analysisInstances,legends,"Left","LowerLimb",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)

    plot.compareKinetic(DATA_PATH,analysisInstances,legends,"Right","LowerLimb",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
            show=False, outputName=comparisonDetails,exportPdf=True)

    plt.show()



    
