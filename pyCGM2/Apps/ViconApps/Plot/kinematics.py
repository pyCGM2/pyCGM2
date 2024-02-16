import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# pyCGM2 settings
import pyCGM2




# pyCGM2 libraries

from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets



def temporal(args):

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
        plot.plotTemporalKinematic(DATA_PATH, modelledFilename, "LowerLimb",
                                   pointLabelSuffix=pointSuffix, exportPdf=True, btkAcq=acq)
        plot.plotTemporalKinematic(DATA_PATH, modelledFilename, "Trunk",
                                   pointLabelSuffix=pointSuffix, exportPdf=True, btkAcq=acq)
        plot.plotTemporalKinematic(DATA_PATH, modelledFilename, "UpperLimb",
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


    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is not None:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        # --- acquisition file and path----
        DATA_PATH, modelledFilenames =eclipse.getCurrentMarkedNodes()
        ECLIPSE_MODE = True

    if not ECLIPSE_MODE:
        LOGGER.logger.info("[pyCGM2] - Script works with the loaded c3d in vicon Nexus")
        # --- acquisition file and path----
        DATA_PATH, modelledFilenameNoExt = nexusTools.getTrialName(NEXUS)
        modelledFilename = modelledFilenameNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.logger.info( "file: "+ modelledFilename)


    subjects = NEXUS.GetSubjectNames()
    subject = nexusTools.getActiveSubject(NEXUS)
    LOGGER.logger.info(  "Subject name : " + subject  )


    #-----------------------SETTINGS---------------------------------------
    normativeData = {"Author" : args.normativeData, "Modality" : args.normativeDataModality}

    if normativeData["Author"] == "Schwartz2008":
        chosenModality = normativeData["Modality"]
    elif normativeData["Author"] == "Pinzone2014":
        chosenModality = normativeData["Modality"]
    nds = normativeDatasets.NormativeData(normativeData["Author"],chosenModality)

    consistencyFlag = True if args.consistency else False
    pointSuffix = args.pointSuffix



    if not ECLIPSE_MODE:

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,modelledFilenameNoExt,subject)
        acq = nacf.build()

        # --------------------------PROCESSING --------------------------------
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            [modelledFilename],
                            eventType="Gait",
                            kineticLabelsDict = None,
                            emgChannels = None,
                            pointLabelSuffix=pointSuffix,
                            btkAcqs=[acq],
                            subjectInfo=None, experimentalInfo=None,modelInfo=None)

        outputName = modelledFilename

    else:
        # --------------------------PROCESSING --------------------------------

        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            modelledFilenames,
                            eventType="Gait",
                            kineticLabelsDict = None,
                            emgChannels = None,
                            pointLabelSuffix=pointSuffix,
                            subjectInfo=None, experimentalInfo=None,modelInfo=None)


        outputName = "Eclipse - NormalizedKinematics"




    if not consistencyFlag:
        plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)
        plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"Trunk",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)
        plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"UpperLimb",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)

    else:
        plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"LowerLimb",nds, pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)
        plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"Trunk",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)
        plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"UpperLimb",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=outputName)



def normalizedComparison(args):

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


    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is None:
        raise Exception("No nodes marked")
    else:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        DATA_PATH = os.getcwd()+"\\"
        # --- acquisition file and path----
        DATA_PATHS, modelledFilenames =eclipse.getCurrentMarkedNodes()
        ECLIPSE_MODE = True
        if len(modelledFilenames)== 1:   raise Exception("Only one node marked")

    subject = nexusTools.getActiveSubject(NEXUS)
    LOGGER.logger.info(  "Subject name : " + subject  )

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

    if  ECLIPSE_MODE:

        if isinstance(DATA_PATHS,list):
            path0 =DATA_PATHS[0]
            path1 =DATA_PATHS[1]
        if isinstance(DATA_PATHS,str):
            path0 = DATA_PATHS
            path1 =DATA_PATHS

        if len(modelledFilenames) == 2:
            analysisInstance1 = analysis.makeAnalysis(path0,
                                [modelledFilenames[0]],
                                eventType="Gait",
                                kineticLabelsDict=None,
                                emgChannels = None,
                                pointLabelSuffix=pointSuffix,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )

            analysisInstance2 =  analysis.makeAnalysis(path1,
                                [modelledFilenames[1]],
                                eventType="Gait",
                                kineticLabelsDict=None,
                                emgChannels = None,
                                pointLabelSuffix=pointSuffix,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )

            # outputName = "Eclipse - CompareNormalizedKinematics"
        #
        analysesToCompare = [analysisInstance1, analysisInstance2]
        comparisonDetails =  modelledFilenames[0] + " Vs " + modelledFilenames[1]
        legends =[modelledFilenames[0],modelledFilenames[1]]

        plot.compareKinematic(DATA_PATH,analysesToCompare,legends,"Left","LowerLimb",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)
        plot.compareKinematic(DATA_PATH,analysesToCompare,legends,"Left","Trunk",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)
        plot.compareKinematic(DATA_PATH,analysesToCompare,legends,"Left","UpperLimb",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)


        plot.compareKinematic(DATA_PATH,analysesToCompare,legends,"Right","LowerLimb",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)
        plot.compareKinematic(DATA_PATH,analysesToCompare,legends,"Right","Trunk",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)
        plot.compareKinematic(DATA_PATH,analysesToCompare,legends,"Right","UpperLimb",nds,plotType=plotType,eventType="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)

        plt.show()
