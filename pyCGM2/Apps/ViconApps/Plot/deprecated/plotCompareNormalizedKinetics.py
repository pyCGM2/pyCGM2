# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/Plot
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

# pyCGM2 settings
import pyCGM2


# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries

from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import  nexusTools
from pyCGM2.Nexus import eclipse

def main():


    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM plot Normalized Kinematics')
    parser.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
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
                                type="Gait",
                                kinematicLabelsDict=None,
                                emgChannels = None,
                                pointLabelSuffix=pointSuffix,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )

            analysisInstance2 = analysisInstance1 = analysis.makeAnalysis(path1,
                                [modelledFilenames[1]],
                                type="Gait",
                                kinematicLabelsDict=None,
                                emgChannels = None,
                                pointLabelSuffix=pointSuffix,
                                subjectInfo=None, experimentalInfo=None,modelInfo=None,
                                )

            # outputName = "Eclipse - CompareNormalizedKinematics"
        #
        analysesToCompare = [analysisInstance1, analysisInstance2]
        comparisonDetails =  modelledFilenames[0] + " Vs " + modelledFilenames[1]
        legends =[modelledFilenames[0],modelledFilenames[1]]

        plot.compareKinetic(DATA_PATH,analysesToCompare,legends,"Left","LowerLimb",nds,plotType=plotType,type="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)

        plot.compareKinetic(DATA_PATH,analysesToCompare,legends,"Right","LowerLimb",nds,plotType=plotType,type="Gait",pointSuffixes=None,
                show=False, outputName=comparisonDetails,exportPdf=True)

        plt.show()

if __name__ == "__main__":


    main()
