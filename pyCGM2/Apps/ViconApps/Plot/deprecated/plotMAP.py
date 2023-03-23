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


# pyCGM2 settings
import pyCGM2


# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from pyCGM2.Nexus import eclipse

def main():


    plt.close("all")

    parser = argparse.ArgumentParser(description='plot MAP')
    parser.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')


    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if not NEXUS_PYTHON_CONNECTED:
        return parser

    args = parser.parse_args()
    #-----------------------SETTINGS---------------------------------------
    pointSuffix = args.pointSuffix


    #-----------------------SETTINGS---------------------------------------
    normativeData = {"Author" : args.normativeData, "Modality" : args.normativeDataModality}

    if normativeData["Author"] == "Schwartz2008":
        chosenModality = normativeData["Modality"]
    elif normativeData["Author"] == "Pinzone2014":
        chosenModality = normativeData["Modality"]
    nds = normativeDatasets.NormativeData(normativeData["Author"],chosenModality)


    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is not None:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        # --- acquisition file and path----
        DATA_PATH, modelledFilenames =eclipse.getCurrentMarkedNodes()
        ECLIPSE_MODE = True

    if not ECLIPSE_MODE:
        LOGGER.logger.info("[pyCGM2] - Script works with the loaded c3d in vicon Nexus")
        # --- acquisition file and path----
        DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()
        modelledFilename = modelledFilenameNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.logger.info( "file: "+ modelledFilename)


    # ----- Subject -----
    # need subject to find input files
    subject = nexusTools.getActiveSubject(NEXUS)
    LOGGER.logger.info(  "Subject name : " + subject  )

    if not ECLIPSE_MODE:
        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,modelledFilenameNoExt,subject)
        acq = nacf.build()

        # --------------------------PROCESSING --------------------------------
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            [modelledFilename],
                            type="Gait",
                            kineticLabelsDict = None,
                            emgChannels = None,
                            pointLabelSuffix=pointSuffix,
                            btkAcqs=[acq],
                            subjectInfo=None, experimentalInfo=None,modelInfo=None)

        outputName = modelledFilename
    else:
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            type="Gait",
                            kineticLabelsDict = None,
                            emgChannels = None,
                            pointLabelSuffix=pointSuffix,
                            subjectInfo=None, experimentalInfo=None,modelInfo=None)

        outputName = "Eclipse-MAP"

    plot.plot_MAP(DATA_PATH,analysisInstance,nds,exportPdf=True,outputName=outputName,pointLabelSuffix=pointSuffix)



if __name__ == "__main__":

    main()
