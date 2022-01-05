# -*- coding: utf-8 -*-
#APIDOC: /Apps/Vicon/Plots
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

from pyCGM2.Nexus import  nexusTools,nexusFilters
from pyCGM2.Eclipse import eclipse

def main():
    """  Plot time-normalized Kinematics from nexus-loaded trial or eclipse nodes from the **same** session

    By default, plot panel display the mean trace and the standard deviation corridor.
    A command argument allows to plot all cycles

    Usage:

    ```bash
        python plotNormalizedKinematics.py
        python plotNormalizedKinematics.py -c -ps CGM1 -nd Schwartz2008 -ndm VerySlow
    ```

    Args:
        [-nd,--normativeData] (str)[Schwartz2008]: normative dataset (Choice : Schwartz2008 or Pinzone2014)
        [--ndm,normativeDataModality] (str) [free]: normative dataset modality (if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo])
        ['-ps','--pointSuffix'] (str): suffix added to model outputs ()
        ['-c','--consistency'] (bool): plot all cycles instead of the mean and sd corridor
    """


    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM plot Normalized Kinematics')
    parser.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('-c','--consistency', action='store_true', help='consistency plots')

    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    ECLIPSE_MODE = False

    if not NEXUS_PYTHON_CONNECTED:
        raise Exception("Vicon Nexus is not running")


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
        # --------------------------PROCESSING --------------------------------

        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            modelledFilenames,
                            type="Gait",
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



if __name__ == "__main__":


    main()
