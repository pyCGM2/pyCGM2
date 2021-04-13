# -*- coding: utf-8 -*-
"""Nexus Operation : **plotMAP**

The script displays Gait-Normalized kinematics

:param -ps, --pointSuffix [string]: suffix adds to the vicon nomenclature outputs
:param -nd, --normativeData [string]: Normative data set ( choice: Schwartz2008 [DEFAULT] or Pinzone2014)
:param -ndm, --normativeDataModality [string]: modalities associated with the selected normative dataset. (choices: if  Schwartz2008: VerySlow,Slow,Free[DEFAULT],Fast,VeryFast.  if Pinzone2014 : CentreOne,CentreTwo)


Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -normativeData=Schwartz2008 --normativeDataModality=VeryFast
    (your gait panel will display as normative data, results from the modality VeryFast of the nomative dataset collected by Schwartz2008)

"""


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

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM plotMAP')
    parser.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')


    args = parser.parse_args()



    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if not NEXUS_PYTHON_CONNECTED:
        raise Exception("Vicon Nexus is not running")


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
        analysisInstance = analysis.makeAnalysis(DATA_PATH,[modelledFilename],None, None, None,pointLabelSuffix=pointSuffix,btkAcqs=[acq]) # analysis structure gathering Time-normalized Kinematic and kinetic CGM outputs
        outputName = modelledFilename
    else:
        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames,None, None, None,pointLabelSuffix=pointSuffix)
        outputName = "Eclipse-MAP"

    plot.plot_MAP(DATA_PATH,analysisInstance,nds,exportPdf=True,outputName=outputName,pointLabelSuffix=pointSuffix)



if __name__ == "__main__":

    main()
