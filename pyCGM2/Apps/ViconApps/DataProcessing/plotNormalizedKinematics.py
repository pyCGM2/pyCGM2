# -*- coding: utf-8 -*-
"""Nexus Operation : **plotNormalizedKinematics**

The script displays Gait-Normalized kinematics

:param -ps, --pointSuffix [string]: suffix adds to the vicon nomenclature outputs
:param -c, --consistency [bool]: display consistency plot ( ie : all gait cycle) instead of a descriptive statistics view
:param -nd, --normativeData [string]: Normative data set ( choice: Schwartz2008 [DEFAULT] or Pinzone2014)
:param -ndm, --normativeDataModality [string]: modalities associated with the selected normative dataset. (choices: if  Schwartz2008: VerySlow,Slow,Free[DEFAULT],Fast,VeryFast.  if Pinzone2014 : CentreOne,CentreTwo)


Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -normativeData=Schwartz2008 --normativeDataModality=VeryFast
    (your gait panel will display as normative data, results from the modality VeryFast of the nomative dataset collected by Schwartz2008)

"""

import logging
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries

from pyCGM2 import enums
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import  nexusTools,nexusFilters
from pyCGM2.Utils import files

def main():

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

    if NEXUS_PYTHON_CONNECTED:

        #-----------------------SETTINGS---------------------------------------
        normativeData = {"Author" : args.normativeData, "Modality" : args.normativeDataModality}

        if normativeData["Author"] == "Schwartz2008":
            chosenModality = normativeData["Modality"]
            nds = normativeDatasets.Schwartz2008(chosenModality)    # modalites : "Very Slow" ,"Slow", "Free", "Fast", "Very Fast"
        elif normativeData["Author"] == "Pinzone2014":
            chosenModality = normativeData["Modality"]
            nds = normativeDatasets.Pinzone2014(chosenModality) # modalites : "Center One" ,"Center Two"

        consistencyFlag = True if args.consistency else False
        pointSuffix = args.pointSuffix

        # --------------------------INPUTS ------------------------------------
        DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()


        modelledFilename = modelledFilenameNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "file: "+ modelledFilename)

        # ----- Subject -----
        # need subject to find input files
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)
        modelVersion = model.version

        # ----- construction of the openMA root instance  -----
        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,modelledFilenameNoExt,subject)
        openmaTrial = trialConstructorFilter.build()


        # --------------------------PROCESSING --------------------------------
        analysisInstance = analysis.makeAnalysis(DATA_PATH,[modelledFilename], pointLabelSuffix=pointSuffix,
                                                openmaTrials=[openmaTrial])

        if not consistencyFlag:
            if model.m_bodypart in [enums.BodyPart.LowerLimb,enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
                plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=modelledFilename)
                    #plot_DescriptiveKinematic(DATA_PATH,analysis,bodyPart,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None):

            if model.m_bodypart in [enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
                plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"Trunk",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=modelledFilename)
            if model.m_bodypart in [enums.BodyPart.UpperLimb, enums.BodyPart.FullBody]:
                pass # TODO plot upperlimb panel

        else:
            if model.m_bodypart in [enums.BodyPart.LowerLimb,enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
                plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"LowerLimb",nds, pointLabelSuffix=pointSuffix, exportPdf=True,outputName=modelledFilename)
            if model.m_bodypart in [enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
                plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"Trunk",nds,pointLabelSuffix=pointSuffix, exportPdf=True,outputName=modelledFilename)
            if model.m_bodypart in [enums.BodyPart.UpperLimb, enums.BodyPart.FullBody]:
                pass # TODO plot upperlimb panel


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":


    main()
