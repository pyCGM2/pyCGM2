# -*- coding: utf-8 -*-
"""Nexus Operation : **plotTemporalKinematics**

The script displays kinematics with time as x-axis

:param -ps, --pointSuffix [string]: suffix adds to the vicon nomenclature outputs

Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -ps=py
    (all points will be suffixed with py (LHipAngles_py))


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
try:
    import ViconNexus
except:
    from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2 import enums
from pyCGM2.Lib import plot

from pyCGM2.Nexus import  nexusTools,nexusFilters
from pyCGM2.Utils import files

def main():

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM plot Temporal Kinematics')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')

    args = parser.parse_args()



    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED:

        pointSuffix = args.pointSuffix
        # --------------------------INPUTS ------------------------------------
        DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()


        modelledFilename = modelledFilenameNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "file: "+ modelledFilename)

        # ----- Subject -----
        # need subject to find input files
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        logging.info(  "Subject name : " + subject  )

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,modelledFilenameNoExt,subject)
        acq = nacf.build()


        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)
        modelVersion = model.version

        # --------------------pyCGM2 MODEL ------------------------------
        if model.m_bodypart in [enums.BodyPart.LowerLimb,enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
            plot.plotTemporalKinematic(DATA_PATH, modelledFilename,"LowerLimb", pointLabelSuffix=pointSuffix,exportPdf=True,btkAcq=acq)
        if model.m_bodypart in [enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
            plot.plotTemporalKinematic(DATA_PATH, modelledFilename,"Trunk", pointLabelSuffix=pointSuffix,exportPdf=True,btkAcq=acq)
        if model.m_bodypart in [enums.BodyPart.UpperLimb, enums.BodyPart.FullBody]:
            pass # TODO plot upperlimb panel



    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    main()
