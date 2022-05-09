# -*- coding: utf-8 -*-

import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse


# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

from pyCGM2.Tools import btkTools

def main():

    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    # parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    # parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    # parser.add_argument('--check', action='store_true', help='force model output suffix')
    # parser.add_argument('--forceLHJC', nargs='+')
    # args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        DATA_PATH, filename = NEXUS.GetTrialName()

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2.log")
        LOGGER.logger.info( " file: "+ filename)


        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS) #checkActivatedSubject(NEXUS,subjects)
        Parameters = NEXUS.GetSubjectParamNames(subject)


        # --------------------------PULL ------------------------------------
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,filename,subject)
        acq = nacf.build()


        # --------------------------process ------------------------------------
        # Work with BTK Here
        values = (acq.GetPoint("LTIAP").GetValues() + acq.GetPoint("LTIB").GetValues()) /2.0
        btkTools.smartAppendPoint(acq,"LTIAD",values, PointType="Marker",desc="",residuals = None)



        # --------------------------PUSH ------------------------------------
        # nexusTools.setTrajectoryFromAcq(NEXUS,subject,"LTIAD",acq)
        # nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,"LHJC", acq,suffix = "")
        # nexusTools.appendAngleFromAcq(NEXUS,subject,str(it.GetLabel()), acq)
        # nexusTools.appendBones(NEXUS,subject,acq,"LFEMUR", model.getSegment("Left Thigh"),
        #     OriginValues = acq.GetPoint("LKJC").GetValues())
        # nexusTools.appendForceFromAcq(NEXUS,subject,"LHipForce", acq)
        # nexusTools.appendMomentFromAcq(NEXUS,subject,"LHipMoment", acq)
        # nexusTools.appendPowerFromAcq(NEXUS,subject,"LHipPower", acq)

if __name__ == '__main__':
    main()
