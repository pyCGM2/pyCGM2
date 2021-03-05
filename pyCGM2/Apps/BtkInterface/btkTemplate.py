# -*- coding: utf-8 -*-
#from __future__ import unicode_literals

import logging
import argparse

# pyCGM2 settings
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

try:
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

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

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( " file: "+ filename)


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
        btkTools.smartAppendPoint(acq,"LTIAD",values, PointType=btk.btkPoint.Marker,desc="",residuals = None)



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
