# -*- coding: utf-8 -*-
#from __future__ import unicode_literals

import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse

# pyCGM2 settings


# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

from pyCGM2.Tools import btkTools
from pyCGM2.Anomaly import AnomalyFilter, AnomalyDetectionProcedure, AnomalyCorrectionProcedure

def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--markers', nargs='+')
    parser.add_argument('--noCorrection', action='store_true', help='disable correction' )

    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        DATA_PATH, filename = NEXUS.GetTrialName()

        LOGGER.logger.info( " Path: "+ DATA_PATH )
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

        markers = args.markers
        # markers = cgm.CGM1.LOWERLIMB_TRACKING_MARKERS

        madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( markers, plot=False, window=10,threshold = 3)
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename,madp)
        anomalyIndexes = adf.run()

        if not args.noCorrection:
            macp = AnomalyCorrectionProcedure.MarkerAnomalyCorrectionProcedure(markers,anomalyIndexes,plot=False,distance_threshold=20)
            acf = AnomalyFilter.AnomalyCorrectionFilter(acq,filename,macp)
            acqo = acf.run()

            # --------------------------PUSH ------------------------------------
            for marker in markers:
                nexusTools.setTrajectoryFromAcq(NEXUS,subject,marker,acqo)
            # nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,"LHJC", acq,suffix = "")
            # nexusTools.appendAngleFromAcq(NEXUS,subject,str(it.GetLabel()), acq)
            # nexusTools.appendBones(NEXUS,subject,acq,"LFEMUR", model.getSegment("Left Thigh"),
            #     OriginValues = acq.GetPoint("LKJC").GetValues())
            # nexusTools.appendForceFromAcq(NEXUS,subject,"LHipForce", acq)
            # nexusTools.appendMomentFromAcq(NEXUS,subject,"LHipMoment", acq)
            # nexusTools.appendPowerFromAcq(NEXUS,subject,"LHipPower", acq)

if __name__ == '__main__':
    main()
