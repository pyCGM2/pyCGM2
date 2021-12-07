# -*- coding: utf-8 -*-
from pyCGM2.Anomaly import AnomalyFilter, AnomalyDetectionProcedure, AnomalyCorrectionProcedure
from pyCGM2.Nexus import nexusFilters, nexusUtils, nexusTools
from viconnexusapi import ViconNexus
import argparse
import pyCGM2
LOGGER = pyCGM2.LOGGER

# vicon nexus

# pyCGM2 libraries

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk


def main():
    """ [**Experimental**] Check marker trajectory anomaly from Nexus.


    After the detection, a correction is applied on the trajectory.
    ( can be disable from the command argument *--noCorrection* )

    Usage:

    ```bash
        python anomalies.py --markers LASI RASI
    ```

    Args:
        [--markers] (lst): list of markers
        [--noCorrection] (bool): disable correction
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--markers', nargs='+')
    parser.add_argument('--noCorrection', action='store_true',
                        help='disable correction')

    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED:  # run Operation

        DATA_PATH, filename = NEXUS.GetTrialName()

        LOGGER.logger.info(" Path: " + DATA_PATH)
        LOGGER.logger.info(" file: " + filename)

        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        # checkActivatedSubject(NEXUS,subjects)
        subject = nexusTools.getActiveSubject(NEXUS)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        # --------------------------PULL ------------------------------------
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, filename, subject)
        acq = nacf.build()

        # --------------------------process ------------------------------------
        # Work with BTK Here

        markers = args.markers
        # markers = cgm.CGM1.LOWERLIMB_TRACKING_MARKERS

        madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure(
            markers, plot=False, window=5, threshold=3)
        adf = AnomalyFilter.AnomalyDetectionFilter(acq, filename, madp)
        anomaly = adf.run()

        if not args.noCorrection:
            macp = AnomalyCorrectionProcedure.MarkerAnomalyCorrectionProcedure(
                markers, anomaly["Output"], plot=False, distance_threshold=20)
            acf = AnomalyFilter.AnomalyCorrectionFilter(acq, filename, macp)
            acqo = acf.run()

            # --------------------------PUSH ------------------------------------
            for marker in markers:
                nexusTools.setTrajectoryFromAcq(NEXUS, subject, marker, acqo)
            # nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,"LHJC", acq,suffix = "")
            # nexusTools.appendAngleFromAcq(NEXUS,subject,str(it.GetLabel()), acq)
            # nexusTools.appendBones(NEXUS,subject,acq,"LFEMUR", model.getSegment("Left Thigh"),
            #     OriginValues = acq.GetPoint("LKJC").GetValues())
            # nexusTools.appendForceFromAcq(NEXUS,subject,"LHipForce", acq)
            # nexusTools.appendMomentFromAcq(NEXUS,subject,"LHipMoment", acq)
            # nexusTools.appendPowerFromAcq(NEXUS,subject,"LHipPower", acq)


if __name__ == '__main__':
    main()
