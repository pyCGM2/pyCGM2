import argparse
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from pyCGM2.Gap import gapFillingProcedures
from pyCGM2.Gap import gapFilters
from viconnexusapi import ViconNexus
from pyCGM2.Tools import btkTools

import pyCGM2
LOGGER = pyCGM2.LOGGER


def main(args=None):
    
    # if args  is None:
    parser = argparse.ArgumentParser(description='clone marker script')
    parser.add_argument('-l','--markerLabel', type=str, help="label of the new mlarker")
    parser.add_argument('-mtc','--markerToClone', type=str, help="label of the marker to clone")
    parser.add_argument('-ot','--outputType', type=str, choices=['modelledMarker', 'trajectory'],
                        help="output type", default='trajectory')
    args = parser.parse_args()


    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if NEXUS_PYTHON_CONNECTED:  # run Operation

        DATA_PATH, filenameLabelledNoExt = NEXUS.GetTrialName()

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.logger.info("file: " + filenameLabelledNoExt)

        # checkActivatedSubject(NEXUS,subjects)
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("Gap filling for subject %s" % (subject))

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,
            DATA_PATH, filenameLabelledNoExt, subject)
        acq = nacf.build()

        values = acq.GetPoint(args.markerToClone).GetValues()
        values[:,2] = values[:,2]+20
        btkTools.smartAppendPoint(acq, args.markerLabel, values )

        if args.outputType  == 'trajectory':      
            nexusTools.setTrajectoryFromAcq(NEXUS, subject, args.markerLabel, acq)
        
        if args.outputType  == 'modelledMarker':
            nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,args.markerLabel, acq,suffix = "")
    else:
        return 0

if __name__ == "__main__":

    main()
