import argparse
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from pyCGM2.Gap import gapFillingProcedures
from pyCGM2.Gap import gapFilters
from viconnexusapi import ViconNexus
import pyCGM2
LOGGER = pyCGM2.LOGGER


def main(args=None):
    if args  is None:
        parser = argparse.ArgumentParser(description='Gloersen PCA-based Gap filling')
        parser.add_argument('--markers', nargs='*', help='list of markers',required=False)


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

        #acq = btkTools.smartReader(str(DATA_PATH+filenameLabelledNoExt+".c3d"))

        gfp = gapFillingProcedures.Gloersen2016GapFillingProcedure()
        gff = gapFilters.GapFillingFilter(gfp, acq)
        markers = args.markers if args.markers else None
        gff.fill(markers= markers)



        filledAcq = gff.getFilledAcq()
        filledMarkers = gff.getFilledMarkers()

        for marker in filledMarkers:
            nexusTools.setTrajectoryFromAcq(NEXUS, subject, marker, filledAcq)
    else:
        return 0

if __name__ == "__main__":

    main(args=None)
