# -*- coding: utf-8 -*-
#APIDOC: /Apps/Vicon/Gap filling

import argparse
import copy
import pyCGM2; LOGGER = pyCGM2.LOGGER

# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2.Model import model, modelFilters
from pyCGM2.Tools import btkTools

from pyCGM2 import enums
from pyCGM2.Nexus import nexusTools,nexusFilters

def main():
    """ rigid gap filling from tracking markers from a loaded trial in nexus.
    This function use the local position of the targeted marker collected during the static trial as reference.
    By default, the targeted marker is constructed from the selected first frame to the slected last frame
    of your trial loaded in Nexus


    Usage:

    load you trial in nexus, then run

    ```bash
        python rigidGapFillingMarkers.py --static="Kevin Cal 01" --target=RASI --trackingMarkers LASI LPSI RPSI
        python rigidGapFillingMarkers.py --static="Kevin Cal 01" --target=RASI --trackingMarkers LASI LPSI RPSI --begin=3589 --last=3600
    ```

    Args:
        [--target] (str): name of the targeted marker to reconstruct
        [--trackingMarkers] (list of 3 str): name of the tracking markers (3 marker at least)
        [--static] (str): name of the static file ( required to be save as c3d)
        [-b,--begin] (int): start frame
        [-e,--end] (int): last frame to process

    """

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='rigidLabelling')
    parser.add_argument('--static', type=str, help='filename of the static',required=True)
    parser.add_argument('--target', type=str, help='marker to reconstruct',required=True)
    parser.add_argument('--trackingMarkers', nargs='*', help='list of tracking markers',required=True)
    parser.add_argument('--begin', type=int, help='initial Frame')
    parser.add_argument('--last', type=int, help='last Frame')
    args = parser.parse_args()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        # enfFiles = eclipse.getEnfTrials(DATA_PATH)

        subject = nexusTools.getActiveSubject(NEXUS)

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acqGait = nacf.build()

        ff = acqGait.GetFirstFrame()-1
        lf = acqGait.GetLastFrame()-1

        # # input arguments management

        # print eclipse.findCalibrationFromEnfs(DATA_PATH,enfFiles)
        staticFilenameNoExt = args.static
        acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilenameNoExt+".c3d"))

        targetMarker = args.target
        trackingMarkers = args.trackingMarkers
        # trackingMarkers.append(targetMarker)

        if args.begin is None and args.last is None: # reconstrution on full frames
            selectInitialFrame = ff
            selectLastFrame = lf
        elif args.begin is not None and args.last is not None: # reconstrution from both selected begin and end frame
            selectInitialFrame = args.begin-1
            selectLastFrame = args.last-1
        elif args.begin is not None and args.last is None: # reconstrution from  selected begin and last frame
            selectInitialFrame = args.begin-1
            selectLastFrame = lf
        elif args.begin is None and args.last is not None: # reconstrution from  first frame and to selected last frame
            selectInitialFrame = ff
            selectLastFrame = args.last-1

        mod=model.Model()
        mod.addSegment("segment",0,enums.SegmentSide.Central,calibration_markers=[targetMarker], tracking_markers = trackingMarkers)


        gcp=modelFilters.GeneralCalibrationProcedure()
        gcp.setDefinition('segment',
                          "TF",
                          sequence='XYZ',
                          pointLabel1=trackingMarkers[0],
                          pointLabel2=trackingMarkers[1],
                          pointLabel3=trackingMarkers[2],
                          pointLabelOrigin=trackingMarkers[0])

        modCal=modelFilters.ModelCalibrationFilter(gcp,acqStatic,mod)
        modCal.compute()

        # if not btkTools.isPointExist(acqGait,targetMarker):
        #     # print "targer Marker not in the c3d"
        #     mod.getSegment("segment").m_tracking_markers.remove(targetMarker)

        modMotion=modelFilters.ModelMotionFilter(gcp,acqGait,mod,enums.motionMethod.Sodervisk)
        modMotion.compute()


        #populate values
        valReconstruct=mod.getSegment('segment').getReferential('TF').getNodeTrajectory(targetMarker)

        if btkTools.isPointExist(acqGait,targetMarker):
            val0 = acqGait.GetPoint(targetMarker).GetValues()
            val_final = copy.deepcopy(val0)
            val_final[selectInitialFrame-ff:selectLastFrame+1-ff,:] = valReconstruct[selectInitialFrame-ff:selectLastFrame+1-ff,:]
        else:
            val_final = valReconstruct

        # nexus display
        nexusTools.setTrajectoryFromArray(NEXUS,subject,targetMarker,val_final,firstFrame = ff)

if __name__ == "__main__":
    main()
