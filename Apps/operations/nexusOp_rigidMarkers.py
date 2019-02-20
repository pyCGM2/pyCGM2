# -*- coding: utf-8 -*-
import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import argparse
import copy
from pyCGM2.Eclipse import vskTools,eclipse

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Model import model, modelFilters
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools
from pyCGM2 import enums

def main(args):

    """
    goal: rigid gap filling. fill gap from 3 markers
    usage : --static="Kevin Cal 01" --target=RASI --trackingMarkers LASI LPSI RPSI --begin=3589 --last=3600

    .. warning:

        target marker must be with the vsk tree to be know as a trajectory

    """


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()



    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        DEBUG = False
        if DEBUG:
            DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\Nantes\\REM_AVC\\Patient 3\\Session 2\\" #pyCGM2.TEST_DATA_PATH +"operations\\miscellaneous\\rigid_labelling_pyCGM2\\"
            reconstructFilenameLabelledNoExt ="19-02-PE-PA-J30-08-06-2012 ProcheBas03"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )

            acqStatic = btkTools.smartReader(str(DATA_PATH+"19-02-PE-PA-J30-08-06-2012 S03.c3d"))
            targetMarker = "PX"
            trackingMarkers = ["IJ","C7","T8","PX"]
            selectInitialFrame = 149
            selectLastFrame = 802


        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        enfFiles = eclipse.getEnfTrials(DATA_PATH)

        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))
        ff = acqGait.GetFirstFrame()-1
        lf = acqGait.GetLastFrame()-1

        print ff
        print lf

        # input arguments management
        if not DEBUG:
            print "----"
            print eclipse.findCalibrationFromEnfs(DATA_PATH,enfFiles)

            staticFilenameNoExt = args.static
            acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilenameNoExt+".c3d"))

            targetMarker = args.target
            trackingMarkers = args.trackingMarkers
            trackingMarkers.append(targetMarker)

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
        mod.addSegment("segment",0,enums.SegmentSide.Central,calibration_markers=[], tracking_markers = trackingMarkers)


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

        if not btkTools.isPointExist(acqGait,targetMarker):
            print "targer Marker not in the c3d"
            mod.getSegment("segment").m_tracking_markers.remove(targetMarker)

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
        print selectInitialFrame
        nexusTools.setTrajectoryFromArray(NEXUS,subject,targetMarker,val_final,firstFrame = ff)

        # btk methods
        # btkTools.smartAppendPoint(acqGait,labelMarkerToreconstruct,val_final)
        # btkTools.smartWriter(acqGait, str(DATA_PATH + reconstructFilenameLabelled))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='rigidLabelling')
    parser.add_argument('--static', type=str, help='filename of the static',required=True)
    parser.add_argument('--target', type=str, help='marker to recosntruct',required=True)
    parser.add_argument('--trackingMarkers', nargs='*', help='list of tracking markers',required=True)
    parser.add_argument('--begin', type=int, help='initial Frame')
    parser.add_argument('--last', type=int, help='last Frame')
    args = parser.parse_args()

    # ---- main script -----
    try:
        main(args)


    except Exception, errormsg:
        print "Error message: %s" % errormsg
        traceback.print_exc()
        print "Press return to exit.."
        raw_input()
