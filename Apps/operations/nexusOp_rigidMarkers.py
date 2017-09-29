# -*- coding: utf-8 -*-
import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import argparse
import copy

import pdb

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# openMA
#import ma.io
#import ma.body

#btk
#import btk


# pyCGM2 libraries
#...
from pyCGM2.Model.CGM2 import model, modelFilters
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools
from pyCGM2 import enums

if __name__ == "__main__":

    """
    goal: rigid gap filling. fill gap from 3 markers
    usage : --static="Kevin Cal 01" --target=RASI --trackingMarkers LASI LPSI RPSI --begin=3589 --last=3600

    .. warning:

        target marker must be with the vsk tree to be know as a trajectory

    """

    DEBUG = False

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='rigidLabelling')
    parser.add_argument('--static', type=str, help='filename of the static',required=True)
    parser.add_argument('--target', type=str, help='marker to recosntruct',required=True)
    parser.add_argument('--trackingMarkers', nargs='*', help='list of tracking markers',required=True)
    parser.add_argument('--begin', type=int, help='initial Frame')
    parser.add_argument('--last', type=int, help='last Frame')
    args = parser.parse_args()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH =pyCGM2.CONFIG.TEST_DATA_PATH +"operations\\miscellaneous\\rigid_labelling_pyCGM2\\"
            reconstructFilenameLabelledNoExt ="KevinAki-VSR1-2"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )


            acqStatic = btkTools.smartReader(str(DATA_PATH+"Kevin Cal 01.c3d"))
            targetMarker = "RASI"
            trackingMarkers = ["LASI","RPSI","LPSI"]
            selectInitialFrame = 3589
            selectLastFrame = 3600



        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))
        ff = acqGait.GetFirstFrame()
        lf = acqGait.GetLastFrame()


        # input arguments management
        if not DEBUG:
            staticFilenameNoExt = args.static
            acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilenameNoExt+".c3d"))

            targetMarker = args.target
            trackingMarkers = args.trackingMarkers
            trackingMarkers.append(targetMarker)

            if args.begin is None and args.last is None: # reconstrution on full frames
                selectInitialFrame = ff-ff
                selectLastFrame = lf-ff
            elif args.begin is not None and args.last is not None: # reconstrution from both selected begin and end frame
                selectInitialFrame = args.begin-ff
                selectLastFrame = args.last-ff
            elif args.begin is not None and args.last is None: # reconstrution from  selected begin and last frame
                selectInitialFrame = args.begin-ff
                selectLastFrame = lf-ff
            elif args.begin is None and args.last is not None: # reconstrution from  first frame and to selected last frame
                selectInitialFrame = ff-ff
                selectLastFrame = args.last-ff


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

        modMotion=modelFilters.ModelMotionFilter(gcp,acqGait,mod,enums.motionMethod.Sodervisk)
        modMotion.compute()

        #populate values
        valReconstruct=mod.getSegment('segment').getReferential('TF').getNodeTrajectory(targetMarker)
        val0 = acqGait.GetPoint(targetMarker).GetValues()
        val_final = copy.deepcopy(val0)
        val_final[selectInitialFrame:selectLastFrame+1,:] = valReconstruct[selectInitialFrame:selectLastFrame+1,:]

        # nexus display
        nexusTools.setTrajectoryFromArray(NEXUS,subject,targetMarker,val_final)

        # btk methods
        # btkTools.smartAppendPoint(acqGait,labelMarkerToreconstruct,val_final)
        # btkTools.smartWriter(acqGait, str(DATA_PATH + reconstructFilenameLabelled))
