# -*- coding: utf-8 -*-

import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse


# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools

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

        DATA_PATH, filename = nexusTools.getTrialName(NEXUS)

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2.log")
        LOGGER.logger.info( " file: "+ filename)


        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS) #checkActivatedSubject(NEXUS,subjects)
        Parameters = NEXUS.GetSubjectParamNames(subject)


        # --------------------------PULL ------------------------------------
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filename,subject)
        acq = nacf.build()


        # --------------------------process ------------------------------------
        # Work with BTK Here

        # example 1 ! mid point
        values = (acq.GetPoint("LTIAP").GetValues() + acq.GetPoint("LTIB").GetValues()) /2.0
        btkTools.smartAppendPoint(acq,"LTIAD",values, PointType="Marker",desc="",residuals = None)

        # example 2 place point on an axis
        from pyCGM2.Model import modelDecorator
        values2  = modelDecorator.midPoint(acq,"RMED","RANK",offset=68)
        btkTools.smartAppendPoint(acq,"RANK",values2, PointType="Marker",desc="",residuals = None)

        # example 3 - rigid filling
        from pyCGM2 import enums
        from pyCGM2.Model import model
        from pyCGM2.Model import modelFilters

        acqStatic = btkTools.smartReader(str(DATA_PATH+"Trial07.c3d"))
        targetMarker = "RANK"
        trackingMarkers = ["RTIAP","RTIAD","RTIB"]
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

        modMotion=modelFilters.ModelMotionFilter(gcp,acq,mod,enums.motionMethod.Sodervisk)
        modMotion.compute()


        #populate values
        valReconstruct=mod.getSegment('segment').getReferential('TF').getNodeTrajectory(targetMarker)

        btkTools.smartAppendPoint(acq,"RANK",valReconstruct, PointType="Marker",desc="",residuals = None)



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
