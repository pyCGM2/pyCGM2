# -*- coding: utf-8 -*-
import logging
import os
import shutil

import pyCGM2
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Lib.CGM import  cgm2_1
from pyCGM2.Utils import files
from pyCGM2.Utils import utils
from pyCGM2.qtm import qtmTools
from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from  pyCGM2.Lib import eventDetector,report
from pyCGM2.Report import normativeDatasets
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

from pyCGM2.Inspect import inspectFilters, inspectProcedures

from pyCGM2 import log;
log.setLogger(level = logging.INFO)



MARKERSETS={"Lower limb tracking markers": cgm.CGM1.LOWERLIMB_TRACKING_MARKERS,
            "Thorax tracking markers": cgm.CGM1.THORAX_TRACKING_MARKERS,
            "Upper limb tracking markers": cgm.CGM1.UPPERLIMB_TRACKING_MARKERS,
            "Calibration markers": ["LKNM","RKNM","LMED","RMED","LKAX","LKD1","LKD2","RKAX","RKD1","RKD2"]}

def command():
    parser = argparse.ArgumentParser(description='CGM21 workflow')
    parser.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")

    args = parser.parse_args()
    sessionFilename = args.sessionFile
    main(sessionFilename)


def main(sessionFilename,createPDFReport=True):
    logging.info("------------------------------------------------")
    logging.info("------------QTM - pyCGM2 Workflow---------------")
    logging.info("------------------------------------------------")

    sessionXML = files.readXml(os.getcwd()+"\\",sessionFilename)
    sessionDate = files.getFileCreationDate(os.getcwd()+"\\"+sessionFilename)


    #---------------------------------------------------------------------------
    #management of the Processed folder
    DATA_PATH = os.getcwd()+"\\"+"processed\\"
    files.createDir(DATA_PATH)

    staticMeasurement = qtmTools.findStatic(sessionXML)
    calibrateFilenameLabelled = qtmTools.getFilename(staticMeasurement)
    if not os.path.isfile(DATA_PATH+calibrateFilenameLabelled):
        shutil.copyfile(os.getcwd()+"\\"+calibrateFilenameLabelled,DATA_PATH+calibrateFilenameLabelled)
        logging.info("qualisys exported c3d file [%s] copied to processed folder"%(calibrateFilenameLabelled))

    dynamicMeasurements= qtmTools.findDynamic(sessionXML)
    for dynamicMeasurement in dynamicMeasurements:
        reconstructFilenameLabelled = qtmTools.getFilename(dynamicMeasurement)

        # marker
        order_marker = int(float(dynamicMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(dynamicMeasurement.Marker_lowpass_filter_frequency.text)

        if not os.path.isfile(DATA_PATH+reconstructFilenameLabelled):
            shutil.copyfile(os.getcwd()+"\\"+reconstructFilenameLabelled,DATA_PATH+reconstructFilenameLabelled)
            logging.info("qualisys exported c3d file [%s] copied to processed folder"%(reconstructFilenameLabelled))

            acq=btkTools.smartReader(str(DATA_PATH+reconstructFilenameLabelled))

            acq,zeniState = eventDetector.zeni(acq,
                                fc_lowPass_marker=fc_marker,
                                order_lowPass_marker=order_marker)

            if zeniState:
                btkTools.smartWriter(acq, str(DATA_PATH + reconstructFilenameLabelled))

                cmd = "Mokka.exe \"%s\""%(str(DATA_PATH + reconstructFilenameLabelled))
                os.system(cmd)

    # --------------------------GLOBAL SETTINGS ------------------------------------
    # global setting ( in user/AppData)

    if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_1-pyCGM2.settings"):
        settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")
    else:
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_1-pyCGM2.settings")
    # --------------------------MP ------------------------------------
    required_mp,optional_mp = qtmTools.SubjectMp(sessionXML)

    # --Check MP
    inspectprocedure = inspectProcedures.AnthropometricDataQualityProcedure(required_mp)
    inspector = inspectFilters.QualityFilter(inspectprocedure)
    inspector.run()


    #  translators management
    translators = files.getTranslators(os.getcwd()+"\\","CGM2_1.translators")
    if not translators:  translators = settings["Translators"]


    # --------------------------MODEL CALIBRATION -----------------------
    logging.info("--------------------------MODEL CALIBRATION -----------------------")
    staticMeasurement = qtmTools.findStatic(sessionXML)
    calibrateFilenameLabelled = qtmTools.getFilename(staticMeasurement)

    logging.info("----- CALIBRATION-  static file [%s]--"%(calibrateFilenameLabelled))

    leftFlatFoot = utils.toBool(sessionXML.Left_foot_normalised_to_static_trial.text)
    rightFlatFoot = utils.toBool(sessionXML.Right_foot_normalised_to_static_trial.text)
    headFlat = utils.toBool(sessionXML.Head_normalised_to_static_trial.text)
    markerDiameter = float(sessionXML.Marker_diameter.text)*1000.0
    hjcMethod = settings["Calibration"]["HJC"]
    pointSuffix = None

    # Calibration checking
    # --------------------
    acqStatic = btkTools.smartReader(DATA_PATH+calibrateFilenameLabelled)
    for key in MARKERSETS.keys():
        logging.info("[pyCGM2] Checking of the %s"%(key))

        # presence
        ip_presence = inspectProcedures.MarkerPresenceQualityProcedure(acqStatic,
                                        markers = MARKERSETS[key])
        inspector = inspectFilters.QualityFilter(ip_presence)
        inspector.run()

        if ip_presence.markersIn !=[]:

            ip_gap = inspectProcedures.GapQualityProcedure(acqStatic,
                                         markers = ip_presence.markersIn)
            inspector = inspectFilters.QualityFilter(ip_gap)
            inspector.run()

            ip_swap = inspectProcedures.SwappingMarkerQualityProcedure(acqStatic,
                                                markers = ip_presence.markersIn)
            inspector = inspectFilters.QualityFilter(ip_swap)
            inspector.run()

            ip_pos = inspectProcedures.MarkerPositionQualityProcedure(acqStatic,
                                         markers = ip_presence.markersIn)
            inspector = inspectFilters.QualityFilter(ip_pos)

    # Calibration operation
    # --------------------
    logging.info("[pyCGM2] --- calibration operation ---")
    model,acqStatic = cgm2_1.calibrate(DATA_PATH,
        calibrateFilenameLabelled,
        translators,
        required_mp,optional_mp,
        leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
        hjcMethod,
        pointSuffix)


    logging.info("----- CALIBRATION-  static file [%s]-----> DONE"%(calibrateFilenameLabelled))

    # --------------------------MODEL FITTING ----------------------------------
    logging.info("--------------------------MODEL FITTING ----------------------------------")
    dynamicMeasurements= qtmTools.findDynamic(sessionXML)

    modelledC3ds = list()
    eventInspectorStates = list()
    for dynamicMeasurement in dynamicMeasurements:

        reconstructFilenameLabelled = qtmTools.getFilename(dynamicMeasurement)

        logging.info("----Processing of [%s]-----"%(reconstructFilenameLabelled))
        mfpa = qtmTools.getForcePlateAssigment(dynamicMeasurement)
        momentProjection_text = sessionXML.Moment_Projection.text
        if momentProjection_text == "Default":
            momentProjection_text = settings["Fitting"]["Moment Projection"]
        if momentProjection_text == "Distal":
            momentProjection = enums.MomentProjection.Distal
        elif momentProjection_text == "Proximal":
            momentProjection =   enums.MomentProjection.Proximal
        elif momentProjection_text == "Global":
            momentProjection =   enums.MomentProjection.Global
        elif momentProjection_text == "JCS":
            momentProjection =  enums.MomentProjection.JCS



        acq = btkTools.smartReader(DATA_PATH+reconstructFilenameLabelled)

        # Fitting checking
        # --------------------
        for key in MARKERSETS.keys():
            if key != "Calibration markers":

                logging.info("[pyCGM2] Checking of the %s"%(key))
                # presence
                ip_presence = inspectProcedures.MarkerPresenceQualityProcedure(acq,
                                                markers = MARKERSETS[key])
                inspector = inspectFilters.QualityFilter(ip_presence)
                inspector.run()

                if ip_presence.markersIn !=[]:

                    ip_gap = inspectProcedures.GapQualityProcedure(acq,
                                                 markers = ip_presence.markersIn)
                    inspector = inspectFilters.QualityFilter(ip_gap)
                    inspector.run()

                    ip_swap = inspectProcedures.SwappingMarkerQualityProcedure(acq,
                                                        markers = ip_presence.markersIn)
                    inspector = inspectFilters.QualityFilter(ip_swap)
                    inspector.run()

                    ip_pos = inspectProcedures.MarkerPositionQualityProcedure(acq,
                                                 markers = ip_presence.markersIn)
                    inspector = inspectFilters.QualityFilter(ip_pos)


        # filtering
        # -----------------------

        # marker
        order_marker = int(float(dynamicMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(dynamicMeasurement.Marker_lowpass_filter_frequency.text)

        # force plate
        order_fp = int(float(dynamicMeasurement.Forceplate_lowpass_filter_order.text))
        fc_fp = float(dynamicMeasurement.Forceplate_lowpass_filter_frequency.text)


        # event checking
        # -----------------------
        inspectprocedureEvents = inspectProcedures.GaitEventQualityProcedure(acq)
        inspector = inspectFilters.QualityFilter(inspectprocedureEvents)
        inspector.run()
        eventInspectorStates.append(inspectprocedureEvents.state)


        # fitting operation
        # -----------------------
        logging.info("[pyCGM2] --- Fitting operation ---")
        acqGait = cgm2_1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,
            markerDiameter,
            pointSuffix,
            mfpa,momentProjection,
            fc_lowPass_marker=fc_marker,
            order_lowPass_marker=order_marker,
            fc_lowPass_forcePlate = fc_fp,
            order_lowPass_forcePlate = order_fp)

        outFilename = reconstructFilenameLabelled
        btkTools.smartWriter(acqGait, str(DATA_PATH + outFilename))
        modelledC3ds.append(outFilename)

        logging.info("----Processing of [%s]-----> DONE"%(reconstructFilenameLabelled))


    # --------------------------GAIT PROCESSING -----------------------
    if not all(eventInspectorStates):
        raise Exception ("[pyCGM2] Impossible to run Gait processing. Badly gait event detection. check the log file")
    logging.info("---------------------GAIT PROCESSING -----------------------")


    if createPDFReport:
        nds = normativeDatasets.NormativeData("Schwartz2008","Free")

        types = qtmTools.detectMeasurementType(sessionXML)
        for type in types:

            modelledTrials = list()
            for dynamicMeasurement in dynamicMeasurements:
                if  qtmTools.isType(dynamicMeasurement,type):
                    filename = qtmTools.getFilename(dynamicMeasurement)
                    modelledTrials.append(filename)

            report.pdfGaitReport(DATA_PATH,model,modelledTrials, nds,pointSuffix, title = type)
            logging.info("----- Gait Processing -----> DONE")
