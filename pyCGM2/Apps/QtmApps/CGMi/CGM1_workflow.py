# -*- coding: utf-8 -*-
import logging
import os
import shutil

import pyCGM2
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Lib.CGM import  cgm1
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

from pyCGM2.Anomaly import AnomalyFilter, AnomalyDetectionProcedure #AnomalyCorrectionProcedure


from pyCGM2 import log; log.setLogger(level = logging.INFO)


# from qtmWebGaitReport import qtmFilters

MARKERSETS={"Lower limb tracking markers": cgm.CGM1.LOWERLIMB_TRACKING_MARKERS,
            "Thorax tracking markers": cgm.CGM1.THORAX_TRACKING_MARKERS,
            "Upper limb tracking markers": cgm.CGM1.UPPERLIMB_TRACKING_MARKERS,
            "Calibration markers": ["LMED","RMED","LKAX","LKD1","LKD2","RKAX","RKD1","RKD2"]}

def command():
    parser = argparse.ArgumentParser(description='CGM1 workflow')
    parser.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")

    args = parser.parse_args()
    sessionFilename = args.sessionFile
    main(sessionFilename)



def main(sessionFilename,createPDFReport=True,checkEventsInMokka=True):
    logging.info("------------------------------------------------")
    logging.info("------------QTM - pyCGM2 Workflow---------------")
    logging.info("------------------------------------------------")

    sessionXML = files.readXml(os.getcwd()+"\\",sessionFilename)
    sessionDate = files.getFileCreationDate(os.getcwd()+"\\"+sessionFilename)

    #---------------------------------------------------------------------------
    #management of the Processed foldercd
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
                if checkEventsInMokka:
                    cmd = "Mokka.exe \"%s\""%(str(DATA_PATH + reconstructFilenameLabelled))
                    os.system(cmd)


    # --------------------------GLOBAL SETTINGS ------------------------------------
    # global setting ( in user/AppData)

    if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM1-pyCGM2.settings"):
        settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")
    else:
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM1-pyCGM2.settings")

    # --------------------------MP ------------------------------------
    required_mp,optional_mp = qtmTools.SubjectMp(sessionXML)

    # --Check MP
    adap = AnomalyDetectionProcedure.AnthropoDataAnomalyProcedure( required_mp)
    adf = AnomalyFilter.AnomalyDetectionFilter(None,None,adap)
    anomaly = adf.run()



    #  translators management
    translators = files.getTranslators(os.getcwd()+"\\","CGM1.translators")
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
    pointSuffix = None

    # Calibration checking
    # --------------------
    acqStatic = btkTools.smartReader(DATA_PATH+calibrateFilenameLabelled)


    for key in MARKERSETS.keys():
        logging.info("[pyCGM2] Checking of the %s"%(key))
        # madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( MARKERSETS[key], plot=False, window=10,threshold = 3)
        mpdp = AnomalyDetectionProcedure.MarkerPresenceDetectionProcedure( MARKERSETS[key])
        adf = AnomalyFilter.AnomalyDetectionFilter(acqStatic,calibrateFilenameLabelled,mpdp)
        anomaly = adf.run()
        foundMarkers = anomaly["Output"]


        if foundMarkers != []:
            madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( foundMarkers, plot=False, window=10,threshold = 3)
            adf = AnomalyFilter.AnomalyDetectionFilter(acqStatic,calibrateFilenameLabelled,madp)
            anomaly = adf.run()
            anomalyIndexes = anomaly["Output"]



    # Calibration operation
    # --------------------
    logging.info("[pyCGM2] --- calibration operation ---")
    model,acqStatic = cgm1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,
                  required_mp,optional_mp,
                  leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
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


        acq = btkTools.smartReader(DATA_PATH+reconstructFilenameLabelled)

        # Fitting checking
        # --------------------
        for key in MARKERSETS.keys():
            if key != "Calibration markers":
                # madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( MARKERSETS[key], plot=False, window=10,threshold = 3)
                mpdp = AnomalyDetectionProcedure.MarkerPresenceDetectionProcedure( MARKERSETS[key])
                adf = AnomalyFilter.AnomalyDetectionFilter(acq,calibrateFilenameLabelled,mpdp)
                anomaly = adf.run()
                foundMarkers = anomaly["Output"]


                if foundMarkers != []:
                    madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( foundMarkers, plot=False, window=10,threshold = 3)
                    adf = AnomalyFilter.AnomalyDetectionFilter(acq,calibrateFilenameLabelled,madp)
                    anomaly = adf.run()
                    anomalyIndexes = anomaly["Output"]

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
        geap = AnomalyDetectionProcedure.GaitEventAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,reconstructFilenameLabelled,geap)
        anomaly = adf.run()
        errorState = anomaly["ErrorState"]
        eventInspectorStates.append(errorState)

        # ForcePlateAnomalyProcedure
        fpap = AnomalyDetectionProcedure.ForcePlateAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,reconstructFilenameLabelled,fpap)
        anomaly = adf.run()

        # fitting operation
        # -----------------------
        logging.info("[pyCGM2] --- Fitting operation ---")
        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
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
    if True in eventInspectorStates:
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
