# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import warnings

import pyCGM2
from pyCGM2 import enums
from pyCGM2.Anomaly import anomalyDetectionProcedures, anomalyFilters
from pyCGM2.Lib import eventDetector, report
from pyCGM2.Lib.CGM import cgm1
from pyCGM2.QTM import qtmTools
from pyCGM2.Report import normativeDatasets
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files, utils

LOGGER = pyCGM2.LOGGER

warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args=None):


    if args is None:
        parser = argparse.ArgumentParser(description='QEvents')
        parser.add_argument('--sessionFile', type=str,
                        help='setting xml file from qtm', default="session.xml")
        args = parser.parse_args()
        sessionFilename = args.sessionFile
    else:
        sessionFilename = args.sessionFile
        sessionFolder = args.session_path
    
    LOGFILE = sessionFolder / "pyCGM2-QTM-CGM2-Events.log"
    LOGGER.setLevel("info")
    LOGGER.set_file_handler(LOGFILE)

    detectAnomaly = False
    qtmFilesWithEvents = args.qtm_files_with_events


    LOGGER.logger.info("------------QTM - pyCGM2 EVENTS---------------")

    DATA_PATH = str(sessionFolder) + "//"


    sessionXML = files.readXml(DATA_PATH, sessionFilename)
    
    dynamicMeasurements = qtmTools.findDynamic(sessionXML)
    for dynamicMeasurement in dynamicMeasurements:
        
        reconstructFilenameLabelled = qtmTools.getFilename(dynamicMeasurement)
        filenameNoExt = reconstructFilenameLabelled[:-4]

        if filenameNoExt in qtmFilesWithEvents:
            LOGGER.logger.info(f"Skipping {filenameNoExt} as it already has events")
            continue
        
        LOGGER.logger.info(f"---File : {reconstructFilenameLabelled}----")

        # marker
        order_marker = int(
            float(dynamicMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(
            dynamicMeasurement.Marker_lowpass_filter_frequency.text)


        acq = btkTools.smartReader(
            str(DATA_PATH+reconstructFilenameLabelled))

        eventNumber = acq.GetEvents().GetItemNumber()

        qlfs =  btkTools.smartGetEvents (acq,"Left Foot Strike","")
        qlfo =  btkTools.smartGetEvents (acq,"Left Foot Off","")
        qrfs =  btkTools.smartGetEvents (acq,"Right Foot Strike","")
        qrfo =  btkTools.smartGetEvents (acq,"Right Foot Off","")



        if not all([qlfs, qlfo, qrfs, qrfo]):
            detectAnomaly = True

            LOGGER.logger.error (f"file [{filenameNoExt}] has no events. Zeni kinematic-based gait event runs")

            acq, zeniState = eventDetector.zeni(acq,
                                                 fc_lowPass_marker=fc_marker,
                                                 order_lowPass_marker=order_marker)
            lfs =  btkTools.smartGetEvents (acq,"Foot Strike","Left")
            lfo =  btkTools.smartGetEvents (acq,"Foot Off","Left")
            rfs =  btkTools.smartGetEvents (acq,"Foot Strike","Right")
            rfo =  btkTools.smartGetEvents (acq,"Foot Off","Right")


            # checking
            geap = anomalyDetectionProcedures.GaitEventAnomalyProcedure()
            adf = anomalyFilters.AnomalyDetectionFilter(
                acq, reconstructFilenameLabelled, geap)
            anomaly_events = adf.run() #anomaly_events["ErrorState"]
            
            out = {"Events":{}}
            out["Events"] = {"LeftFootStrike":  [x / acq.GetPointFrequency() for x in lfs],
                   "LeftFootOff":[x / acq.GetPointFrequency() for x in lfo],
                   "RightFootStrike":[x / acq.GetPointFrequency() for x in rfs],
                   "RightFootOff":[x / acq.GetPointFrequency() for x in rfo]}
            # also save to c3d
            btkTools.smartWriter(acq, str(DATA_PATH + reconstructFilenameLabelled))
            files.saveJson( DATA_PATH, reconstructFilenameLabelled[:-4]+"-events.json", out)
            
            LOGGER.logger.warning (f"file [{filenameNoExt}]- Update your gait events in QTM and check them")     

        else:
            LOGGER.logger.error (f"file [{filenameNoExt}] already contains events")

            if qlfs !=[]:
                for it in qlfs:
                    btkTools.smartCreateEvent(acq, "Foot Strike", "Left", it, type="Automatic", subject="", desc="",id=1)
            if qlfo !=[]:
                for it in qlfo:
                    btkTools.smartCreateEvent(acq, "Foot Off", "Left", it, type="Manual", subject="", desc="",id=2)
            if qrfs !=[]:
                for it in qrfs:
                    btkTools.smartCreateEvent(acq, "Foot Strike", "Right", it, type="Manual", subject="", desc="",id=1)
            if qrfo !=[]:
                for it in qrfo:
                    btkTools.smartCreateEvent(acq, "Foot Off", "Right", it, type="Manual", subject="", desc="",id=2)

            # checking
            geap = anomalyDetectionProcedures.GaitEventAnomalyProcedure()
            adf = anomalyFilters.AnomalyDetectionFilter(
                acq, reconstructFilenameLabelled, geap)
            anomaly_events = adf.run()


            if anomaly_events["ErrorState"]:
                detectAnomaly = True
                LOGGER.logger.error(f"file [{filenameNoExt}] bad gait event detected -  check you QTM events")
        LOGGER.logger.info("----------------------------------------------")

    if detectAnomaly:
        LOGGER.logger.error("QTM Gait Events need to be updated or checked. see content of the log file")
        LOGGER.logger.error("Then regenerate c3d")
    else:
        
        LOGGER.logger.info("[QTM GAIT EVENT---> OK ]- Command *CGM2 processing* can be executed")

    # os.startfile( os.getcwd()+"\\"+LOGFILE)


            

