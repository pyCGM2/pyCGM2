# -*- coding: utf-8 -*-

import warnings
from pyCGM2.Report import normativeDatasets
from pyCGM2.Lib import eventDetector
from pyCGM2.Lib import report
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
from pyCGM2.QTM import qtmTools
from pyCGM2.Utils import utils
from pyCGM2.Utils import files
from pyCGM2.Lib.CGM import cgm1
import shutil
import os
from pyCGM2.Anomaly import anomalyFilters
from pyCGM2.Anomaly import anomalyDetectionProcedures
import argparse
import pyCGM2
LOGGER = pyCGM2.LOGGER

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args=None):

    LOGFILE = "pyCGM2-QTM-CGM2-Processing.log"
    LOGGER.set_file_handler(LOGFILE)

    if args is None:
        parser = argparse.ArgumentParser(description='QTM processing')
        parser.add_argument('--sessionFile', type=str,
                        help='setting xml file from qtm', default="session.xml")
        args = parser.parse_args()
        sessionFilename = args.sessionFile
    else:
        sessionFilename="session.xml"
    
    detectAnomaly = False


    LOGGER.logger.info("------------QTM - pyCGM2 CGM Processing---------------")

    sessionXML = files.readXml(os.getcwd()+"\\", sessionFilename)
    sessionDate = files.getFileCreationDate(os.getcwd()+"\\"+sessionFilename)
    CGM2_Model = sessionXML.Subsession.CGM2_Model.text


    LOGGER.logger.info(f"----> {CGM2_Model} <------")
    LOGGER.logger.info(f"--------------------------")

    checkEventsInMokka = bool(sessionXML.Subsession.Check_Events_In_Mokka.text)
    createPDFReport = bool(sessionXML.Subsession.Create_PDF_report.text)
    anomalyException = bool(sessionXML.Subsession.Anomaly_Exception.text)


    #---------------------------------------------------------------------------
    #management of the Processed foldercd
    DATA_PATH = os.getcwd()+"\\"

    staticMeasurement = qtmTools.findStatic(sessionXML)
    calibrateFilenameLabelled = qtmTools.getFilename(staticMeasurement)

    dynamicMeasurements = qtmTools.findDynamic(sessionXML)
    for dynamicMeasurement in dynamicMeasurements:
        reconstructFilenameLabelled = qtmTools.getFilename(dynamicMeasurement)

        acq = btkTools.smartReader(
            str(DATA_PATH+reconstructFilenameLabelled))
        
        lfs =  btkTools.smartGetEvents (acq,"Left Foot Strike","")
        lfo =  btkTools.smartGetEvents (acq,"Left Foot Off","")
        rfs =  btkTools.smartGetEvents (acq,"Right Foot Strike","")
        rfo =  btkTools.smartGetEvents (acq,"Right Foot Off","")

        acq.ClearEvents()

        if lfs !=[]:
            for it in lfs:
                btkTools.smartCreateEvent(acq, "Foot Strike", "Left", it, type="Automatic", subject="", desc="",id=1)
        if lfo !=[]:
            for it in lfo:
                btkTools.smartCreateEvent(acq, "Foot Off", "Left", it, type="Manual", subject="", desc="",id=2)
        if rfs !=[]:
            for it in rfs:
                btkTools.smartCreateEvent(acq, "Foot Strike", "Right", it, type="Manual", subject="", desc="",id=1)
        if rfo !=[]:
            for it in rfo:
                btkTools.smartCreateEvent(acq, "Foot Off", "Right", it, type="Manual", subject="", desc="",id=2)
            
        btkTools.smartWriter(
                acq, str(DATA_PATH + reconstructFilenameLabelled))
        
    
    pointSuffix = None
    # --------------------------GAIT PROCESSING -----------------------


    if createPDFReport:
        nds = normativeDatasets.NormativeData("Schwartz2008", "Free")
        types = qtmTools.detectMeasurementType(sessionXML)
        for type in types:
            modelledTrials = []
            for dynamicMeasurement in dynamicMeasurements:
                if qtmTools.isType(dynamicMeasurement, type):
                    filename = qtmTools.getFilename(dynamicMeasurement)
                    # event checking
                    # -----------------------
                    acq = btkTools.smartReader(DATA_PATH+filename)
                    geap = anomalyDetectionProcedures.GaitEventAnomalyProcedure()
                    adf = anomalyFilters.AnomalyDetectionFilter(
                        acq, filename, geap)
                    anomaly_events = adf.run()
                    if anomaly_events["ErrorState"]:
                        detectAnomaly = True
                        LOGGER.logger.warning(
                            "file [%s] not used for generating the gait report. bad gait event detected" % (filename))
                    else:
                        modelledTrials.append(filename)
            try:
                report.pdfGaitReport(
                    DATA_PATH, modelledTrials, nds, pointSuffix, title=type)
                LOGGER.logger.info("Generation of Gait report complete")
            except:
                LOGGER.logger.error("Generation of Gait report failed")

    LOGGER.logger.info(
        "-------------------------------------------------------")
    if detectAnomaly:
        LOGGER.logger.error(
            "Anomalies has been detected - Find Error messages, then check warning message in the log file")
    else:
        LOGGER.logger.info("workflow return with NO detected anomalies")
    
    os.startfile( os.getcwd()+"\\"+LOGFILE)


if __name__ == '__main__':
    main(args=None) 