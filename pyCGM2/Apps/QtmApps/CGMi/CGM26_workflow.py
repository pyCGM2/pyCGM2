# -*- coding: utf-8 -*-


import pyCGM2; LOGGER = pyCGM2.LOGGER
import os
import shutil

import pyCGM2
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.CGM2 import cgm2
from pyCGM2.Lib.CGM import  cgm2_5
from pyCGM2.Lib.CGM import  kneeCalibration
from pyCGM2.Utils import files
from pyCGM2.Utils import utils
from pyCGM2.QTM import qtmTools
from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from pyCGM2.Lib import eventDetector
from pyCGM2.Lib import report
from pyCGM2.Report import normativeDatasets
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

from pyCGM2.Anomaly import anomalyFilters
from pyCGM2.Anomaly import anomalyDetectionProcedures


MODEL = "CGM2.6"


def command():

    parser = argparse.ArgumentParser(description='CGM26 workflow')
    parser.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')

    try:
        args = parser.parse_args()
        sessionFilename = args.sessionFile
        main(sessionFilename, anomalyException=args.anomalyException)
    except:
        return parser


def main(sessionFilename,createPDFReport=True,checkEventsInMokka=True,anomalyException=False):

    detectAnomaly = False
    LOGGER.set_file_handler("pyCGM2-QTM-Workflow.log")


    LOGGER.logger.info("------------QTM - pyCGM2 Workflow---------------")

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
        LOGGER.logger.info("qualisys exported c3d file [%s] copied to processed folder"%(calibrateFilenameLabelled))


    leftKneeFuncMeasurement = qtmTools.findKneeCalibration(sessionXML,"Left")
    rightKneeFuncMeasurement = qtmTools.findKneeCalibration(sessionXML,"Right")

    if leftKneeFuncMeasurement is not None:
        shutil.copyfile(os.getcwd()+"\\"+qtmTools.getFilename(leftKneeFuncMeasurement),
                        DATA_PATH+qtmTools.getFilename(leftKneeFuncMeasurement))
    if rightKneeFuncMeasurement is not None:
        shutil.copyfile(os.getcwd()+"\\"+qtmTools.getFilename(rightKneeFuncMeasurement),
                        DATA_PATH+qtmTools.getFilename(rightKneeFuncMeasurement))



    dynamicMeasurements= qtmTools.findDynamic(sessionXML)
    for dynamicMeasurement in dynamicMeasurements:
        reconstructFilenameLabelled = qtmTools.getFilename(dynamicMeasurement)

        # marker
        order_marker = int(float(dynamicMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(dynamicMeasurement.Marker_lowpass_filter_frequency.text)

        if not os.path.isfile(DATA_PATH+reconstructFilenameLabelled):
            shutil.copyfile(os.getcwd()+"\\"+reconstructFilenameLabelled,DATA_PATH+reconstructFilenameLabelled)
            LOGGER.logger.info("qualisys exported c3d file [%s] copied to processed folder"%(reconstructFilenameLabelled))

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
    settings = files.loadModelSettings(DATA_PATH,"CGM2_5-pyCGM2.settings")

    # --------------------------MP ------------------------------------
    required_mp,optional_mp = qtmTools.SubjectMp(sessionXML)


    #  translators management
    translators = files.getTranslators(os.getcwd()+"\\","CGM2_5.translators")
    if not translators:  translators = settings["Translators"]

    #  ikweight
    ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_5.ikw")
    if not ikWeight: ikWeight = settings["Fitting"]["Weight"]



    # --------------------------MODEL CALIBRATION -----------------------
    LOGGER.logger.info("--------------------------MODEL CALIBRATION -----------------------")
    staticMeasurement = qtmTools.findStatic(sessionXML)
    calibrateFilenameLabelled = qtmTools.getFilename(staticMeasurement)

    LOGGER.logger.info("----- CALIBRATION-  static file [%s]--"%(calibrateFilenameLabelled))

    leftFlatFoot = utils.toBool(sessionXML.Left_foot_normalised_to_static_trial.text)
    rightFlatFoot = utils.toBool(sessionXML.Right_foot_normalised_to_static_trial.text)
    headFlat = utils.toBool(sessionXML.Head_normalised_to_static_trial.text)
    markerDiameter = float(sessionXML.Marker_diameter.text)*1000.0
    hjcMethod = settings["Calibration"]["HJC"]
    pointSuffix = None

    # Calibration checking
    # --------------------
    acqStatic = btkTools.smartReader(DATA_PATH+calibrateFilenameLabelled)

    # Calibration operation
    # --------------------
    model,acqStatic,detectAnomaly = cgm2_5.calibrate(DATA_PATH,
        calibrateFilenameLabelled,
        translators,settings,
        required_mp,optional_mp,
        False,
        leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
        hjcMethod,
        pointSuffix,
        anomalyException=anomalyException)


    LOGGER.logger.info("----- CALIBRATION-  static file [%s]-----> DONE"%(calibrateFilenameLabelled))


    # --------------------------Knee Calibration ----------------------------------
    LOGGER.logger.info("--------------------------Knee Calibration ----------------------------------")


    if leftKneeFuncMeasurement is not None:
        reconstructFilenameLabelled = qtmTools.getFilename(leftKneeFuncMeasurement)

        order_marker = int(float(leftKneeFuncMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(leftKneeFuncMeasurement.Marker_lowpass_filter_frequency.text)

        if qtmTools.getKneeFunctionCalibMethod(leftKneeFuncMeasurement) =="Calibration2Dof":
            model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                                  DATA_PATH,reconstructFilenameLabelled,translators,
                                  "Left",None,None,None,
                                  fc_lowPass_marker = order_marker,
                                  order_lowPass_marker = fc_marker)

        if qtmTools.getKneeFunctionCalibMethod(leftKneeFuncMeasurement) =="SARA":
            model,acqFunc,side = kneeCalibration.Sara(model,
                                  DATA_PATH,reconstructFilenameLabelled,translators,
                                  "Left",None,None,None,
                                  fc_lowPass_marker = order_marker,
                                  order_lowPass_marker = fc_marker)

        LOGGER.logger.info("Left Knee functional Calibration ----> Done")


    if rightKneeFuncMeasurement is not None:
        reconstructFilenameLabelled = qtmTools.getFilename(rightKneeFuncMeasurement)

        order_marker = int(float(rightKneeFuncMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(rightKneeFuncMeasurement.Marker_lowpass_filter_frequency.text)


        if qtmTools.getKneeFunctionCalibMethod(rightKneeFuncMeasurement) =="Calibration2Dof":
            model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                                  DATA_PATH,reconstructFilenameLabelled,translators,
                                  "Right",None,None,None,
                                  fc_lowPass_marker = order_marker,
                                  order_lowPass_marker = fc_marker)

        if qtmTools.getKneeFunctionCalibMethod(rightKneeFuncMeasurement) =="SARA":
            model,acqFunc,side = kneeCalibration.Sara(model,
                                  DATA_PATH,reconstructFilenameLabelled,translators,
                                  "Right",None,None,None,
                                  fc_lowPass_marker = order_marker,
                                  order_lowPass_marker = fc_marker)


        LOGGER.logger.info("Right Knee functional Calibration ----> Done")


    # --------------------------MODEL FITTING ----------------------------------
    LOGGER.logger.info("--------------------------MODEL FITTING ----------------------------------")
    dynamicMeasurements= qtmTools.findDynamic(sessionXML)

    ik_flag = True

    modelledC3ds = list()
    eventInspectorStates = list()
    for dynamicMeasurement in dynamicMeasurements:

        reconstructFilenameLabelled = qtmTools.getFilename(dynamicMeasurement)

        LOGGER.logger.info("----Processing of [%s]-----"%(reconstructFilenameLabelled))
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



        # filtering
        # -----------------------

        # marker
        order_marker = int(float(dynamicMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(dynamicMeasurement.Marker_lowpass_filter_frequency.text)

        # force plate
        order_fp = int(float(dynamicMeasurement.Forceplate_lowpass_filter_order.text))
        fc_fp = float(dynamicMeasurement.Forceplate_lowpass_filter_frequency.text)

        # ik accuracy
        ikAccuracy = float(dynamicMeasurement.IkAccuracy.text)


        if dynamicMeasurement.First_frame_to_process.text != "":
            vff = int(dynamicMeasurement.First_frame_to_process.text)
        else: vff = None

        if dynamicMeasurement.Last_frame_to_process.text != "":
            vlf = int(dynamicMeasurement.Last_frame_to_process.text)
        else: vlf = None


        # fitting operation
        # -----------------------
        LOGGER.logger.info("[pyCGM2] --- Fitting operation ---")
        acqGait,detectAnomaly = cgm2_5.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,settings,
            ik_flag,markerDiameter,
            pointSuffix,
            mfpa,
            momentProjection,
            fc_lowPass_marker=fc_marker,
            order_lowPass_marker=order_marker,
            fc_lowPass_forcePlate = fc_fp,
            order_lowPass_forcePlate = order_fp,
            anomalyException=anomalyException,
            ikAccuracy = ikAccuracy,
            frameInit= vff, frameEnd= vlf )


        outFilename = reconstructFilenameLabelled
        btkTools.smartWriter(acqGait, str(DATA_PATH + outFilename))
        modelledC3ds.append(outFilename)

        LOGGER.logger.info("----Processing of [%s]-----> DONE"%(reconstructFilenameLabelled))



    LOGGER.logger.info("---------------------GAIT PROCESSING -----------------------")
    if createPDFReport:
        nds = normativeDatasets.NormativeData("Schwartz2008","Free")
        types = qtmTools.detectMeasurementType(sessionXML)
        for type in types:
            modelledTrials = list()
            for dynamicMeasurement in dynamicMeasurements:
                if  qtmTools.isType(dynamicMeasurement,type):
                    filename = qtmTools.getFilename(dynamicMeasurement)
                    # event checking
                    # -----------------------
                    acq = btkTools.smartReader(DATA_PATH+filename)
                    geap = anomalyDetectionProcedures.GaitEventAnomalyProcedure()
                    adf = anomalyFilters.AnomalyDetectionFilter(acq,filename,geap)
                    anomaly_events = adf.run()
                    if anomaly_events["ErrorState"]:
                        detectAnomaly = True
                        LOGGER.logger.warning("file [%s] not used for generating the gait report. bad gait event detected"%(filename))
                    else:
                        modelledTrials.append(filename)
            try:
                report.pdfGaitReport(DATA_PATH,model,modelledTrials, nds,pointSuffix, title = type)
                LOGGER.logger.error("Generation of Gait report complete")
            except:
                LOGGER.logger.error("Generation of Gait report failed")


    LOGGER.logger.info("-------------------------------------------------------")
    if detectAnomaly:
        LOGGER.logger.error("Anomalies has been detected - Find Error messages, then check warning message in the log file")
    else:
        LOGGER.logger.info("workflow return with no detected anomalies")
