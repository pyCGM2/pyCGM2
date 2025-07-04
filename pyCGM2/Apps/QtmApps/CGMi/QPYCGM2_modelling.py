# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import warnings

import pyCGM2
from pyCGM2 import enums
from pyCGM2.Lib.CGM import cgm1, cgm1_1, cgm2_1, cgm2_4, cgm2_5, kneeCalibration
from pyCGM2.Lib.CGM.musculoskeletal import cgm2_2, cgm2_3
from pyCGM2.QTM import qtmTools
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files, utils

LOGGER = pyCGM2.LOGGER

warnings.simplefilter(action='ignore', category=FutureWarning)




def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='QTM CGM Modelling')
        parser.add_argument('--sessionFile', type=str,
                        help='setting xml file from qtm', default="session.xml")
        args = parser.parse_args()
        sessionFilename = args.sessionFile
    else:
        sessionFilename = args.sessionFile
        sessionFolder = args.session_path
    
    LOGFILE = sessionFolder / "pyCGM2-QTM-CGM2-Modelling.log"
    LOGGER.set_file_handler(LOGFILE)

    if args.debug:
        LOGGER.setLevel("debug")

    detectAnomaly = False

    

    LOGGER.logger.info("------------QTM - pyCGM2 Modelling---------------")

    
    #---------------------------------------------------------------------------
    DATA_PATH = str(sessionFolder)+"\\"
    sessionXML = files.readXml(DATA_PATH, sessionFilename)
    CGM2_Model = sessionXML.Subsession.CGM2_Model.text
    if "CGM2.6" in CGM2_Model:
        CGM2_Model = "CGM2.6-Knee Calibration" # change name to the expected one here

    LOGGER.logger.info(f"----> {CGM2_Model} <------")
    LOGGER.logger.info(f"--------------------------")

    anomalyException = False # bool(sessionXML.Subsession.Anomaly_Exception.text)

    staticMeasurement = qtmTools.findStatic(sessionXML)
    calibrateFilenameLabelled = qtmTools.getFilename(staticMeasurement)

    if CGM2_Model != "CGM2.6-Knee Calibration":
        if qtmTools.findKneeCalibration(sessionXML, "Left") is not None or qtmTools.findKneeCalibration(sessionXML, "Right") is not None:
            LOGGER.logger.info(
                " the %s not accept functional knee calibration !!" % (CGM2_Model))

   
        
    
    # --------------------------GLOBAL SETTINGS ------------------------------------
    # global setting ( in user/AppData)
    if CGM2_Model == "CGM1.0":
        settings = files.loadModelSettings(DATA_PATH, "CGM1-pyCGM2.settings")
    elif CGM2_Model == "CGM1.1":
        settings = files.loadModelSettings(DATA_PATH, "CGM1_1-pyCGM2.settings")
    elif CGM2_Model == "CGM2.1-HJC":
        settings = files.loadModelSettings(DATA_PATH, "CGM2_1-pyCGM2.settings")
    elif CGM2_Model == "CGM2.2-IK":
        settings = files.loadModelSettings(DATA_PATH, "CGM2_2-pyCGM2.settings")
    elif CGM2_Model == "CGM2.3-skinClusters":
        settings = files.loadModelSettings(DATA_PATH, "CGM2_3-pyCGM2.settings")
    elif CGM2_Model == "CGM2.4-ForeFoot":
        settings = files.loadModelSettings(DATA_PATH, "CGM2_4-pyCGM2.settings")
    elif CGM2_Model == "CGM2.5-UpperLimb":
        settings = files.loadModelSettings(DATA_PATH, "CGM2_5-pyCGM2.settings")
    elif CGM2_Model == "CGM2.6-Knee Calibration":
        settings = files.loadModelSettings(DATA_PATH, "CGM2_5-pyCGM2.settings")


    # --------------------------MP ------------------------------------
    required_mp, optional_mp = qtmTools.SubjectMp(sessionXML)

    #  translators management
    if CGM2_Model == "CGM1.0":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM1.translators")
    elif CGM2_Model == "CGM1.1":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM1_1.translators")
    elif CGM2_Model == "CGM2.1-HJC":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM2_1.translators")
    elif CGM2_Model == "CGM2.2-IK":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM2_2.translators")
    elif CGM2_Model == "CGM2.3-skinClusters":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM2_3.translators")
    elif CGM2_Model == "CGM2.4-ForeFoot":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM2_4.translators")
    elif CGM2_Model == "CGM2.5-UpperLimb":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM2_5.translators")
    elif CGM2_Model == "CGM2.6-Knee Calibration":
        translators = files.getTranslators(os.getcwd()+"\\", "CGM2_5.translators")
    if not translators:
        translators = settings["Translators"]

    if CGM2_Model == "CGM2.2-IK":
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_2.ikw")
    elif CGM2_Model == "CGM2.3-skinClusters":
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_3.ikw")
    elif CGM2_Model == "CGM2.4-ForeFoot":
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_4.ikw")
    elif CGM2_Model == "CGM2.5-UpperLimb":
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_5.ikw")
    elif CGM2_Model == "CGM2.6-Knee Calibration":
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_5.ikw")
    else:
        ikWeight = True
    if not ikWeight: ikWeight = settings["Fitting"]["Weight"]

    # --------------------------MODEL CALIBRATION -----------------------
    LOGGER.logger.info(
        "--------------------------MODEL CALIBRATION -----------------------")
    staticMeasurement = qtmTools.findStatic(sessionXML)
    calibrateFilenameLabelled = qtmTools.getFilename(staticMeasurement)

    LOGGER.logger.info(
        "----- CALIBRATION-  static file [%s]--" % (calibrateFilenameLabelled))

    leftFlatFoot = utils.toBool(
        sessionXML.Left_foot_normalised_to_static_trial.text)
    rightFlatFoot = utils.toBool(
        sessionXML.Right_foot_normalised_to_static_trial.text)
    headFlat = utils.toBool(sessionXML.Head_normalised_to_static_trial.text)
    markerDiameter = float(sessionXML.Marker_diameter.text)*1000.0
    pointSuffix = None




    acqStatic = btkTools.smartReader(DATA_PATH+calibrateFilenameLabelled)

    # Calibration operation
    # --------------------
    if CGM2_Model == "CGM1.0":
    
        model, acqStatic, detectAnomaly = cgm1.calibrate(DATA_PATH, calibrateFilenameLabelled, translators,
                                                        required_mp, optional_mp,
                                                        leftFlatFoot, rightFlatFoot, headFlat, markerDiameter,
                                                        pointSuffix,
                                                        anomalyException=anomalyException)
    elif CGM2_Model == "CGM1.1":
    
        model,acqStatic,detectAnomaly = cgm1_1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,
                  required_mp,optional_mp,
                  leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
                  pointSuffix,anomalyException=anomalyException)
    
    elif CGM2_Model == "CGM2.1-HJC":
        hjcMethod = settings["Calibration"]["HJC"]

        model,acqStatic,detectAnomaly = cgm2_1.calibrate(DATA_PATH,
        calibrateFilenameLabelled,
        translators,
        required_mp,optional_mp,
        leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
        hjcMethod,
        pointSuffix,
        anomalyException=anomalyException)

    elif CGM2_Model == "CGM2.2-IK":
        hjcMethod = settings["Calibration"]["HJC"]

        model,acqStatic,detectAnomaly = cgm2_2.calibrate(DATA_PATH,
                calibrateFilenameLabelled,
                translators,settings,
                required_mp,optional_mp,
                False,
                leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
                hjcMethod,
                pointSuffix,
                anomalyException=anomalyException)
        

    elif CGM2_Model == "CGM2.3-skinClusters":
        hjcMethod = settings["Calibration"]["HJC"]

        model,acqStatic,detectAnomaly = cgm2_3.calibrate(DATA_PATH,
            calibrateFilenameLabelled,
            translators,settings,
            required_mp,optional_mp,
            False,
            leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
            hjcMethod,
            pointSuffix,
            anomalyException=anomalyException)


    elif CGM2_Model == "CGM2.4-ForeFoot":
        hjcMethod = settings["Calibration"]["HJC"]

        model,acqStatic,detectAnomaly = cgm2_4.calibrate(DATA_PATH,
            calibrateFilenameLabelled,
            translators,settings,
            required_mp,optional_mp,
            False,
            leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
            hjcMethod,
            pointSuffix,
            anomalyException=anomalyException)

    elif CGM2_Model == "CGM2.5-UpperLimb":
        hjcMethod = settings["Calibration"]["HJC"]

        model,acqStatic,detectAnomaly = cgm2_5.calibrate(DATA_PATH,
            calibrateFilenameLabelled,
            translators,settings,
            required_mp,optional_mp,
            False,
            leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
            hjcMethod,
            pointSuffix,
            anomalyException=anomalyException)
    
    elif CGM2_Model == "CGM2.6-Knee Calibration":
        hjcMethod = settings["Calibration"]["HJC"]

        model,acqStatic,detectAnomaly = cgm2_5.calibrate(DATA_PATH,
            calibrateFilenameLabelled,
            translators,settings,
            required_mp,optional_mp,
            False,
            leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
            hjcMethod,
            pointSuffix,
            anomalyException=anomalyException)

    btkTools.cleanAcq(acqStatic)    
    
    LOGGER.logger.info(
        "----- CALIBRATION-  static file [%s]-----> DONE" % (calibrateFilenameLabelled))

    # --------------------------KNEE CALIBRATION ----------------------------------
    if CGM2_Model == "CGM2.6-Knee Calibration":
        leftKneeFuncMeasurement = qtmTools.findKneeCalibration(sessionXML,"Left")
        rightKneeFuncMeasurement = qtmTools.findKneeCalibration(sessionXML,"Right")

        LOGGER.logger.info("--------------------------Knee Calibration ----------------------------------")

        # if os.getcwd() + "\\" != DATA_PATH: # since cwd and data path are not related anymore, this is obsolete
        #     if leftKneeFuncMeasurement is not None:
        #         shutil.copyfile(os.getcwd()+"\\"+qtmTools.getFilename(leftKneeFuncMeasurement),
        #                         DATA_PATH+qtmTools.getFilename(leftKneeFuncMeasurement))
        #     if rightKneeFuncMeasurement is not None:
        #         shutil.copyfile(os.getcwd()+"\\"+qtmTools.getFilename(rightKneeFuncMeasurement),
        #                         DATA_PATH+qtmTools.getFilename(rightKneeFuncMeasurement))

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
    LOGGER.logger.info(
        "--------------------------MODEL FITTING ----------------------------------")
    dynamicMeasurements = qtmTools.findDynamic(sessionXML)
    modelledC3ds = []

    for dynamicMeasurement in dynamicMeasurements:
        reconstructFilenameLabelled = qtmTools.getFilename(dynamicMeasurement)

        LOGGER.logger.info(
            "----Processing of [%s]-----" % (reconstructFilenameLabelled))
        mfpa = qtmTools.getForcePlateAssigment(dynamicMeasurement)
        momentProjection_text = sessionXML.Moment_Projection.text


        if momentProjection_text == "Default":
            momentProjection_text = settings["Fitting"]["Moment Projection"]
        if momentProjection_text == "Distal":
            momentProjection = enums.MomentProjection.Distal
        elif momentProjection_text == "Proximal":
            momentProjection = enums.MomentProjection.Proximal
        elif momentProjection_text == "Global":
            momentProjection = enums.MomentProjection.Global
        elif momentProjection_text == "JCS":
            momentProjection =  enums.MomentProjection.JCS



        acq = btkTools.smartReader(DATA_PATH+reconstructFilenameLabelled)

        # filtering
        # -----------------------
        # marker
        order_marker = int(
            float(dynamicMeasurement.Marker_lowpass_filter_order.text))
        fc_marker = float(
            dynamicMeasurement.Marker_lowpass_filter_frequency.text)

        # force plate
        order_fp = int(
            float(dynamicMeasurement.Forceplate_lowpass_filter_order.text))
        fc_fp = float(
            dynamicMeasurement.Forceplate_lowpass_filter_frequency.text)

        if dynamicMeasurement.First_frame_to_process.text != "":
            vff = int(dynamicMeasurement.First_frame_to_process.text)
        else:
            vff = None

        if dynamicMeasurement.Last_frame_to_process.text != "":
            vlf = int(dynamicMeasurement.Last_frame_to_process.text)
        else:
            vlf = None

        # fitting operation
        # -----------------------
        if CGM2_Model == "CGM1.0":
            acqGait, detectAnomaly = cgm1.fitting(model, DATA_PATH, reconstructFilenameLabelled,
                                                translators,
                                                markerDiameter,
                                                pointSuffix,
                                                mfpa, momentProjection,
                                                fc_lowPass_marker=fc_marker,
                                                order_lowPass_marker=order_marker,
                                                fc_lowPass_forcePlate=fc_fp,
                                                order_lowPass_forcePlate=order_fp,
                                                anomalyException=anomalyException,
                                                frameInit=vff, frameEnd=vlf)
        elif CGM2_Model == "CGM1.1":
            acqGait,detectAnomaly = cgm1_1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,
            markerDiameter,
            pointSuffix,
            mfpa,momentProjection,
            fc_lowPass_marker=fc_marker,
            order_lowPass_marker=order_marker,
            fc_lowPass_forcePlate = fc_fp,
            order_lowPass_forcePlate = order_fp,
            anomalyException=anomalyException,
            frameInit= vff, frameEnd= vlf )
        
        elif CGM2_Model == "CGM2.1-HJC":
            acqGait,detectAnomaly = cgm2_1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,
            markerDiameter,
            pointSuffix,
            mfpa,momentProjection,
            fc_lowPass_marker=fc_marker,
            order_lowPass_marker=order_marker,
            fc_lowPass_forcePlate = fc_fp,
            order_lowPass_forcePlate = order_fp,
            anomalyException=anomalyException,
            frameInit= vff, frameEnd= vlf )

        elif CGM2_Model == "CGM2.2-IK":
            ik_flag = True
            ikAccuracy = float(dynamicMeasurement.IkAccuracy.text)
            
            acqGait,detectAnomaly = cgm2_2.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                        translators,settings,
                        ik_flag,
                        markerDiameter,
                        pointSuffix,
                        mfpa,momentProjection,
                        fc_lowPass_marker=fc_marker,
                        order_lowPass_marker=order_marker,
                        fc_lowPass_forcePlate = fc_fp,
                        order_lowPass_forcePlate = order_fp,
                        anomalyException=anomalyException,
                        ikAccuracy =ikAccuracy,
                        frameInit= vff, frameEnd= vlf )         
            
        elif CGM2_Model == "CGM2.3-skinClusters":
            ikAccuracy = float(dynamicMeasurement.IkAccuracy.text)
            ik_flag = True


            acqGait,detectAnomaly = cgm2_3.fitting(model,DATA_PATH, reconstructFilenameLabelled,
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
                frameInit= vff, frameEnd= vlf,
                muscleLength=True)
            
        elif CGM2_Model == "CGM2.4-ForeFoot":
            ikAccuracy = float(dynamicMeasurement.IkAccuracy.text)
            ik_flag = True

            acqGait,detectAnomaly = cgm2_4.fitting(model,DATA_PATH, reconstructFilenameLabelled,
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

        elif CGM2_Model == "CGM2.5-UpperLimb":
            ikAccuracy = float(dynamicMeasurement.IkAccuracy.text)
            ik_flag = True

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

        elif CGM2_Model == "CGM2.6-Knee Calibration":
            ikAccuracy = float(dynamicMeasurement.IkAccuracy.text)
            ik_flag = True

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

        btkTools.cleanAcq(acqGait)

        outFilename = reconstructFilenameLabelled
        btkTools.smartWriter(acqGait, str(DATA_PATH + outFilename))
        modelledC3ds.append(outFilename)

        LOGGER.logger.info(
            "----Processing of [%s]-----> DONE" % (reconstructFilenameLabelled))

    LOGGER.logger.info(
        "-------------------------------------------------------")
    if detectAnomaly:
        LOGGER.logger.error(
            "Anomalies has been detected - Find Error messages, then check warning message in the log file")
    else:
        LOGGER.logger.info("workflow return with no detected anomalies")

    # os.startfile( os.getcwd()+"\\"+ LOGFILE)


if __name__ == '__main__':
    main(args=None) 