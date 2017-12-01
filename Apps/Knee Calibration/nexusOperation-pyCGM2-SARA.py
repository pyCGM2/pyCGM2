# -*- coding: utf-8 -*-
#import ipdb
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Model.CGM2 import cgm2
from pyCGM2.Model import modelFilters, modelDecorator
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

def detectSide(acq,left_markerLabel,right_markerLabel):

    flag,vff,vlf = btkTools.findValidFrames(acq,[left_markerLabel,right_markerLabel])

    left = acq.GetPoint(left_markerLabel).GetValues()[vff:vlf,2]
    right = acq.GetPoint(right_markerLabel).GetValues()[vff:vlf,2]

    side = "Left" if np.max(left)>np.max(right) else "Right"

    return side


if __name__ == "__main__":

    plt.close("all")
    DEBUG = True

    parser = argparse.ArgumentParser(description='SARA Functional Knee Calibration')
    parser.add_argument('-s','--side', type=str, help="Side : Left or Right")
    parser.add_argument('-b','--beginFrame', type=int, help="begin frame")
    parser.add_argument('-e','--endFrame', type=int, help="end frame")
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------PATH + FILE ------------------------------------

        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "pyCGM2-Data\\Datasets Tests\\fraser\\New Session\\"
            reconstructedFilenameLabelledNoExt = "15KUFC01_Trial07"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructedFilenameLabelledNoExt), 30 )

            side = "Left"
            args.beginFrame=1073
            args.endFrame=2961

        else:
            DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "reconstructed file: "+ reconstructFilenameLabelled)

       # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        logging.info("loaded model : %s" %(model.version ))
        # --------------------------CONFIG ------------------------------------

        # --------------------CHECKING ------------------------------
        if model.version in ["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e"] :
            raise Exception ("Can t use SARA method with your model %s [minimal version : CGM2.3]"%(model.version))
        elif model.version == "CGM2.3":
            settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.settings")
        elif model.version == "CGM2.3e":
            settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_3-Expert-pyCGM2.settings")
        elif model.version == "CGM2.4":
            settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")
        elif model.version == "CGM2.4e":
            settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_4-Expert-pyCGM2.settings")
        else:
            raise Exception ("model version not found [contact admin]")

        # --------------------------SESSION INFOS ------------------------------------
        # info file
        info = files.manage_pycgm2SessionInfos(DATA_PATH,subject)

        #  translators management
        if model.version in  ["CGM2.3","CGM2.3e"]:
            translators = files.manage_pycgm2Translators(DATA_PATH,"CGM2-3.translators")
        elif model.version in  ["CGM2.4","CGM2.4e"]:
            translators = files.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")
        if not translators:
           translators = settings["Translators"]

        # --------------------------ACQ WITH TRANSLATORS --------------------------------------

        # --- btk acquisition ----
        acqFunc = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))
        btkTools.checkMultipleSubject(acqFunc)
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)

        #---get frame range of interest---
        ff = acqFunc.GetFirstFrame()
        lf = acqFunc.GetLastFrame()

        initFrame = args.beginFrame if args.beginFrame is not None else ff
        endFrame = args.endFrame if args.endFrame is not None else lf

        iff=initFrame-ff
        ilf=endFrame-ff


        #---motion side of the lower limb---
        if args.side is None:
            side = detectSide(acqFunc,"LANK","RANK")
            logging.info("Detected motion side : %s" %(side) )
        else:
            side = args.side

        # --------------------------RESET OF THE STATIC File---------

        # load btkAcq from static file
        staticFilename = model.m_staticFilename
        acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilename))
        btkTools.checkMultipleSubject(acqStatic)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        # initial calibration ( i.e calibration from Calibration operation)
        leftFlatFoot = model.m_properties["CalibrationParameters"]["leftFlatFoot"]
        rightFlatFoot = model.m_properties["CalibrationParameters"]["rightFlatFoot"]
        markerDiameter = model.m_properties["CalibrationParameters"]["markerDiameter"]

        if side == "Left":
            # remove other functional calibration
            model.mp_computed["LeftKneeFuncCalibrationOffset"] = 0

        if side == "Right":
            model.mp_computed["RightKneeFuncCalibrationOffset"] = 0



        # initial calibration ( zero previous KneeFunc offset on considered side )
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                               markerDiameter=markerDiameter).compute()



        #btkTools.smartWriter(acqStatic, "acqStatic0-test.c3d")

        if model.version in  ["CGM2.3","CGM2.3e","CGM2.3","CGM2.4e"]:
            if side == "Left":
                thigh_markers = model.getSegment("Left Thigh").m_tracking_markers
                shank_markers = model.getSegment("Left Shank").m_tracking_markers

            elif side == "Right":
                thigh_markers = model.getSegment("Right Thigh").m_tracking_markers
                shank_markers = model.getSegment("Right Shank").m_tracking_markers

            validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,thigh_markers+shank_markers)

            proximalSegmentLabel=str(side+" Thigh")
            distalSegmentLabel=str(side+" Shank")

            # segment Motion
            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
            modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])

            # decorator
            modelDecorator.KneeCalibrationDecorator(model).sara(side,
                                                                indexFirstFrame = iff,
                                                                indexLastFrame = ilf )


            # --------------------------FINAL CALIBRATION OF THE STATIC File---------

            modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                               markerDiameter=markerDiameter).compute()


            #btkTools.smartWriter(acqStatic, "acqStatic1-test.c3d")

            # ----------------------SAVE-------------------------------------------
            files.saveModel(model,DATA_PATH,subject)
            logging.warning("model updated with a  %s knee calibrated with SARA method" %(side))



            # ----------------------VICON INTERFACE-------------------------------------------
            #--- update mp
            nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)



            #---points from first motio filter

            # -- add nexus Bones
            if side == "Left":
                nexusTools.appendBones(NEXUS,subject,acqFunc,"LFE0", model.getSegment("Left Thigh"),OriginValues = acqFunc.GetPoint("LKJC").GetValues() )
            elif side == "Right":
                nexusTools.appendBones(NEXUS,subject,acqFunc,"RFE0", model.getSegment("Right Thigh"),OriginValues = acqFunc.GetPoint("RKJC").GetValues() )

            # add modelled markers
            meanOr_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_Sara")
            meanAxis_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_SaraAxis")
            btkTools.smartAppendPoint(acqFunc,side+"_KJC_Sara",meanOr_inThigh)
            btkTools.smartAppendPoint(acqFunc,side+"_KJC_SaraAxis",meanAxis_inThigh)

            nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KJC_Sara", acqFunc)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KJC_SaraAxis", acqFunc)


            #---Second model motion filter
            # consider new anatomical frame

            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
            modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])


            # projection of the Sara axis in the transversale plane
            projMeanAxis_inThigh = model.getSegment(proximalSegmentLabel).anatomicalFrame.getNodeTrajectory("proj_saraAxis")
            btkTools.smartAppendPoint(acqFunc,side+"_proj_SaraAxis",projMeanAxis_inThigh)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_proj_SaraAxis", acqFunc)

            # -- add nexus Bones
            if side == "Left":
                nexusTools.appendBones(NEXUS,subject,acqFunc,"LFE1", model.getSegment("Left Thigh"),OriginValues = acqFunc.GetPoint("LKJC").GetValues() )
                print model.mp_computed["LeftKneeFuncCalibrationOffset"]
                logging.warning("offset %s" %(str(model.mp_computed["LeftKneeFuncCalibrationOffset"] )))
            elif side == "Right":
                nexusTools.appendBones(NEXUS,subject,acqFunc,"RFE1", model.getSegment("Right Thigh"),OriginValues = acqFunc.GetPoint("RKJC").GetValues() )
                logging.warning("offset %s" %(str(model.mp_computed["RightKneeFuncCalibrationOffset"] )))0
                print model.mp_computed["RightKneeFuncCalibrationOffset"]


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
