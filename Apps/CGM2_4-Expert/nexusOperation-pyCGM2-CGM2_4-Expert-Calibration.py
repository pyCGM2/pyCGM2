# -*- coding: utf-8 -*-
#import ipdb
import logging
import argparse
import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# vicon nexus
import ViconNexus


# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model import modelFilters, modelDecorator
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.apps import cgmUtils


if __name__ == "__main__":

    plt.close("all")
    DEBUG = False

    parser = argparse.ArgumentParser(description='CGM2.4-Expert Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser.add_argument('--resetMP', action='store_false', help='reset optional mass parameters')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------GLOBAL SETTINGS ------------------------------

        # global setting ( in user/AppData)
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_4-Expert-pyCGM2.settings")
        # --------------------------LOADING------------------------------

        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\c3dOnly\\"
            calibrateFilenameLabelledNoExt = "static" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )
            args.noIk=False

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)


        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp,optional_mp = nexusUtils.getNexusSubjectMp(NEXUS,subject,resetFlag=args.resetMP)


        # --------------------------SESSION INFOS -----------------------------
        info = files.manage_pycgm2SessionInfos(DATA_PATH,subject)

        #  translators management
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")
        if not translators:
           translators = settings["Translators"]

        # --------------------------CONFIG ------------------------------------
        argsManager = cgmUtils.argsManager_cgm(settings,args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.4e")

        hjcMethod = settings["Calibration"]["HJC regression"]
        ik_flag = False if args.noIk else True

        # --------------------------STATIC FILE WITH TRANSLATORS --------------------------------------
        # ---btk acquisition---
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))
        btkTools.checkMultipleSubject(acqStatic)

        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        validFrames,vff,vlf = btkTools.findValidFrames(acqStatic,cgm2.CGM2_4LowerLimbs.MARKERS)



        # --------------------------MODEL--------------------------------------
        # ---definition---
        model=cgm2.CGM2_4LowerLimbs()
        model.setVersion("CGM2.4e")
        model.configure()

        model.addAnthropoInputParameters(required_mp,optional=optional_mp)

        # --store calibration parameters--
        model.setStaticFilename(calibrateFilenameLabelled)
        model.setCalibrationProperty("leftFlatFoot",leftFlatFoot)
        model.setCalibrationProperty("rightFlatFoot",rightFlatFoot)
        model.setCalibrationProperty("markerDiameter",markerDiameter)


        # ---check marker set used----
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)

        # --------------------------STATIC CALBRATION--------------------------
        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        # ---initial calibration filter----
        # use if all optional mp are zero
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                                            markerDiameter=markerDiameter,
                                            ).compute()

        # ---- Decorators -----
        # Goal = modified calibration according the identified marker set or if offsets manually set

        cgmUtils.applyDecorators_CGM(smc, model,acqStatic,optional_mp,markerDiameter)
        cgmUtils.applyHJCDecorators(model,hjcMethod)


        # ----Final Calibration filter if model previously decorated -----
        if model.decoratedModel:
            # initial static filter
            modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                               markerDiameter=markerDiameter).compute()



        # ----------------------CGM MODELLING----------------------------------
        # ----motion filter----
        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Determinist,
                                                  markerDiameter=markerDiameter)

        modMotion.compute()



        if ik_flag:

            # ---Marker decomp filter----
            mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqStatic)
            mtf.decompose()

            #                        ---OPENSIM IK---

            # --- opensim calibration Filter ---
            osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
            markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset - expert.xml" # markerset
            cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

            oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                    model,
                                                    cgmCalibrationprocedure)
            oscf.addMarkerSet(markersetFile)
            scalingOsim = oscf.build()


            # --- opensim Fitting Filter ---
            iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-expert-ikSetUp_template.xml" # ik tool file

            cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model,expertMode = True) # procedure
            cgmFittingProcedure.updateMarkerWeight("LASI",settings["Fitting"]["Weight"]["LASI"])
            cgmFittingProcedure.updateMarkerWeight("RASI",settings["Fitting"]["Weight"]["RASI"])
            cgmFittingProcedure.updateMarkerWeight("LPSI",settings["Fitting"]["Weight"]["LPSI"])
            cgmFittingProcedure.updateMarkerWeight("RPSI",settings["Fitting"]["Weight"]["RPSI"])
            cgmFittingProcedure.updateMarkerWeight("RTHI",settings["Fitting"]["Weight"]["RTHI"])
            cgmFittingProcedure.updateMarkerWeight("RKNE",settings["Fitting"]["Weight"]["RKNE"])
            cgmFittingProcedure.updateMarkerWeight("RTIB",settings["Fitting"]["Weight"]["RTIB"])
            cgmFittingProcedure.updateMarkerWeight("RANK",settings["Fitting"]["Weight"]["RANK"])
            cgmFittingProcedure.updateMarkerWeight("RHEE",settings["Fitting"]["Weight"]["RHEE"])
            cgmFittingProcedure.updateMarkerWeight("RTOE",settings["Fitting"]["Weight"]["RTOE"])
            cgmFittingProcedure.updateMarkerWeight("LTHI",settings["Fitting"]["Weight"]["LTHI"])
            cgmFittingProcedure.updateMarkerWeight("LKNE",settings["Fitting"]["Weight"]["LKNE"])
            cgmFittingProcedure.updateMarkerWeight("LTIB",settings["Fitting"]["Weight"]["LTIB"])
            cgmFittingProcedure.updateMarkerWeight("LANK",settings["Fitting"]["Weight"]["LANK"])
            cgmFittingProcedure.updateMarkerWeight("LHEE",settings["Fitting"]["Weight"]["LHEE"])
            cgmFittingProcedure.updateMarkerWeight("LTOE",settings["Fitting"]["Weight"]["LTOE"])

            cgmFittingProcedure.updateMarkerWeight("LASI_posAnt",settings["Fitting"]["Weight"]["LASI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LASI_medLat",settings["Fitting"]["Weight"]["LASI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LASI_supInf",settings["Fitting"]["Weight"]["LASI_supInf"])

            cgmFittingProcedure.updateMarkerWeight("RASI_posAnt",settings["Fitting"]["Weight"]["RASI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RASI_medLat",settings["Fitting"]["Weight"]["RASI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RASI_supInf",settings["Fitting"]["Weight"]["RASI_supInf"])

            cgmFittingProcedure.updateMarkerWeight("LPSI_posAnt",settings["Fitting"]["Weight"]["LPSI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LPSI_medLat",settings["Fitting"]["Weight"]["LPSI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LPSI_supInf",settings["Fitting"]["Weight"]["LPSI_supInf"])

            cgmFittingProcedure.updateMarkerWeight("RPSI_posAnt",settings["Fitting"]["Weight"]["RPSI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RPSI_medLat",settings["Fitting"]["Weight"]["RPSI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RPSI_supInf",settings["Fitting"]["Weight"]["RPSI_supInf"])


            cgmFittingProcedure.updateMarkerWeight("RTHI_posAnt",settings["Fitting"]["Weight"]["RTHI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTHI_medLat",settings["Fitting"]["Weight"]["RTHI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTHI_proDis",settings["Fitting"]["Weight"]["RTHI_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RKNE_posAnt",settings["Fitting"]["Weight"]["RKNE_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RKNE_medLat",settings["Fitting"]["Weight"]["RKNE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RKNE_proDis",settings["Fitting"]["Weight"]["RKNE_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RTIB_posAnt",settings["Fitting"]["Weight"]["RTIB_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTIB_medLat",settings["Fitting"]["Weight"]["RTIB_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTIB_proDis",settings["Fitting"]["Weight"]["RTIB_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RANK_posAnt",settings["Fitting"]["Weight"]["RANK_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RANK_medLat",settings["Fitting"]["Weight"]["RANK_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RANK_proDis",settings["Fitting"]["Weight"]["RANK_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RHEE_supInf",settings["Fitting"]["Weight"]["RHEE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RHEE_medLat",settings["Fitting"]["Weight"]["RHEE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RHEE_proDis",settings["Fitting"]["Weight"]["RHEE_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RTOE_supInf",settings["Fitting"]["Weight"]["RTOE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RTOE_medLat",settings["Fitting"]["Weight"]["RTOE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTOE_proDis",settings["Fitting"]["Weight"]["RTOE_proDis"])






            cgmFittingProcedure.updateMarkerWeight("LTHI_posAnt",settings["Fitting"]["Weight"]["LTHI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTHI_medLat",settings["Fitting"]["Weight"]["LTHI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTHI_proDis",settings["Fitting"]["Weight"]["LTHI_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LKNE_posAnt",settings["Fitting"]["Weight"]["LKNE_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LKNE_medLat",settings["Fitting"]["Weight"]["LKNE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LKNE_proDis",settings["Fitting"]["Weight"]["LKNE_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LTIB_posAnt",settings["Fitting"]["Weight"]["LTIB_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTIB_medLat",settings["Fitting"]["Weight"]["LTIB_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTIB_proDis",settings["Fitting"]["Weight"]["LTIB_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LANK_posAnt",settings["Fitting"]["Weight"]["LANK_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LANK_medLat",settings["Fitting"]["Weight"]["LANK_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LANK_proDis",settings["Fitting"]["Weight"]["LANK_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LHEE_supInf",settings["Fitting"]["Weight"]["LHEE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LHEE_medLat",settings["Fitting"]["Weight"]["LHEE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LHEE_proDis",settings["Fitting"]["Weight"]["LHEE_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LTOE_supInf",settings["Fitting"]["Weight"]["LTOE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LTOE_medLat",settings["Fitting"]["Weight"]["LTOE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTOE_proDis",settings["Fitting"]["Weight"]["LTOE_proDis"])


            cgmFittingProcedure.updateMarkerWeight("LTHAP_posAnt",settings["Fitting"]["Weight"]["LTHAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTHAP_medLat",settings["Fitting"]["Weight"]["LTHAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTHAP_proDis",settings["Fitting"]["Weight"]["LTHAP_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LTHAD_posAnt",settings["Fitting"]["Weight"]["LTHAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTHAD_medLat",settings["Fitting"]["Weight"]["LTHAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTHAD_proDis",settings["Fitting"]["Weight"]["LTHAD_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RTHAP_posAnt",settings["Fitting"]["Weight"]["RTHAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTHAP_medLat",settings["Fitting"]["Weight"]["RTHAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTHAP_proDis",settings["Fitting"]["Weight"]["RTHAP_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RTHAD_posAnt",settings["Fitting"]["Weight"]["RTHAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTHAD_medLat",settings["Fitting"]["Weight"]["RTHAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTHAD_proDis",settings["Fitting"]["Weight"]["RTHAD_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LTIAP_posAnt",settings["Fitting"]["Weight"]["LTIAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTIAP_medLat",settings["Fitting"]["Weight"]["LTIAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTIAP_proDis",settings["Fitting"]["Weight"]["LTIAP_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LTIAD_posAnt",settings["Fitting"]["Weight"]["LTIAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTIAD_medLat",settings["Fitting"]["Weight"]["LTIAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTIAD_proDis",settings["Fitting"]["Weight"]["LTIAD_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RTIAP_posAnt",settings["Fitting"]["Weight"]["RTIAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTIAP_medLat",settings["Fitting"]["Weight"]["RTIAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTIAP_proDis",settings["Fitting"]["Weight"]["RTIAP_proDis"])


            cgmFittingProcedure.updateMarkerWeight("RTIAD_posAnt",settings["Fitting"]["Weight"]["RTIAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTIAD_medLat",settings["Fitting"]["Weight"]["RTIAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTIAD_proDis",settings["Fitting"]["Weight"]["RTIAD_proDis"])


            cgmFittingProcedure.updateMarkerWeight("LSMH_supInf",settings["Fitting"]["Weight"]["LSMH_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LSMH_medLat",settings["Fitting"]["Weight"]["LSMH_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LSMH_proDis",settings["Fitting"]["Weight"]["LSMH_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LFMH_supInf",settings["Fitting"]["Weight"]["LFMH_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LFMH_medLat",settings["Fitting"]["Weight"]["LFMH_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LFMH_proDis",settings["Fitting"]["Weight"]["LFMH_proDis"])

            cgmFittingProcedure.updateMarkerWeight("LVMH_supInf",settings["Fitting"]["Weight"]["LVMH_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LVMH_medLat",settings["Fitting"]["Weight"]["LVMH_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LVMH_proDis",settings["Fitting"]["Weight"]["LVMH_proDis"])


            cgmFittingProcedure.updateMarkerWeight("RSMH_supInf",settings["Fitting"]["Weight"]["RSMH_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RSMH_medLat",settings["Fitting"]["Weight"]["RSMH_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RSMH_proDis",settings["Fitting"]["Weight"]["RSMH_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RFMH_supInf",settings["Fitting"]["Weight"]["RFMH_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RFMH_medLat",settings["Fitting"]["Weight"]["RFMH_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RFMH_proDis",settings["Fitting"]["Weight"]["RFMH_proDis"])

            cgmFittingProcedure.updateMarkerWeight("RVMH_supInf",settings["Fitting"]["Weight"]["RVMH_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RVMH_medLat",settings["Fitting"]["Weight"]["RVMH_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RVMH_proDis",settings["Fitting"]["Weight"]["RVMH_proDis"])


#            cgmFittingProcedure.updateMarkerWeight("LPAT",settings["Fitting"]["Weight"]["LPAT"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_posAnt",settings["Fitting"]["Weight"]["LPAT_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_medLat",settings["Fitting"]["Weight"]["LPAT_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_proDis",settings["Fitting"]["Weight"]["LPAT_proDis"])
#
#            cgmFittingProcedure.updateMarkerWeight("RPAT",settings["Fitting"]["Weight"]["RPAT"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_posAnt",settings["Fitting"]["Weight"]["RPAT_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_medLat",settings["Fitting"]["Weight"]["RPAT_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_proDis",settings["Fitting"]["Weight"]["RPAT_proDis"])

#            cgmFittingProcedure.updateMarkerWeight("LTHLD",settings["Fitting"]["Weight"]["LTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_posAnt",settings["Fitting"]["Weight"]["LTHLD_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_medLat",settings["Fitting"]["Weight"]["LTHLD_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_proDis",settings["Fitting"]["Weight"]["LTHLD_proDis"])
#
#            cgmFittingProcedure.updateMarkerWeight("RTHLD",settings["Fitting"]["Weight"]["RTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_posAnt",settings["Fitting"]["Weight"]["RTHLD_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_medLat",settings["Fitting"]["Weight"]["RTHLD_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_proDis",settings["Fitting"]["Weight"]["RTHLD_proDis"])

            osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                              scalingOsim,
                                                              cgmFittingProcedure,
                                                              str(DATA_PATH) )
            acqStaticIK = osrf.run(acqStatic,str(DATA_PATH + calibrateFilenameLabelled ))



        # eventual static acquisition to consider for joint kinematics
        finalAcqStatic = acqStaticIK if ik_flag else acqStatic

        # --- final pyCGM2 model motion Filter ---
        # use fitted markers
        modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqStatic,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotionFitted.compute()

        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,finalAcqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        # detection of traveling axis
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(finalAcqStatic,["LASI","RASI","RPSI","LPSI"])

        # absolute angles
        modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqStatic,
                                               segmentLabels=["Left HindFoot","Right HindFoot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                eulerSequences=["TOR","TOR", "ROT"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)



        # ----------------------SAVE-------------------------------------------
        files.saveModel(model,DATA_PATH,subject)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)
        nexusFilters.NexusModelFilter(NEXUS,
                                      model,finalAcqStatic,subject,
                                      pointSuffix,
                                      staticProcessing=True).run()


        # ========END of the nexus OPERATION if run from Nexus  =========


        if DEBUG:
            NEXUS.SaveTrial(30)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
