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

if __name__ == "__main__":

    plt.close("all")
    DEBUG = False

    parser = argparse.ArgumentParser(description='CGM2.2 Calibration')
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

        # --------------------------GLOBAL SETTINGS ------------------------------------
        # global setting ( in user/AppData)
        settings = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_2-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")

        # --------------------------LOADING ------------------------------------

        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH +"CGM2\\cgm2.2\\c3dOnly\\"
            calibrateFilenameLabelledNoExt = "MRI-US-01, 2008-08-08, 3DGA 02" #"static Cal 01-noKAD-noAnkleMed" #
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

        # --------------------------SESSIONS INFOS -----------------------------------
        info = files.manage_pycgm2SessionInfos(DATA_PATH,subject)

        #  translators management
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM2-2.translators")
        if not translators:
           translators = settings["Translators"]

        # --------------------------CONFIG -----------------------------------

        hjcMethod = settings["Calibration"]["HJC regression"]

        if args.leftFlatFoot is not None:
            flag_leftFlatFoot = bool(args.leftFlatFoot)
        else:
            flag_leftFlatFoot = bool(settings["Calibration"]["Left flat foot"])

        if args.rightFlatFoot is not None:
            flag_rightFlatFoot = bool(args.rightFlatFoot)
        else:
            flag_rightFlatFoot =  bool(settings["Calibration"]["Right flat foot"])


        if args.markerDiameter is not None:
            markerDiameter = float(args.markerDiameter)
            logging.warning("marker diameter forced : %s", str(float(args.markerDiameter)))
        else:
            markerDiameter = float(settings["Global"]["Marker diameter"])


        if args.check:
            pointSuffix="cgm2.2"
        else:
            if args.pointSuffix is not None:
                pointSuffix = args.pointSuffix
            else:
                pointSuffix = settings["Global"]["Point suffix"]

        ik_flag = False if args.noIk else True

        # --------------------ACQUISITION------------------------------

        # ---btk acquisition---
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))
        btkTools.checkMultipleSubject(acqStatic)

        acqStatic =  btkTools.applyTranslators(acqStatic,translators)


        # ---definition---
        model=cgm2.CGM2_2LowerLimbs()
        model.configure()

        model.setStaticFilename(calibrateFilenameLabelled)

        model.addAnthropoInputParameters(required_mp,optional=optional_mp)

        # ---check marker set used----
        staticMarkerConfiguration= cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)


        # --------------------------STATIC CALBRATION--------------------------
        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        # ---initial calibration filter----
        # use if all optional mp are zero
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                            markerDiameter=markerDiameter,
                                            ).compute()

        # ---- Decorators -----
        # Goal = modified calibration according the identified marker set or if offsets manually set


        # hip joint centres ---
        useLeftHJCnodeLabel = "LHJC_cgm1"
        useRightHJCnodeLabel = "RHJC_cgm1"
        if hjcMethod == "Hara":
            modelDecorator.HipJointCenterDecorator(model).hara()
            useLeftHJCnodeLabel = "LHJC_Hara"
            useRightHJCnodeLabel = "RHJC_Hara"


        # knee - ankle centres ---

        # initialisation of node label
        useLeftKJCnodeLabel = "LKJC_chord"
        useLeftAJCnodeLabel = "LAJC_chord"
        useRightKJCnodeLabel = "RKJC_chord"
        useRightAJCnodeLabel = "RAJC_chord"

        # case 1 : NO kad, NO medial ankle BUT thighRotation different from zero ( mean manual modification or new calibration from a previous one )
        if not staticMarkerConfiguration["leftKadFlag"]  and not staticMarkerConfiguration["leftMedialAnkleFlag"] and not staticMarkerConfiguration["leftMedialKneeFlag"] and optional_mp["LeftThighRotation"] !=0:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - Origine - manual offsets")
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],markerDiameter,optional_mp["LeftTibialTorsion"],optional_mp["LeftShankRotation"])
            useLeftKJCnodeLabel = "LKJC_mo"
            useLeftAJCnodeLabel = "LAJC_mo"
        if not staticMarkerConfiguration["rightKadFlag"]  and not staticMarkerConfiguration["rightMedialAnkleFlag"] and not staticMarkerConfiguration["rightMedialKneeFlag"] and optional_mp["RightThighRotation"] !=0:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - Origine - manual offsets")
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])
            useRightKJCnodeLabel = "RKJC_mo"
            useRightAJCnodeLabel = "RAJC_mo"

        # case 2 : kad FOUND and NO medial Ankle
        if staticMarkerConfiguration["leftKadFlag"]:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD variant")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
            useLeftKJCnodeLabel = "LKJC_kad"
            useLeftAJCnodeLabel = "LAJC_kad"
        if staticMarkerConfiguration["rightKadFlag"]:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD variant")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
            useRightKJCnodeLabel = "RKJC_kad"
            useRightAJCnodeLabel = "RAJC_kad"


        # case 3 : medial knee FOUND
        # notice: cgm1Behaviour is enable mean effect on AJC location
        if staticMarkerConfiguration["leftMedialKneeFlag"]:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - medial knee ")
            modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left",cgm1Behaviour=True)
            useLeftKJCnodeLabel = "LKJC_mid"
            useLeftAJCnodeLabel = "LAJC_midKnee"
        if staticMarkerConfiguration["rightMedialKneeFlag"]:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - medial knee ")
            modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right",cgm1Behaviour=True)
            useRightKJCnodeLabel = "RKJC_mid"
            useRightAJCnodeLabel = "RAJC_midKnee"


        # case 4 : medial ankle FOUND
        if staticMarkerConfiguration["leftMedialAnkleFlag"]:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - medial ankle ")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
            useLeftAJCnodeLabel = "LAJC_mid"
        if staticMarkerConfiguration["rightMedialAnkleFlag"]:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - medial ankle ")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
            useRightAJCnodeLabel = "RAJC_mid"

        properties_initialCalibration=dict()
        properties_initialCalibration["LHJC_node"] = useLeftHJCnodeLabel
        properties_initialCalibration["RHJC_node"] = useRightHJCnodeLabel
        properties_initialCalibration["LKJC_node"] = useLeftKJCnodeLabel
        properties_initialCalibration["RKJC_node"] = useRightKJCnodeLabel
        properties_initialCalibration["LAJC_node"] = useLeftAJCnodeLabel
        properties_initialCalibration["RAJC_node"] = useRightAJCnodeLabel
        properties_initialCalibration["rightFlatFoot"] = useRightAJCnodeLabel
        properties_initialCalibration["leftFlatFoot"] = flag_rightFlatFoot
        properties_initialCalibration["markerDiameter"] = markerDiameter


        # ----Final Calibration filter if model previously decorated -----
        if model.decoratedModel:
            # initial static filter
            modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               useLeftHJCnode=useLeftHJCnodeLabel, useRightHJCnode=useRightHJCnodeLabel,
                               useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
                               useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
                               leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                               markerDiameter=markerDiameter).compute()

        # set initial calibration as model property
        model.m_properties["CalibrationParameters0"] = properties_initialCalibration

        # ----------------------CGM MODELLING----------------------------------
        # ----motion filter----
        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Determinist,
                                                  markerDiameter=markerDiameter)

        modMotion.compute()


        if ik_flag:
            #                        ---OPENSIM IK---

            # --- opensim calibration Filter ---
            osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
            markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-markerset.xml" # markerset
            cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

            oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                    model,
                                                    cgmCalibrationprocedure)
            oscf.addMarkerSet(markersetFile)
            scalingOsim = oscf.build()


            # --- opensim Fitting Filter ---
            iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-ikSetUp_template.xml" # ik tool file

            cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
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


            osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                              scalingOsim,
                                                              cgmFittingProcedure,
                                                              str(DATA_PATH) )
            acqStaticIK = osrf.run(acqStatic,str(DATA_PATH + calibrateFilenameLabelled ))



            # --- final pyCGM2 model motion Filter ---
            # use fitted markers
            modMotionFitted=modelFilters.ModelMotionFilter(scp,acqStaticIK,model,pyCGM2Enums.motionMethod.Sodervisk)

            modMotionFitted.compute()


        # eventual static acquisition to consider for joint kinematics
        finalAcqStatic = acqStaticIK if ik_flag else acqStatic

        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,finalAcqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        # detection of traveling axis
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(finalAcqStatic,["LASI","RASI","RPSI","LPSI"])

        # absolute angles
        modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqStatic,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                eulerSequences=["TOR","TOR", "ROT"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)



        # ----------------------SAVE-------------------------------------------
        #pyCGM2.model
        files.saveModel(model,DATA_PATH,subject)



        # ----------------------DISPLAY ON VICON-------------------------------
        nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)
        nexusFilters.NexusModelFilter(NEXUS,
                                      model,acqStatic,subject,
                                      pointSuffix,
                                      staticProcessing=True).run()

        # ========END of the nexus OPERATION if run from Nexus  =========


        if DEBUG:
            NEXUS.SaveTrial(30)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
