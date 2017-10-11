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
from pyCGM2.Model import modelFilters, modelDecorator,bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim import opensimFilters



if __name__ == "__main__":


    DEBUG = False


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='CGM2-4 Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-mfpa',type=str,  help='manual assignment of force plates')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    args = parser.parse_args()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------GLOBAL SETTINGS ------------------------------
        # global setting ( in user/AppData)
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")

        # ----------------------LOADING-------------------------------------------
        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\c3dOnly\\"
            reconstructFilenameLabelledNoExt = "gait 01"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )
            args.noIk = False
        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()


        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "reconstructed file: "+ reconstructFilenameLabelled)

        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        # --------------------------CHECKING -----------------------------------
        # check model
        logging.info("loaded model : %s" %(model.version))
        if model.version != "CGM2.4":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM2.3e calibration pipeline"%subject)

        # --------------------------SESSION INFOS -----------------------------
        # info file
        info = files.manage_pycgm2SessionInfos(DATA_PATH,subject)

        #  translators management
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")
        if not translators:
           translators = settings["Translators"]

        # --------------------------CONFIG -----------------------------

        if args.markerDiameter is not None:
            markerDiameter = float(args.markerDiameter)
            logging.warning("marker diameter forced : %s", str(float(args.markerDiameter)))
        else:
            markerDiameter = float(settings["Global"]["Marker diameter"])


        if args.check:
            pointSuffix="cgm2.4"
        else:
            if args.pointSuffix is not None:
                pointSuffix = args.pointSuffix
            else:
                pointSuffix = settings["Global"]["Point suffix"]

        if args.proj is not None:
            if args.proj == "Distal":
                momentProjection = pyCGM2Enums.MomentProjection.Distal
            elif args.proj == "Proximal":
                momentProjection = pyCGM2Enums.MomentProjection.Proximal
            elif args.proj == "Global":
                momentProjection = pyCGM2Enums.MomentProjection.Global
            elif args.proj == "JCS":
                momentProjection = pyCGM2Enums.MomentProjection.JCS
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

        else:
            if settings["Fitting"]["Moment Projection"] == "Distal":
                momentProjection = pyCGM2Enums.MomentProjection.Distal
            elif settings["Fitting"]["Moment Projection"] == "Proximal":
                momentProjection = pyCGM2Enums.MomentProjection.Proximal
            elif settings["Fitting"]["Moment Projection"] == "Global":
                momentProjection = pyCGM2Enums.MomentProjection.Global
            elif settings["Fitting"]["Moment Projection"] == "JCS":
                momentProjection = pyCGM2Enums.MomentProjection.JCS
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

        ik_flag = False if args.noIk else True

        # --------------------------ACQ WITH TRANSLATORS --------------------------------------

        # --- btk acquisition ----
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        btkTools.checkMultipleSubject(acqGait)

        acqGait =  btkTools.applyTranslators(acqGait,translators)
        validFrames,vff,vlf = btkTools.findValidFrames(acqGait,cgm2.CGM2_4LowerLimbs.MARKERS)



        # --- initial motion Filter ---
        scp=modelFilters.StaticCalibrationProcedure(model)
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        if ik_flag:
            #                        ---OPENSIM IK---

            # --- opensim calibration Filter ---
            osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
            markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml" # markerset
            cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

            oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                    model,
                                                    cgmCalibrationprocedure)
            oscf.addMarkerSet(markersetFile)
            scalingOsim = oscf.build()


            # --- opensim Fitting Filter ---
            iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml" # ik tl file

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


            cgmFittingProcedure.updateMarkerWeight("LTHAP",settings["Fitting"]["Weight"]["LTHAP"])
            cgmFittingProcedure.updateMarkerWeight("LTHAD",settings["Fitting"]["Weight"]["LTHAD"])
            cgmFittingProcedure.updateMarkerWeight("LTIAP",settings["Fitting"]["Weight"]["LTIAP"])
            cgmFittingProcedure.updateMarkerWeight("LTIAD",settings["Fitting"]["Weight"]["LTIAD"])
            cgmFittingProcedure.updateMarkerWeight("RTHAP",settings["Fitting"]["Weight"]["RTHAP"])
            cgmFittingProcedure.updateMarkerWeight("RTHAD",settings["Fitting"]["Weight"]["RTHAD"])
            cgmFittingProcedure.updateMarkerWeight("RTIAP",settings["Fitting"]["Weight"]["RTIAP"])
            cgmFittingProcedure.updateMarkerWeight("RTIAD",settings["Fitting"]["Weight"]["RTIAD"])

            cgmFittingProcedure.updateMarkerWeight("LSMH",settings["Fitting"]["Weight"]["LSMH"])
            cgmFittingProcedure.updateMarkerWeight("LFMH",settings["Fitting"]["Weight"]["LFMH"])
            cgmFittingProcedure.updateMarkerWeight("LVMH",settings["Fitting"]["Weight"]["LVMH"])

            cgmFittingProcedure.updateMarkerWeight("RSMH",settings["Fitting"]["Weight"]["RSMH"])
            cgmFittingProcedure.updateMarkerWeight("RFMH",settings["Fitting"]["Weight"]["RFMH"])
            cgmFittingProcedure.updateMarkerWeight("RVMH",settings["Fitting"]["Weight"]["RVMH"])


    #       cgmFittingProcedure.updateMarkerWeight("LTHL",settings["Fitting"]["Weight"]["LTHL"])
    #       cgmFittingProcedure.updateMarkerWeight("LTHLD",settings["Fitting"]["Weight"]["LTHLD"])
    #       cgmFittingProcedure.updateMarkerWeight("LPAT",settings["Fitting"]["Weight"]["LPAT"])
    #       cgmFittingProcedure.updateMarkerWeight("LTIBL",settings["Fitting"]["Weight"]["LTIBL"])
    #       cgmFittingProcedure.updateMarkerWeight("RTHL",settings["Fitting"]["Weight"]["RTHL"])
    #       cgmFittingProcedure.updateMarkerWeight("RTHLD",settings["Fitting"]["Weight"]["RTHLD"])
    #       cgmFittingProcedure.updateMarkerWeight("RPAT",settings["Fitting"]["Weight"]["RPAT"])
    #       cgmFittingProcedure.updateMarkerWeight("RTIBL",settings["Fitting"]["Weight"]["RTIBL"])


            osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                              scalingOsim,
                                                              cgmFittingProcedure,
                                                              str(DATA_PATH) )
            acqIK = osrf.run(acqGait,str(DATA_PATH + reconstructFilenameLabelled ))


        # eventual gait acquisition to consider for joint kinematics
        finalAcqGait = acqIK if ik_flag else acqGait

        # --- final pyCGM2 model motion Filter ---
        # use fitted markers
        modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqGait,model,pyCGM2Enums.motionMethod.Sodervisk ,
                                                  markerDiameter=markerDiameter)

        modMotionFitted.compute()


        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,finalAcqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        # detection of traveling axis
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(finalAcqGait,["LASI","LPSI","RASI","RPSI"])


        # absolute angles
        modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqGait,
                                               segmentLabels=["Left HindFoot","Right HindFoot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                eulerSequences=["TOR","TOR", "ROT"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        #---- Body segment parameters----
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # --- force plate handling----
        # find foot  in contact
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(finalAcqGait)
        forceplates.addForcePlateGeneralEvents(finalAcqGait,mappedForcePlate)
        logging.info("Force plate assignment : %s" %mappedForcePlate)

        if args.mfpa is not None:
            if len(args.mfpa) != len(mappedForcePlate):
                raise Exception("[pyCGM2] manual force plate assignment badly sets. Wrong force plate number. %s force plate require" %(str(len(mappedForcePlate))))
            else:
                mappedForcePlate = args.mfpa
                forceplates.addForcePlateGeneralEvents(finalAcqGait,mappedForcePlate)
                logging.warning("Force plates assign manually")

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,finalAcqGait,mappedForcePlate,
                                 leftSegmentLabel="Left HindFoot",
                                 rightSegmentLabel="Right HindFoot").compute()

        #---- Joint kinetics----
        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             finalAcqGait,
                             procedure = idp,
                             projection = momentProjection
                             ).compute(pointLabelSuffix=pointSuffix)

        #---- Joint energetics----
        modelFilters.JointPowerFilter(model,finalAcqGait).compute(pointLabelSuffix=pointSuffix)

        #---- zero unvalid frames ---
        btkTools.applyValidFramesOnOutput(finalAcqGait,validFrames)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(NEXUS,model,finalAcqGait,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,finalAcqGait,["Left-FP","Right-FP"])
        # ========END of the nexus OPERATION if run from Nexus  =========

        if DEBUG:

            NEXUS.SaveTrial(30)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
