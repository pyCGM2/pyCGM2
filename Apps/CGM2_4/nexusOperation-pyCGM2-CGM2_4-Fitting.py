# -*- coding: utf-8 -*-

import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
from shutil import copyfile
import json
from collections import OrderedDict
import argparse

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus


# pyCGM2 libraries
from pyCGM2.Tools import btkTools,nexusTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm2, modelFilters, forceplates,bodySegmentParameters
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Utils import fileManagement

from pyCGM2 import viconInterface



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
        inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_4-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)


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
        if not os.path.isfile(DATA_PATH + subject + "-pyCGM2.model"):
            raise Exception ("%s-pyCGM2.model file doesn't exist. Run Calibration operation"%subject)
        else:
            f = open(DATA_PATH + subject + '-pyCGM2.model', 'r')
            model = cPickle.load(f)
            f.close()

        # --------------------------CHECKING -----------------------------------
        # check model
        logging.info("loaded model : %s" %(model.version))
        if model.version != "CGM2.4":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM2.3e calibration pipeline"%subject)

        # --------------------------SESSION INFOS -----------------------------
        # info file
        infoSettings = fileManagement.manage_pycgm2SessionInfos(DATA_PATH,subject)

        #  translators management
        translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")
        if not translators:
           translators = inputs["Translators"]

        # --------------------------CONFIG -----------------------------

        if args.markerDiameter is not None:
            markerDiameter = float(args.markerDiameter)
            logging.warning("marker diameter forced : %s", str(float(args.markerDiameter)))
        else:
            markerDiameter = float(inputs["Global"]["Marker diameter"])


        if args.check:
            pointSuffix="cgm2.4"
        else:
            if args.pointSuffix is not None:
                pointSuffix = args.pointSuffix
            else:
                pointSuffix = inputs["Global"]["Point suffix"]

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
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your inputs. choice is Proximal, Distal or Global")

        else:
            if inputs["Fitting"]["Moment Projection"] == "Distal":
                momentProjection = pyCGM2Enums.MomentProjection.Distal
            elif inputs["Fitting"]["Moment Projection"] == "Proximal":
                momentProjection = pyCGM2Enums.MomentProjection.Proximal
            elif inputs["Fitting"]["Moment Projection"] == "Global":
                momentProjection = pyCGM2Enums.MomentProjection.Global
            elif inputs["Fitting"]["Moment Projection"] == "JCS":
                momentProjection = pyCGM2Enums.MomentProjection.JCS
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your inputs. choice is Proximal, Distal or Global")

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
            cgmFittingProcedure.updateMarkerWeight("LASI",inputs["Fitting"]["Weight"]["LASI"])
            cgmFittingProcedure.updateMarkerWeight("RASI",inputs["Fitting"]["Weight"]["RASI"])
            cgmFittingProcedure.updateMarkerWeight("LPSI",inputs["Fitting"]["Weight"]["LPSI"])
            cgmFittingProcedure.updateMarkerWeight("RPSI",inputs["Fitting"]["Weight"]["RPSI"])
            cgmFittingProcedure.updateMarkerWeight("RTHI",inputs["Fitting"]["Weight"]["RTHI"])
            cgmFittingProcedure.updateMarkerWeight("RKNE",inputs["Fitting"]["Weight"]["RKNE"])
            cgmFittingProcedure.updateMarkerWeight("RTIB",inputs["Fitting"]["Weight"]["RTIB"])
            cgmFittingProcedure.updateMarkerWeight("RANK",inputs["Fitting"]["Weight"]["RANK"])
            cgmFittingProcedure.updateMarkerWeight("RHEE",inputs["Fitting"]["Weight"]["RHEE"])
            cgmFittingProcedure.updateMarkerWeight("RTOE",inputs["Fitting"]["Weight"]["RTOE"])
            cgmFittingProcedure.updateMarkerWeight("RCUN",inputs["Fitting"]["Weight"]["RCUN"])
            cgmFittingProcedure.updateMarkerWeight("RD1M",inputs["Fitting"]["Weight"]["RD1M"])
            cgmFittingProcedure.updateMarkerWeight("RD5M",inputs["Fitting"]["Weight"]["RD5M"])

            cgmFittingProcedure.updateMarkerWeight("LTHI",inputs["Fitting"]["Weight"]["LTHI"])
            cgmFittingProcedure.updateMarkerWeight("LKNE",inputs["Fitting"]["Weight"]["LKNE"])
            cgmFittingProcedure.updateMarkerWeight("LTIB",inputs["Fitting"]["Weight"]["LTIB"])
            cgmFittingProcedure.updateMarkerWeight("LANK",inputs["Fitting"]["Weight"]["LANK"])
            cgmFittingProcedure.updateMarkerWeight("LHEE",inputs["Fitting"]["Weight"]["LHEE"])
            cgmFittingProcedure.updateMarkerWeight("LTOE",inputs["Fitting"]["Weight"]["LTOE"])
            cgmFittingProcedure.updateMarkerWeight("LCUN",inputs["Fitting"]["Weight"]["LCUN"])
            cgmFittingProcedure.updateMarkerWeight("LD1M",inputs["Fitting"]["Weight"]["LD1M"])
            cgmFittingProcedure.updateMarkerWeight("LD5M",inputs["Fitting"]["Weight"]["LD5M"])


            cgmFittingProcedure.updateMarkerWeight("LTHIAP",inputs["Fitting"]["Weight"]["LTHIAP"])
            cgmFittingProcedure.updateMarkerWeight("LTHIAD",inputs["Fitting"]["Weight"]["LTHIAD"])
            cgmFittingProcedure.updateMarkerWeight("LTIBAP",inputs["Fitting"]["Weight"]["LTIBAP"])
            cgmFittingProcedure.updateMarkerWeight("LTIBAD",inputs["Fitting"]["Weight"]["LTIBAD"])
            cgmFittingProcedure.updateMarkerWeight("RTHIAP",inputs["Fitting"]["Weight"]["RTHIAP"])
            cgmFittingProcedure.updateMarkerWeight("RTHIAD",inputs["Fitting"]["Weight"]["RTHIAD"])
            cgmFittingProcedure.updateMarkerWeight("RTIBAP",inputs["Fitting"]["Weight"]["RTIBAP"])
            cgmFittingProcedure.updateMarkerWeight("RTIBAD",inputs["Fitting"]["Weight"]["RTIBAD"])

    #       cgmFittingProcedure.updateMarkerWeight("LTHL",inputs["Fitting"]["Weight"]["LTHL"])
    #       cgmFittingProcedure.updateMarkerWeight("LTHLD",inputs["Fitting"]["Weight"]["LTHLD"])
    #       cgmFittingProcedure.updateMarkerWeight("LPAT",inputs["Fitting"]["Weight"]["LPAT"])
    #       cgmFittingProcedure.updateMarkerWeight("LTIBL",inputs["Fitting"]["Weight"]["LTIBL"])
    #       cgmFittingProcedure.updateMarkerWeight("RTHL",inputs["Fitting"]["Weight"]["RTHL"])
    #       cgmFittingProcedure.updateMarkerWeight("RTHLD",inputs["Fitting"]["Weight"]["RTHLD"])
    #       cgmFittingProcedure.updateMarkerWeight("RPAT",inputs["Fitting"]["Weight"]["RPAT"])
    #       cgmFittingProcedure.updateMarkerWeight("RTIBL",inputs["Fitting"]["Weight"]["RTIBL"])


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
        viconInterface.ViconInterface(NEXUS,model,finalAcqGait,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,finalAcqGait,["Left-FP","Right-FP"])

        # ========END of the nexus OPERATION if run from Nexus  =========

        if DEBUG:

            NEXUS.SaveTrial(30)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
