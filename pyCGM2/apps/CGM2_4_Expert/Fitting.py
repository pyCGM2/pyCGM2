# -*- coding: utf-8 -*-

import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
import json
from collections import OrderedDict
import argparse
from pyCGM2.Model.Opensim import opensimFilters

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import  cgm, cgm2, modelFilters, forceplates,bodySegmentParameters
#

from pyCGM2.Utils import fileManagement



if __name__ == "__main__":

    DEBUG = False
    parser = argparse.ArgumentParser(description='CGM2-4-Expert Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-mfpa',type=str,  help='manual assignment of force plates')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')    
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--ik', action='store_true', help='inverse kinematic',default=True)
    args = parser.parse_args()




    # --------------------GOBAL SETTINGS ------------------------------

    # global setting ( in user/AppData)
    inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_4-Expert-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)

    # --------------------SESSION SETTINGS ------------------------------
        # --------------------SESSION SETTINGS ------------------------------
    if DEBUG:
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Datasets Tests\\Florent Moissenet\\sample\\"
        infoSettings = json.loads(open(DATA_PATH + 'pyCGM2.info').read(),object_pairs_hook=OrderedDict)
    else:
        DATA_PATH =os.getcwd()+"\\"
        infoSettings = json.loads(open('pyCGM2.info').read(),object_pairs_hook=OrderedDict)
    



    # --------------------CONFIGURATION ------------------------------
    if args.markerDiameter is not None:
        markerDiameter = float(args.markerDiameter)
        logging.warning("marker diameter forced : %s", str(float(args.markerDiameter)))
    else:
        markerDiameter = float(inputs["Global"]["Marker diameter"])

    if args.check:
        pointSuffix="cgm2.4e"
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



    # --------------------------TRANSLATORS ------------------------------------
    #  translators management
    translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")
    if not translators:
       translators = inputs["Translators"]
       

    # ------------------ pyCGM2 MODEL -----------------------------------

    if not os.path.isfile(DATA_PATH +  "pyCGM2.model"):
        raise Exception ("pyCGM2.model file doesn't exist. Run Calibration operation")
    else:
        f = open(DATA_PATH + 'pyCGM2.model', 'r')
        model = cPickle.load(f)
        f.close()

    # --------------------------CHECKING -----------------------------------    
    # check model is the CGM2.4e
    logging.info("loaded model : %s" %(model.version ))
    if model.version != "CGM2.4e":
        raise Exception ("pyCGM2.model file was not calibrated from the CGM2.4e calibration pipeline"%model.version)

    # --------------------------MODELLLING--------------------------
    motionTrials = infoSettings["Modelling"]["Trials"]["Motion"]


    for trial in motionTrials:

        acqGait = btkTools.smartReader(str(DATA_PATH + trial))

        btkTools.checkMultipleSubject(acqGait)
        acqGait =  btkTools.applyTranslators(acqGait,translators)
        validFrames,vff,vlf = btkTools.findValidFrames(acqGait,cgm.CGM2_4LowerLimbs.MARKERS)

        scp=modelFilters.StaticCalibrationProcedure(model) # procedure

        # ---Motion filter----
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                  markerDiameter=markerDiameter)

        modMotion.compute()


        if args.ik:
            #                        ---OPENSIM IK---
    
            # ---Marker decomp filter----
            mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
            mtf.decompose()
    
    
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
            iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-expert-ikSetUp_template.xml" # ik tl file
    
            cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model,expertMode = True) # procedure
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
            cgmFittingProcedure.updateMarkerWeight("LTHI",inputs["Fitting"]["Weight"]["LTHI"])
            cgmFittingProcedure.updateMarkerWeight("LKNE",inputs["Fitting"]["Weight"]["LKNE"])
            cgmFittingProcedure.updateMarkerWeight("LTIB",inputs["Fitting"]["Weight"]["LTIB"])
            cgmFittingProcedure.updateMarkerWeight("LANK",inputs["Fitting"]["Weight"]["LANK"])
            cgmFittingProcedure.updateMarkerWeight("LHEE",inputs["Fitting"]["Weight"]["LHEE"])
            cgmFittingProcedure.updateMarkerWeight("LTOE",inputs["Fitting"]["Weight"]["LTOE"])
    
            cgmFittingProcedure.updateMarkerWeight("LASI_posAnt",inputs["Fitting"]["Weight"]["LASI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LASI_medLat",inputs["Fitting"]["Weight"]["LASI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LASI_supInf",inputs["Fitting"]["Weight"]["LASI_supInf"])
    
            cgmFittingProcedure.updateMarkerWeight("RASI_posAnt",inputs["Fitting"]["Weight"]["RASI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RASI_medLat",inputs["Fitting"]["Weight"]["RASI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RASI_supInf",inputs["Fitting"]["Weight"]["RASI_supInf"])
    
            cgmFittingProcedure.updateMarkerWeight("LPSI_posAnt",inputs["Fitting"]["Weight"]["LPSI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LPSI_medLat",inputs["Fitting"]["Weight"]["LPSI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LPSI_supInf",inputs["Fitting"]["Weight"]["LPSI_supInf"])
    
            cgmFittingProcedure.updateMarkerWeight("RPSI_posAnt",inputs["Fitting"]["Weight"]["RPSI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RPSI_medLat",inputs["Fitting"]["Weight"]["RPSI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RPSI_supInf",inputs["Fitting"]["Weight"]["RPSI_supInf"])
    
    
            cgmFittingProcedure.updateMarkerWeight("RTHI_posAnt",inputs["Fitting"]["Weight"]["RTHI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTHI_medLat",inputs["Fitting"]["Weight"]["RTHI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTHI_proDis",inputs["Fitting"]["Weight"]["RTHI_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RKNE_posAnt",inputs["Fitting"]["Weight"]["RKNE_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RKNE_medLat",inputs["Fitting"]["Weight"]["RKNE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RKNE_proDis",inputs["Fitting"]["Weight"]["RKNE_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RTIB_posAnt",inputs["Fitting"]["Weight"]["RTIB_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTIB_medLat",inputs["Fitting"]["Weight"]["RTIB_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTIB_proDis",inputs["Fitting"]["Weight"]["RTIB_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RANK_posAnt",inputs["Fitting"]["Weight"]["RANK_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RANK_medLat",inputs["Fitting"]["Weight"]["RANK_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RANK_proDis",inputs["Fitting"]["Weight"]["RANK_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RHEE_supInf",inputs["Fitting"]["Weight"]["RHEE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RHEE_medLat",inputs["Fitting"]["Weight"]["RHEE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RHEE_proDis",inputs["Fitting"]["Weight"]["RHEE_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RTOE_supInf",inputs["Fitting"]["Weight"]["RTOE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RTOE_medLat",inputs["Fitting"]["Weight"]["RTOE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTOE_proDis",inputs["Fitting"]["Weight"]["RTOE_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RCUN_supInf",inputs["Fitting"]["Weight"]["RCUN_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RCUN_medLat",inputs["Fitting"]["Weight"]["RCUN_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RCUN_proDis",inputs["Fitting"]["Weight"]["RCUN_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RD1M_supInf",inputs["Fitting"]["Weight"]["RD1M_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RD1M_medLat",inputs["Fitting"]["Weight"]["RD1M_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RD1M_proDis",inputs["Fitting"]["Weight"]["RD1M_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RD5M_supInf",inputs["Fitting"]["Weight"]["RD5M_supInf"])
            cgmFittingProcedure.updateMarkerWeight("RD5M_medLat",inputs["Fitting"]["Weight"]["RD5M_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RD5M_proDis",inputs["Fitting"]["Weight"]["RD5M_proDis"])
    
    
            cgmFittingProcedure.updateMarkerWeight("LTHI_posAnt",inputs["Fitting"]["Weight"]["LTHI_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTHI_medLat",inputs["Fitting"]["Weight"]["LTHI_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTHI_proDis",inputs["Fitting"]["Weight"]["LTHI_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LKNE_posAnt",inputs["Fitting"]["Weight"]["LKNE_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LKNE_medLat",inputs["Fitting"]["Weight"]["LKNE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LKNE_proDis",inputs["Fitting"]["Weight"]["LKNE_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LTIB_posAnt",inputs["Fitting"]["Weight"]["LTIB_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTIB_medLat",inputs["Fitting"]["Weight"]["LTIB_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTIB_proDis",inputs["Fitting"]["Weight"]["LTIB_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LANK_posAnt",inputs["Fitting"]["Weight"]["LANK_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LANK_medLat",inputs["Fitting"]["Weight"]["LANK_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LANK_proDis",inputs["Fitting"]["Weight"]["LANK_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LHEE_supInf",inputs["Fitting"]["Weight"]["LHEE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LHEE_medLat",inputs["Fitting"]["Weight"]["LHEE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LHEE_proDis",inputs["Fitting"]["Weight"]["LHEE_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LTOE_supInf",inputs["Fitting"]["Weight"]["LTOE_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LTOE_medLat",inputs["Fitting"]["Weight"]["LTOE_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTOE_proDis",inputs["Fitting"]["Weight"]["LTOE_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LCUN_supInf",inputs["Fitting"]["Weight"]["LCUN_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LCUN_medLat",inputs["Fitting"]["Weight"]["LCUN_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LCUN_proDis",inputs["Fitting"]["Weight"]["LCUN_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LD1M_supInf",inputs["Fitting"]["Weight"]["LD1M_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LD1M_medLat",inputs["Fitting"]["Weight"]["LD1M_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LD1M_proDis",inputs["Fitting"]["Weight"]["LD1M_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LD5M_supInf",inputs["Fitting"]["Weight"]["LD5M_supInf"])
            cgmFittingProcedure.updateMarkerWeight("LD5M_medLat",inputs["Fitting"]["Weight"]["LD5M_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LD5M_proDis",inputs["Fitting"]["Weight"]["LD5M_proDis"])
    
    
    
            cgmFittingProcedure.updateMarkerWeight("LTHIAP_posAnt",inputs["Fitting"]["Weight"]["LTHIAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTHIAP_medLat",inputs["Fitting"]["Weight"]["LTHIAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTHIAP_proDis",inputs["Fitting"]["Weight"]["LTHIAP_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LTHIAD_posAnt",inputs["Fitting"]["Weight"]["LTHIAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTHIAD_medLat",inputs["Fitting"]["Weight"]["LTHIAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTHIAD_proDis",inputs["Fitting"]["Weight"]["LTHIAD_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RTHIAP_posAnt",inputs["Fitting"]["Weight"]["RTHIAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTHIAP_medLat",inputs["Fitting"]["Weight"]["RTHIAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTHIAP_proDis",inputs["Fitting"]["Weight"]["RTHIAP_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RTHIAD_posAnt",inputs["Fitting"]["Weight"]["RTHIAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTHIAD_medLat",inputs["Fitting"]["Weight"]["RTHIAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTHIAD_proDis",inputs["Fitting"]["Weight"]["RTHIAD_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LTIBAP_posAnt",inputs["Fitting"]["Weight"]["LTIBAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTIBAP_medLat",inputs["Fitting"]["Weight"]["LTIBAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTIBAP_proDis",inputs["Fitting"]["Weight"]["LTIBAP_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("LTIBAD_posAnt",inputs["Fitting"]["Weight"]["LTIBAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("LTIBAD_medLat",inputs["Fitting"]["Weight"]["LTIBAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("LTIBAD_proDis",inputs["Fitting"]["Weight"]["LTIBAD_proDis"])
    
            cgmFittingProcedure.updateMarkerWeight("RTIBAP_posAnt",inputs["Fitting"]["Weight"]["RTIBAP_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTIBAP_medLat",inputs["Fitting"]["Weight"]["RTIBAP_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTIBAP_proDis",inputs["Fitting"]["Weight"]["RTIBAP_proDis"])
    
    
            cgmFittingProcedure.updateMarkerWeight("RTIBAD_posAnt",inputs["Fitting"]["Weight"]["RTIBAD_posAnt"])
            cgmFittingProcedure.updateMarkerWeight("RTIBAD_medLat",inputs["Fitting"]["Weight"]["RTIBAD_medLat"])
            cgmFittingProcedure.updateMarkerWeight("RTIBAD_proDis",inputs["Fitting"]["Weight"]["RTIBAD_proDis"])
    
    #            cgmFittingProcedure.updateMarkerWeight("LPAT",inputs["Fitting"]["Weight"]["LPAT"])
    #            cgmFittingProcedure.updateMarkerWeight("LPAT_posAnt",inputs["Fitting"]["Weight"]["LPAT_posAnt"])
    #            cgmFittingProcedure.updateMarkerWeight("LPAT_medLat",inputs["Fitting"]["Weight"]["LPAT_medLat"])
    #            cgmFittingProcedure.updateMarkerWeight("LPAT_proDis",inputs["Fitting"]["Weight"]["LPAT_proDis"])
    #
    #            cgmFittingProcedure.updateMarkerWeight("RPAT",inputs["Fitting"]["Weight"]["RPAT"])
    #            cgmFittingProcedure.updateMarkerWeight("RPAT_posAnt",inputs["Fitting"]["Weight"]["RPAT_posAnt"])
    #            cgmFittingProcedure.updateMarkerWeight("RPAT_medLat",inputs["Fitting"]["Weight"]["RPAT_medLat"])
    #            cgmFittingProcedure.updateMarkerWeight("RPAT_proDis",inputs["Fitting"]["Weight"]["RPAT_proDis"])
    #
    #            cgmFittingProcedure.updateMarkerWeight("LTHLD",inputs["Fitting"]["Weight"]["LTHLD"])
    #            cgmFittingProcedure.updateMarkerWeight("LTHLD_posAnt",inputs["Fitting"]["Weight"]["LTHLD_posAnt"])
    #            cgmFittingProcedure.updateMarkerWeight("LTHLD_medLat",inputs["Fitting"]["Weight"]["LTHLD_medLat"])
    #            cgmFittingProcedure.updateMarkerWeight("LTHLD_proDis",inputs["Fitting"]["Weight"]["LTHLD_proDis"])
    #
    #            cgmFittingProcedure.updateMarkerWeight("RTHLD",inputs["Fitting"]["Weight"]["RTHLD"])
    #            cgmFittingProcedure.updateMarkerWeight("RTHLD_posAnt",inputs["Fitting"]["Weight"]["RTHLD_posAnt"])
    #            cgmFittingProcedure.updateMarkerWeight("RTHLD_medLat",inputs["Fitting"]["Weight"]["RTHLD_medLat"])
    #            cgmFittingProcedure.updateMarkerWeight("RTHLD_proDis",inputs["Fitting"]["Weight"]["RTHLD_proDis"])
    
    
            osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                              scalingOsim,
                                                              cgmFittingProcedure,
                                                              str(DATA_PATH) )
    
    
            acqIK = osrf.run(acqGait,str(DATA_PATH + trial ))

        # eventual gait acquisition to consider for joint kinematics
        finalAcqGait = acqIK if args.ik else acqGait

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
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
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
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()

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

        # ----------------------SAVE-------------------------------------------
        btkTools.smartWriter(finalAcqGait, str(DATA_PATH+trial[:-4]+"-modelled.c3d"))
