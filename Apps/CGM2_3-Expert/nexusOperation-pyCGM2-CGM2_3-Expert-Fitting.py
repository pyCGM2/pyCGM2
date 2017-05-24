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

# openMA
#import ma.io
#import ma.body

#btk
#import btk


# pyCGM2 libraries
from pyCGM2.Tools import btkTools,nexusTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm,cgm2, modelFilters, forceplates,bodySegmentParameters
from pyCGM2.Model.Opensim import opensimFilters
#
from pyCGM2 import viconInterface



if __name__ == "__main__":


    DEBUG = False

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='CGM2-3-Expert Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-mfpa',type=str,  help='manual assignment of force plates')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')  
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    args = parser.parse_args()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.MAIN_BENCHMARK_PATH + "True equinus\\S01\\CGM2.3Expert\\"
            reconstructFilenameLabelledNoExt = "gait trial 01"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )

        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()


        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)


        # --- btk acquisition ----
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        #   check if acq was saved with only one  activated subject
        if acqGait.GetPoint(0).GetLabel().count(":"):
            raise Exception("[pyCGM2] Your Trial c3d was saved with two activate subject. Re-save it with only one before pyCGM2 calculation") 


        validFrames,vff,vlf = btkTools.findValidFrames(acqGait,cgm.CGM1LowerLimbs.MARKERS) # TODO - fix with cgm2

#        # --relabel PIG output if processing previously---
#        n_angles,n_forces ,n_moments,  n_powers = btkTools.getNumberOfModelOutputs(acqGait)
#        if any([n_angles,n_forces ,n_moments,  n_powers])==1:            
#            cgm.CGM.reLabelOldOutputs(acqGait) 

        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects,"LASI")
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 INPUT FILES ------------------------------
        if not os.path.isfile(DATA_PATH + subject + "-CGM2_3-Expert-pyCGM2.model"):
            raise Exception ("%s-CGM2_3-Expert-pyCGM2.model file doesn't exist. Run Calibration operation"%subject)
        else:
            f = open(DATA_PATH + subject + '-CGM2_3-Expert-pyCGM2.model', 'r')
            model = cPickle.load(f)
            f.close()
            

        # global setting ( in user/AppData)
        inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_3-Expert-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)

        # info file
        if not os.path.isfile( DATA_PATH + subject+"-pyCGM2.info"):
            copyfile(str(pyCGM2.CONFIG.PYCGM2_SESSION_SETTINGS_FOLDER+"pyCGM2.info"), str(DATA_PATH + subject+"-pyCGM2.info"))
            logging.warning("Copy of pyCGM2.info from pyCGM2 Settings folder")
            infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)
        else:
            infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)

        # ---- configuration parameters ----
        if args.markerDiameter is not None: 
            markerDiameter = float(args.markerDiameter)
            logging.warning("marker diameter forced : %s", str(float(args.markerDiameter)))
        else:
            markerDiameter = float(inputs["Global"]["Marker diameter"])
            
            
        if args.check:
            pointSuffix="cgm2.3-Expert"
        else:
            pointSuffix = inputs["Global"]["Point suffix"]

        if args.proj is not None:        
            if args.proj == "Distal":
                momentProjection = pyCGM2Enums.MomentProjection.Distal
            elif args.proj == "Proximal":
                momentProjection = pyCGM2Enums.MomentProjection.Proximal
            elif args.proj == "Global":
                momentProjection = pyCGM2Enums.MomentProjection.Global
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your inputs. choice is Proximal, Distal or Global")

        else:        
            if inputs["Fitting"]["Moment Projection"] == "Distal":
                momentProjection = pyCGM2Enums.MomentProjection.Distal
            elif inputs["Fitting"]["Moment Projection"] == "Proximal":
                momentProjection = pyCGM2Enums.MomentProjection.Proximal
            elif inputs["Fitting"]["Moment Projection"] == "Global":
                momentProjection = pyCGM2Enums.MomentProjection.Global
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your inputs. choice is Proximal, Distal or Global")      
        
        # --------------------------MODELLLING--------------------------
        
        # --------------------------STATIC FILE WITH TRANSLATORS --------------------------------------
        acqGait =  btkTools.applyTranslators(acqGait,inputs["Translators"])
        
      
        
        # --- initial motion Filter ---
        scp=modelFilters.StaticCalibrationProcedure(model) 
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        #                        ---OPENSIM IK---

        # ---Marker decomp filter----           
        mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
        mtf.decompose()
        
        
        # --- opensim calibration Filter ---
        osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile        
        markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset - expert.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure
    
        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()
         
        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-expert-ikSetUp_template.xml" # ik tl file

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
        
                                                          
        acqIK = osrf.run(acqGait,str(DATA_PATH + reconstructFilenameLabelled ))        
        
        
        
        # --- final pyCGM2 model motion Filter ---
        # use fitted markers             
        modMotionFitted=modelFilters.ModelMotionFilter(scp,acqIK,model,pyCGM2Enums.motionMethod.Sodervisk ,
                                                  markerDiameter=markerDiameter)

        modMotionFitted.compute()


        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,acqIK).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        # detection of traveling axis
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxis(acqIK,"LASI")

        # absolute angles        
        modelFilters.ModelAbsoluteAnglesFilter(model,acqIK,
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
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqIK)
        logging.info("Force plate assignment : %s" %mappedForcePlate)

        if args.mfpa is not None:
            if len(args.mfpa) != len(mappedForcePlate):
                raise Exception("[pyCGM2] manual force plate assignment badly sets. Wrong force plate number. %s force plate require" %(str(len(mappedForcePlate))))
            else:
                mappedForcePlate = args.mfpa
                logging.warning("Force plates assign manually")

        # assembly foot and force plate        
        modelFilters.ForcePlateAssemblyFilter(model,acqIK,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()

        #---- Joint kinetics----
        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqIK,
                             procedure = idp,
                             projection = momentProjection
                             ).compute(pointLabelSuffix=pointSuffix)

        #---- Joint energetics----
        modelFilters.JointPowerFilter(model,acqIK).compute(pointLabelSuffix=pointSuffix)

        #---- zero unvalid frames ---
        btkTools.applyValidFramesOnOutput(acqIK,validFrames)  

        # ----------------------DISPLAY ON VICON-------------------------------
        viconInterface.ViconInterface(NEXUS,model,acqIK,subject,pointSuffix).run()

        # ========END of the nexus OPERATION if run from Nexus  =========

        if DEBUG:

            NEXUS.SaveTrial(30)
    
            
    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
