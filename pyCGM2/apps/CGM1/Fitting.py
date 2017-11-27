# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import argparse

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Model import modelFilters, bodySegmentParameters
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Utils import files
from pyCGM2.apps import cgmUtils


if __name__ == "__main__":

    DEBUG = False
    parser = argparse.ArgumentParser(description='CGM1 Fitting')
    parser.add_argument('--infoFile', type=str, help='infoFile')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-mfpa',type=str,  help='manual assignment of force plates')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('-fs','--fileSuffix', type=str, help='suffix of output file')
    args = parser.parse_args()




    # --------------------GOBAL SETTINGS ------------------------------

    # global setting ( in user/AppData)
    settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")


    # --------------------SESSION SETTINGS ------------------------------
    if DEBUG:
        DATA_PATH = "C:\\Users\\HLS501\\Google Drive\\Paper_for BJSM\\BJSM_trials\\FMS_Screening\\15KUFC01\\Session 2\\"
        infoFilename = "pyCGM2.info"
        info = files.openJson(DATA_PATH,infoFilename)


    else:
        DATA_PATH =os.getcwd()+"\\"
        infoFilename = "pyCGM2.info" if args.infoFile is None else  args.infoFile
        info = files.openJson(DATA_PATH,infoFilename)



    # --------------------------CONFIG ------------------------------------
    argsManager = cgmUtils.argsManager_cgm1(settings,args)
    markerDiameter = argsManager.getMarkerDiameter()
    pointSuffix = argsManager.getPointSuffix("cgm1")
    momentProjection =  argsManager.getMomentProjection()
    mfpa = argsManager.getManualForcePlateAssign()


    # --------------------------TRANSLATORS ------------------------------------
    #  translators management
    translators = files.manage_pycgm2Translators(DATA_PATH,"CGM1.translators")
    if not translators:
       translators = settings["Translators"]


    # ------------------ pyCGM2 MODEL -----------------------------------
    model = files.loadModel(DATA_PATH,None)

    # --------------------------CHECKING -----------------------------------
    # check model is the CGM1
    logging.info("loaded model : %s" %(model.version ))
    if model.version != "CGM1.0":
        raise Exception ("pyCGM2.model file was not calibrated from the CGM1.0 calibration pipeline"%model.version)

    # --------------------------MODELLLING--------------------------
    motionTrials = info["Modelling"]["Trials"]["Motion"]


    for trial in motionTrials:

        acqGait = btkTools.smartReader(str(DATA_PATH + trial))

        btkTools.checkMultipleSubject(acqGait)
        acqGait =  btkTools.applyTranslators(acqGait,translators)
        validFrames,vff,vlf = btkTools.findValidFrames(acqGait,cgm.CGM1LowerLimbs.MARKERS)

        scp=modelFilters.StaticCalibrationProcedure(model) # procedure

        # ---Motion filter----
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                  markerDiameter=markerDiameter,
                                                  viconCGM1compatible=True)

        modMotion.compute()


        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        # detection of traveling axis
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","RASI","RPSI","LPSI"])

        # absolute angles
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                eulerSequences=["TOR","TOR", "TOR"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        #---- Body segment parameters----
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # --- force plate handling----
        # find foot  in contact
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        forceplates.addForcePlateGeneralEvents(acqGait,mappedForcePlate)
        logging.info("Force plate assignment : %s" %mappedForcePlate)

        if mfpa is not None:
            if len(mfpa) != len(mappedForcePlate):
                raise Exception("[pyCGM2] manual force plate assignment badly sets. Wrong force plate number. %s force plate require" %(str(len(mappedForcePlate))))
            else:
                mappedForcePlate = mfpa
                logging.warning("Force plates assign manually")
                forceplates.addForcePlateGeneralEvents(acqGait,mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute()

        #---- Joint kinetics----
        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = momentProjection,
                             viconCGM1compatible=True
                             ).compute(pointLabelSuffix=pointSuffix)

        #---- Joint energetics----
        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix=pointSuffix)

        # ----------------------SAVE-------------------------------------------
        # new static file
        if args.fileSuffix is not None:
            btkTools.smartWriter(acqGait, str(DATA_PATH+trial[:-4]+"-modelled-"+args.fileSuffix+".c3d"))
        else:
            btkTools.smartWriter(acqGait, str(DATA_PATH+trial[:-4]+"-modelled.c3d"))
