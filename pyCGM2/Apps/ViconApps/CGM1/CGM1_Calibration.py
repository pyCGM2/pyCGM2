import warnings
import argparse
from pyCGM2.Lib.CGM import cgm1
from pyCGM2.Apps.ViconApps import CgmArgsManager
from pyCGM2.Utils import files
import pyCGM2
import os
from pyCGM2.Tools import btkTools

LOGGER = pyCGM2.LOGGER
warnings.filterwarnings("ignore")


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='CGM1 Calibration')
        parser.add_argument('-l', '--leftFlatFoot', type=int,
                            help='left flat foot option')
        parser.add_argument('-r', '--rightFlatFoot', type=int,
                            help='right flat foot option')
        parser.add_argument('-hf', '--headFlat', type=int,
                            help='head flat option')
        parser.add_argument('-md', '--markerDiameter',
                            type=float, help='marker diameter')
        parser.add_argument('-ps', '--pointSuffix', type=str,
                            help='suffix of the model outputs')
        parser.add_argument('--check', action='store_true',
                            help='force cgm1 as model ouput suffix')
        parser.add_argument('--resetMP', action='store_true',
                            help='reset optional anthropometric parameters')
        parser.add_argument('--forceMP', action='store_true',
                            help='force the use of MP offsets to compute knee and ankle joint centres')
        parser.add_argument('-ae', '--anomalyException',
                            action='store_true', help='raise an exception if an anomaly is detected')
        parser.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)

        args = parser.parse_args()

    NEXUS_PYTHON_CONNECTED = False
    OFFLINE_MODE = False if args.offline is None else True

    if not OFFLINE_MODE:
        try:
            from viconnexusapi import ViconNexus 
            from pyCGM2.Nexus import nexusFilters
            from pyCGM2.Nexus import nexusUtils
            from pyCGM2.Nexus import nexusTools

            NEXUS = ViconNexus.ViconNexus()
            NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
        except:
            LOGGER.logger.error("Vicon nexus not connected")
    else:
        LOGGER.logger.info("You are working in offlinemode")

    if NEXUS_PYTHON_CONNECTED or OFFLINE_MODE: # run Operation

        # --------------------------LOADING ------------------------------------
        if NEXUS_PYTHON_CONNECTED:
            DATA_PATH, calibrateFilenameLabelledNoExt = nexusTools.getTrialName(NEXUS)
            calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"
        else:
            DATA_PATH = os.getcwd()+"\\"
            calibrateFilenameLabelled = args.offline[1]
            if not os.path.exists(DATA_PATH+calibrateFilenameLabelled):
                raise Exception("[pyCGM2]  file [%s] not found in the folder"%(calibrateFilenameLabelled))
       
        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Calibration.log")
        LOGGER.logger.info( "calibration file: "+ calibrateFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH, "CGM1-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm1(settings, args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        headFlat = argsManager.getHeadFlat()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1")

        # --------------------------SUBJECT ------------------------------------
        # Notice : Work with ONE subject by session
        if NEXUS_PYTHON_CONNECTED:
            subjects = NEXUS.GetSubjectNames()
            subject = nexusTools.getActiveSubject(NEXUS)
            Parameters = NEXUS.GetSubjectParamNames(subject)
            required_mp,optional_mp = nexusUtils.getNexusSubjectMp(NEXUS,subject,resetFlag=args.resetMP)
            mpInfo,mpFilename = files.getMpFileContent(DATA_PATH,"mp.pyCGM2",subject)
        else:
            
            subject = args.offline[0]
            if not os.path.exists(DATA_PATH+subject+"-mp.pyCGM2"):
                raise Exception("[pyCGM2]  the mp file [%s] not found in the folder"%(subject+"-mp.pyCGM2"))
            
            mpFilename = subject+"-mp.pyCGM2"
            mpInfo, required_mp, optional_mp = files.loadMp(DATA_PATH,mpFilename)


        #  translators management
        translators = files.getTranslators(DATA_PATH, "CGM1.translators")
        if not translators:
            translators = settings["Translators"]

        # btkAcq builder
        if NEXUS_PYTHON_CONNECTED:
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,calibrateFilenameLabelledNoExt,subject)
            acq = nacf.build()
        else:
            acq=btkTools.smartReader(DATA_PATH+calibrateFilenameLabelled)

        # --------------------------MODELLING PROCESSING -----------------------
        model, acqStatic, detectAnomaly = cgm1.calibrate(DATA_PATH, calibrateFilenameLabelled, translators,
                                                         required_mp, optional_mp,
                                                         leftFlatFoot, rightFlatFoot, headFlat, markerDiameter,
                                                         pointSuffix, forceBtkAcq=acq, anomalyException=args.anomalyException,forceMP=args.forceMP)

        # ----------------------SAVE-------------------------------------------
        #pyCGM2.model
        files.saveModel(model, DATA_PATH, subject)

        # save mp
        files.saveMp(mpInfo, model, DATA_PATH, mpFilename)

        # ----------------------DISPLAY ON VICON-------------------------------
        if NEXUS_PYTHON_CONNECTED:
            nexusUtils.updateNexusSubjectMp(NEXUS, model, subject)
            nexusFilters.NexusModelFilter(NEXUS,
                                        model, acqStatic, subject,
                                        pointSuffix,
                                        staticProcessing=True).run()
        else: 
            btkTools.smartWriter(acqStatic, DATA_PATH+calibrateFilenameLabelled[:-4]+"-offlineProcessed.c3d")

        # ========END of the nexus OPERATION if run from Nexus  =========
    else:
        return 0


if __name__ == "__main__":

    main(args=None)
