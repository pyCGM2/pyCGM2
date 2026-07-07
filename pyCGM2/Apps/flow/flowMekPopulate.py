# coding: utf-8
import os
from pyCGM2.Utils import files
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import pyCGM2 
LOGGER = pyCGM2.LOGGER
# LOGGER.setLevel("info")
# LOGGER.set_file_handler("pyCGM2-Mek.log")



from pyCGM2.Utils import files
from pyCGM2.Tools import uiTools
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.flow import settingsHandler
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Nexus import nexus


import argparse
from argparse import Namespace
from pathlib import Path


try:
    from pyCGM2.Mek.mek import mekOperations
    from pyCGM2.Mek import mekConstants
    from pyCGM2.Mek.mek import mekInit
    from pyCGM2.Mek.mek import mekExtract
    from pyCGM2.Mek.mek import mekNormalize
    from pyCGM2.Mek.lib import mekLib
except ImportError as e:
    LOGGER.logger.error(f"Error importing Mek modules: {e}. Mek functionalities will not be available.")
    raise e    


def main(args=None):


    if args is None:
        parser = argparse.ArgumentParser(description='Process flow report from Eclipse')
        parser.add_argument('-u', '--userSettings', type=str,
                            help='userSettings file name, should be in the data folder',
                            required=True)
        parser.add_argument('-dp', '--data_path', type=str,
                            default=None)  

        parser.add_argument('-a', '--analysisID', type=int,
                            help='analysis identification',
                            required=False)
        parser.add_argument('-up', '--update', 
                            action='store_true', help='enable update of the analysis')
       
        parser.add_argument('-c', '--conditions', nargs='*', help='list of conditions',required=False)


        args = parser.parse_args()
    


    userSettings = args.userSettings
    analysisId = args.analysisID if args.analysisID is not None else 1
    updateMode = args.update
    forcedConditions = args.conditions if args.conditions is not None else []
    data_path = args.data_path

    if data_path is None:
        nexusCon = nexus.NexusConnection()
        if nexusCon.isConnected():
            try:
                data_path, trialFilename = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS)
                
            except Exception as e:
                LOGGER.logger.warning(f"No trial  loaded in Nexus: {e}, fallback to ui selection")
                data_path = uiTools.uiGetDir()
        else:
            data_path = uiTools.uiGetDir() 
                            

    # run 
    userSettingsFile = userSettings+".settings" if not userSettings.endswith(".settings") else userSettings
    userSettings = files.openFile(data_path,userSettingsFile)
    userSettingsFileNoExt = userSettingsFile.replace(".settings","")
    processedPath = data_path+f"Processing_{userSettingsFileNoExt}\\"

    subject_path = files.get_parent_directory(data_path)


    # recherche du fichier h5
    ipp =  userSettings["SubjectInfo"]["Ipp"]
    sessionNumber = userSettings["VisitInfo"]["SessionNumber"]
    session_dir = f"Session {sessionNumber}"
    modelVersion =  userSettings["Global"]["ModelVersion"]

    # tratement des conditions et insertion dans l'analyse 1 avec ajout des settings

    h5fileOut = f"{ipp}-storage.h5"
    h5pathFileOut = subject_path + h5fileOut
    if os.path.exists(h5pathFileOut):
        h5pathFile = h5pathFileOut
    else :
        h5pathFile = None


    storage = mekInit.Storage(storagePathFile=h5pathFile,updateFlag=updateMode)
    ds = storage.getStorage()

    continueFlag = True
    if ds.root().exists_group(f"{session_dir}/Analysis {analysisId}"):
        if not updateMode:
            LOGGER.logger.error(f"the group ({session_dir}/Analysis {analysisId}) already exists.")
            raise Exception("groupError")
    else:
        ds.root().create_group(f"{session_dir}/Analysis {analysisId}")



    # add attribute a session
    mekLib.setDictToYamlAttribute(ds,userSettings,  f"{session_dir}/Analysis {analysisId}","flow-userSettings")
    group = ds.root().retrieve_group(f"{session_dir}/Analysis {analysisId}")
    group.create_attribute("userSettingsFile", userSettingsFile)    



    #
    exists = (Path(data_path) / "musculoskeletal_modelling" / "pose_standstill").exists()
    if exists:
        modelVersionShort = modelVersion.replace(".","") 
        muscleLengths0 = opensimIO.OpensimDataFrame(data_path,
                                                    f"musculoskeletal_modelling/pose_standstill/{modelVersionShort}-Pose[standstill]_MuscleAnalysis_Length.sto")
        muscleLengths0Dict = muscleLengths0.dataFrameToDict()

        poseGroup = f"{session_dir}/Analysis {analysisId}/Poses/standstill"

        for muscle in muscleLengths0Dict:
            print(muscle)
            values = muscleLengths0Dict[muscle]
            if ds.root().exists_set(f"{poseGroup}/MTUL/{muscle}"):
                ds.root().retrieve_set(f"{poseGroup}/MTUL/{muscle}").write(values)
            else:
                ds.root().create_set(f"{poseGroup}/MTUL/{muscle}", values)





    conditions = settingsHandler.list_conditions(userSettings)
    for condition in conditions:
        if condition in forcedConditions or forcedConditions == []:

            if not ds.root().exists_group(f"{session_dir}/Analysis {analysisId}/{condition}"):
                ds.root().create_group(f"{session_dir}/Analysis {analysisId}/{condition}")

            trialnames = settingsHandler.get_trials_by_condition(userSettings, condition)

            emgTrialNames = settingsHandler.get_emg_trials_by_condition(userSettings, condition)
            emgConfiguration = settingsHandler.get_emg_configuration(userSettings, condition, outputType="dict")

            task = settingsHandler.get_condition_details(userSettings,condition)["Task"]            

            trials =   list(set(trialnames).union(emgTrialNames))

            if "gait" in task.lower():
                mekOperations.compute_spatio_temporal_parametersOperation( ds.root().retrieve_group(f"{session_dir}/Analysis {analysisId}/{condition}"),  data_path, trials)
            else:
                LOGGER.logger.warning(f"No STP computed for the condition {condition} - task ({task})")   




            mergedLabelContext_env = [f"{label}_Rectify_Env:{context}={context[0]}{muscle}" for label, context, muscle in zip(emgConfiguration["Labels"], emgConfiguration["Contexts"],emgConfiguration["Muscles"]) if muscle is not None]
            mergedLabelContext_rect = [f"{label}_Rectify:{context}={context[0]}{muscle}" for label, context, muscle in zip(emgConfiguration["Labels"], emgConfiguration["Contexts"],emgConfiguration["Muscles"]) if muscle is not None]

            
            scheme = {
                        "Kinematics/Angles": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Right"]]],
                        "Kinetics/Moments": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_MOMENTS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_MOMENTS["Right"]]],
                        "Kinetics/Forces": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_FORCES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_FORCES["Right"]]],
                        "Kinetics/Powers": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_POWERS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_POWERS["Right"]]],
                        "EMG/rectify/Envelop": [[processedPath+filename for filename in emgTrialNames], mergedLabelContext_env],
                        "EMG/rectify": [[processedPath+filename for filename in emgTrialNames], mergedLabelContext_rect],
                }

            
            if modelVersion in ["CGM2.2","CGM2.3"]:
                osimInterface = opensimInterface.osimInterface(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\"+modelVersion.replace(".","")+"\\", "pycgm2-gait2392_simbody.osim")
                muscleDict = osimInterface.getMuscles_bySide(addToName="[MuscleLength]")
                
                scheme["MuscleKinematics/MTUL"] = [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in muscleDict["Left"]] + [it+":Right" for it  in muscleDict["Right"]]]

            filter = mekExtract.mekExtractFilter(ds,group=f"{session_dir}/Analysis {analysisId}/{condition}")
            filter.run(scheme)


            normalize_filter = mekNormalize.mekNormalizeFilter(ds,group=f"{session_dir}/Analysis {analysisId}/{condition}")
            normalize_filter.run(scheme,cropToForcePlateGroups =["Kinetics/Moments", "Kinetics/Forces"]) 


            if not storage.updateFlag:
                ds.dump(h5pathFileOut)

            
            LOGGER.logger.info(f"✅ Session : { session_dir}-analysis {analysisId}-condition {condition}  processed successfully")

    
    # file copied to server
    LOGGER.logger.info(f"copy to server")
    try:
        import companion
        destination = f"{companion.FLOW_PUSH_FOLDER_PATH}{ipp}"
        copiedFlag = files.robocopyFile(subject_path, destination, h5fileOut)
    except:
        pass
    


if __name__ == "__main__":

    main(args=None)
