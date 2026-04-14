# coding: utf-8
import os
from pyCGM2.Mek.mek import mekOperations, mekTransform
from pyCGM2.Utils import files
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import pyCGM2 
LOGGER = pyCGM2.LOGGER
# LOGGER.setLevel("info")
# LOGGER.set_file_handler("pyCGM2-Mek.log")



from pyCGM2.Utils import files
from pyCGM2.flow import flowFilters
from pyCGM2.flow.procedures import eclipseFlowProcedure
from pyCGM2.Tools import uiTools

from pyCGM2.flow import settingsHandler
from pyCGM2.Lib import emg
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Nexus import nexus


import argparse
from argparse import Namespace

try:
    from pyCGM2.Mek.mek import mekOperations
    from pyCGM2.Mek import mekConstants
    from pyCGM2.Mek.mek import mekInit
    from pyCGM2.Mek.mek import mekExtract
    from pyCGM2.Mek.mek import mekNormalize
    from pyCGM2.Mek.mek import mekFlow
    from pyCGM2.Mek.mek import mekTransform
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
     
        parser.add_argument('-c', '--conditions', nargs='*', help='list of conditions',required=False)
        parser.add_argument('--ui', action='store_true',
                    help='open PySide6 dialog to fill arguments')
      

        args = parser.parse_args()
    
    

    userSettings = args.userSettings
    forcedConditions = args.conditions if args.conditions is not None else []
    data_path = args.data_path

    h5fileOut = f"storage.h5"


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


    h5pathFileOut = data_path + h5fileOut
    h5pathFile = h5pathFileOut if os.path.exists(h5pathFileOut) else None

    storagefilter = mekInit.mekInitStorageFilter(storagePathFile=h5pathFile)
    ds = storagefilter.getStorage()


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

    proc = mekTransform.mekViconTrialTransformProcedure(cgmVersion=modelVersion)

    conditions = settingsHandler.list_conditions(userSettings)
    for condition in conditions:
        if condition in forcedConditions or forcedConditions == []:

            trialnames = settingsHandler.get_trials_by_condition(userSettings, condition)    

            for trial in trialnames:
                filter = mekTransform.mekTrialTransformFilter(ds,procedure=proc)
                filter.run(data_path,trial)
        
        LOGGER.logger.info(f"✅ Session : { session_dir}-condition {condition}  imported successfully")

    
    if not storagefilter.updateFlag:
        ds.dump(h5pathFileOut)

    



if __name__ == "__main__":

    main(args=None)
