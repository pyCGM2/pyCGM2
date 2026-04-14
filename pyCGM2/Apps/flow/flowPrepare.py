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
from pyCGM2.flow import flowFilters
from pyCGM2.flow.procedures import eclipseFlowProcedure
from pyCGM2.Tools import uiTools
from pyCGM2 import connection
from pyCGM2.flow import settingsHandler
from pyCGM2.Lib import emg
from pyCGM2.Nexus import nexus


import argparse
from argparse import Namespace

def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='Process flow report from Eclipse')
        parser.add_argument('-u', '--userSettings', type=str,
                            help='userSettings file name, should be in the data folder',
                            required=False)
        parser.add_argument('-dp', '--data_path', type=str,
                            default=None)  
        parser.add_argument('-c', '--conditions', nargs='*', help='list of conditions',required=False)
        args = parser.parse_args()

    data_path = args.data_path
    userSettings = args.userSettings
    forcedConditions = args.conditions if args.conditions is not None else []

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

    files.createDir(processedPath)


    conditions = settingsHandler.list_conditions(userSettings)
    for condition in conditions:
        if condition in forcedConditions or forcedConditions == []:

            emgConfiguration = settingsHandler.get_emg_configuration(userSettings, condition, outputType="dict")
            emgProcessingParameters = settingsHandler.get_emg_processing(userSettings, condition) 

            trialnames = settingsHandler.get_trials_by_condition(userSettings, condition)    
            emgTrialNames = settingsHandler.get_emg_trials_by_condition(userSettings, condition)


            trials =   list(set(trialnames).union(emgTrialNames))
            for trial in trials:
                files.copyPaste(  data_path+trial, processedPath+trial)

            conditionInfo = settingsHandler.get_condition(userSettings,condition)

            if emgTrialNames != []:     
                emg.processEMG(processedPath,
                                    emgTrialNames,
                                    emgConfiguration["Labels"],
                                    highPassFrequencies = emgProcessingParameters["BandpassFrequencies"],
                                    envelopFrequency= emgProcessingParameters["EnvelopLowpassFrequency"],
                                    fileSuffix=None,
                                    outDataPath=None)




if __name__ == "__main__":

    main(args=None)
