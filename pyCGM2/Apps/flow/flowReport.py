
import os
import warnings
import argparse

import pyCGM2;
LOGGER = pyCGM2.LOGGER
LOGGER.setLevel("info")

from pyCGM2.Utils import files
from pyCGM2.flow import settingsHandler 
from pyCGM2.Tools import uiTools
from pyCGM2.Nexus import nexus
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot

from pyCGM2.Lib import emg
from pyCGM2.Report import normativeDatasets


from pyCGM2.Report import report

TEMPLATE_PATH = pyCGM2.PYCGM2_SETTINGS_FOLDER+"templates\\"

import time

def wait_until_closed(path, timeout=300):
    """Attend que le fichier soit libérable (Word fermé)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            # tentative d'ouverture exclusive : échoue tant que Word tient le verrou
            with open(path, "r+b"):
                return True
        except (PermissionError, OSError):
            time.sleep(0.5)
    return False


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='Process flow report from Eclipse')
        parser.add_argument('-u', '--userSettings', type=str,
                            help='userSettings file name, should be in the data folder',
                            required=True)
        parser.add_argument('-dp', '--data_path', type=str,
                            default=None)       
        parser.add_argument('-c', '--conditions', nargs='*', help='list of conditions',required=False)


        args = parser.parse_args()
    


    userSettings = args.userSettings
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
    report_path = processedPath+"ReportFiles\\"
    files.createDir(report_path)


    # recherche du fichier h5
    ipp =  userSettings["SubjectInfo"]["Ipp"]
    sessionNumber = userSettings["VisitInfo"]["SessionNumber"]
    session_dir = f"Session {sessionNumber}"
    modelVersion =  userSettings["Global"]["ModelVersion"]



    conditionId = "Condition1"
    conditionDetails="marche spontanee"
    analysisIndex=1

    modelledFilenames = settingsHandler.get_trials_by_condition(userSettings,conditionId)
    analysisInstance = analysis.makeAnalysis(processedPath, modelledFilenames, type="Gait")

    settingsHandler.get_emg_configuration(userSettings,"Condition1")
    representativeEmgTrial = settingsHandler.get_EmgRepresentative(userSettings,"Condition1")
    
    nds = normativeDatasets.NormativeData("CGM23-msm","Spont")

    GAITREPORT = report.Report()
    GAITREPORT.add_analysisSection("Analyse Marche Spontanée") # add a new section to the report

    try:
        pst1_fig, pst1_file = plot.plot_spatioTemporal(processedPath, analysisInstance, 
                                                   OUT_PATH = report_path,
                                                   exportPng=True,
                                                   outputName=conditionDetails,show=False)
    except Exception as e:
            LOGGER.logger.warning(f"Failed to generate spatio-temporel plot: {e}")
    else:
        GAITREPORT.add_pageToAnalysis(analysisIndex, pst1_file , "spatio-temporel")

    #-----Kinematic----
    try:
        kine1_fig, kine1_file = plot.plot_DescriptiveKinematic(processedPath,analysisInstance,"LowerLimb",nds,
                                                           OUT_PATH = report_path,
                                                           exportPng=True, 
                                                           outputName=conditionDetails,show=False)
    except Exception as e:
        LOGGER.logger.warning(f"Failed to generate kinematic plot: {e}")
    else:
        GAITREPORT.add_pageToAnalysis(analysisIndex, kine1_file , "Cinematique")
    

    #-----Kinetic----    
    try:
        dyn1_fig, dyn1_file = plot.plot_DescriptiveKinetic(processedPath,analysisInstance,"LowerLimb",nds,
                                                       OUT_PATH = report_path,
                                                       exportPng=True,
                                                       outputName=conditionDetails,show=False)
    except Exception as e:
        LOGGER.logger.warning(f"Failed to generate kinetic plot: {e}")
    else:
        GAITREPORT.add_pageToAnalysis(analysisIndex, dyn1_file , "Dynamique")        

    #-----GRF----
    try:
        grf1_fig, grf1_file = plot.plot_DescriptiveGRF(processedPath,analysisInstance,None,
                                                   OUT_PATH = report_path,
                                                   exportPng=True,
                                                   outputName=conditionDetails,show=False)
    except Exception as e:
        LOGGER.logger.warning(f"Failed to generate Ground reaction force plot: {e}")
    else:
        GAITREPORT.add_pageToAnalysis(analysisIndex, grf1_file , "Dynamique-reaction")
    
    #----EMG----
    
    
    try:
        files.copyPaste(data_path+"emg.settings", processedPath+"emg.settings")
        emg1_figs, emg1_files = plot.plotTemporalEMG(processedPath,representativeEmgTrial,rectify=True,
                                        show=False,
                                        exportPng=True,
                                        outputName=conditionDetails,
                                        OUT_PATH=report_path)
    except Exception as e:
        LOGGER.logger.warning(f"Failed to generate emg plot: {e}")
    else:
        GAITREPORT.add_pageToAnalysis(analysisIndex, emg1_files , "EMG")


    #-----Scores----
    try:
        map1_fig, map1_file = plot.plot_MAP(processedPath,analysisInstance,nds,
                                        OUT_PATH = report_path,
                                        exportPng=True,
                                        outputName=conditionDetails,show=False)
    except Exception as e:
        LOGGER.logger.warning(f"Failed to generate scores plot: {e}")
    else:
        GAITREPORT.add_pageToAnalysis(analysisIndex, map1_file , "Scores")



    #-----PFKE----
    try:
        pfke, pfke_file = plot.plotPFKE(processedPath,
                                        analysisInstance,nds,
                                        OUT_PATH = report_path,
                                        exportPng=True,
                                        outputName=conditionDetails,show=False)
    except Exception as e:
        LOGGER.logger.warning(f"Failed to generate pfke plot: {e}")
    else:
        GAITREPORT.add_pageToAnalysis(analysisIndex, pfke_file , "Plantar flexor - knee extensor")
    
    
    #-----Gage----
    # try:
    #     gage_fig, gage_file = plot.plotSaggitalGagePanel(processedPath,analysisInstance,nds,emgType="Envelope",
    #                                                     show=False,
    #                                                     exportPng=True,outputName=conditionDetails,
    #                                                     OUT_PATH=report_path)
    # except Exception as e:
    #     LOGGER.logger.warning(f"Failed to generate kinetic plot: {e}")
    # else:
    #     GAITREPORT.add_pageToAnalysis(analysisIndex, dyn1_file , "Dynamique")


    #-----msucleLength----
    # muscleLength_figs,muscleLength_files = plot.plot_DescriptiveMuscleLength(processedPath,analysisInstance,normativeDataset_ml,
    #             normalizedSuffix= "PoseNormalized",
    #             exportPng=True,outputName=conditionDetails,OUT_PATH=REPORT_PATH_OUT,
    #             show=False, analyticalData = analytics
    #             )





    # # Generation
    files.createDir(data_path+"Doc")
 
    template = TEMPLATE_PATH+'gaitreport_Nantes.docxtpl'

    GAITREPORT.generateWord(template, data_path+"Doc\\", report_path, processedInfos=userSettings)
    os.startfile(data_path+"Doc//gaitReport.docx")

    source =  data_path+"Doc//gaitReport.docx"
 
 
    if wait_until_closed(source):
        destinationDir = f"{pyCGM2.FLOW_GAITLAB_FOLDER_PATH}{ipp}\\"
        files.copyPaste(source, destinationDir+session_dir+"\\Doc\\gaitReport.docx")



if __name__ == "__main__":
    main()