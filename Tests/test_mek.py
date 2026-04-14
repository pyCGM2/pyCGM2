# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_mek.py::Test_mek::test_gaitScheme
from logging import root
import sys
import time
import trace
from tracemalloc import start
from tracemalloc import start
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

import pyCGM2

from pyCGM2 import normativeDataHelpers
from pyCGM2.EMG import normalActivation
from pyCGM2.Tools import btkTools


LOGGER = pyCGM2.LOGGER
from pyCGM2.Lib import emg
from pyCGM2.Utils import files

LAYOUT_PATH = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Mek\\layout\\"

from pyCGM2.Utils import utils
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.flow import settingsHandler



try:
    import moveck
    MOVECK_AVAILABLE = True
except ImportError:
    MOVECK_AVAILABLE = False
    LOGGER.warning("moveck pipe is not installed")




if MOVECK_AVAILABLE:

    from pyCGM2.Mek import mekConstants
    from pyCGM2.Mek.mek import mekInit
    from pyCGM2.Mek.mek import mekExtract
    from pyCGM2.Mek.mek import mekNormalize
    from pyCGM2.Mek.mek import mekFlow
    from pyCGM2.Mek.lib import mekLib
    from pyCGM2.Mek.mek import mekTools
    from pyCGM2.Mek.mek import mekPlot
    from pyCGM2.Mek.mek import mekOperations
    from pyCGM2.Mek.mek import mekTransform
    from pyCGM2.Mek.mek import mekScore


    class Test_mekImporter:
        
        def test_mekViconTransform(self):

            path = pyCGM2.TEST_DATA_PATH + "NantesSamples\AQM Adultes\\BOUCHE Alain\\Session 1\\"
            modelledFilenames = ["20260203-AB-PRE-S-NNCN-dyn 01.c3d"]

            ds = moveck.data_store()

            proc = mekTransform.mekViconTrialTransformProcedure(cgmVersion="CGM2.3")
            filter = mekTransform.mekTrialTransformFilter(ds,procedure=proc)
            filter.run(path,modelledFilenames[0])

            ds.dump("storageViconTransformer.h5")
            
           




    class Test_mekScheme:
        def test_gaitScheme(self):

            # path = pyCGM2.TEST_DATA_PATH + "mek\\gaitdata\\"
            # modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]


            data_path = pyCGM2.TEST_DATA_PATH + "NantesSamples\\AQM Enfants\\BILLAUD Maxence\\Session 2\\"
           # modelledFilenames = ["20201209-AO-PONC-S-NNNN-dyn 05.c3d","20201209-AO-PONC-S-NNNN-dyn 06.c3d"]

            userSettingsFile = "CGM23_v2.settings"
            userSettings = files.openFile(data_path,"CGM23_v2.settings")
            userSettingsFileNoExt = userSettingsFile.replace(".settings","")
            processedPath = data_path+f"Processing_{userSettingsFileNoExt}_test\\"

            files.createDir(processedPath)


            condition = "Condition1"
            session_dir =  "Session 2"
            analysisId="1"

            ## preparation -------

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
            
            ## populate -------

            mergedLabelContext_env = [f"{label}_Rectify_Env:{context}={context[0]}{muscle}" for label, context, muscle in zip(emgConfiguration["Labels"], emgConfiguration["Contexts"],emgConfiguration["Muscles"]) if muscle is not None]
            mergedLabelContext_rect = [f"{label}_Rectify:{context}={context[0]}{muscle}" for label, context, muscle in zip(emgConfiguration["Labels"], emgConfiguration["Contexts"],emgConfiguration["Muscles"]) if muscle is not None]

            
            # if variableName=="LRECFEM":
            #                 import ipdb; ipdb.set_trace()
            #                 pos,burstDuration=normalActivation.getNormalBurstActivity_fromCycles(normalActivationLabel,cycleIt.firstFrame,cycleIt.begin, cycleIt.m_contraFO, cycleIt.end, cycleIt.appf)


            scheme = {
                        "Kinematics/Angles": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Right"]]],
                        "Kinetics/Moments": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_MOMENTS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_MOMENTS["Right"]]],
                        "Kinetics/Forces": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_FORCES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_FORCES["Right"]]],
                        "Kinetics/Powers": [[processedPath+filename for filename in trialnames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_POWERS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_POWERS["Right"]]],
                        "EMG/rectify/Envelop": [[processedPath+filename for filename in emgTrialNames], mergedLabelContext_env],
                        "EMG/rectify": [[processedPath+filename for filename in emgTrialNames], mergedLabelContext_rect],
                }


            storage = mekInit.Storage()
            ds = storage.getStorage()
            ds.root().create_group(f"{session_dir}/Analysis {analysisId}/{condition}")


            mekLib.setDictToYamlAttribute(ds,userSettings,  f"{session_dir}/Analysis {analysisId}","flow-userSettings")


            # osimInterface = opensimInterface.osimInterface(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "pycgm2-gait2354_simbody.osim")
            # muscleDict = osimInterface.getMuscles_bySide(addToName="[MuscleLength]")


            mekOperations.compute_spatio_temporal_parametersOperation( ds.root().retrieve_group(f"{session_dir}/Analysis {analysisId}/{condition}"), 
                                                                       data_path, trials)

            filter = mekExtract.mekExtractFilter(ds,group=f"{session_dir}/Analysis {analysisId}/{condition}")
            filter.run(scheme)    


            normalize_filter = mekNormalize.mekNormalizeFilter(ds,group=f"{session_dir}/Analysis {analysisId}/{condition}")
            normalize_filter.run(scheme,cropToForcePlateGroups =["Kinetics/Moments", "Kinetics/Forces"]) 

            ds.dump("storageGaitScheme.h5")
            

    class Test_mekLib:
        def test_iter(self):
            path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1")

            for path, set_obj in mekTools.iter_sets(group):
                print(f"set path  : {path}")

            for path, set_obj in mekTools.iter_grp(group):
                print(f"group path  : {path}")



        def test_gather(self):
            path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"session-storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1/Normalize/Kinematics/Angles")
            cycleValues ,attrs = mekLib.gatherCycles(group,"LAnkleAngles")
            cycleValues[:, :, 0].mean(axis=0) # return frame by frame mean of col #0




    class Test_mekPlot:
        def test_plotKinematicsDescriptive(self):
            
            # path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            # ds = moveck.data_store(path+"storage.h5")
            path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\flow\\Nantes\\ESNAULT Oceane\\"#pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"010280945-storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1")

            layoutfile = LAYOUT_PATH + "normalizedKinematics.layouts"
            layout = files.openFile(None,layoutfile)

            mpp = mekPlot.mekPlotSingleGroupLayoutProcedure(layout = layout["lowerLimbKinematics"],consistency=False)
            mpp.setData(group)

            mpf = mekPlot.mekPlotFilter(procedure=mpp)
            mpf.run()



        def test_plotKineticsDescriptive(self):


            path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\flow\\Nantes\\ESNAULT Oceane\\"#pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"010280945-storage.h5")            
            # path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            # ds = moveck.data_store(path+"storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1")

            layoutfile = LAYOUT_PATH + "normalizedKinetics.layouts"
            layout = files.openFile(None,layoutfile)


            mpp = mekPlot.mekPlotSingleGroupLayoutProcedure(layout = layout["lowerLimbKinetics"],consistency=False)
            mpp.setData(group)

            mpf = mekPlot.mekPlotFilter(procedure=mpp)
            mpf.run()

        def test_plotKinematicsConsistency(self):
            
            # path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            # ds = moveck.data_store(path+"storage.h5")
            path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\flow\\Nantes\\ESNAULT Oceane\\"#pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"010280945-storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1")

            layoutfile = LAYOUT_PATH + "normalizedKinematics.layouts"
            layout = files.openFile(None,layoutfile)

            mpp = mekPlot.mekPlotSingleGroupLayoutProcedure(layout = layout["lowerLimbKinematics"],consistency=False)
            mpp.setData(group)

            mpf = mekPlot.mekPlotFilter(procedure=mpp)
            mpf.run()



        def test_plotKineticsConsistency(self):


            path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\flow\\Nantes\\ESNAULT Oceane\\"#pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"010280945-storage.h5")            
            # path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            # ds = moveck.data_store(path+"storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1")

            layoutfile = LAYOUT_PATH + "normalizedKinetics.layouts"
            layout = files.openFile(None,layoutfile)


            mpp = mekPlot.mekPlotSingleGroupLayoutProcedure(layout = layout["lowerLimbKinetics"],consistency=False)
            mpp.setData(group)

            mpf = mekPlot.mekPlotFilter(procedure=mpp)
            mpf.run()


        def test_plotEmgTemporal(self):






            path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\flow\\Nantes\\ESNAULT Oceane\\"#pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"010280945-storage.h5")


            userSettings = mekFlow.read_pycgm2Settings_attribute(ds, "Session 1/Analysis 1")

            group = ds.root().retrieve_group("Session 1/Analysis 1/Condition1/Extraction/EMG/rectify/20171025_EO_PONC_S_NNNN dyn 01.c3d")




            layoutfile = LAYOUT_PATH + "temporalEmg.layouts"
            layout = files.openFile(None,layoutfile)

            labels,muscles,contexts,normalActivities = settingsHandler.get_emg_configuration(userSettings, "Condition1")
            
            index = 0
            for plotIt in layout["EMG1_10"]["plots"]:
                plotIt["title"] = muscles[index]
                plotIt["normalActivation"] = normalActivities[index]
                plotIt["curves"][0]["data"] = contexts[index][0]+muscles[index]
                plotIt["curves"][0]["legendLabel"] = contexts[index]
                plotIt["curves"][0]["eventContext"] = contexts[index]
                index+=1



            mpp = mekPlot.mekPlotTemporalEmgLayoutProcedure(layout = layout["EMG1_10"])
            mpp.setData(group)

            mpf = mekPlot.mekPlotFilter(procedure=mpp)
            mpf.run()
            

        def test_plotEmgNormalized(self):


            
            path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"010280945-storage.h5")


            userSettings = mekFlow.read_pycgm2Settings_attribute(ds, "Session 1/Analysis 1")

            group = ds.root().retrieve_group("Session 1/Analysis 1/Condition1/Normalize/EMG/rectify/Envelop/Norm")


            layoutfile = LAYOUT_PATH + "normalizedEmg.layouts"
            layout = files.openFile(None,layoutfile)

            labels,muscles,contexts,normalActivities = settingsHandler.get_emg_configuration(userSettings, "Condition1")
             
            nmuscles = len(muscles) 
            
            for i in range(nmuscles):
                layout["EMG1_16"]["plots"][i]["title"] = muscles[i]
                layout["EMG1_16"]["plots"][i]["normalActivation"] = normalActivities[i]
                layout["EMG1_16"]["plots"][i]["curves"][0]["data"] = contexts[i][0]+muscles[i]
                layout["EMG1_16"]["plots"][i]["curves"][0]["legendLabel"] = contexts[i]
                layout["EMG1_16"]["plots"][i]["curves"][0]["eventContext"] = contexts[i]



            mpp = mekPlot.mekPlotNormalizedEMGLayoutProcedure(layout = layout["EMG1_16"])
            mpp.setData(group)

            mpf = mekPlot.mekPlotFilter(procedure=mpp)
            mpf.run()





    class Test_mekApp:
        def test_FlowSettingsToDataframe(self):

            path = pyCGM2.TEST_DATA_PATH+"mek\\storageSample\\"

            storage = mekInit.Storage(storagePathFile=path+"session-storage.h5")
            ds = storage.getStorage()

            df = mekFlow.build_session_conditions_dataframe(storage)            

            print (df)


        def test_stp(self):

            path = pyCGM2.TEST_DATA_PATH+"mek\\storageSample\\"

            storage = mekInit.Storage(storagePathFile=path+"session-storage.h5")
            ds = storage.getStorage()

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1/Preparation/SpatioTemporalParameters")
            cycleValuesL ,attrs = mekLib.gatherCycles(group,"Lspeed")
            cycleValuesR ,attrs = mekLib.gatherCycles(group,"Rspeed")
            np.mean([cycleValuesR,cycleValuesL])


            #list of all spatio temporal parameters :
            #['Lcadence', 'LdoubleStance1', 'LdoubleStance1Duration', 'LdoubleStance2', 'LdoubleStance2Duration', 'Lduration', 'LsimpleStance', 'LsimpleStanceDuration', 
            # 'Lspeed', 'LstanceDuration', 'LstancePhase', 'LstepDuration', 'LstepLength', 'LstepPhase', 'LstrideLength', 'LstrideWidth', 'LswingDuration', 'LswingPhase', 
            # 'Rcadence', 'RdoubleStance1', 'RdoubleStance1Duration', 'RdoubleStance2', 'RdoubleStance2Duration', 'Rduration', 'RsimpleStance', 'RsimpleStanceDuration', 
            # 'Rspeed', 'RstanceDuration', 'RstancePhase', 'RstepDuration', 'RstepLength', 'RstepPhase', 'RstrideLength', 'RstrideWidth', 'RswingDuration', 'RswingPhase']

            settings = mekLib.readYamlAttribute(ds,f"Session 2/Analysis 1","flow-userSettings")
            age = settings["VisitInfo"]["Age"]

            import ipdb; ipdb.set_trace()

            normativeStpHelper = normativeDataHelpers.NormativeStpHelper()

            normRange = normativeStpHelper.getMinMax("speed",age=age) 

            



        def test_emgRectify(self):

            path = pyCGM2.TEST_DATA_PATH+"mek\\"
            
            storage = mekInit.Storage(storagePathFile=path+"session-storage.h5")
            ds =  storage.getStorage()


            settings = mekLib.readYamlAttribute(ds,f"Session 2/Analysis 1","flow-userSettings")
            
            # settings = mekFlow.read_flowSettings_attribute(ds, f"Session 2/Analysis 1")
            emgRepTrialName = settingsHandler.get_EmgRepresentative(settings,"Condition1")

            group = ds.root().retrieve_group(f"Session 2/Analysis 1/Condition1/Extraction/EMG/rectify/{emgRepTrialName}")
            
            values = group.retrieve_set("LGASTRO").read()
            sampleRate = group.retrieve_set("LGASTRO").retrieve_attribute("SampleRate").read()
            startTime = group.retrieve_set("LGASTRO").retrieve_attribute("StartTime").read()
            channel = group.retrieve_set("LGASTRO").retrieve_attribute("Channel").read().split("_")[0]


            emgConfig =    settingsHandler.get_emg_configuration(settings,"Condition1",outputType="dict")
            index = emgConfig["Labels"].index(channel)
            normalActivityLabel =emgConfig["NormalActivity"][index]




            lfs = group.retrieve_set("events/Foot Strike/Left").read() 
            lfo = group.retrieve_set("events/Foot Off/Left").read() 
            rfs = group.retrieve_set("events/Foot Strike/Right").read()  
            rfo = group.retrieve_set("events/Foot Off/Right").read() 

            data_path = pyCGM2.TEST_DATA_PATH + "EMG\\emgTrials\\gait\\"
            filename = "20210908_NZ-PRE-S-NNNN-dyn 02.c3d"


            time = startTime + np.arange(values.shape[0]) / sampleRate
            # time = utils.timeRange(startTime,  sampleRate, values.shape[0])
        

            onsets, durations,signal = normalActivation.getNormalGaitEmgActivities(lfs,lfo, normalActivityLabel, time=time)
            import ipdb; ipdb.set_trace()
        

            print(onsets)
            plt.plot(time,signal)
            plt.plot(time,values)
            plt.axvline(x=lfs[0], color='b', linestyle='--', linewidth=0.8, label='FS')
            plt.show()



        def test_emgEnvComparison(self):

            path = path = pyCGM2.TEST_DATA_PATH+"mek\\storageSample\\"
            ds = moveck.data_store(path+"ipp-storage.h5")

            # settings = mekFlow.read_flowSettings_attribute(ds, f"Session 2/Analysis 1")
            # emgRepTrialName = settingsHandler.get_EmgRepresentative(settings,"Condition1")


            group = ds.root().retrieve_group("Session 1/Analysis 1/Condition1/Normalize/EMG/rectify/Envelop")
            cycleValues1 ,attrs = mekLib.gatherCycles(group,"LRECFEM")

            group = ds.root().retrieve_group("Session 1/Analysis 1/Condition2/Normalize/EMG/rectify/Envelop")
            cycleValues2 ,attrs = mekLib.gatherCycles(group,"LRECFEM")


            import ipdb; ipdb.set_trace()
            denominator = np.array([it.max()  for it in cycleValues1]).mean()

            cycleValues1 = cycleValues1 / denominator
            cycleValues2 = cycleValues2 /denominator

            meanFootOff =np.mean([it["footOff"] for it in attrs])
            controlateral_footOff =np.mean([it["controlateral_footOff"] for it in attrs])
            controlateral_footStrike =np.mean([it["controlateral_footStrike"] for it in attrs])

            plt.plot(cycleValues1[:, :, 0].mean(axis=0),"-r")
            plt.plot(cycleValues2[:, :, 0].mean(axis=0),"-b")
            plt.show()


            



        def test_plot(self):
            path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"session-storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1/Normalize/Kinematics/Angles")
            cycleValues ,attrs = mekLib.gatherCycles(group,"LAnkleAngles")
            avg = cycleValues[:, :, 0].mean(axis=0) # return frame by frame mean of col #0

            cycleValues ,attrs = mekLib.gatherCycles(group,"LAnkleAngles")
            meanFootOff =np.mean([it["footOff"] for it in attrs])
            controlateral_footOff =np.mean([it["controlateral_footOff"] for it in attrs])
            controlateral_footStrike =np.mean([it["controlateral_footStrike"] for it in attrs])


            plt.plot(avg)
            plt.show()

        def test_scoreGPS(self):
            
            path = pyCGM2.TEST_DATA_PATH+"mek\\storageSample\\"
        
            storage = mekInit.Storage(storagePathFile=path+"session-storage.h5")
            ds =  storage.getStorage()

            group = ds.root().retrieve_group("Session 1/Analysis 1/Condition1")

            import time
                
            # start = time.perf_counter()
            # newNormativeData = openFile("C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\pyCGM2\\pyCGM2\\Data\\normativeData\\",
            #                               "CGM23-msm.json")["CGM23"]["Spont"]
            # elapsed = time.perf_counter() - start
            # print(f"Temps d'exécution : {elapsed:.3f} s")

            start = time.perf_counter()
            newNormativeData = files.openJson(pyCGM2.NORMATIVE_DATABASE_PATH, "CGM23-msm.json")["CGM23"]["Spont"]
            elapsed = time.perf_counter() - start
            print(f"Temps d'exécution : {elapsed:.3f} s")

            # start = time.perf_counter()
            # newNormativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")
            # elapsed = time.perf_counter() - start
            # print(f"Temps d'exécution : {elapsed:.3f} s")


            gps =mekScore.CGM1_GPS()
            scf = mekScore.ScoreFilter(gps,group, newNormativeData)
            gvs,gpsByContext,gpsOverall = scf.compute()

            print(gpsOverall)
        #     {'mean': array([5.63951725]), 'std': array([1.17288645]), 'median': array([5.07650152]), 'values': array([4.87285455, 4.17551561, 4.6721065 , 4.48255292, 4.86252503,
        #    5.28014849, 7.28559112, 6.50077662, 6.89219175, 7.37090989])}
            import ipdb; ipdb.set_trace()









