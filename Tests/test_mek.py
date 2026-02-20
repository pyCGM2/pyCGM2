# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_mek.py::Test_mek::test_gaitScheme
from logging import root
import sys
import trace
import pandas as pd

import pyCGM2

LOGGER = pyCGM2.LOGGER
from pyCGM2.Lib import emg
from pyCGM2.Utils import files

LAYOUT_PATH = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Mek\\layout\\"

try:
    import moveck
    MOVECK_AVAILABLE = True
except ImportError:
    MOVECK_AVAILABLE = False
    LOGGER.warning("moveck pipe is not installed")



from pyCGM2.Utils import utils
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.flow import settingsHandler

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


    class Test_mekImporter:
        
        def test_mekViconTransform(self):

            path = pyCGM2.TEST_DATA_PATH + "NantesSamples\AQM Adultes\\BOUCHE Alain\\Session 1\\"
            modelledFilenames = ["20260203-AB-PRE-S-NNCN-dyn 01.c3d"]

            ds = moveck.data_store()

            proc = mekTransform.mekViconTrialTransformProcedure(cgmVersion="CGM2.3")
            filter = mekTransform.mekTrialTransformFilter(ds,procedure=proc)
            filter.run(path,modelledFilenames[0])

            ds.dump("storageViconTransformer.h5")
            

    class Test_mekFlow:
        def test_readSettings(self):

            path = pyCGM2.TEST_DATA_PATH+"mek\\storageSample\\"

            storagefilter = mekInit.mekInitStorageFilter(storagePathFile=path+"019146680-zaidi-storage-multiSessionsAndConditions.h5")
            ds = storagefilter.getStorage()

            df = mekFlow.build_session_conditions_dataframe(ds)

            print (df)

           




    class Test_mekScheme:
        def test_gaitScheme(self):

            # path = pyCGM2.TEST_DATA_PATH + "mek\\gaitdata\\"
            # modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]


            path = pyCGM2.TEST_DATA_PATH + "Nantes\\OSMANOV Akhmed\\Session 3\\"
            modelledFilenames = ["20201209-AO-PONC-S-NNNN-dyn 05.c3d","20201209-AO-PONC-S-NNNN-dyn 06.c3d"]

            userSettings = files.openFile(path,"CGM23.userSettings")
                        
            storagefilter = mekInit.mekInitStorageFilter()
            storagefilter.createGroup("Session 1/Analysis 1/Condition 1")
            ds = storagefilter.getStorage()


            osimInterface = opensimInterface.osimInterface(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "pycgm2-gait2354_simbody.osim")
            muscleDict = osimInterface.getMuscles_bySide(addToName="[MuscleLength]")

            # preparation

            emg.processEMG(path,
                                modelledFilenames,
                                ['Voltage.EMG1', 'Voltage.EMG2', 'Voltage.EMG3', 'Voltage.EMG4', 'Voltage.EMG5', 'Voltage.EMG6', 'Voltage.EMG7', 'Voltage.EMG8', 'Voltage.EMG9', 'Voltage.EMG10'],
                                highPassFrequencies = [20,200],
                                envelopFrequency= 6,
                                fileSuffix="test",
                                outDataPath=None)
            
            
            emgFilenames = ["20201209-AO-PONC-S-NNNN-dyn 05_test.c3d","20201209-AO-PONC-S-NNNN-dyn 06_test.c3d"]


            mekOperations.compute_spatio_temporal_parametersOperation( ds.root().retrieve_group("Session 1/Analysis 1/Condition 1"),
                                                               path, modelledFilenames)
            

            # extraction and normalization

            scheme = {
                    "Kinematics/Angles": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Right"]]],
                    "Kinetics/Moments": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_MOMENTS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_MOMENTS["Right"]]],
                    "Kinetics/Forces": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_FORCES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_FORCES["Right"]]],
                    "EMG/rectify/Envelop": [[path+filename for filename in emgFilenames], ["Voltage.EMG1_Rectify_Env:Left=RECTFEM"]  ],    
                    "MuscleKinematics/MTUL" : [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in muscleDict["Left"]] + [it+":Right" for it  in muscleDict["Right"]]]
            }
            
            mekFlow.create_pycgm2Settings_attribute(userSettings, ds, "Session 1/Analysis 1")


            filter = mekExtract.mekExtractFilter(ds,group="Session 1/Analysis 1/Condition 1")
            filter.run(scheme)    


            normalize_filter = mekNormalize.mekNormalizeFilter(ds,group="Session 1/Analysis 1/Condition 1")
            normalize_filter.run(scheme,cropToForcePlateGroups =["Kinetics/Moments", "Kinetics/Forces"]) 

            ds.dump(path+"storageGaitScheme.h5")
            

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
            ds = moveck.data_store(path+"storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1")

            values = mekLib.gather(group,"LAnkleAngles")
            values[:, :, 0].mean(axis=0) # return frame by frame mean of col #0

        def test_gatherStp(self):
            path = pyCGM2.TEST_DATA_PATH + "mek\\gaitdata\\"
            ds = moveck.data_store(path+"storage2.h5")

            group = ds.root().retrieve_group("Session 1/Analysis 1/MultiConditions/SpatioTemporalParameters")
            import ipdb; ipdb.set_trace()
            values = mekLib.gather(group,"Lspeed")

            group = ds.root().retrieve_group("Session 1/Analysis 1/MultiConditions/SpatioTemporalParameters")
            values = mekLib.gather(group,"Rspeed")

            print(values)


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











