# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_mek.py::Test_mek::test_gaitScheme
import sys
import pandas as pd

import pyCGM2

from pyCGM2.Utils import files

try:
    import moveck
    MOVECK_AVAILABLE = True
except ImportError:
    MOVECK_AVAILABLE = False
    LOGGER.warning("moveck pipe is not installed")



from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.Opensim.interface import opensimInterface


if MOVECK_AVAILABLE:

    from pyCGM2.Mek import mekConstants
    from pyCGM2.Mek import mekInit
    from pyCGM2.Mek import mekExtract
    from pyCGM2.Mek import mekNormalize
    from pyCGM2.Mek import mekParametrize
    from pyCGM2.Mek import mekLib
    from pyCGM2.Mek import mekTools


    class Test_mekSettings:
        def test_cgm23(self):

            path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\mek\\gaitdata\\"
            userSettings = files.openFile(path,"CGM23.userSettings")
            modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

            storagefilter = mekInit.mekInitStorageFilter()
            storagefilter.createGroup("Session 1/Analysis 1/Condition 1")
            storagefilter.createGroup("Session 1/Analysis 1")
            ds = storagefilter.getStorage()    

            mekParametrize.create_pycgm2Settings_attribute(userSettings, ds, "Session 1/Analysis 1")
            settings_dict = mekParametrize.read_pycgm2Settings_attribute(ds, "Session 1/Analysis 1")

            import ipdb; ipdb.set_trace()

            ds.dump("storageSettings.h5")


    class Test_mek:
        def test_simpleScheme(self):

                path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\mek\\gaitdata\\"
                filenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

                # ds = moveck.data_store()
                # root = ds.root()

                # root.create_group("Session 1/Condition 1")
                
                storagefilter = mekInit.mekInitStorageFilter()
                storagefilter.createGroup("Session 1/Analysis 1/Condition 1")
                ds = storagefilter.getStorage()    

                scheme = {
                    "Kinematics/Angles": [[path+filename for filename in filenames],  ["LHipAngles:Left", "RHipAngles:Right"]],
                    "Kinematics/Angles2": [[path+filename for filename in filenames],  ["LHipAngles:Left=LHipAngles2", "RHipAngles:Right"]],
                }


                filter = mekExtract.mekExtractFilter(ds,group="Session 1/Analysis 1/Condition 1")
                filter.run(scheme)    

                normalize_filter = mekNormalize.mekNormalizeFilter(ds,group="Session 1/Analysis 1/Condition 1")
                normalize_filter.run(scheme)  

                ds.dump("storage.h5")


        def test_gaitScheme(self):

            path = pyCGM2.TEST_DATA_PATH + "mek\\gaitdata\\"
            modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]
            userSettings = files.openFile(path,"CGM23.userSettings")
                        
            storagefilter = mekInit.mekInitStorageFilter()
            storagefilter.createGroup("Session 1/Analysis 1/Condition 1")
            ds = storagefilter.getStorage()


            osimInterface = opensimInterface.osimInterface(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "pycgm2-gait2354_simbody.osim")
            muscleDict = osimInterface.getMuscles_bySide(addToName="[MuscleLength]")
            

            # emgTrialNames = settingsHandler.get_emg_trials_by_condition(userSettings, condition)
            # emgConfiguration = settingsHandler.get_emg_configuration(userSettings, condition, outputType="dict")
            # mergedLabelContext = [f"{label}_Rectify_Env:{context}={context[0]}{muscle}" for label, context, muscle in zip(emgConfiguration["Labels"], emgConfiguration["Contexts"],emgConfiguration["Muscles"]) if context is not None]
           
            scheme = {
                    "Kinematics/Angles": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Right"]]],
                    "Kinetics/Moments": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_MOMENTS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_MOMENTS["Right"]]],
                    "Kinetics/Forces": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_FORCES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_FORCES["Right"]]],
            }
            
            mekParametrize.create_pycgm2Settings_attribute(userSettings, ds, "Session 1/Analysis 1")

            filter = mekExtract.mekExtractFilter(ds,group="Session 1/Analysis 1/Condition 1")
            filter.run(scheme)    


            normalize_filter = mekNormalize.mekNormalizeFilter(ds,group="Session 1/Analysis 1/Condition 1")
            normalize_filter.run(scheme,cropToForcePlateGroups =["Kinetics/Moments", "Kinetics/Forces"]) 

            

            ds.dump(path + "storage2.h5")
            

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


        def test_plot(self):
            path = pyCGM2.TEST_DATA_PATH + "mek\\storageSample\\"
            ds = moveck.data_store(path+"storage.h5")

            group = ds.root().retrieve_group("Session 2/Analysis 1/Condition1")

            layout = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Mek\\layout\\lowerLimbKinematics.layout"

            mekLib.plot(group, layout)


            import ipdb; ipdb.set_trace()










