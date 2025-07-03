# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_mek.py::Test_mek::test_gaitScheme
import sys

from pyCGM2.Mek import mekConstants
from pyCGM2.Mek import mekInit
from pyCGM2.Mek import mekExtract
from pyCGM2.Mek import mekNormalize
from pyCGM2.Mek import mekParametrize
from pyCGM2.Mek import mekLib

import pyCGM2

from pyCGM2.Utils import files

MOVECKPATH = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\moveck\\"
sys.path.append(MOVECKPATH+"Moveck_pipe-2024.1.0-win64-pipeline_install\\packages")
import moveck

from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.Opensim.interface import opensimInterface


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


        mekParametrize.addPyCGM2SettingsToGroup(path + "CGM23.userSettings", ds, "Session 1/Analysis 1")

        settings_dict = mekParametrize.readPyCGM2Settings(ds, "Session 1/Analysis 1")


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
            }


            filter = mekExtract.mekExtractFilter(ds,group="Session 1/Analysis 1/Condition 1")
            filter.run(scheme)    

            normalize_filter = mekNormalize.mekNormalizeFilter(ds,group="Session 1/Analysis 1/Condition 1")
            normalize_filter.run(scheme)  

            ds.dump("storage.h5")


    def test_gaitScheme(self):
        # path = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

        # opensimSettings = files.loadSettings(path,"opensim.settings")
        
        # modelledFilenames = ["gait1.c3d", "gait2.c3d"]
        
        

        path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\mek\\gaitdata\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        # ds = moveck.data_store()
        # root = ds.root()

                    
        storagefilter = mekInit.mekInitStorageFilter()
        storagefilter.createGroup("Session 1/Analysis 1/Condition 1")
        ds = storagefilter.getStorage()


        osimInterface = opensimInterface.osimInterface(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "pycgm2-gait2354_simbody.osim")
        muscleDict = osimInterface.getMuscles_bySide(addToName="[MuscleLength]")
        
        scheme = {
                "Kinematics/Angles": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Right"]]],
                "Kinetics/Moments": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_MOMENTS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_MOMENTS["Right"]]],
                "Kinetics/Forces": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_FORCES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_FORCES["Right"]]],
        }
        
        filter = mekExtract.mekExtractFilter(ds,group="Session 1/Analysis 1/Condition 1")
        filter.run(scheme)    


        normalize_filter = mekNormalize.mekNormalizeFilter(ds,group="Session 1/Analysis 1/Condition 1")
        normalize_filter.run(scheme,cropToForcePlateGroups =["Kinetics/Moments", "Kinetics/Forces"]) 



        ds.dump("storage2.h5")
        
        
        
        # analysisInstance = analysis.makeAnalysis(DATA_PATH,
        #                 modelledFilenames,
        #                 type="Gait",
        #                 emgChannels=emgChannels,
        #                 geometryMuscleLabelsDict={"Left": ["glut_med1_l[MuscleLength]" , "bifemlh_l[MuscleLength]"],
        #                                           "Right" : ["glut_med1_r[MuscleLength]" , "bifemlh_r[MuscleLength]"]},
        #                 dynamicMuscleLabelsDict = None)


    def test_parametrizeGaitScheme(self):
        # path = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

        # opensimSettings = files.loadSettings(path,"opensim.settings")
        
        # modelledFilenames = ["gait1.c3d", "gait2.c3d"]
        
        
        path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\mek\\gaitdata\\"
        userSettings = files.openFile(path,"CGM23.userSettings")
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        # ds = moveck.data_store()
        # root = ds.root()

                    
        storagefilter = mekInit.mekInitStorageFilter()
        storagefilter.createGroup("Session 1/Analysis 1/Condition 1")
        ds = storagefilter.getStorage()


        filter = mekParametrize.mekAnalysisParametrizeFilter(ds,group="Session 1/Analysis 1/Condition 1")
        filter.run(userSettings,"Condition1")
        ds = storagefilter.getStorage()


        osimInterface = opensimInterface.osimInterface(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "pycgm2-gait2354_simbody.osim")
        muscleDict = osimInterface.getMuscles_bySide(addToName="[MuscleLength]")
        
        scheme = {
                "Kinematics/Angles": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINEMATICS_ANGLES["Right"]]],
                "Kinetics/Moments": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_MOMENTS["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_MOMENTS["Right"]]],
                "Kinetics/Forces": [[path+filename for filename in modelledFilenames],  [it+":Left" for it  in mekConstants.CGM_KINETICS_FORCES["Left"]] + [it+":Right" for it  in mekConstants.CGM_KINETICS_FORCES["Right"]]],
        }
        
        filter = mekExtract.mekExtractFilter(ds,group="Session 1/Analysis 1/Condition 1")
        filter.run(scheme)    


        normalize_filter = mekNormalize.mekNormalizeFilter(ds,group="Session 1/Analysis 1/Condition 1")
        normalize_filter.run(scheme,cropToForcePlateGroups =["Kinetics/Moments", "Kinetics/Forces"]) 



        ds.dump("storage2.h5")

class Test_mekLib:
    def test_iter(self):
        ds = moveck.data_store("storage2.h5")
        group = ds.root().retrieve_group("Session 1/Analysis 1/Condition 1")

        for path, set_obj in mekTools.iter_sets(group):
            print(f"path  : {path}")


    def test_gather(self):
        ds = moveck.data_store("storage2.h5")
        group = ds.root().retrieve_group("Session 1/Analysis 1/Condition 1")


        values = mekLib.gather(group,"LAnkleAngles")
        values[:, :, 0].mean(axis=0) # return frame by frame mean of col #0


        layout = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Mek\\layout\\lowerLimbKinematics.layout"

        mekLib.plot(group, layout)


        import ipdb; ipdb.set_trace()












