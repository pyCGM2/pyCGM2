# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_IO::test_readXML
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_misc::test_prepareData
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_osim::test_osimInterface
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_IO::test_cgmOutputsToMot

from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Tools import opensimTools
from pyCGM2.Tools import btkTools
from bs4 import BeautifulSoup
from pyCGM2.Lib import opensimtk
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface import opensimInterface

import pyCGM2
LOGGER = pyCGM2.LOGGER



class Test_IO:

    def test_readXML(self):

        scaleFile = pyCGM2.TEST_DATA_PATH + "OpenSim/IO/CGM23_scaleSetup_template.xml"
        soup = BeautifulSoup(open(scaleFile), "xml")
        

    def test_opensimDataframe_motFile(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/IO\\"

        motDf = opensimIO.OpensimDataFrame(DATA_PATH, "gait1.mot")
        motDf.getDataFrame()["pelvis_tilt"] = 0.0
        motDf.save(filename="_mot_out.mot")

    def test_opensimDataframe_issueStoFile(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/IO\\"

        motDf = opensimIO.OpensimDataFrame(DATA_PATH, "issue_ik_model_marker_locations.sto")

    def test_zeroing_motFile(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/IO\\"

        motDf = opensimIO.OpensimDataFrame(DATA_PATH, "gait1.mot")
        for col in motDf.getDataFrame().columns:
            if col != "time":
                motDf.getDataFrame()[col] = 0.0
        
        motDf.save(filename="_motzeroing_out.mot")


    def test_opensimDataframe_stoFile(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/IO\\"
        
        stoDf = opensimIO.OpensimDataFrame(
            DATA_PATH, "cgm2-Osim-scaled_MuscleAnalysis_ActiveFiberForce.sto")
        
        stoDf.save(filename="_sto_out.mot")


    def test_createFootGrf_file(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/CGM23\\gait\\"
        gaitFilename = "gait1.c3d"

        acqGait = btkTools.smartReader(str(DATA_PATH + gaitFilename))

        opensimTools.footReactionMotFile(
            acqGait, DATA_PATH+ "_"+gaitFilename[:-4]+"_grf.mot","X",False)

    def test_nexusC3d_withOutputs(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"


        acq1 = btkTools.smartReader(DATA_PATH+"gait1.c3d")
        btkTools.smartWriter(acq1, DATA_PATH+"gait1verif.c3d", extension=None)

        acq2 = btkTools.smartReader(DATA_PATH+"gait2.c3d")
        btkTools.smartWriter(acq2, DATA_PATH+"gait2verif.c3d", extension=None)

    def test_export_cgmToMot(self):

        data_path = pyCGM2.TEST_DATA_PATH + "OpenSim\CGM23\\CGM23-progressionX-test\\"
        staticFilename = "static.c3d" 
        gaitFilename = "gait1.c3d"

        osimInterface = opensimInterface.osimInterface(data_path,"static-CGM23-ScaledModel.osim")
        #osimModel.getCoordinates()

        acqGait = btkTools.smartReader(str(data_path + gaitFilename))

        opensimTools.export_CgmToMot(
            acqGait,data_path, "motGenerated.mot",osimInterface)


class Test_preparation:

    def test_prepareData_gait(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/prepareData/gait\\"

        opensimtk.prepareC3dFiles(DATA_PATH,
            staticFilename="static.c3d", 
            dynamicData=[["gait1.c3d",None], ["gait2.c3d",None]])

class Test_osim:
    def test_osimInterface(self): 
        osimInterface = opensimInterface.osimInterface(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "pycgm2-gait2392_simbody.osim")
        muscles = osimInterface.getMuscles()
        print (muscles)

        bySide = osimInterface.getMuscles_bySide()
        print (bySide)