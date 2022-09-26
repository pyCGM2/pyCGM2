# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_IO::test_motFile
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_misc::test_prepareData
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Tools import opensimTools
from pyCGM2.Tools import btkTools
from bs4 import BeautifulSoup
from pyCGM2.Lib import opensimtk
import ipdb

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

class Test_preparation:

    def test_prepareData_gait(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/prepareData/gait\\"

        opensimtk.prepareC3dFiles(DATA_PATH,
            staticFilename="static.c3d", 
            dynamicData=[["gait1.c3d",None], ["gait2.c3d",None]])