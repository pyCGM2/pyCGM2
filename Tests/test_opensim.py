# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_IO::test_motFile
from pyCGM2.Processing import progressionFrame
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2 import enums
from pyCGM2.Model import modelFilters, modelDecorator
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.Model.CGM2 import cgm, cgm2
from pyCGM2.Tools import opensimTools
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
from pyCGM2 import opensim4 as opensim
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import pyCGM2
LOGGER = pyCGM2.LOGGER


class Test_xml:
    def test_readScaling(self):

        scaleFile = "C:/Users/fleboeuf/Documents/Programmation/pyCGM2/pyCGM2/Sandbox/opensim/setUpXmlFiles/CGM23_scaleSetup_template.xml"
        soup = BeautifulSoup(open(scaleFile), "xml")
        import ipdb
        ipdb.set_trace()


class Test_basics:
    def test_c3dAdapter(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/CGM23\\gait\\"

        opensimTools.createGroundReactionForceMOT_file(DATA_PATH, 'gait1.c3d')


class Test_IO:
    def test_motFile(self):


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/IO\\"

        motDf = opensimIO.OpensimDataFrame(DATA_PATH, "gait1.mot")
        motDf.getDataFrame()["pelvis_tilt"] = 0.0
        motDf.save(filename="gait1_2.mot")


    def test_stoFile(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/IO\\"
        stoDf = opensimIO.OpensimDataFrame(DATA_PATH, "cgm2-Osim-scaled_MuscleAnalysis_ActiveFiberForce.sto")





class Test_misc:

    def test_foot_grf_file(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/CGM23\\gait\\"
        gaitFilename = "gait1.c3d"

        acqGait = btkTools.smartReader(str(DATA_PATH + gaitFilename))

        opensimTools.footReactionMotFile(
            acqGait, DATA_PATH+gaitFilename[:-4]+"_grf.mot")
