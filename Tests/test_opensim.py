# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_InverseDynamics::test_cgm23
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2
from pyCGM2 import opensim4 as opensim

from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools
from pyCGM2.Tools import  opensimTools
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2 import enums
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Model.Opensim import osimProcessing


class Test_xml:
    def test_readScaling(self):

        scaleFile = "C:/Users/fleboeuf/Documents/Programmation/pyCGM2/pyCGM2/Sandbox/opensim/setUpXmlFiles/CGM23_scaleSetup_template.xml"
        soup = BeautifulSoup(open(scaleFile), "xml")
        import ipdb; ipdb.set_trace()


class Test_basics:
    def test_c3dAdapter(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/CGM23\\gait\\"

        opensimTools.createGroundReactionForceMOT_file(DATA_PATH,'gait1.c3d')
