# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensim.py::Test_misc::test_foot_grf_file
from pyCGM2.Processing import progressionFrame
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Model.Opensim import opensimFilters
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


class Test_misc:

    def test_foot_grf_file(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim/CGM23\\gait\\"
        gaitFilename = "gait1.c3d"

        acqGait = btkTools.smartReader(str(DATA_PATH + gaitFilename))

        opensimTools.footReactionMotFile(acqGait)

        # #forceplates.appendForcePlateCornerAsMarker(acqGait)
        # mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        #
        # pfp = progressionFrame.PelvisProgressionFrameProcedure()
        # pff = progressionFrame.ProgressionFrameFilter(acqGait, pfp)
        # pff.compute()
        # progressionAxis = pff.outputs["progressionAxis"]
        # forwardProgression = pff.outputs["forwardProgression"]
        #
        # R_LAB_OSIM = opensimTools.setGlobalTransormation_lab_osim(
        #     progressionAxis, forwardProgression)
        #
        # grw1 = btkTools.getForcePlateWrench(acqGait, fpIndex=1)

        import ipdb
        ipdb.set_trace()
        # opensimTools.globalTransformationLabToOsim(
        #     self.m_acqMotion_forIK, R_LAB_OSIM)
        # opensimTools.smartTrcExport(
        #     self.m_acqMotion_forIK, self.m_DATA_PATH + self.m_dynamicFileNoExt[:-4])