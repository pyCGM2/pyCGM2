# coding: utf-8
# pytest -s --disable-pytest-warnings  test_eclipse.py::Test_eclipse::test_modipyForcePlate
from __future__ import unicode_literals

import pyCGM2
from pyCGM2 import enums
from pyCGM2.Nexus import eclipse

from pyCGM2.Tools import  btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Nexus import vskTools

class Test_VSK:

    def test_reader(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\\CGM1\\fullBody-native-noOptions-customMP\\"
        staticFilename = "static.c3d"

        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vsk = vskTools.Vsk(DATA_PATH + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)



class Test_eclipse:

    def test_eclipse(self):

        files = eclipse.getEnfFiles(pyCGM2.TEST_DATA_PATH+"LowLevel\\eclipse\\Hännibål\\", enums.EclipseType.Trial)


    def test_modipyForcePlate(self):

        path = pyCGM2.TEST_DATA_PATH+"LowLevel\\eclipse\\ForcePlate\\"

        gaitFilename="PN01OP01S01SS03.c3d"
        acqGait = btkTools.smartReader(str(path +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)


        trial = eclipse.TrialEnfReader(path,"PN01OP01S01SS03.Trial.enf")
        trial.setForcePlates(mappedForcePlate)
        trial.save()
