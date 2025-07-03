# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_singleBody.py

import pytest
import pyCGM2
from pyCGM2.Tools import btkTools

from pyCGM2.Model.Models import singleBody
class Test_singleBody:
    def test_0(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\singleBody\\"

        acqStatic = btkTools.smartReader(DATA_PATH + "static.c3d")
        acqDyn = btkTools.smartReader(DATA_PATH + "motion 01.c3d")

        
        rigidBody = singleBody.SingleBody(["rigidO","rigidX","rigidY","rigidO"],
                                    ["rigidO","rigidX","rigidY","rigidZ"])
        rigidBody.calibrate(acqStatic) #,calibFramesOfInterest=[66,66])
        rigidBody.fit(acqDyn)

        globalpos = acqStatic.GetPoint("rigidZ").GetValues().mean(axis=0)
        rigidBody.addNode("new",globalpos,positionType="Global")

        traj = rigidBody.getTrajectory("new")

        btkTools.smartAppendPoint(acqDyn,"new",traj)
        btkTools.smartWriter(acqDyn,"new.c3d") 

