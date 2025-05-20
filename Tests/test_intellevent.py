import pytest
# pytest -s --disable-pytest-warnings  test_intellevent.py
# pyCGM2
import pyCGM2


from pyCGM2.Tools import  btkTools
from pyCGM2.Events import eventFilters

from pyCGM2.Events.procedures import intellEventProcedures


from pyCGM2.Lib.Processing import progression

from pyCGM2.Lib import eventDetector

class Test_Intellevent:
    def test_0_lowLevel(self):


        path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\events\\intellevent\\"

        acq = btkTools.smartReader(path+"gait_vicon.c3d")

        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=False)
        initialContactModelFile = pyCGM2.PYCGM2_APPDATA_PATH+"intellEventModels\\version0\\ic_intellevent.onnx"
        footOffModelFile = pyCGM2.PYCGM2_APPDATA_PATH+"intellEventModels\\version0\\fo_intellevent.onnx"

        evp = intellEventProcedures.IntellEventProcedure(progressionAxis,forwardProgression)
        evp.setModels(initialContactModelFile,footOffModelFile)

        evf = eventFilters.EventFilter(evp,acq)
        evf.detect()


        btkTools.smartWriter(acq,"intellevent.c3d")

    def test_0_highLevel(self):


        path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\events\\intellevent\\"
        acq = btkTools.smartReader(path+"gait_vicon.c3d")

        eventDetector.intellEvent(acq)

        btkTools.smartWriter(acq,"intellevent2.c3d")
