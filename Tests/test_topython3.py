# coding: utf-8
# pytest -s --disable-pytest-warnings  test_topython3.py::Test_topy3::test_c3dManager

from __future__ import unicode_literals
import pytest
import numpy as np

import pyCGM2
from pyCGM2 import btk
from pyCGM2.Lib import analysis
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import utils

from pyCGM2.Processing import c3dManager, cycle, analysis
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Report import plot

import numpy as np


class Test_topy3():

    # def test_sortEvent(self):
    #     DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hånnibøl Lecter\\"
    #
    #     acq = btkTools.smartReader(DATA_PATH+"gait Trial 01.c3d")
    #
    #     left_fs_frames=list()
    #     for ev in btk.Iterate(acq.GetEvents()):
    #         if ev.GetContext() == "Left" and ev.GetLabel() == "Foot Strike":
    #             left_fs_frames.append(ev.GetFrame())
    #
    #     print left_fs_frames
    #     left_fs_frames=list()
    #     btkTools.sortedEvents(acq)
    #
    #     for ev in btk.Iterate(acq.GetEvents()):
    #         if ev.GetContext() == "Left" and ev.GetLabel() == "Foot Strike":
    #             left_fs_frames.append(ev.GetFrame())
    #
    #     print left_fs_frames
    #     import ipdb; ipdb.set_trace()

    def test_btkTools(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hånnibøl Lecter\\"

        modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]


        acqs,filenames = btkTools.buildTrials(DATA_PATH,modelledFilenames)

        flag,kineticEvent_times,kineticEvent_times_left,kineticEvent_times_right = btkTools.isKineticFlag(acqs[0])

        kineticAcqs,kineticFilenames,flag_kinetics = automaticKineticDetection = btkTools.automaticKineticDetection(DATA_PATH, modelledFilenames)


    def test_c3dManager(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hånnibøl Lecter\\"

        modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]

        acq = btkTools.smartReader(DATA_PATH+"gait Trial 01.c3d")


        c3dmanagerProcedure = c3dManager.DistinctC3dSetProcedure(DATA_PATH, modelledFilenames,
            modelledFilenames, modelledFilenames, None)

        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableSpatioTemporal(True)
        cmf.enableKinematic(True)
        cmf.enableKinetic(True)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #----cycles
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                               kinematicAcqs = trialManager.kinematic["Acqs"],
                                               kineticAcqs = trialManager.kinetic["Acqs"],
                                               emgAcqs=trialManager.emg["Acqs"])

        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()

        #----analysis
        kinematicLabelsDict = cgm.CGM.ANALYSIS_KINEMATIC_LABELS_DICT
        kineticLabelsDict = cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT
        pointLabelSuffix = None
        # emgLabelList  = [label+"_Rectify_Env" for label in emgChannels]


        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                              kinematicLabelsDict = kinematicLabelsDict,
                                              kineticLabelsDict = kineticLabelsDict,
                                              pointlabelSuffix = pointLabelSuffix,
                                              emgLabelList = None)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setInfo(subject=None, model=None, experimental=None)
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.build()

        analysisInstance = analysisFilter.analysis

        import ipdb; ipdb.set_trace()
