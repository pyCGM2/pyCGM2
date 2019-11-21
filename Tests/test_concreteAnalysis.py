# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# pytest -s --disable-pytest-warnings  test_concreteAnalysis.py::Test_compare2TestingEMG::test_onlyAnalogsInAcq_withFsBoundaries


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Utils import files
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Lib import configuration

from pyCGM2.Report import normativeDatasets

class Test_compare2TestingEMG():

    def test_onlyAnalogsInAcq_withFsBoundaries(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"LowLevel\\c3OnlyAnalog\\gait\\"

        EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT,NORMAL_ACTIVITIES = configuration.getEmgConfiguration(None, None)
        emgTrials_pre = ["20191120-CN-PRE-Testing-NNNS-dyn 09_withFSboundaries.c3d"]
        emgTrials_post = ["20191120-CN-POST-testing-NNNNS-dyn 20_withFSBoundaries.c3d"]

        analysis.processEMG(DATA_PATH, emgTrials_pre, EMG_LABELS, highPassFrequencies=[20,200], envelopFrequency=6,fileSuffix=None)
        preAnalysis = analysis.makeEmgAnalysis(DATA_PATH,emgTrials_pre,EMG_LABELS,type="testing")
        analysis.normalizedEMG(preAnalysis,EMG_LABELS,EMG_CONTEXT,method="MeanMax")

        analysis.processEMG(DATA_PATH, emgTrials_post, EMG_LABELS, highPassFrequencies=[20,200], envelopFrequency=6,fileSuffix=None)
        postAnalysis = analysis.makeEmgAnalysis(DATA_PATH,emgTrials_post,EMG_LABELS,type="testing")
        analysis.normalizedEMG(postAnalysis,EMG_LABELS,EMG_CONTEXT,method="MeanMax")


        plot.compareEmgEnvelops([preAnalysis,postAnalysis],["pre","post"],EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT,NORMAL_ACTIVITIES,type="testing")
        plot.compareSelectedEmgEvelops([preAnalysis,postAnalysis],["pre","post"],["Voltage.EMG1","Voltage.EMG1"],["Left","Left"],type="testing",title = "comparison",plotType="Descriptive")
