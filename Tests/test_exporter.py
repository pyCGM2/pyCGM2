# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_exporter.py::TestExport::test_analysisAdvanced

import ipdb
import matplotlib.pyplot as plt
import logging

import pyCGM2
import pyCGM2.Lib.analysis


from pyCGM2.Model.CGM2 import cgm

from pyCGM2.Processing import exporter,c3dManager,cycle,analysis
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files


class TestExport:

    def test_analysisAdvanced(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        files.createDir(DATA_PATH_OUT)

        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------
        # ----INFOS-----
        modelInfo={"Type":"cgm2", "hjc":"hara"}
        subjectInfo={"Id":"1", "Name":"Lecter"}
        experimentalInfo={"Condition":"Barefoot", "context":"block"}

        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis(DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("Hän-advancedExport", path=DATA_PATH_OUT,excelFormat = "xls",mode="Advanced")


    def test_analysisBasic(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        files.createDir(DATA_PATH_OUT)


        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        # ----INFOS-----
        modelInfo={"Type":"cgm2", "hjc":"hara"}
        subjectInfo={"Id":"1", "Name":"Lecter"}
        experimentalInfo={"Condition":"Barefoot", "context":"block"}


        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis(DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("Hän-basicExport", path=DATA_PATH_OUT,excelFormat = "xls",mode="Basic")


    def test_analysisJson(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        files.createDir(DATA_PATH_OUT)

        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        # ----INFOS-----
        modelInfo={"Type":"cgm2", "hjc":"hara"}
        subjectInfo={"Id":"1", "Name":"Lecter"}
        experimentalInfo={"Condition":"Barefoot", "context":"block"}

        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis(DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.AnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("Hän-jsonExport", path=DATA_PATH_OUT)


    def test_analysisC3d(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        files.createDir(DATA_PATH_OUT)

        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        # ----INFOS-----
        modelInfo={"Type":"cgm2", "hjc":"hara"}
        subjectInfo={"Id":"1", "Name":"Lecter"}
        experimentalInfo={"Condition":"Barefoot", "context":"block"}

        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis(DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.AnalysisC3dExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("Hän-c3dExport", path=DATA_PATH_OUT)

    def test_analysisAdvancedEMG(self):

        # ----DATA-----
# ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\\EMG\\Hånnibøl Lecter-nerve block\\"
        inputFile = ["PRE-gait trial 01.c3d"]

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"GaitData\\EMG\\Hånnibøl Lecter-nerve block\\"
        files.createDir(DATA_PATH_OUT)

        EMG_LABELS = ["EMG1","EMG2"]
#
        for file in inputFile:
            acq = btkTools.smartReader(DATA_PATH+file)
            pyCGM2.Lib.analysis.processEMG_fromBtkAcq(acq, EMG_LABELS, highPassFrequencies=[20,200],envelopFrequency=6.0)
            btkTools.smartWriter(acq,DATA_PATH_OUT+file[:-4]+"-emgProcessed.c3d")

        inputFileProcessed =   [file[:-4]+"-emgProcessed.c3d" for file in inputFile]

        emgAnalysis =  pyCGM2.Lib.analysis.makeEmgAnalysis(DATA_PATH_OUT, inputFileProcessed, EMG_LABELS,None, None)

        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("Hän-Emg_advancedExport", path=DATA_PATH_OUT,excelFormat = "xls",mode="Advanced")

        exportFilter = exporter.AnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("Hän-Emg_export.json", path=DATA_PATH_OUT)

        exportFilter = exporter.AnalysisC3dExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("Hän-emg_C3dExport", path=DATA_PATH_OUT)
