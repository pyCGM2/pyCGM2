# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_exporter.py::TestExport::test_analysisAdvanced

import ipdb
import matplotlib.pyplot as plt
import logging

import pyCGM2
import pyCGM2.Lib.analysis


from pyCGM2.Model.CGM2 import cgm

from pyCGM2.Processing import exporter,c3dManager,cycle,analysis
from pyCGM2.Tools import btkTools


class TestExport:

    def test_analysisAdvanced(self):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("advancedExport", path=DATA_PATH,excelFormat = "xls",mode="Advanced")


    def test_analysisBasic(self):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("basicExport", path=None,excelFormat = "xls",mode="Basic")


    def test_analysisJson(self):

        # ----DATA-----

        #DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\gait\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d", "gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("jsonExport", path=None)


    def test_analysisC3d(self):

        # ----DATA-----

        #DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\gait\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d", "gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("c3dExport", path=DATA_PATH)

    def test_analysisAdvancedEMG(self):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gaitEMG\\"
        inputFile = ["pre.c3d","post.c3d"]

        EMG_LABELS = ["EMG1","EMG2"]
#
        for file in inputFile:
            acq = btkTools.smartReader(DATA_PATH+file)
            pyCGM2.Lib.analysis.processEMG_fromBtkAcq(acq, EMG_LABELS, highPassFrequencies=[20,200],envelopFrequency=6.0)
            btkTools.smartWriter(acq,DATA_PATH+file[:-4]+"-emgProcessed.c3d")

        inputFileProcessed =   [file[:-4]+"-emgProcessed.c3d" for file in inputFile]

        emgAnalysis =  pyCGM2.Lib.analysis.makeEmgAnalysis(DATA_PATH, inputFileProcessed, EMG_LABELS,None, None)

        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("Emg_advancedExport", path=DATA_PATH,excelFormat = "xls",mode="Advanced")

        exportFilter = exporter.AnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("Emg_export.json", path=DATA_PATH)

        exportFilter = exporter.AnalysisC3dExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("emg_C3dExport", path=DATA_PATH)


class TestExport_encoding:

    def test_analysisAdvanced(self):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait_latin1_çà\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("advancedExport", path=DATA_PATH,excelFormat = "xls",mode="Advanced")


    def test_analysisBasic(self):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait_latin1_çà\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("basicExport", path=None,excelFormat = "xls",mode="Basic")


    def test_analysisJson(self):

        # ----DATA-----

        #DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\gait_latin1_çà\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d", "gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("jsonExport", path=None)


    def test_analysisC3d(self):

        # ----DATA-----

        #DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\gait_latin1_çà\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d", "gait Trial 03 - viconName.c3d" ]

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
        exportFilter.export("c3dExport", path=DATA_PATH)

    def test_analysisAdvancedEMG(self):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gaitEMG_latin1_çà\\"
        inputFile = ["pre.c3d","post.c3d"]

        EMG_LABELS = ["EMG1","EMG2"]
#
        for file in inputFile:
            acq = btkTools.smartReader(DATA_PATH+file)
            pyCGM2.Lib.analysis.processEMG_fromBtkAcq(acq, EMG_LABELS, highPassFrequencies=[20,200],envelopFrequency=6.0)
            btkTools.smartWriter(acq,DATA_PATH+file[:-4]+"-emgProcessed.c3d")

        inputFileProcessed =   [file[:-4]+"-emgProcessed.c3d" for file in inputFile]

        emgAnalysis =  pyCGM2.Lib.analysis.makeEmgAnalysis(DATA_PATH, inputFileProcessed, EMG_LABELS,None, None)

        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("Emg_advancedExport", path=DATA_PATH,excelFormat = "xls",mode="Advanced")

        exportFilter = exporter.AnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("Emg_export.json", path=DATA_PATH)

        exportFilter = exporter.AnalysisC3dExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("emg_C3dExport", path=DATA_PATH)
