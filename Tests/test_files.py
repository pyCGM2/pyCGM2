# coding: utf-8
# pytest -s --disable-pytest-warnings  test_files.py::Test_UtilsFiles::test_loadAndSaveAnalysis

from pyCGM2.Utils import testingUtils,files
import ipdb
import os
import logging
from pyCGM2.Eclipse import vskTools,eclipse
from pyCGM2 import enums

import pytest
from pyCGM2 import btk

class Test_UtilsFiles:
    def test_openFile(self):

        contentYaml = files.openFile("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","file.yaml")
        contentJson = files.openFile("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","file.json")

        assert contentYaml !=False
        assert contentJson !=False

        files.saveJson("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\", "testJsonOut", contentJson)


    def test_loadAndSaveModel(self):

        model = files.loadModel("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","PIG-KAD")
        assert model !=False
        files.saveModel(model,"C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","testModelOut")

    def test_loadAndSaveAnalysis(self):

        ana = files.loadAnalysis("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","file")
        assert ana !=False
        files.saveAnalysis(ana,"C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","testAnalysisOut")

    def test_getTranslators(self):
        translators = files.getTranslators("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\", translatorType = "CGM1.translators")
        assert translators !=False

    def test_getIKweightSet(self):
        ikws = files.getIKweightSet("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\", "CGM2_2.ikw")
        assert ikws !=False

    def test_getMpFileContent(self):
        mp = files.getMpFileContent("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","mp.pyCGM2","Nick")
        assert mp !=False

    def test_getFiles(self):
        fs = files.getFiles("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\","json")


    def test_copySessionFolder(self):
        files.copySessionFolder("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\", "folder", "newFolder", selectedFiles=None)

    def test_createDir(self):
        files.createDir("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\latin1_iæøå_test")

    def test_getDirs(self):
        dirs = files.getDirs("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\")
        print dirs


    def test_getFileCreationDate(self):
        files.getFileCreationDate("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\file.c3d")

class Test_vsk:

    def test_vskFiles(self):
        vskFile = vskTools.getVskFiles("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\")
        assert vskFile !=False

        vskTools.checkSetReadOnly("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\PIG-KAD.vsk")

        vskInstance = vskTools.Vsk("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\PIG-KAD.vsk")

# class Test_eclipse:
#
#     def test_eclipse(self):
#
#         files = eclipse.getEnfFiles("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\", enums.EclipseType.Trial)
#
#     def test_findCalibration(self):
#         calib = eclipse.findCalibration("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Eclipse\\Lecter-iæøå\\session\\")
#         assert calib == "PN01OP01S01STAT.Trial.enf"
#
#         motions = eclipse.findMotions("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Eclipse\\Lecter-iæøå\\session\\",ignoreSelect=False)
#         import ipdb; ipdb.set_trace()
# def test_generateEmptyENF(self):
#     eclipse.generateEmptyENF("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\LowLevelTests\\latin1_iæøå\\")
