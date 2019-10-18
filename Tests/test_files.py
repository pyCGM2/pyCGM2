# coding: utf-8
# pytest -s --disable-pytest-warnings  test_files.py::Test_UtilsFiles::test_openFile
from __future__ import unicode_literals
import pyCGM2
from pyCGM2.Utils import testingUtils,files
import ipdb
import os
import logging
from pyCGM2.Eclipse import vskTools,eclipse
from pyCGM2 import enums

import pytest
from pyCGM2.Tools import btkTools



class Test_UtilsFiles:
    def test_openFile(self):

        contentYaml = files.openFile("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","file.yaml")
        contentJson = files.openFile("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","file.json")

        assert contentYaml !=False
        assert contentJson !=False

        files.saveJson("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\", "testJsonOut", contentJson)


    def test_loadAndSaveModel(self):

        model = files.loadModel("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","PIG-KAD")
        assert model !=False
        files.saveModel(model,"C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","testModelOut")

    def test_loadAndSaveAnalysis(self):

        ana = files.loadAnalysis("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","file")
        assert ana !=False
        files.saveAnalysis(ana,"C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","testAnalysisOut")

    def test_getTranslators(self):
        translators = files.getTranslators("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\", translatorType = "CGM1.translators")
        assert translators !=False

    def test_getIKweightSet(self):
        ikws = files.getIKweightSet("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\", "CGM2_2.ikw")
        assert ikws !=False

    def test_getMpFileContent(self):
        mp = files.getMpFileContent("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","mp.pyCGM2","Nick")
        assert mp !=False

    def test_getFiles(self):
        fs = files.getFiles("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\","json")


    def test_copySessionFolder(self):
        files.copySessionFolder("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\", "folder", "newFolder", selectedFiles=None)

    def test_createDir(self):
        files.createDir("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests-OUT\\LowLevel\\IO\\Hänibàl_files\\latin1_iæøå_test")

    def test_getDirs(self):
        dirs = files.getDirs("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\")
        print dirs


    def test_getFileCreationDate(self):
        files.getFileCreationDate("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\file.c3d")

class Test_vsk:

    def test_vskFiles(self):
        vskFile = vskTools.getVskFiles("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\")
        assert vskFile !=False

        vskTools.checkSetReadOnly("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\PIG-KAD.vsk")

        vskInstance = vskTools.Vsk("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\PIG-KAD.vsk")

# class Test_eclipse:
#
#     def test_eclipse(self):
#
#         files = eclipse.getEnfFiles("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\", enums.EclipseType.Trial)
#
#     def test_findCalibration(self):
#         calib = eclipse.findCalibration("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Eclipse\\Lecter-iæøå\\session\\")
#         assert calib == "PN01OP01S01STAT.Trial.enf"
#
#         motions = eclipse.findMotions("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Eclipse\\Lecter-iæøå\\session\\",ignoreSelect=False)
#         import ipdb; ipdb.set_trace()
# def test_generateEmptyENF(self):
#     eclipse.generateEmptyENF("C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data-Tests\\LowLevel\\IO\\Hänibàl_files\\")
