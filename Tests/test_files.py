# coding: utf-8
# pytest -s --disable-pytest-warnings  test_files.py::Test_UtilsFiles::test_loadAndSaveAnalysis
# from __future__ import unicode_literals
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

        contentYaml = files.openFile(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","file.yaml")
        contentJson = files.openFile(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","file.json")

        assert contentYaml !=False
        assert contentJson !=False

        files.saveJson(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\", "testJsonOut", contentJson)


    # def test_loadAndSaveModel(self):
    #
    #     model = files.loadModel(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","COTTREL Simon")
    #     assert model !=False
    #     files.saveModel(model,pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","testModelOut")
    #
    # def test_loadAndSaveAnalysis(self):
    #
    #     ana = files.loadAnalysis(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","Condition1")
    #     assert ana !=False
    #
    #     files.saveAnalysis(ana,pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","testAnalysisOut")

    def test_getTranslators(self):
        translators = files.getTranslators(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\", translatorType = "CGM1.translators")
        assert translators !=False

    def test_getIKweightSet(self):
        ikws = files.getIKweightSet(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\", "CGM2_2.ikw")
        assert ikws !=False

    def test_getMpFileContent(self):
        mp = files.getMpFileContent(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","mp.pyCGM2","Nick")
        assert mp !=False

    def test_getFiles(self):
        fs = files.getFiles(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\","json")

    def test_getFiles2(self):
        fs = files.getFiles(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hannibal_c3d\\","Trial.enf")

    def test_copySessionFolder(self):
        files.copySessionFolder(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\", "folder", "newFolder", selectedFiles=None)

    def test_createDir(self):
        files.createDir(pyCGM2.TEST_DATA_PATH + "-OUT\\LowLevel\\IO\\Hanibal_files\\latin1_iæøå_test")

    def test_getDirs(self):
        dirs = files.getDirs(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\")
        print(dirs)


    def test_getFileCreationDate(self):
        files.getFileCreationDate(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\file.c3d")

class Test_vsk:

    def test_vskFiles(self):
        vskFile = vskTools.getVskFiles(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\")
        assert vskFile !=False

        vskTools.checkSetReadOnly(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\PIG-KAD.vsk")

        vskInstance = vskTools.Vsk(pyCGM2.TEST_DATA_PATH + "\\LowLevel\\IO\\Hanibal_files\\PIG-KAD.vsk")
