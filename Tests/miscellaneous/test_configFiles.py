# -*- coding: utf-8 -*-

import yaml
import json
import ipdb
from pyCGM2.Utils import files
import logging
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

import pyCGM2


class configFile_tests():

    @classmethod
    def yamltest(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\"
        ymlConfigFile = "pipeline_yml.pyCGM2"

        struct = files.openYaml(DATA_PATH,ymlConfigFile)


    @classmethod
    def jsontest(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\"
        jsonConfigFile = "pipeline.pyCGM2"

        struct = files.openJson(DATA_PATH,jsonConfigFile )

    @classmethod
    def detectTest_yaml(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\"
        configFile = "pipeline_yml.pyCGM2"

        content = files.openPipelineFile(DATA_PATH,configFile)

    @classmethod
    def detectTest_json(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\"
        configFile = "pipeline.pyCGM2"

        content = files.openPipelineFile(DATA_PATH,configFile)


if __name__ == "__main__":

    configFile_tests.yamltest()
    configFile_tests.jsontest()
    configFile_tests.detectTest_yaml()
    configFile_tests.detectTest_json()
