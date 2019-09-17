import logging
import pyCGM2
from pyCGM2.Utils import files


from pyCGM2.Tools import btkTools

import sys
import os
import json
# from qtmWebGaitReport import parserUploader


# class WebReportFilter(object):
#     def __init__(self,workingDirectory):
#
#         if os.path.isfile(pyCGM2.PYCGM2_SETTINGS_FOLDER + 'config.json'):
#             with open(pyCGM2.PYCGM2_SETTINGS_FOLDER + 'config.json') as jsonDataFile:
#                 configData = json.load(jsonDataFile)
#         else:
#             print "Config.json not found at " + os.getcwd()
#
#
#         self.processing = parserUploader.ParserUploader(workingDirectory,configData)
#
#
#     def exportJson(self):
#
#         jsonData = self.processing.createReportJson()
#         files.saveJson("","jsonData.json",jsonData)
#
#
#     def upload(self):
#         self.processing.Upload()
