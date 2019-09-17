# -*- coding: utf-8 -*-
import logging
import os
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

from pyCGM2.qtm import qtmTools, qtmFilters
import ipdb


class QtmReportTests():

    @classmethod
    def reportTest(cls):
        workingDirectory = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2-Qualisys\\Data\WebReport\\"
        report =  qtmFilters.WebReportFilter(workingDirectory)
        report.exportJson()

        #report.upload()

    @classmethod
    def noEnf_reportTest(cls):
        workingDirectory = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2-Qualisys\\Data\WebReport - noEnf\\"
        report =  qtmFilters.WebReportFilter(workingDirectory)
        report.exportJson()


if __name__ == "__main__":


    QtmReportTests.noEnf_reportTest()
