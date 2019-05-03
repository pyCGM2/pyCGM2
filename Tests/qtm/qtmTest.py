# -*- coding: utf-8 -*-
import logging
import os
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Utils import files
from pyCGM2.Utils.utils import *
from pyCGM2.qtm import qtmTools



class QtmTests():

    @classmethod
    def sessionReaderTest(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\QUALISYS DATA\\pyCGM2-Data\\sessionXML\\"
        file="sessionHUG.xml"
        soup = files.readXml(DATA_PATH,file)


        staticMeasurement = qtmTools.findStatic(soup)
        dynamicMeaurements= qtmTools.findDynamic(soup)
        qtmTools.SubjectMp(soup)

        types = qtmTools.detectMeasurementType(soup)

        mfpa = qtmTools.getForcePlateAssigment(dynamicMeaurements[0])

        qtmTools.isType(dynamicMeaurements[0],"Gait")






if __name__ == "__main__":

    QtmTests.sessionReaderTest()
