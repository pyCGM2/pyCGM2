# -*- coding: utf-8 -*-
import ipdb
import logging

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2 libraries
from pyCGM2 import ma
from pyCGM2.Tools import btkTools,trialTools




class Tests():

    @classmethod
    def croppedC3d(cls):

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"

        acq = btkTools.smartReader(DATA_PATH+"gait_cropped.c3d")


        root = trialTools.convertBtkAcquisition(acq,returnType="Root")

        ma.io.write(root,"gait_cropped_checked.c3d")

    @classmethod
    def noCroppedC3d(cls):

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"

        acq = btkTools.smartReader(DATA_PATH+"gait_noCropped.c3d")


        root = trialTools.convertBtkAcquisition(acq,returnType="Root")

        ma.io.write(root,"gait_noCropped_checked.c3d")





if __name__ == "__main__":

    Tests.croppedC3d()
    Tests.noCroppedC3d()
