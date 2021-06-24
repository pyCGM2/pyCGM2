import pyCGM2
from pyCGM2.Lib import eventDetector
from pyCGM2.Tools import btkTools

def example1():

    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\Session 1 - CGM1\\"

    trialName = "03367_05136_20200604-GBNNN-VDEF-01.c3d"

    acqGait = btkTools.smartReader(DATA_PATH+trialName)
    eventDetector.zeni(acqGait)
    btkTools.smartWriter(acqGait, DATA_PATH+trialName[:-4]+"-event.c3d")



if __name__ == '__main__':
    example1()
