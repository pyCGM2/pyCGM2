# coding: utf-8
# pytest -s --disable-pytest-warnings  test_signal.py::Test_filtering::test_markerFilteringCGM23

import pytest
import matplotlib.pyplot as plt


import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Signal import signal_processing
from pyCGM2.Utils import files




class Test_filtering:
    def test_markerFiltering_noGap(self):

        PATH = pyCGM2.TEST_DATA_PATH+"LowLevel\\filtering\\"
        acq = btkTools.smartReader(PATH+"verif0.c3d")

        import ipdb; ipdb.set_trace()


        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"LowLevel\\filtering\\"
        files.createDir(DATA_PATH_OUT)


        markers = ['LASI', 'RASI', 'RPSI', 'LPSI', 'LTHI', 'LKNE', 'LTHAP', 'LTHAD', 'LTIB', 'LANK',
                    'LTIAP', 'LTIAD', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTHAP', 'RTHAD', 'RTIB', 'RANK',
                     'RTIAP', 'RTIAD', 'RHEE', 'RTOE', 'C7', 'T10', 'CLAV', 'STRN', 'LSHO', 'LELB', 'LWRA',
                      'LWRB', 'LFIN', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'RFIN', 'LFHD', 'LBHD', 'RFHD', 'RBHD']

        marker = "LFHD"
        array0 = acq.GetPoint(marker).GetValues()[:,1]
        signal_processing.markerFiltering(acq,markers,zerosFiltering=True,order=4, fc =6.0)

        btkTools.smartWriter(acq,DATA_PATH_OUT+"test_markerFilteringCGM23.c3d")

        array1 = acq.GetPoint(marker).GetValues()[:,1]

        plt.plot(array0,'-b')
        plt.plot(array1,'-r')


        plt.show()



    def test_markerFiltering_gaps(self):

        PATH = pyCGM2.TEST_DATA_PATH+"LowLevel\\filtering\\"
        acq = btkTools.smartReader(PATH+"verif0.c3d")

        marker = "LFHD"

        # artificial gap
        for i in range(0,10):
            acq.GetPoint(marker).SetValue(i,1,0)

        for i in range(100,150):
            acq.GetPoint(marker).SetValue(i,1,0)

        array0 = acq.GetPoint(marker).GetValues()[:,1]
        signal_processing.markerFiltering(acq,[marker],zerosFiltering=True,order=4, fc =6.0)

        array1 = acq.GetPoint(marker).GetValues()[:,1]

        plt.plot(array0,'-b')
        plt.plot(array1,'-r')

        plt.show()
