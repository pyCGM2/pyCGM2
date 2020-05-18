# coding: utf-8
# pytest -s --disable-pytest-warnings  test_signal.py::Test_filtering::test_markerFiltering

import pytest
import matplotlib.pyplot as plt


from pyCGM2.Tools import btkTools
from pyCGM2.Signal import signal_processing




class Test_filtering:
    def test_markerFiltering(self):

        PATH = "C:\\Users\\FLEBOEUF.CHU-NANTES\\Documents\\DATA\\Vicon data\\pyCGM2-Data-Tests\\LowLevel\\gap\\gap_begin\\"
        acq = btkTools.smartReader(PATH+"Gait - CGM2 1.c3d")

        marker = "CLAV"

        array0 = acq.GetPoint(marker).GetValues()[:,1]

        signal_processing.markerFiltering(acq)

        array1 = acq.GetPoint(marker).GetValues()[:,1]

        plt.plot(array0,'-b')
        plt.plot(array1,'-r')


        plt.show()
