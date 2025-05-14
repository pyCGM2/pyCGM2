# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_forcePlate_lowLevel.py::Test_ForcePlateTypeReader::test_ForcePlateType5

import numpy as np
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Signal import signal_processing
import btk

import matplotlib.pyplot as plt


class Test_ForcePlateTypeReader():

    def test_ForcePlateType5(self):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        #files.createDir(DATA_PATH_OUT)

        btkAcq = btkTools.smartReader(MAIN_PATH + "HUG_gait_type5_origin.c3d")
        btkAcq_correct = btkTools.smartReader(MAIN_PATH + "HUG_gait_type5_convert.c3d")


        flabels = ["Force.Fx0","Force.Fx1", "Force.Fy0","Force.Fy1", "Force.Fz0","Force.Fz1"]
        mlabels = ["Moment.Mx0","Moment.Mx1", "Moment.My0","Moment.My1", "Moment.Mz0","Moment.Mz1"]

        for label in flabels:
            np.testing.assert_almost_equal(btkAcq.GetAnalog(label).GetValues(),btkAcq_correct.GetAnalog(label).GetValues(),decimal = 2)

        for label in mlabels:
            np.testing.assert_almost_equal(btkAcq.GetAnalog(label).GetValues(),btkAcq_correct.GetAnalog(label).GetValues(),decimal = 2)


class Test_ForcePlateFiltering():

    def test_type2(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        btkAcq = btkTools.smartReader(DATA_PATH + "HUG_gait_type5_convert.c3d")

        acqClone = btk.btkAcquisition.Clone(btkAcq)

        pfe0 = btk.btkForcePlatformsExtractor()
        grwf0 = btk.btkGroundReactionWrenchFilter()
        pfe0.SetInput(acqClone)
        pfc0 = pfe0.GetOutput()
        grwf0.SetInput(pfc0)
        grwc0 = grwf0.GetOutput()
        grwc0.Update()

        signal_processing.forcePlateFiltering(btkAcq,order=4, fc =5)


        pfe = btk.btkForcePlatformsExtractor()
        grwf = btk.btkGroundReactionWrenchFilter()
        pfe.SetInput(btkAcq)
        pfc = pfe.GetOutput()
        grwf.SetInput(pfc)
        grwc = grwf.GetOutput()
        grwc.Update()

        plt.plot(grwc0.GetItem(0).GetForce().GetValues()[:,2])
        plt.plot(grwc.GetItem(0).GetForce().GetValues()[:,2],"-g")
        plt.show()


    def test_bertecType3(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Issues\\qualisys\\issue_digitalBertec_3FP\\Bertec data\\"
        staticFilename = "Static LB - CGM2 2-fromQTM.c3d"
        reconstructFilenameLabelled= "Gait LB-CGM21-fromVincent.c3d"
        btkAcq = btkTools.smartReader(str(DATA_PATH +  reconstructFilenameLabelled))

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        btkAcq = btkTools.smartReader(DATA_PATH + "HUG_gait_type5_convert.c3d")

        acqClone = btk.btkAcquisition.Clone(btkAcq)

        pfe0 = btk.btkForcePlatformsExtractor()
        grwf0 = btk.btkGroundReactionWrenchFilter()
        pfe0.SetInput(acqClone)
        pfc0 = pfe0.GetOutput()
        grwf0.SetInput(pfc0)
        grwc0 = grwf0.GetOutput()
        grwc0.Update()

        signal_processing.forcePlateFiltering(btkAcq,order=4, fc =5)


        pfe = btk.btkForcePlatformsExtractor()
        grwf = btk.btkGroundReactionWrenchFilter()
        pfe.SetInput(btkAcq)
        pfc = pfe.GetOutput()
        grwf.SetInput(pfc)
        grwc = grwf.GetOutput()
        grwc.Update()

        plt.plot(grwc0.GetItem(0).GetForce().GetValues()[:,2])
        plt.plot(grwc.GetItem(0).GetForce().GetValues()[:,2],"-g")
        plt.show()
 
