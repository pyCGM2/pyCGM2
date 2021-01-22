# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_forcePlate_lowLevel.py::Test_ForcePlateTypeReader::test_ForcePlateType5

import numpy as np
import pyCGM2
from pyCGM2.Tools import btkTools



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
