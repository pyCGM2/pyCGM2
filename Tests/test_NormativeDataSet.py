# coding: utf-8
#pytest -s --disable-pytest-warnings  test_NormativeDataSet.py::Test_NormativeDataSet::test_Schwartz2008

from __future__ import unicode_literals
import pytest

import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Lib import analysis, plot
from pyCGM2.Processing import scores
from pyCGM2.Utils import files

from pyCGM2.Report import normativeDatasets






class Test_NormativeDataSet:

    def test_Schwartz2008(self):


        nds = normativeDatasets.NormativeData("Schwartz2008","Free")

        nds2 = normativeDatasets.Schwartz2008("Free")
        nds2.constructNormativeData()

        np.testing.assert_almost_equal(nds.data["PelvisAngles"]["mean"] ,nds2.data["Pelvis.Angles"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["HipAngles"]["mean"] ,nds2.data["Hip.Angles"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["KneeAngles"]["mean"] ,nds2.data["Knee.Angles"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["AnkleAngles"]["mean"] ,nds2.data["Ankle.Angles"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["FootAngles"]["mean"] ,nds2.data["Foot.Angles"]["mean"],decimal=3)

        np.testing.assert_almost_equal(nds.data["HipMoment"]["mean"] ,nds2.data["Hip.Moment"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["KneeMoment"]["mean"] ,nds2.data["Knee.Moment"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["AnkleMoment"]["mean"] ,nds2.data["Ankle.Moment"]["mean"],decimal=3)

        np.testing.assert_almost_equal(nds.data["HipPower"]["mean"] ,nds2.data["Hip.Power"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["KneePower"]["mean"] ,nds2.data["Knee.Power"]["mean"],decimal=3)
        np.testing.assert_almost_equal(nds.data["AnklePower"]["mean"] ,nds2.data["Ankle.Power"]["mean"],decimal=3)


        # # import ipdb; ipdb.set_trace()
        # plt.plot(nds.data["KneeAngles"]["mean"])
        # plt.plot(nds2.data["Knee.Angles"]["mean"],"o")
        # plt.show()
        #
        # import ipdb; ipdb.set_trace()
