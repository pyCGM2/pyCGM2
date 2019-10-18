# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyCGM2.Math import numeric
import logging
from pyCGM2.Utils import utils


#---------TESTS----------

def test_offset(value,acq,viconLabel, decimal=3):
    np.testing.assert_almost_equal(value,
    np.rad2deg(acq.GetMetaData().FindChild(utils.str("PROCESSING")).value().FindChild(utils.str(viconLabel)).value().GetInfo().ToDouble()[0]) , decimal = decimal)


def test_point(acq,RefLabel,LabelToTest,decimal = 3):
    np.testing.assert_almost_equal(acq.GetPoint(utils.str(RefLabel)).GetValues(),acq.GetPoint(utils.str(LabelToTest)).GetValues(),decimal = decimal)


def test_point_rms(acq,RefLabel,LabelToTest,threshold):
    np.testing.assert_array_less(numeric.rms((acq.GetPoint(utils.str(RefLabel)).GetValues()-acq.GetPoint(utils.str(LabelToTest)).GetValues()[init:end,:]), axis = 0),
                                 threshold)



#---------DISPLAY----------
def print_offset(value,acq,viconLabel, decimal=3):
    logging.info(" offset [%s] => %f ( my value) = %f ( reference)"%(viconLabel,
                            value,
                            np.rad2deg(acq.GetMetaData().FindChild(utils.str("PROCESSING")).value().FindChild(utils.str(viconLabel)).value().GetInfo().ToDouble()[0])))


def plotComparisonOfPoint(acq,label,suffix):

    fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    ax1.plot(acq.GetPoint(utils.str(label)).GetValues()[:,0],"-r")
    ax1.plot(acq.GetPoint(utils.str(label+"_"+suffix)).GetValues()[:,0],"-b")


    ax2.plot(acq.GetPoint(utils.str(label)).GetValues()[:,1],"-r")
    ax2.plot(acq.GetPoint(utils.str(label+"_"+suffix)).GetValues()[:,1],"-b")

    ax3.plot(acq.GetPoint(utils.str(label)).GetValues()[:,2],"-r")
    ax3.plot(acq.GetPoint(utils.str(label+"_"+suffix)).GetValues()[:,2],"-b")

    plt.show()

def plotValuesComparison(values0,values1):

    fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    ax1.plot(values0[:,0],"-r")
    ax1.plot(values1[:,0],"-b")


    ax2.plot(values0[:,1],"-r")
    ax2.plot(values1[:,1],"-b")

    ax3.plot(values0[:,2],"-r")
    ax3.plot(values1[:,2],"-b")

    plt.show()
