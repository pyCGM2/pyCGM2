# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyCGM2.Math import numeric
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import utils


#---------TESTS----------

def test_offset(value,acq,viconLabel, decimal=3):
    np.testing.assert_almost_equal(value,
    np.rad2deg(acq.GetMetaData().FindChild("PROCESSING").value().FindChild(viconLabel).value().GetInfo().ToDouble()[0]) , decimal = decimal)


def test_point(acq,RefLabel,LabelToTest,decimal = 3,init=-1,end=-1):
    init = 0  if init == -1 else  init - acq.GetFirstFrame()
    end = acq.GetLastFrame()- acq.GetFirstFrame()  if end == -1 else  end - acq.GetFirstFrame()

    np.testing.assert_almost_equal(acq.GetPoint(RefLabel).GetValues()[init:end,:],acq.GetPoint(LabelToTest).GetValues()[init:end,:],decimal = decimal)


def test_point_rms(acq,RefLabel,LabelToTest,threshold,init=-1,end=-1):
    init = 0  if init == -1 else  init - acq.GetFirstFrame()
    end = acq.GetLastFrame()- acq.GetFirstFrame()  if end == -1 else  end - acq.GetFirstFrame()

    np.testing.assert_array_less(numeric.rms( acq.GetPoint(RefLabel).GetValues()[init:end,:] -
        acq.GetPoint(LabelToTest).GetValues()[init:end,:], axis = 0),threshold)



def test_point_compareToRef(acqRef,acqToTest, labelToTest,decimal = 3):

    ref = acqRef.GetPoint(labelToTest).GetValues()
    values = acqToTest.GetPoint(labelToTest).GetValues()

    np.testing.assert_almost_equal(ref,values,decimal = decimal)


def test_point_rms_compareToRef(acqRef,acqToTest,labelToTest,threshold):

    ref = acqRef.GetPoint(labelToTest).GetValues()
    values = acqToTest.GetPoint(labelToTest).GetValues()


    np.testing.assert_array_less(numeric.rms( ref -
        values, axis = 0),threshold)



#---------DISPLAY----------
def print_offset(value,acq,viconLabel, decimal=3):
    LOGGER.logger.info(" offset [%s] => %f ( my value) = %f ( reference)"%(viconLabel,
                            value,
                            np.rad2deg(acq.GetMetaData().FindChild("PROCESSING").value().FindChild(viconLabel).value().GetInfo().ToDouble()[0])))


def plotComparisonOfPoint(acq,label,suffix,title=None,init=-1,end=-1):
    init = 0  if init == -1 else  init - acq.GetFirstFrame()
    end = acq.GetLastFrame()- acq.GetFirstFrame()  if end == -1 else  end - acq.GetFirstFrame()

    fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    ax1.plot(acq.GetPoint(label).GetValues()[init:end,0],"-r")
    ax1.plot(acq.GetPoint(label+"_"+suffix).GetValues()[init:end,0],"-b")


    ax2.plot(acq.GetPoint(label).GetValues()[init:end,1],"-r")
    ax2.plot(acq.GetPoint(label+"_"+suffix).GetValues()[init:end,1],"-b")

    ax3.plot(acq.GetPoint(label).GetValues()[init:end,2],"-r")
    ax3.plot(acq.GetPoint(label+"_"+suffix).GetValues()[init:end,2],"-b")

    if title is not None: plt.title(title)

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

def plotComparison_MomentPanel(acq,suffix1,suffix2,side,title=None,init=-1,end=-1):


    if side=="Left":
        sideLetter = "L"
    elif side=="Right":
        sideLetter = "R"
    else:
        raise Exception ("side is Left or Right only")

    suffix1  = "_"+suffix1  if suffix1 is not None else ""
    suffix2  = "_"+suffix2  if suffix2 is not None else ""

    init = 0  if init == -1 else  init - acq.GetFirstFrame()
    end = end = acq.GetLastFrame()- acq.GetFirstFrame()  if end == -1 else  end - acq.GetFirstFrame()

    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    if title is not None: fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax0 = plt.subplot(3,4,1)# Hip X extensor
    ax1 = plt.subplot(3,4,2)# Hip Y abductor
    ax2 = plt.subplot(3,4,3)# Hip Z rotation
    ax3 = plt.subplot(3,4,4)# Knee Z power

    ax4 = plt.subplot(3,4,5)# knee X extensor
    ax5 = plt.subplot(3,4,6)# knee Y abductor
    ax6 = plt.subplot(3,4,7)# knee Z rotation
    ax7 = plt.subplot(3,4,8)# knee Z power

    ax8 = plt.subplot(3,4,9)# ankle X plantar flexion
    ax9 = plt.subplot(3,4,10)# ankle Y rotation
    ax10 = plt.subplot(3,4,11)# ankle Z everter
    ax11 = plt.subplot(3,4,12)# ankle Z power

    ax0.plot(acq.GetPoint(sideLetter+ "HipMoment"+suffix1).GetValues()[init:end,0],"-r")
    ax0.plot(acq.GetPoint(sideLetter+ "HipMoment"+suffix2).GetValues()[init:end,0],"-b")

    ax1.plot(acq.GetPoint(sideLetter+ "HipMoment"+suffix1).GetValues()[init:end,1],"-r")
    ax1.plot(acq.GetPoint(sideLetter+ "HipMoment"+suffix2).GetValues()[init:end,1],"-b")

    ax2.plot(acq.GetPoint(sideLetter+ "HipMoment"+suffix1).GetValues()[init:end,2],"-r")
    ax2.plot(acq.GetPoint(sideLetter+ "HipMoment"+suffix2).GetValues()[init:end,2],"-b")


    ax4.plot(acq.GetPoint(sideLetter+ "KneeMoment"+suffix1).GetValues()[init:end,0],"-r")
    ax4.plot(acq.GetPoint(sideLetter+ "KneeMoment"+suffix2).GetValues()[init:end,0],"-b")

    ax5.plot(acq.GetPoint(sideLetter+ "KneeMoment"+suffix1).GetValues()[init:end,1],"-r")
    ax5.plot(acq.GetPoint(sideLetter+ "KneeMoment"+suffix2).GetValues()[init:end,1],"-b")

    ax6.plot(acq.GetPoint(sideLetter+ "KneeMoment"+suffix1).GetValues()[init:end,2],"-r")
    ax6.plot(acq.GetPoint(sideLetter+ "KneeMoment"+suffix2).GetValues()[init:end,2],"-b")


    ax8.plot(acq.GetPoint(sideLetter+ "AnkleMoment"+suffix1).GetValues()[init:end,0],"-r")
    ax8.plot(acq.GetPoint(sideLetter+ "AnkleMoment"+suffix2).GetValues()[init:end,0],"-b")

    ax9.plot(acq.GetPoint(sideLetter+ "AnkleMoment"+suffix1).GetValues()[init:end,1],"-r")
    ax9.plot(acq.GetPoint(sideLetter+ "AnkleMoment"+suffix2).GetValues()[init:end,1],"-b")

    ax10.plot(acq.GetPoint(sideLetter+ "AnkleMoment"+suffix1).GetValues()[init:end,2],"-r")
    ax10.plot(acq.GetPoint(sideLetter+ "AnkleMoment"+suffix2).GetValues()[init:end,2],"-b")

    plt.show()


def plotComparison_ForcePanel(acq,suffix1,suffix2,side,title=None,init=-1,end=-1):

    if side=="Left":
        sideLetter = "L"
    elif side=="Right":
        sideLetter = "R"
    else:
        raise Exception ("side is Left or Right only")


    suffix1  = "_"+suffix1  if suffix1 is not None else ""
    suffix2  = "_"+suffix2  if suffix2 is not None else ""

    init = 0  if init == -1 else  init - acq.GetFirstFrame()
    end = acq.GetLastFrame()- acq.GetFirstFrame()  if end == -1 else  end - acq.GetFirstFrame()


    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    if title is not None: fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax0 = plt.subplot(3,4,1)# Hip X extensor
    ax1 = plt.subplot(3,4,2)# Hip Y abductor
    ax2 = plt.subplot(3,4,3)# Hip Z rotation
    ax3 = plt.subplot(3,4,4)# Knee Z power

    ax4 = plt.subplot(3,4,5)# knee X extensor
    ax5 = plt.subplot(3,4,6)# knee Y abductor
    ax6 = plt.subplot(3,4,7)# knee Z rotation
    ax7 = plt.subplot(3,4,8)# knee Z power

    ax8 = plt.subplot(3,4,9)# ankle X plantar flexion
    ax9 = plt.subplot(3,4,10)# ankle Y rotation
    ax10 = plt.subplot(3,4,11)# ankle Z everter
    ax11 = plt.subplot(3,4,12)# ankle Z power

    ax0.plot(acq.GetPoint(sideLetter+ "HipForce"+suffix1).GetValues()[init:end,0],"-r")
    ax0.plot(acq.GetPoint(sideLetter+ "HipForce"+suffix2).GetValues()[init:end,0],"-b")

    ax1.plot(acq.GetPoint(sideLetter+ "HipForce"+suffix1).GetValues()[init:end,1],"-r")
    ax1.plot(acq.GetPoint(sideLetter+ "HipForce"+suffix2).GetValues()[init:end,1],"-b")

    ax2.plot(acq.GetPoint(sideLetter+ "HipForce"+suffix1).GetValues()[init:end,2],"-r")
    ax2.plot(acq.GetPoint(sideLetter+ "HipForce"+suffix2).GetValues()[init:end,2],"-b")


    ax4.plot(acq.GetPoint(sideLetter+ "KneeForce"+suffix1).GetValues()[init:end,0],"-r")
    ax4.plot(acq.GetPoint(sideLetter+ "KneeForce"+suffix2).GetValues()[init:end,0],"-b")

    ax5.plot(acq.GetPoint(sideLetter+ "KneeForce"+suffix1).GetValues()[init:end,1],"-r")
    ax5.plot(acq.GetPoint(sideLetter+ "KneeForce"+suffix2).GetValues()[init:end,1],"-b")

    ax6.plot(acq.GetPoint(sideLetter+ "KneeForce"+suffix1).GetValues()[init:end,2],"-r")
    ax6.plot(acq.GetPoint(sideLetter+ "KneeForce"+suffix2).GetValues()[init:end,2],"-b")


    ax8.plot(acq.GetPoint(sideLetter+ "AnkleForce"+suffix1).GetValues()[init:end,0],"-r")
    ax8.plot(acq.GetPoint(sideLetter+ "AnkleForce"+suffix2).GetValues()[init:end,0],"-b")

    ax9.plot(acq.GetPoint(sideLetter+ "AnkleForce"+suffix1).GetValues()[init:end,1],"-r")
    ax9.plot(acq.GetPoint(sideLetter+ "AnkleForce"+suffix2).GetValues()[init:end,1],"-b")

    ax10.plot(acq.GetPoint(sideLetter+ "AnkleForce"+suffix1).GetValues()[init:end,2],"-r")
    ax10.plot(acq.GetPoint(sideLetter+ "AnkleForce"+suffix2).GetValues()[init:end,2],"-b")

    plt.show()
