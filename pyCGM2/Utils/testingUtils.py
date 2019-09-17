# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

#TODO : Resume functions


def pointsComparison(acq,label,suffix):

    fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    ax1.plot(acq.GetPoint(label).GetValues()[:,0],"-r")
    ax1.plot(acq.GetPoint(label+"_"+suffix).GetValues()[:,0],"-b")


    ax2.plot(acq.GetPoint(label).GetValues()[:,1],"-r")
    ax2.plot(acq.GetPoint(label+"_"+suffix).GetValues()[:,1],"-b")

    ax3.plot(acq.GetPoint(label).GetValues()[:,2],"-r")
    ax3.plot(acq.GetPoint(label+"_"+suffix).GetValues()[:,2],"-b")

    plt.show()

def TestingPoint(acq,RefLabel,LabelToTest,meanValue=False,decimal = 3):

    if meanValue:
        np.testing.assert_almost_equal(acq.GetPoint(RefLabel).GetValues().mean(axis=0),acq.GetPoint(LabelToTest).GetValues().mean(axis=0),decimal = decimal)
    else:
        np.testing.assert_almost_equal(acq.GetPoint(RefLabel).GetValues(),acq.GetPoint(LabelToTest).GetValues(),decimal = decimal)


def TestingOffset_lowerLimb(acqStatic,model):

    # tibial torsion
    ltt_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LTibialTorsion").value().GetInfo().ToDouble()[0])
    rtt_vicon =np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RTibialTorsion").value().GetInfo().ToDouble()[0])

    # shank abAdd offset
    abdAdd_l = model.getViconAnkleAbAddOffset("Left")
    abdAdd_r = model.getViconAnkleAbAddOffset("Right")

    # thigh and shank Offsets
    lto = model.getViconThighOffset("Left")
    lso = model.getViconShankOffset("Left")
    rto = model.getViconThighOffset("Right")
    rso = model.getViconShankOffset("Right")

    # foot offsets
    spf_l,sro_l= model.getViconFootOffset("Left")
    spf_r,sro_r= model.getViconFootOffset("Right")

    vicon_abdAdd_l = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LAnkleAbAdd").value().GetInfo().ToDouble()[0])
    vicon_abdAdd_r = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RAnkleAbAdd").value().GetInfo().ToDouble()[0])


    lto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0])
    lso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0])

    rto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0])
    rso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0])

    vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
    vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
    vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
    vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])



    # tibial torsion
    np.testing.assert_almost_equal(-ltt_vicon,model.mp_computed["LeftTibialTorsionOffset"] , decimal = 3)
    np.testing.assert_almost_equal(rtt_vicon,model.mp_computed["RightTibialTorsionOffset"] , decimal = 3)


    # thigh and shank Offsets
    np.testing.assert_almost_equal(lto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)
    np.testing.assert_almost_equal(lso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)

    np.testing.assert_almost_equal(rto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)
    np.testing.assert_almost_equal(rso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)

    # shank abAdd offset
    np.testing.assert_almost_equal(abdAdd_l ,vicon_abdAdd_l  , decimal = 3)
    np.testing.assert_almost_equal(abdAdd_r ,vicon_abdAdd_r  , decimal = 3)

    # foot offsets
    np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
    np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
    np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
    np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)


def TestingRotationMatrix(acq,model,frameIndex,originLabel, proximalLabel, lateralLabel, sequence,decimal =3):

    """    TestingRotationMatrix(acq,model,10, "TRXO", "TRXA", "TRXL", "XZY")"""

    pt1 = acq.GetPoint(originLabel).GetValues()[frameIndex,:]
    pt2 = acq.GetPoint(proximalLabel).GetValues()[frameIndex,:]
    pt3 = acq.GetPoint(lateralLabel).GetValues()[frameIndex,:]

    a1 = (pt2-pt1)
    a1 = a1/np.linalg.norm(a1)
    v = (pt3-pt1)
    v = v/np.linalg.norm(v)
    a2 = np.cross(a1,v)
    a2 = a2/np.linalg.norm(a2)
    x,y,z,R_vicon = frame.setFrameData(a1,a2,sequence)

    R_model= model.getSegment("Thorax").anatomicalFrame.motion[frameIndex].getRotation()

    np.testing.assert_almost_equal( R_model,
                            R_vicon, decimal =decimal)
