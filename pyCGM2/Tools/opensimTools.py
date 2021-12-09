# -*- coding: utf-8 -*-
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Processing import progressionFrame
from pyCGM2.Tools import btkTools
import os
import numpy as np

import pyCGM2
LOGGER = pyCGM2.LOGGER

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim


def createGroundReactionForceMOT_file(DATA_PATH, c3dFile):

    c3dFileAdapter = opensim.C3DFileAdapter()
    # ForceLocation_OriginOfForcePlate , ForceLocation_CenterOfPressure
    c3dFileAdapter.setLocationForForceExpression(
        opensim.C3DFileAdapter.ForceLocation_PointOfWrenchApplication)
    tables = c3dFileAdapter.read(DATA_PATH + c3dFile)

    forces = c3dFileAdapter.getForcesTable(tables)
    forcesFlat = forces.flatten()

    forcesFilename = DATA_PATH+c3dFile[:-4] + '_GRF.mot'
    stoAdapter = opensim.STOFileAdapter()
    stoAdapter.write(forcesFlat, forcesFilename)


def smartTrcExport(acq, filenameNoExt):
    writerDyn = btk.btkAcquisitionFileWriter()
    writerDyn.SetInput(acq)
    writerDyn.SetFilename(filenameNoExt + ".trc")
    writerDyn.Update()


def setGlobalTransormation_lab_osim(axis, forwardProgression):
    """ Todo : incomplet, il faut traiter tous les cas """
    if axis == "X":
        if forwardProgression:
            R_LAB_OSIM = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        else:
            R_LAB_OSIM = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    elif axis == "Y":
        if forwardProgression:
            R_LAB_OSIM = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        else:
            R_LAB_OSIM = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

    else:
        raise Exception("[pyCGM2] - Global Referential not configured yet")

    return R_LAB_OSIM


def globalTransformationLabToOsim(acq, R_LAB_OSIM):

    points = acq.GetPoints()
    for it in btk.Iterate(points):
        if it.GetType() == btk.btkPoint.Marker:
            values = np.zeros(it.GetValues().shape)
            for i in range(0, it.GetValues().shape[0]):
                values[i, :] = np.dot(R_LAB_OSIM, it.GetValues()[i, :])
            it.SetValues(values)


def sto2pointValues(storageObject, label, R_LAB_OSIM):

    # storageObject = opensim.Storage(stoFilename)
    index_x = storageObject.getStateIndex(label+"_tx")
    index_y = storageObject.getStateIndex(label+"_ty")
    index_z = storageObject.getStateIndex(label+"_tz")

    array_x = opensim.ArrayDouble()
    storageObject.getDataColumn(index_x, array_x)

    array_y = opensim.ArrayDouble()
    storageObject.getDataColumn(index_y, array_y)

    array_z = opensim.ArrayDouble()
    storageObject.getDataColumn(index_z, array_z)

    n = array_x.getSize()
    pointValues = np.zeros((n, 3))
    for i in range(0, n):
        pointValues[i, 0] = array_x.getitem(i)
        pointValues[i, 1] = array_y.getitem(i)
        pointValues[i, 2] = array_z.getitem(i)

    for i in range(0, n):
        pointValues[i, :] = np.dot(R_LAB_OSIM.T, pointValues[i, :])*1000.0

    return pointValues


def mot2pointValues(motFilename, labels, orientation=[1, 1, 1]):
    storageObject = opensim.Storage(motFilename)

    index_x = storageObject.getStateIndex(labels[0])
    index_y = storageObject.getStateIndex(labels[1])
    index_z = storageObject.getStateIndex(labels[2])

    array_x = opensim.ArrayDouble()
    storageObject.getDataColumn(index_x, array_x)

    array_y = opensim.ArrayDouble()
    storageObject.getDataColumn(index_y, array_y)

    array_z = opensim.ArrayDouble()
    storageObject.getDataColumn(index_z, array_z)

    n = array_x.getSize()
    pointValues = np.zeros((n, 3))
    for i in range(0, n):
        pointValues[i, 0] = orientation[0]*array_x.getitem(i)
        pointValues[i, 1] = orientation[1]*array_y.getitem(i)
        pointValues[i, 2] = orientation[2]*array_z.getitem(i)

    return pointValues


def footReactionMotFile(acq,filename):

    mappedForcePlate = forceplates.matchingFootSideOnForceplate(acq)

    pfp = progressionFrame.PelvisProgressionFrameProcedure()
    pff = progressionFrame.ProgressionFrameFilter(acq, pfp)
    pff.compute()
    progressionAxis = pff.outputs["progressionAxis"]
    forwardProgression = pff.outputs["forwardProgression"]

    R_LAB_OSIM = setGlobalTransormation_lab_osim(
        progressionAxis, forwardProgression)

    leftFootWrench, rightFootWrench = forceplates.combineForcePlate(
        acq, mappedForcePlate)

    analogFrames = acq.GetAnalogFrameNumber()
    analogFreq = acq.GetAnalogFrequency()
    time = np.arange(0, analogFrames/analogFreq, 1/analogFreq)

    LForce_osim = np.zeros((analogFrames, 3))
    LMoment_osim = np.zeros((analogFrames, 3))
    LPosition_osim = np.zeros((analogFrames, 3))
    if leftFootWrench is not None:
        for i in range(0, analogFrames):
            LForce_osim[i, :] = np.dot(
                R_LAB_OSIM, leftFootWrench.GetForce().GetValues()[i, :])
            LMoment_osim[i, :] = np.dot(
                R_LAB_OSIM, leftFootWrench.GetMoment().GetValues()[i, :])
            LPosition_osim[i, :] = np.dot(
                R_LAB_OSIM, leftFootWrench.GetPosition().GetValues()[i, :])

    RForce_osim = np.zeros((analogFrames, 3))
    RMoment_osim = np.zeros((analogFrames, 3))
    RPosition_osim = np.zeros((analogFrames, 3))

    if rightFootWrench is not None:
        for i in range(0, analogFrames):
            RForce_osim[i, :] = np.dot(
                R_LAB_OSIM, rightFootWrench.GetForce().GetValues()[i, :])
            RMoment_osim[i, :] = np.dot(
                R_LAB_OSIM, rightFootWrench.GetMoment().GetValues()[i, :])
            RPosition_osim[i, :] = np.dot(
                R_LAB_OSIM, rightFootWrench.GetPosition().GetValues()[i, :])


    file1 = open(filename, "w")
    # file1.write("DataRate = %.5f\n" % (analogFreq))
    # file1.write("DataType=double\n")
    # file1.write("version=3\n")
    # file1.write("OpenSimVersion=4.1\n")
    # file1.write("endheader\n")

    file1.write("subject01_walk1_grf.mot\n")
    file1.write("version=1\n")
    file1.write("nRows=%i\n"%(analogFrames))
    file1.write("nColumns=19\n")
    file1.write("inDegrees=yes\n")
    file1.write("endheader\n")


    file1.write("time	ground_force_vx	ground_force_vy	ground_force_vz	ground_force_px	ground_force_py	ground_force_pz	1_ground_force_vx	1_ground_force_vy	1_ground_force_vz	1_ground_force_px	1_ground_force_py	1_ground_force_pz	ground_torque_x	ground_torque_y	ground_torque_z	1_ground_torque_x	1_ground_torque_y	1_ground_torque_z\n")
    #file1.write("time\tLeftFoot_Fx\tLeftFoot_Fy\tLeftFoot_Fz\tLeftFoot_Mx\tLeftFoot_My\tLeftFoot_Mz\tLeftFoot_Px\tLeftFoot_Py\tLeftFoot_Pz\tRightFoot_Fx\tRightFoot_Fy\tRightFoot_Fz\tRightFoot_Mx\tRightFoot_My\tRightFoot_Mz\tRightFoot_Px\tRightFoot_Py\tRightFoot_Pz\n")
    for i in range(0, analogFrames):
        file1.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"% (time[i],
        RForce_osim[i, 0],    RForce_osim[i, 1],    RForce_osim[i, 2],
        RPosition_osim[i, 0]/1000.0, RPosition_osim[i, 1]/1000.0, RPosition_osim[i, 2]/1000.0,
        LForce_osim[i, 0],    LForce_osim[i, 1],    LForce_osim[i, 2],
        LPosition_osim[i, 0]/1000.0, LPosition_osim[i, 1]/1000.0, LPosition_osim[i, 2]/1000.0,
        RMoment_osim[i, 0]/1000.0,   RMoment_osim[i, 1]/1000.0,   RMoment_osim[i, 2]/1000.0,
        LMoment_osim[i, 0]/1000.0,   LMoment_osim[i, 1]/1000.0,   LMoment_osim[i, 2]/1000.0))


            # LForce_osim[i, 0],    LForce_osim[i, 1],    LForce_osim[i, 2],
            # LMoment_osim[i, 0],   LMoment_osim[i, 1],   LMoment_osim[i, 2],
            # LPosition_osim[i, 0], LPosition_osim[i, 1], LPosition_osim[i, 2],
            # RForce_osim[i, 0],    RForce_osim[i, 1],    RForce_osim[i, 2],
            # RMoment_osim[i, 0],   RMoment_osim[i, 1],   RMoment_osim[i, 2],
            # RPosition_osim[i, 0], RPosition_osim[i, 1], RPosition_osim[i, 2]))

    file1.close()
