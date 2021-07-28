# -*- coding: utf-8 -*-
import os
import numpy as np

import pyCGM2; LOGGER = pyCGM2.LOGGER


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


def createGroundReactionForceMOT_file(DATA_PATH,c3dFile):

    c3dFileAdapter = opensim.C3DFileAdapter()
    c3dFileAdapter.setLocationForForceExpression(opensim.C3DFileAdapter.ForceLocation_PointOfWrenchApplication); #ForceLocation_OriginOfForcePlate , ForceLocation_CenterOfPressure
    tables = c3dFileAdapter.read(DATA_PATH + c3dFile)

    forces = c3dFileAdapter.getForcesTable(tables)
    forcesFlat = forces.flatten()

    forcesFilename = DATA_PATH+c3dFile[:-4] + '_GRF.mot'
    stoAdapter = opensim.STOFileAdapter()
    stoAdapter.write(forcesFlat, forcesFilename)


def smartTrcExport(acq,filenameNoExt):
    writerDyn = btk.btkAcquisitionFileWriter()
    writerDyn.SetInput(acq)
    writerDyn.SetFilename(filenameNoExt + ".trc")
    writerDyn.Update()


def setGlobalTransormation_lab_osim(axis,forwardProgression):
    """ Todo : incomplet, il faut traiter tous les cas """
    if axis =="X":
        if forwardProgression:
            R_LAB_OSIM=np.array([[1,0,0],[0,0,1],[0,-1,0]])
        else:
            R_LAB_OSIM=np.array([[-1,0,0],[0,0,1],[0,1,0]])

    elif axis =="Y":
        if forwardProgression:
            R_LAB_OSIM=np.array([[0,1,0],[0,0,1],[1,0,0]])
        else:
            R_LAB_OSIM=np.array([[0,-1,0],[0,0,1],[-1,0,0]])

    else:
        raise Exception("[pyCGM2] - Global Referential not configured yet")


    return R_LAB_OSIM



def globalTransformationLabToOsim(acq,R_LAB_OSIM):

    points = acq.GetPoints()
    for it in btk.Iterate(points):
        if it.GetType() == btk.btkPoint.Marker:
            values = np.zeros(it.GetValues().shape)
            for i in range(0,it.GetValues().shape[0]):
                values[i,:] = np.dot(R_LAB_OSIM,it.GetValues()[i,:])
            it.SetValues(values)


def sto2pointValues(storageObject,label,R_LAB_OSIM):

    # storageObject = opensim.Storage(stoFilename)
    index_x =storageObject.getStateIndex(label+"_tx")
    index_y =storageObject.getStateIndex(label+"_ty")
    index_z =storageObject.getStateIndex(label+"_tz")

    array_x=opensim.ArrayDouble()
    storageObject.getDataColumn(index_x,array_x)

    array_y=opensim.ArrayDouble()
    storageObject.getDataColumn(index_y,array_y)

    array_z=opensim.ArrayDouble()
    storageObject.getDataColumn(index_z,array_z)

    n= array_x.getSize()
    pointValues = np.zeros((n,3))
    for i in range(0,n):
        pointValues[i,0] = array_x.getitem(i)
        pointValues[i,1] = array_y.getitem(i)
        pointValues[i,2] = array_z.getitem(i)


    for i in range(0,n):
        pointValues[i,:] = np.dot(R_LAB_OSIM.T,pointValues[i,:])*1000.0


    return pointValues


def mot2pointValues(motFilename,labels,orientation =[1,1,1]):
    storageObject = opensim.Storage(motFilename)

    index_x =storageObject.getStateIndex(labels[0])
    index_y =storageObject.getStateIndex(labels[1])
    index_z =storageObject.getStateIndex(labels[2])

    array_x=opensim.ArrayDouble()
    storageObject.getDataColumn(index_x,array_x)

    array_y=opensim.ArrayDouble()
    storageObject.getDataColumn(index_y,array_y)

    array_z=opensim.ArrayDouble()
    storageObject.getDataColumn(index_z,array_z)

    n= array_x.getSize()
    pointValues = np.zeros((n,3))
    for i in range(0,n):
        pointValues[i,0] = orientation[0]*array_x.getitem(i)
        pointValues[i,1] = orientation[1]*array_y.getitem(i)
        pointValues[i,2] = orientation[2]*array_z.getitem(i)


    return pointValues
