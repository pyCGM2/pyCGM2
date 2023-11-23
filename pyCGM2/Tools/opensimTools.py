# -*- coding: utf-8 -*-
import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.ForcePlates import forceplates
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface.opensimInterface import osimInterface
import numpy as np
import pandas as pd

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")


try:
    import opensim
except:
    try:
        from pyCGM2 import opensim4 as opensim
    except:
        LOGGER.logger.error("[pyCGM2] opensim not found on your system")

from typing import List, Tuple, Dict, Optional

# def createGroundReactionForceMOT_file(DATA_PATH, c3dFile):

#     c3dFileAdapter = opensim.C3DFileAdapter()
#     # ForceLocation_OriginOfForcePlate , ForceLocation_CenterOfPressure
#     c3dFileAdapter.setLocationForForceExpression(
#         opensim.C3DFileAdapter.ForceLocation_PointOfWrenchApplication)
#     tables = c3dFileAdapter.read(DATA_PATH + c3dFile)

#     forces = c3dFileAdapter.getForcesTable(tables)
#     forcesFlat = forces.flatten()

#     forcesFilename = DATA_PATH+c3dFile[:-4] + '_GRF.mot'
#     stoAdapter = opensim.STOFileAdapter()
#     stoAdapter.write(forcesFlat, forcesFilename)




def rotationMatrix_labToOsim(axis:str,forwardProgression:bool):
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
        raise Exception("[pyCGM2] - Global Referential with Z axis as progression not configured yet")
    
    return R_LAB_OSIM


def transformMarker_ToOsimReferencial(acq:btk.btkAcquisition, axis:str, forwardProgression:bool):

    R_LAB_OSIM = rotationMatrix_labToOsim(axis,forwardProgression)

    points = acq.GetPoints()
    for it in btk.Iterate(points):
        if it.GetType() == btk.btkPoint.Marker:
            values = np.zeros(it.GetValues().shape)
            for i in range(0, it.GetValues().shape[0]):
                values[i, :] = np.dot(R_LAB_OSIM, it.GetValues()[i, :])
            it.SetValues(values)


def sto2pointValues(storageObject:opensim.Storage, label:str, R_LAB_OSIM:np.ndarray):

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


def mot2pointValues(motFilename:str, labels:list[str], orientation:list[int]=[1, 1, 1]):
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

def smartGetValues(DATA_PATH:str,filename:str,label:str):
    storageObject = opensim.Storage(DATA_PATH+filename)
    labels = storageObject.getColumnLabels()
    for index in range(1,labels.getSize()): #1 because 0 is time
        if label == labels.get(index):
            index_x = storageObject.getStateIndex(labels.get(index))
            array_x = opensim.ArrayDouble()
            storageObject.getDataColumn(index_x, array_x)
            n = array_x.getSize()
            values = np.zeros((n))
            for i in range(0, n):
                values[i] = array_x.getitem(i)
    return values

def footReactionMotFile(acq:btk.btkAcquisition,filename:str,progressionAxis:str,forwardProgression:bool,mfpa:Optional[str]=None):

    
    mappedForcePlate = forceplates.matchingFootSideOnForceplate(acq,mfpa=mfpa)

    R_LAB_OSIM = rotationMatrix_labToOsim(progressionAxis,forwardProgression)

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

    file1.write(files.getFilename(filename)+"\n")
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

    file1.close()




def export_CgmToMot(acq:btk.btkAcquisition,datapath:str,filename:str,osimModelInterface:osimInterface):
    """Export the CGM kinematics outputs in a mot opensim-format

    Args:
        acq (btk.Acquisition): acquisition
        datapath (str): data path
        filename (str): filenama
        osimModelInterface (pyCGM2.opensimInterface.osimInterface): opensim Interface
    """

    osim2cgm_converter = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"opensim\\interface\\CGM23\\CGMtoOsim.settings")

    coordinateNames = osimModelInterface.getCoordinates()
    coordNamesStr = "time"
    for it in coordinateNames:
        coordNamesStr = coordNamesStr +" " + it

    pointFrames = acq.GetPointFrameNumber()
    freq = acq.GetPointFrequency()
    time = np.arange(0, pointFrames/freq, 1/freq)

    dataFrame = pd.DataFrame()
    dataFrame["time"] = time


    for it in osim2cgm_converter["Angles"]:
        orientation =  int(osim2cgm_converter["Angles"][it].split(".")[0])
        name = osim2cgm_converter["Angles"][it].split(".")[1]
        axis = int(osim2cgm_converter["Angles"][it].split(".")[2])
        
        try:
            values =  orientation*acq.GetPoint(name).GetValues()[:,axis]
            

            if dataFrame.shape[0] == values.shape[0]+1:
                dataFrame[it] = np.append(values,values[-1])
            elif values.shape[0] == dataFrame.shape[0]+1:
                 dataFrame[it] = np.append(values,values[:-1])
            else:
                dataFrame[it] = values

        except RuntimeError:
            pass
        
    dataFrame.fillna(0,inplace=True)
    dataFrame.to_csv(datapath+filename, index=False, sep="\t")

    
    with open(datapath+filename, 'r') as original: 
        data = original.read()
    with open(datapath+filename, 'w') as modified: 
        modified.write(files.getFilename(datapath+filename)+"\n")
        modified.write("version=1\n")
        modified.write("nRows=%i\n"%(pointFrames))
        modified.write("nColumns=%i\n"%(dataFrame.shape[1]))
        modified.write("inDegrees=yes\n")
        modified.write("endheader\n" + data)
