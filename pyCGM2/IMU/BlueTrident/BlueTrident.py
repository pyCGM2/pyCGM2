# coding: utf-8
import re
import numpy as np
import pandas as pd

from pyCGM2.IMU import imu

import pyCGM2
LOGGER = pyCGM2.LOGGER


try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")


def readmultipleBlueTridentCsv(fullfilenames,freq):

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx,array[idx]

    datasets= list()
    for fullfilename in fullfilenames:
        datasets.append( pd.read_csv(fullfilename))

    times0 = list()
    for dataset in datasets:
        times0.append(dataset["time_s"].values[0])
    time_0 =  max(times0)

    datasets_0 = list()
    numberOfFrames = list()
    for dataset in datasets:
        idx0,value0 = find_nearest(dataset["time_s"].values,time_0)
        numberOfFrames.append(dataset.iloc[idx0:].shape[0])
        datasets_0.append(dataset.iloc[idx0:])
    nFrames =  min(numberOfFrames)

    datasets_equal = list()
    for dataset in datasets_0:
        datasets_equal.append(dataset[0:nFrames])

    imuInstances = list()
    for data in datasets_equal:
        imuInstance = imu.Imu(freq)

        imuInstance.setAcceleration("X",data["ax_m/s/s"].to_numpy())
        imuInstance.setAcceleration("Y",data["ay_m/s/s"].to_numpy())
        imuInstance.setAcceleration("Z",data["az_m/s/s"].to_numpy())

        imuInstance.setGyro("X",data["gx_deg/s"].to_numpy())
        imuInstance.setGyro("Y",data["gy_deg/s"].to_numpy())
        imuInstance.setGyro("Z",data["gz_deg/s"].to_numpy())

        imuInstances.append(imuInstance)


    return imuInstances



def readBlueTridentCsv(fullfilename,freq):
    data = pd.read_csv(fullfilename)

    imuInstance = imu.Imu(freq)

    imuInstance.setAcceleration("X",data["ax_m/s/s"].to_numpy())
    imuInstance.setAcceleration("Y",data["ay_m/s/s"].to_numpy())
    imuInstance.setAcceleration("Z",data["az_m/s/s"].to_numpy())

    imuInstance.setGyro("X",data["gx_deg/s"].to_numpy())
    imuInstance.setGyro("Y",data["gy_deg/s"].to_numpy())
    imuInstance.setGyro("Z",data["gz_deg/s"].to_numpy())

    return imuInstance


def correctBlueTridentIds(acq):
    correctFlag = False

    # extract id from description
    ids = list()
    for it in btk.Iterate(acq.GetAnalogs()):
        desc = it.GetDescription()
        analoglabel = it.GetLabel()
        if "Vicon BlueTrident" in desc:
            id = re.findall( "\[(.*?)\]", desc)[0].split(",")

            if id[0] not in ids:  ids.append(id[0])

    if ids[0] !=1:
        correctFlag = True
        LOGGER.logger.warning("Blue trident ids do not start from 1 ")

    # correct description
    if correctFlag:
        newIds = list(np.arange(1,len(ids)+1))

        for it in btk.Iterate(acq.GetAnalogs()):
            desc = it.GetDescription()
            analoglabel = it.GetLabel()
            if "Vicon BlueTrident" in desc:
                id = re.findall( "\[(.*?)\]", desc)[0].split(",")[0]
                index = ids.index(id)
                newdesc = desc.replace("["+id, "["+str(newIds[index]))
                it.SetDescription(newdesc)

    return acq


def getBlueTrident(acq, id):

    imuInstance = imu.Imu(acq.GetAnalogFrequency())

    channels = []
    for it in btk.Iterate(acq.GetAnalogs()):
        desc = it.GetDescription()
        label = it.GetLabel()
        values= it.GetValues()
        nAnalogframes = values.shape[0]

        if "Vicon BlueTrident" in desc:
            deviceId = re.findall( "\[(.*?)\]", desc)[0].split(",")[0]
            if deviceId == id:
                channels.append(it)

    for it in ["accel.x","accel.y","accel.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     imuInstance.setAcceleration("X",channel.GetValues().reshape(nAnalogframes))
                if ".y" in it:
                    imuInstance.setAcceleration("Y",channel.GetValues().reshape(nAnalogframes))
                if ".z" in it:
                   imuInstance.setAcceleration("Z",channel.GetValues().reshape(nAnalogframes))

    for it in ["gyro.x","gyro.y","gyro.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     imuInstance.setGyro("X",channel.GetValues().reshape(nAnalogframes))
                if ".y" in it:
                    imuInstance.setGyro("Y",channel.GetValues().reshape(nAnalogframes))
                if ".z" in it:
                   imuInstance.setGyro("Z",channel.GetValues().reshape(nAnalogframes))

    return imuInstance
