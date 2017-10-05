# -*- coding: utf-8 -*-
import os
import numpy as np

# pyCGM2 settings
import pyCGM2

# vicon nexus
import ViconNexus

from pyCGM2.Tools import btkTools

# vicon nexus
NEXUS = ViconNexus.ViconNexus()


if __name__ == "__main__":
    # check if behaviour of vicon MomentTypes : Torque or TorqueNormalized.
    # after saving, all mmoments from TorqueNormalized are multiply by 1000 !

    DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1-NexusPlugin\\pyCGM2- CGM1-KAD\\"
    modelledFilenameNoExt = "Gait Trial 02" #"static Cal 01-noKAD-noAnkleMed" #
    NEXUS.OpenTrial( str(DATA_PATH+modelledFilenameNoExt), 30 )

    vskName = "PIG-KAD"
    label = "LKneeMoment"
    newlabel = "LKneeMoment2"

    acq = btkTools.smartReader(str(path + name + ".c3d"))
    values = acq.GetPoint("LKneeMoment").GetValues()

    #["TorqueNormalized","TorqueNormalized","TorqueNormalized"]
    NEXUS.CreateModelOutput( vskName, newlabel, "Moments", ["X","Y","Z"], ["Torque","Torque","Torque"])#

    ff = acq.GetFirstFrame()
    lf = acq.GetLastFrame()
    framecount = NEXUS.GetFrameCount()


    data =[list(np.zeros((framecount))), list(np.zeros((framecount))),list(np.zeros((framecount)))]
    exists = [False]*framecount

    j=0
    for i in range(ff-1,lf):
        exists[i] = True
        data[0][i] = values[j,0]
        data[1][i] = values[j,1]
        data[2][i] = values[j,2]
        j+=1

    NEXUS.SetModelOutput( vskName, newlabel, data, exists )
