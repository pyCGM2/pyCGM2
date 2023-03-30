# -*- coding: utf-8 -*-
import pyCGM2; LOGGER = pyCGM2.LOGGER

# vicon nexus
from viconnexusapi import ViconNexus
from viconnexusapi.ViconUtils import RotationMatrixFromAngleAxis, QuaternionFromMatrix, EulerFromMatrix

import argparse
import pandas as pd
import numpy as np

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools



def convertIMU(vicon,args):
    filePath, trialName = vicon.GetTrialName()

    d = {}
    for name in args.TridentNames:
        d[name] = pd.DataFrame()

        sensorID = vicon.GetDeviceIDFromName(name)
        sensorOutputID = vicon.GetDeviceOutputIDFromName(sensorID, 'Global Angle')
        GA_X = np.deg2rad(vicon.GetDeviceChannelGlobal(sensorID, sensorOutputID, 1)[0]) # Store GA_X in radians for future conversions
        GA_Y = np.deg2rad(vicon.GetDeviceChannelGlobal(sensorID, sensorOutputID, 2)[0]) # Store GA_Y in radians for future conversions
        GA_Z = np.deg2rad(vicon.GetDeviceChannelGlobal(sensorID, sensorOutputID, 3)[0]) # Store GA_Z in radians for future conversions

        zipped = zip(GA_X, GA_Y, GA_Z) # create a tuple for each frame with the GA X, Y, and Z to pass to conversion functions

        rotList = []
        quatList = []
        eulerXYZList = []
        eulerZYXList = []
        eulerXZYList = []
        eulerYZXList = []
        eulerYXZList = []
        eulerZXYList = []

        for tup in zipped:
            rotList.append(RotationMatrixFromAngleAxis(tup))
            a = RotationMatrixFromAngleAxis(tup)

            if args.Quaternion:
                quatList.append(QuaternionFromMatrix(a))

            if args.EulerXYZ:
                eulerXYZList.append(EulerFromMatrix(a, 'xyz'))
            if args.EulerZYX:
                eulerZYXList.append(EulerFromMatrix(a, 'zyx'))
            if args.EulerXZY:
                eulerXZYList.append(EulerFromMatrix(a, 'xzy'))
            if args.EulerYZX:
                eulerYZXList.append(EulerFromMatrix(a, 'yzx'))
            if args.EulerYXZ:
                eulerYXZList.append(EulerFromMatrix(a, 'yxz'))
            if args.EulerZXY:
                eulerZXYList.append(EulerFromMatrix(a, 'zxy'))


        if args.Quaternion:
            d[name]['qx'] = [item[0] for item in quatList]
            d[name]['qy'] = [item[1] for item in quatList]
            d[name]['qz'] = [item[2] for item in quatList]
            d[name]['qr'] = [item[3] for item in quatList]

        if args.RotationMatrix:
            d[name]['r1'] = [item[0] for item in rotList]
            d[name]['r2'] = [item[1] for item in rotList]
            d[name]['r3'] = [item[2] for item in rotList]

        if args.EulerXYZ:
            if args.Radians:
                d[name]['XYZ_1 (rad)'] = [item[0] for item in eulerXYZList]
                d[name]['XYZ_2 (rad)'] = [item[1] for item in eulerXYZList]
                d[name]['XYZ_3 (rad)'] = [item[2] for item in eulerXYZList]
            else:
                d[name]['XYZ_1 (deg)'] = [np.rad2deg(item[0]) for item in eulerXYZList]
                d[name]['XYZ_2 (deg)'] = [np.rad2deg(item[1]) for item in eulerXYZList]
                d[name]['XYZ_3 (deg)'] = [np.rad2deg(item[2]) for item in eulerXYZList]
        if args.EulerZYX:
            if args.Radians:
                d[name]['ZYX_1 (rad)'] = [item[0] for item in eulerZYXList]
                d[name]['ZYX_2 (rad)'] = [item[1] for item in eulerZYXList]
                d[name]['ZYX_3 (rad)'] = [item[2] for item in eulerZYXList]
            else:
                d[name]['ZYX_1 (deg)'] = [np.rad2deg(item[0]) for item in eulerZYXList]
                d[name]['ZYX_2 (deg)'] = [np.rad2deg(item[1]) for item in eulerZYXList]
                d[name]['ZYX_3 (deg)'] = [np.rad2deg(item[2]) for item in eulerZYXList]
        if args.EulerXZY:
            if args.Radians:
                d[name]['XZY_1 (rad)'] = [item[0] for item in eulerXZYList]
                d[name]['XZY_2 (rad)'] = [item[1] for item in eulerXZYList]
                d[name]['XZY_3 (rad)'] = [item[2] for item in eulerXZYList]
            else:
                d[name]['XZY_1 (deg)'] = [np.rad2deg(item[0]) for item in eulerXZYList]
                d[name]['XZY_2 (deg)'] = [np.rad2deg(item[1]) for item in eulerXZYList]
                d[name]['XZY_3 (deg)'] = [np.rad2deg(item[2]) for item in eulerXZYList]
        if args.EulerYZX:
            if args.Radians:
                d[name]['YZX_1 (rad)'] = [item[0] for item in eulerYZXList]
                d[name]['YZX_2 (rad)'] = [item[1] for item in eulerYZXList]
                d[name]['YZX_3 (rad)'] = [item[2] for item in eulerYZXList]
            else:
                d[name]['YZX_1 (deg)'] = [np.rad2deg(item[0]) for item in eulerYZXList]
                d[name]['YZX_2 (deg)'] = [np.rad2deg(item[1]) for item in eulerYZXList]
                d[name]['YZX_3 (deg)'] = [np.rad2deg(item[2]) for item in eulerYZXList]
        if args.EulerYXZ:
            if args.Radians:
                d[name]['YXZ_1 (rad)'] = [item[0] for item in eulerYXZList]
                d[name]['YXZ_2 (rad)'] = [item[1] for item in eulerYXZList]
                d[name]['YXZ_3 (rad)'] = [item[2] for item in eulerYXZList]
            else:
                d[name]['YXZ_1 (deg)'] = [np.rad2deg(item[0]) for item in eulerYXZList]
                d[name]['YXZ_2 (deg)'] = [np.rad2deg(item[1]) for item in eulerYXZList]
                d[name]['YXZ_3 (deg)'] = [np.rad2deg(item[2]) for item in eulerYXZList]
        if args.EulerZXY:
            if args.Radians:
                d[name]['ZXY_1 (rad)'] = [item[0] for item in eulerZXYList]
                d[name]['ZXY_2 (rad)'] = [item[1] for item in eulerZXYList]
                d[name]['ZXY_3 (rad)'] = [item[2] for item in eulerZXYList]
            else:
                d[name]['ZXY_1 (deg)'] = [np.rad2deg(item[0]) for item in eulerZXYList]
                d[name]['ZXY_2 (deg)'] = [np.rad2deg(item[1]) for item in eulerZXYList]
                d[name]['ZXY_3 (deg)'] = [np.rad2deg(item[2]) for item in eulerZXYList]

        d[name].columns = pd.MultiIndex.from_product([[name], d[name].columns])
        d[name].index +=1


    df = pd.concat(d.values(), axis = 1, ignore_index = False)
    df.to_csv(filePath + trialName + '.IMU.csv')



def main():

    parser = argparse.ArgumentParser(description='This script converts angle axis outputs from Blue Trident into Quaternions')

    parser.add_argument('TridentNames', type=str, nargs = '+', help='These are the names you have given the Blue Trident sensors')

    # Euler, Matrix, Quaternion
    parser.add_argument('--Quaternion', action = 'store_true', help = 'If this flag is present, the output file will include the Quaternions')
    parser.add_argument('--RotationMatrix', action = 'store_true', help = 'If this flag is present, the output file will include the rotation matrix')
    parser.add_argument('--EulerXYZ', action = 'store_true', help = 'If this flag is present, the output file will include the Euler XYZ')
    parser.add_argument('--EulerZYX', action = 'store_true', help = 'If this flag is present, the output file will include the Euler ZYX')
    parser.add_argument('--EulerXZY', action = 'store_true', help = 'If this flag is present, the output file will include the Euler XZY')
    parser.add_argument('--EulerYZX', action = 'store_true', help = 'If this flag is present, the output file will include the Euler YZX')
    parser.add_argument('--EulerYXZ', action = 'store_true', help = 'If this flag is present, the output file will include the Euler YXZ')
    parser.add_argument('--EulerZXY', action = 'store_true', help = 'If this flag is present, the output file will include the Euler ZXY')
    # Radians vs Degrees for Euler outputs
    parser.add_argument('--Radians', action = 'store_true', help = 'If this flag is present, the output of any Euler angles will be in radians')


    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED: # run Operation
        DATA_PATH, filename = NEXUS.GetTrialName()

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2.log")
        LOGGER.logger.info( " file: "+ filename)


        # --------------------------SUBJECT ------------------------------------

        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS) #checkActivatedSubject(NEXUS,subjects)
        # Parameters = NEXUS.GetSubjectParamNames(subject)
        if isinstance(subject,tuple):
            subject=""


        # --------------------------PULL ------------------------------------
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,filename,subject)
        acq = nacf.build()
        btkTools.smartWriter(acq, "verif.c3d")

        for it in btk.Iterate(acq.GetAnalogs()):
            print(it.GetLabel())
        import ipdb; ipdb.set_trace()


        convertIMU(NEXUS,args)










if __name__ == '__main__':
    main()
