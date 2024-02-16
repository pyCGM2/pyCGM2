# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files


# pyCGM2 libraries
from pyCGM2.Tools import btkTools


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(prog='pyCGM2-Nexus-Device')
        args = parser.parse_args()
    
    NEXUS_PYTHON_CONNECTED = False
    try:
        from viconnexusapi import ViconNexus

        from pyCGM2.Nexus import nexusFilters
        from pyCGM2.Nexus import nexusUtils
        from pyCGM2.Nexus import nexusTools 

        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
    

    if NEXUS_PYTHON_CONNECTED: # run Operation

        data_path, filename = nexusTools.getTrialName(NEXUS)

        LOGGER.logger.info( "data Path: "+ data_path )
        LOGGER.set_file_handler(data_path+"pyCGM2.log")
        LOGGER.logger.info( " file: "+ filename)


        data = []
        deviceIDs = NEXUS.GetDeviceIDs()
        if(len(deviceIDs) > 0):
            for deviceID in deviceIDs:
                details = NEXUS.GetDeviceDetails(deviceID)
                data.append([filename,deviceID, details[0], details[1]])
            df = pd.DataFrame(data, columns=[" filename",'Vicon DeviceID', 'Name', 'Type'])

            print("-------------------------Device details-------------------------")
            print(df)
            print("----------------------------------------------------------------")


if __name__ == '__main__':
    main(args=None) 