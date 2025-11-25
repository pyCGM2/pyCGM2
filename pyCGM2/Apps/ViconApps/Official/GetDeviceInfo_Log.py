# -*- coding: utf-8 -*-
"""
This script is to be used to help identify connected devices in the trial when working with the offline SDK to extract data in Python/Matlab
@author: kHerber
"""
log_separator = "=============================================================================================================================="

from ViconNexus import *


def GetDeviceInfo(vicon):
  (path, name) = vicon.GetTrialName()

  # list of devices
  deviceIDs = vicon.GetDeviceIDs()
  if( len(deviceIDs) > 0 ):
    for deviceID in deviceIDs:
      deviceName = vicon.GetDeviceDetails( deviceID )[0]
      deviceType = vicon.GetDeviceDetails( deviceID )[1]
      deviceOutputIDs = vicon.GetDeviceDetails( deviceID )[3]
            
      for deviceOutputID in deviceOutputIDs:
        deviceOutputChannelIDs = vicon.GetDeviceOutputDetails( deviceID , deviceOutputID)[5]
      if deviceName == "":
          print(log_separator)
          print ("Consider giving your device a name to help identify each device and its properties")
          print ("Device ID: ["+ str(deviceID)+ "]  |  Name: ["+ deviceName+ "]  |  Type: ["+ deviceType+ "]  |  Device Output IDs: "+ str(deviceOutputIDs)+"  |  Channel IDs: " + str(deviceOutputChannelIDs)+"")
          print(log_separator)
      else: 
          print("Device ID: ["+ str(deviceID)+ "]  |  Name: ["+ deviceName+ "]  |  Type: ["+ deviceType+ "]  |  Device Output IDs: "+ str(deviceOutputIDs)+"  |  Channel IDs: " + str(deviceOutputChannelIDs)+"")

  else:
      print(log_separator)
      print("No devices found")
      print(log_separator)
      
if __name__ == "__main__":
    vicon = ViconNexus()
    print(log_separator)
    GetDeviceInfo(vicon) 
    print(log_separator)
  


