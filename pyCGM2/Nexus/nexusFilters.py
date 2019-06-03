# -*- coding: utf-8 -*-
import pyCGM2
import ViconNexus
import numpy as np
import logging

from pyCGM2 import btk
from pyCGM2.Tools import btkTools

from pyCGM2.Nexus import nexusTools,Devices

NEXUS = ViconNexus.ViconNexus()

class NexusModelFilter(object):
    def __init__(self,NEXUS,iModel, iAcq, vskName,pointSuffix, staticProcessing = False ):
        """
            Constructor

            :Parameters:
                - `NEXUS` () - Nexus environment
                - `iModel` (pyCGM2.Model.CGM2.Model) - model instance
                - `vskName` (str) . subject name create in Nexus
                - `staticProcessingFlag` (bool`) : flag indicating only static model ouput will be export

        """
        self.m_model = iModel
        self.m_acq = iAcq
        self.m_vskName = vskName
        self.NEXUS = NEXUS
        self.staticProcessing = staticProcessing
        self.m_pointSuffix = pointSuffix if pointSuffix is None else str("_"+pointSuffix)

    def run(self):
        """
            method calling embedded-model method : viconExport
        """
        self.m_model.viconExport(self.NEXUS,self.m_acq, self.m_vskName,self.m_pointSuffix,self.staticProcessing)



class NexusConstructAcquisitionFilter(object):
    def __init__(self,filenameNoExt,subject):

        """
        """
        self.filenameNoExt = filenameNoExt
        self.subject = subject

        self.__output = None

    def run(self):
        framerate = NEXUS.GetFrameRate()
        frames = NEXUS.GetFrameCount()
        analogFrameRate = NEXUS.GetDeviceDetails(1)[2]
        analogFrameNumber = frames * int(analogFrameRate/framerate)


        acq = btk.btkAcquisition()
        acq.Init(0, int(frames),0, int(analogFrameRate/framerate))
        acq.SetPointFrequency(framerate)


        nexusForcePlates = list()
        nexusAnalogDevices = list()

        # list of devices
        devices = list()

        deviceIDs = NEXUS.GetDeviceIDs()
        if( len(deviceIDs) > 0 ):
            for deviceID in deviceIDs:
                if NEXUS.GetDeviceDetails( deviceID )[1] == "ForcePlate":
                    devices.append( Devices.ForcePlate(deviceID))
                else:
                    devices.append(Devices.AnalogDevice(deviceID))



        fp_count=0
        for device in devices:

            if isinstance(device,pyCGM2.Nexus.Devices.ForcePlate):
                nexusForcePlates.append(device)
                forceLocal = device.getLocalReactionForce()
                momentLocal = device.getLocalReactionMoment()

                forceLabels =["Force.Fx"+str(fp_count+1), "Force.Fy"+str(fp_count+1),"Force.Fz"+str(fp_count+1)]
                for j in range(0,3):
                    analog = btk.btkAnalog()
                    analog.SetLabel(forceLabels[j])
                    analog.SetUnit("N")#nexusForcePlate.getForceUnit())
                    analog.SetFrameNumber(analogFrameNumber)
                    analog.SetValues(forceLocal[:,j])
                    analog.SetDescription(device.getDescription())
                    #analog.SetGain(btk.btkAnalog.PlusMinus10)

                    acq.AppendAnalog(analog)

                momentLabels =["Moment.Mx"+str(fp_count+1), "Moment.My"+str(fp_count+1),"Moment.Mz"+str(fp_count+1)]
                for j in range(0,3):
                    analog = btk.btkAnalog()
                    analog.SetLabel(momentLabels[j])
                    analog.SetUnit("Nmm")#nexusForcePlate.getMomentUnit())
                    analog.SetFrameNumber(analogFrameNumber)
                    analog.SetValues(momentLocal[:,j])
                    analog.SetDescription(device.getDescription())
                    #analog.GetGain(btk.btkAnalog.PlusMinus10)
                    acq.AppendAnalog(analog)

                fp_count+=1

            else:
                nexusAnalogDevices.append(device)
                channels = device.getChannels()

                for channel in channels:
                    analog = btk.btkAnalog()
                    analog.SetLabel(channel.getLabel())
                    analog.SetUnit(channel.getUnit())
                    analog.SetFrameNumber(analogFrameNumber)
                    analog.SetValues(channel.getValues())
                    analog.SetDescription(channel.getDescription())
                    acq.AppendAnalog(analog)


        #
        markersLoaded = NEXUS.GetMarkerNames(self.subject) # nexus2.7 return all makers, even calibration only
        markers =[]
        for i in range(0,len(markersLoaded)):
            data = NEXUS.GetTrajectory(self.subject,markersLoaded[i])
            if data != ([],[],[],[]):
                markers.append(markersLoaded[i])


        for marker in markers:
            rawDataX, rawDataY, rawDataZ, E = NEXUS.GetTrajectory(self.subject,marker)

            E = np.asarray(E).astype("float")-1
            values =np.array([np.asarray(rawDataX),np.asarray(rawDataY),np.asarray(rawDataZ)]).T
            btkTools.smartAppendPoint(acq,str(marker),values, PointType=btk.btkPoint.Marker,desc="",residuals=E)


        # metadata for platform type2
        md_force_platform = btk.btkMetaData('FORCE_PLATFORM') # create main metadata
        btk.btkMetaDataCreateChild(md_force_platform, "USED", 4)# a
        btk.btkMetaDataCreateChild(md_force_platform, "ZERO", [1,0]) #btk.btkDoubleArray(12, 0.8))# add a child
        btk.btkMetaDataCreateChild(md_force_platform, "TYPE", [2,2,2,2]) #btk.btkDoubleArray(12, 0.8))# add a child
        acq.GetMetaData().AppendChild(md_force_platform)


        origins = []
        for nexusForcePlate in nexusForcePlates:
            origins.append(-1.0*nexusForcePlate.getLocalOrigin())

        md_origin = btk.btkMetaData('ORIGIN')
        md_origin.SetInfo(btk.btkMetaDataInfo([3,4], np.concatenate(origins)))
        md_force_platform.AppendChild(md_origin)



        corners = []
        for nexusForcePlate in nexusForcePlates:
            corners.append(nexusForcePlate.getCorners().T.flatten())

        md_corners = btk.btkMetaData('CORNERS')
        md_corners.SetInfo(btk.btkMetaDataInfo([3,4,4], np.concatenate(corners)))
        md_force_platform.AppendChild(md_corners)


        md_channel = btk.btkMetaData('CHANNEL')
        md_channel.SetInfo(btk.btkMetaDataInfo([6,4], np.arange(1,25)))
        md_force_platform.AppendChild(md_channel)

        self.__output = acq
        return acq
