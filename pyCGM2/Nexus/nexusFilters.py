"""
This module contains convenient classes interacting with the nexus API
"""


from pyCGM2.Nexus import Devices
from pyCGM2.Tools import btkTools
from pyCGM2.Model import model

import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER

import btk


try:
    from viconnexusapi import ViconNexus
except ImportError as e:
    LOGGER.logger.error(f"viconnexusapi not installed: {e}")


from typing import List, Tuple, Dict, Optional,Union

class NexusModelFilter(object):
    """
    Nexus Model Filter is an interface running the method `viconExport` of a pyCGM2 Model instance.

    Args:
        NEXUS (ViconNexus.ViconNexus): Vicon Nexus handle.
        iModel (model.Model): A model instance.
        iAcq (btk.btkAcquisition): A BTK acquisition.
        vskName (str): Name of the VSK file.
        pointSuffix (str): Suffix added to the model output name.
        staticProcessing (bool, optional): Enable static mode processing. Default is False.

    """

    def __init__(self, NEXUS:ViconNexus.ViconNexus, 
                 iModel:model.Model, 
                 iAcq:btk.btkAcquisition, 
                 vskName:str, pointSuffix:str, staticProcessing:bool=False):
        """Initializes the NexusModelFilter class."""

        self.m_model = iModel
        self.m_acq = iAcq
        self.m_vskName = vskName
        self.NEXUS = NEXUS
        self.staticProcessing = staticProcessing
        self.m_pointSuffix = pointSuffix if pointSuffix is None else "_"+pointSuffix

    def run(self):
        """
        Runs the viconExport method of the pyCGM2 Model instance.
        """
        self.m_model.viconExport(
            self.NEXUS, self.m_acq, self.m_vskName, self.m_pointSuffix, self.staticProcessing)


class NexusConstructAcquisitionFilter(object):
    """
    Filter for constructing a btk.Acquisition from Nexus API.

    Args:
        dataPath (str): Data folder path.
        filenameNoExt (str): Filename without its extension.
        subject (str): Subject name (equivalent to VSK name).

    """

    def __init__(self, NEXUS:ViconNexus.ViconNexus,
                 dataPath:str, filenameNoExt:str, subject:str):
        """Initializes the NexusConstructAcquisitionFilter class."""
        self.NEXUS = NEXUS
        self.m_dataPath = dataPath
        self.m_filenameNoExt = filenameNoExt
        self.m_subject = subject

        self.m_framerate = self.NEXUS.GetFrameRate()
        #self.m_frames = NEXUS.GetTrialRange()[1]
        self.m_rangeROI = self.NEXUS.GetTrialRegionOfInterest()
        self.m_trialRange = self.NEXUS.GetTrialRange()
        # might be different from 1 if corpped and no x2d
        self.m_trialFirstFrame = self.m_trialRange[0]

        self.m_firstFrame = self.m_rangeROI[0]
        self.m_lastFrame = self.m_rangeROI[1]
        self.m_frames = self.m_lastFrame-(self.m_firstFrame-1)

        deviceIDs = self.NEXUS.GetDeviceIDs()
        self.m_analogFrameRate = self.NEXUS.GetDeviceDetails(
            deviceIDs[0])[2] if (len(deviceIDs) > 0) else self.m_framerate

        self.m_numberAnalogSamplePerFrame = int(
            self.m_analogFrameRate/self.m_framerate)
        self.m_analogFrameNumber = self.m_frames * self.m_numberAnalogSamplePerFrame

        self.m_nexusForcePlates = []
        self.m_nexusAnalogDevices = []

        if(len(deviceIDs) > 0):
            for deviceID in deviceIDs:
                if self.NEXUS.GetDeviceDetails(deviceID)[1] == "ForcePlate":
                    self.m_nexusForcePlates.append(
                        Devices.ForcePlate(self.NEXUS,deviceID))
                else:
                    self.m_nexusAnalogDevices.append(
                        Devices.AnalogDevice(self.NEXUS,deviceID))

        self.m_acq = btk.btkAcquisition()
        self.m_acq.Init(0, int(self.m_frames), 0,
                        self.m_numberAnalogSamplePerFrame)
        self.m_acq.SetPointFrequency(self.m_framerate)
        self.m_acq.SetFirstFrame(self.m_firstFrame)

    def appendEvents(self):
        """ Appends events to the acquisition object."""

        eventType = "Foot Strike"
        eventContext = "Left"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Automatic)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "Foot Off"
        eventContext = "Left"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Automatic)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "Foot Strike"
        eventContext = "Right"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Automatic)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "Foot Off"
        eventContext = "Right"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Automatic)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "Event"
        eventContext = "General"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Manual)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "Left-FP"
        eventContext = "General"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Manual)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "Right-FP"
        eventContext = "General"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Manual)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "Start"
        eventContext = "Left"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Manual)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

        eventType = "End"
        eventContext = "Left"
        if self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0] != []:
            for frame in self.NEXUS.GetEvents(self.m_subject, eventContext, eventType)[0]:
                if frame >= self.m_firstFrame and frame <= self.m_lastFrame:
                    time = (frame-1)/self.m_framerate
                    ev = btk.btkEvent(eventType, time, int(
                        frame), eventContext, btk.btkEvent.Manual)
                    ev.SetSubject(self.m_subject)
                    self.m_acq.AppendEvent(ev)

    def appendMarkers(self):
        """ Appends markers to the acquisition object."""

        markersLoaded = self.NEXUS.GetMarkerNames(self.m_subject)
        markers = []
        for i in range(0, len(markersLoaded)):
            data = self.NEXUS.GetTrajectory(self.m_subject, markersLoaded[i])
            if data != ([], [], [], []):
                markers.append(markersLoaded[i])

        for marker in markers:
            rawDataX, rawDataY, rawDataZ, E = self.NEXUS.GetTrajectory(
                self.m_subject, marker)

            E = np.asarray(E).astype("float")-1
            values = np.array([np.asarray(rawDataX), np.asarray(
                rawDataY), np.asarray(rawDataZ)]).T

            start = self.m_firstFrame - self.m_trialFirstFrame
            end = self.m_lastFrame - self.m_trialFirstFrame

            values_cut = values[start:end+1, :]
            E_cut = E[start:end+1]

            btkTools.smartAppendPoint(self.m_acq, marker, values_cut, PointType="Marker", desc="",
                                      residuals=E_cut)

    def appendAnalogs(self):
        """ Appends analogs to the acquisition object."""

        ftr = self.NEXUS.GetTrialRange()[0]

        for nexusAnalogDevice in self.m_nexusAnalogDevices:

            start = self.m_firstFrame - 1  # self.m_trialFirstFrame
            end = self.m_lastFrame - 1  # self.m_trialFirstFrame

            deviceName = nexusAnalogDevice.getDeviceName() 
            channels = nexusAnalogDevice.getChannels()
            for channel in channels:
                analog = btk.btkAnalog()
                analog.SetLabel(channel.getLabel())
                analog.SetUnit(channel.getUnit())
                analog.SetFrameNumber(self.m_analogFrameNumber)
                analog.SetValues(channel.getValues()[
                                 start*self.m_numberAnalogSamplePerFrame:(end+1)*self.m_numberAnalogSamplePerFrame])
                analog.SetDescription(channel.getDescription())
                #if deviceName == "":
                analog.SetDescription(channel.getDescription()) 
                # else:
                #    analog.SetDescription("("+deviceName+ ") "+channel.getDescription()) 


                self.m_acq.AppendAnalog(analog)

    def appendForcePlates(self):
        """ Appends force plates to the acquisition object."""

        forcePlateNumber = len(self.m_nexusForcePlates)

        fp_count = 0
        for nexusForcePlate in self.m_nexusForcePlates:
            # row number =  self.NEXUS.getTrialRange[1] not FrameCount
            forceLocal = nexusForcePlate.getLocalReactionForce()
            momentLocal = nexusForcePlate.getLocalReactionMoment()

            start = self.m_firstFrame - 1  # -1 because Nexus frame start at 1
            end = self.m_lastFrame - 1  # - self.m_trialFirstFrame

            forceLabels = [
                "Force.Fx"+str(fp_count+1), "Force.Fy"+str(fp_count+1), "Force.Fz"+str(fp_count+1)]
            for j in range(0, 3):
                analog = btk.btkAnalog()
                analog.SetLabel(forceLabels[j])
                analog.SetUnit("N")  # nexusForcePlate.getForceUnit())
                analog.SetFrameNumber(self.m_analogFrameNumber)
                analog.SetValues(forceLocal[(
                    start)*self.m_numberAnalogSamplePerFrame:(end+1)*self.m_numberAnalogSamplePerFrame, j])
                analog.SetDescription(nexusForcePlate.getDescription())
                #analog.SetGain(btk.btkAnalog.PlusMinus10)

                self.m_acq.AppendAnalog(analog)

            momentLabels = [
                "Moment.Mx"+str(fp_count+1), "Moment.My"+str(fp_count+1), "Moment.Mz"+str(fp_count+1)]
            for j in range(0, 3):
                analog = btk.btkAnalog()
                analog.SetLabel(momentLabels[j])
                analog.SetUnit("Nmm")  # nexusForcePlate.getMomentUnit())
                analog.SetFrameNumber(self.m_analogFrameNumber)
                analog.SetValues(momentLocal[(
                    start)*self.m_numberAnalogSamplePerFrame:(end+1)*self.m_numberAnalogSamplePerFrame, j])
                analog.SetDescription(nexusForcePlate.getDescription())
                #analog.GetGain(btk.btkAnalog.PlusMinus10)
                self.m_acq.AppendAnalog(analog)

            fp_count += 1

        # metadata for platform type2
        md_force_platform = btk.btkMetaData('FORCE_PLATFORM')
        btk.btkMetaDataCreateChild(
            md_force_platform, "USED", int(forcePlateNumber))  # a
        btk.btkMetaDataCreateChild(md_force_platform, "ZERO", [1, 0])
        btk.btkMetaDataCreateChild(md_force_platform, "TYPE", btk.btkDoubleArray(
            forcePlateNumber, 2))  # btk.btkDoubleArray(forcePlateNumber, 2))# add a child

        self.m_acq.GetMetaData().AppendChild(md_force_platform)

        origins = []
        for nexusForcePlate in self.m_nexusForcePlates:
            origins.append(-1.0*nexusForcePlate.getLocalOrigin())

        md_origin = btk.btkMetaData('ORIGIN')
        md_origin.SetInfo(btk.btkMetaDataInfo(
            [3, int(forcePlateNumber)], np.concatenate(origins)))
        md_force_platform.AppendChild(md_origin)

        corners = []
        for nexusForcePlate in self.m_nexusForcePlates:
            corners.append(nexusForcePlate.getCorners().T.flatten())

        md_corners = btk.btkMetaData('CORNERS')
        md_corners.SetInfo(btk.btkMetaDataInfo(
            [3, 4, int(forcePlateNumber)], np.concatenate(corners)))
        md_force_platform.AppendChild(md_corners)

        md_channel = btk.btkMetaData('CHANNEL')
        md_channel.SetInfo(btk.btkMetaDataInfo(
            [6, int(forcePlateNumber)], np.arange(1, int(forcePlateNumber)*6+1).tolist()))
        md_force_platform.AppendChild(md_channel)

    def appendModelOutputs(self):
        """ Appends model output data to the acquisition object."""

        modelOutputNames = self.NEXUS.GetModelOutputNames(self.m_subject)

        if modelOutputNames != []:
            for modelOutputName in modelOutputNames:
                data, E = self.NEXUS.GetModelOutput(self.m_subject, modelOutputName)
                type = self.NEXUS.GetModelOutputDetails(
                    self.m_subject, modelOutputName)[0]

                if type in ["Angles", "Forces", "Moments", "Powers", "Modeled Markers"]:

                    E = np.asarray(E).astype("float")-1
                    values = np.array(
                        [np.asarray(data[0]), np.asarray(data[1]), np.asarray(data[2])]).T

                    start = self.m_firstFrame - self.m_trialFirstFrame
                    end = self.m_lastFrame - self.m_trialFirstFrame

                    values_cut = values[start:end+1, :]
                    E_cut = E[start:end+1]

                    if type == "Angles":
                        btkTools.smartAppendPoint(self.m_acq, modelOutputName, values_cut, PointType="Angle", desc="",
                                                  residuals=E_cut)
                    elif type == "Forces":
                        btkTools.smartAppendPoint(self.m_acq, modelOutputName, values_cut, PointType="Force", desc="",
                                                  residuals=E_cut)
                    elif type == "Moments":
                        btkTools.smartAppendPoint(self.m_acq, modelOutputName, values_cut, PointType="Moment", desc="",
                                                  residuals=E_cut)
                    elif type == "Powers":
                        btkTools.smartAppendPoint(self.m_acq, modelOutputName, values_cut, PointType="Power", desc="",
                                                  residuals=E_cut)
                    elif type == "Modeled Markers":
                        btkTools.smartAppendPoint(self.m_acq, modelOutputName, values_cut, PointType="Marker", desc="",
                                                  residuals=E_cut)
                    else:
                        LOGGER.logger.debug("[pyCGM2] : type unknown")

                else:
                    LOGGER.logger.debug(
                        "[pyCGM2] : Model Output (%s) from Nexus not added to the btk acquisition" % (modelOutputName))

    def initMetaData(self):
        """ Initializes metadata with an ANALYSIS section in the acquisition object. """
        # ANALYSIS Section
        self.m_acq.GetMetaData().AppendChild(btk.btkMetaData("ANALYSIS"))
        self.m_acq.GetMetaData().FindChild(
            "ANALYSIS").value().AppendChild(btk.btkMetaData('USED', 0))

    def build(self):
        """
        Builds the acquisition object with appended data.
        Returns:
            btk.btkAcquisition: The built BTK acquisition object.
        """
        self.appendEvents()
        self.appendMarkers()
        if self.m_nexusForcePlates != []:
            self.appendForcePlates()
        if self.m_nexusAnalogDevices != []:
            self.appendAnalogs()
        self.appendModelOutputs()
        self.initMetaData()

        return self.m_acq

    def exportC3d(self, filenameNoExt:Optional[str]=None):
        """
        Exports the built acquisition to a C3D file.
        Args:
            filenameNoExt (Optional[str], optional): Specific filename without its extension.
        """

        if filenameNoExt is None:
            btkTools.smartWriter(
                self.m_acq, self.m_dataPath+self.m_filenameNoExt+".c3d")
        else:
            btkTools.smartWriter(
                self.m_acq, self.m_dataPath+filenameNoExt+".c3d")
