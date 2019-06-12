# -*- coding: utf-8 -*-
import pyCGM2
import ViconNexus
import numpy as np
import logging

from pyCGM2 import btk
from pyCGM2 import ma
from pyCGM2.ma import io

from pyCGM2.Tools import btkTools
from pyCGM2.Tools import trialTools

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
    def __init__(self,dataPath,filenameNoExt,subject):

        """
        """
        self.m_dataPath = dataPath
        self.m_filenameNoExt = filenameNoExt
        self.m_subject = subject

        self.m_framerate = NEXUS.GetFrameRate()
        #self.m_frames = NEXUS.GetTrialRange()[1]
        self.m_rangeROI = NEXUS.GetTrialRegionOfInterest()
        self.m_trialRange = NEXUS.GetTrialRange()

        self.m_firstFrame = self.m_rangeROI[0]
        self.m_lastFrame = self.m_rangeROI[1]
        self.m_frames = self.m_lastFrame-(self.m_firstFrame-1)



        deviceIDs = NEXUS.GetDeviceIDs()
        self.m_analogFrameRate = NEXUS.GetDeviceDetails(deviceIDs[0])[2] if ( len(deviceIDs) > 0 ) else self.m_framerate


        self.m_numberAnalogSamplePerFrame = int(self.m_analogFrameRate/self.m_framerate)
        self.m_analogFrameNumber = self.m_frames * self.m_numberAnalogSamplePerFrame


        self.m_nexusForcePlates = list()
        self.m_nexusAnalogDevices = list()


        if( len(deviceIDs) > 0 ):
            for deviceID in deviceIDs:
                if NEXUS.GetDeviceDetails( deviceID )[1] == "ForcePlate":
                    self.m_nexusForcePlates.append( Devices.ForcePlate(deviceID))
                else:
                    self.m_nexusAnalogDevices.append(Devices.AnalogDevice(deviceID))


        self.m_acq = btk.btkAcquisition()
        self.m_acq.Init(0, int(self.m_frames),0, self.m_numberAnalogSamplePerFrame)
        self.m_acq.SetPointFrequency(self.m_framerate)
        self.m_acq.SetFirstFrame(self.m_firstFrame)


    def appendEvents(self):


        eventType = "Foot Strike"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext, btk.btkEvent.Automatic)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)


        eventType = "Foot Off"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext, btk.btkEvent.Automatic)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)

        eventType = "Foot Strike"
        eventContext = "Right"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext, btk.btkEvent.Automatic)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)

        eventType = "Foot Off"
        eventContext = "Right"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext, btk.btkEvent.Automatic)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)

        eventType = "Event"
        eventContext = "General"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext, btk.btkEvent.Manual)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)

        eventType = "Left-FP"
        eventContext = "General"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext,btk.btkEvent.Manual)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)

        eventType = "Right-FP"
        eventContext = "General"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext,btk.btkEvent.Manual)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)


        eventType = "Start"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time, int(frame), eventContext,btk.btkEvent.Manual)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)

        eventType = "End"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = btk.btkEvent(eventType,time , int(frame), eventContext, btk.btkEvent.Manual)
                ev.SetSubject(str(self.m_subject))
                self.m_acq.AppendEvent(ev)



    def appendMarkers(self):

        markersLoaded = NEXUS.GetMarkerNames(self.m_subject)
        markers =[]
        for i in range(0,len(markersLoaded)):
            data = NEXUS.GetTrajectory(self.m_subject,markersLoaded[i])
            if data != ([],[],[],[]):
                markers.append(markersLoaded[i])


        for marker in markers:
            rawDataX, rawDataY, rawDataZ, E = NEXUS.GetTrajectory(self.m_subject,marker)

            E = np.asarray(E).astype("float")-1
            values =np.array([np.asarray(rawDataX),np.asarray(rawDataY),np.asarray(rawDataZ)]).T

            if values.shape[0]<self.m_lastFrame:
                values_cut = values
                E_cut = E
            else:
                values_cut = values[(self.m_firstFrame-1):self.m_lastFrame,:]
                E_cut = E[(self.m_firstFrame-1):self.m_lastFrame]

            btkTools.smartAppendPoint(self.m_acq,str(marker),values_cut, PointType=btk.btkPoint.Marker,desc="",
                                      residuals=E_cut)

    def appendAnalogs(self):

        for nexusAnalogDevice in self.m_nexusAnalogDevices:

            channels = nexusAnalogDevice.getChannels()
            for channel in channels:
                analog = btk.btkAnalog()
                analog.SetLabel(channel.getLabel())
                analog.SetUnit(channel.getUnit())
                analog.SetFrameNumber(self.m_analogFrameNumber)
                analog.SetValues(channel.getValues()[(self.m_firstFrame-1)*self.m_numberAnalogSamplePerFrame:self.m_lastFrame*self.m_numberAnalogSamplePerFrame])
                analog.SetDescription(channel.getDescription())

                self.m_acq.AppendAnalog(analog)

    def appendForcePlates(self):

        forcePlateNumber = len(self.m_nexusForcePlates)

        fp_count=0
        for nexusForcePlate in self.m_nexusForcePlates:
            forceLocal = nexusForcePlate.getLocalReactionForce()
            momentLocal = nexusForcePlate.getLocalReactionMoment()

            forceLabels =["Force.Fx"+str(fp_count+1), "Force.Fy"+str(fp_count+1),"Force.Fz"+str(fp_count+1)]
            for j in range(0,3):
                analog = btk.btkAnalog()
                analog.SetLabel(forceLabels[j])
                analog.SetUnit("N")#nexusForcePlate.getForceUnit())
                analog.SetFrameNumber(self.m_analogFrameNumber)
                analog.SetValues(forceLocal[(self.m_firstFrame-1)*self.m_numberAnalogSamplePerFrame:self.m_lastFrame*self.m_numberAnalogSamplePerFrame,j])
                analog.SetDescription(nexusForcePlate.getDescription())
                #analog.SetGain(btk.btkAnalog.PlusMinus10)

                self.m_acq.AppendAnalog(analog)

            momentLabels =["Moment.Mx"+str(fp_count+1), "Moment.My"+str(fp_count+1),"Moment.Mz"+str(fp_count+1)]
            for j in range(0,3):
                analog = btk.btkAnalog()
                analog.SetLabel(momentLabels[j])
                analog.SetUnit("Nmm")#nexusForcePlate.getMomentUnit())
                analog.SetFrameNumber(self.m_analogFrameNumber)
                analog.SetValues(momentLocal[(self.m_firstFrame-1)*self.m_numberAnalogSamplePerFrame:self.m_lastFrame*self.m_numberAnalogSamplePerFrame,j])
                analog.SetDescription(nexusForcePlate.getDescription())
                #analog.GetGain(btk.btkAnalog.PlusMinus10)
                self.m_acq.AppendAnalog(analog)

            fp_count+=1

        # metadata for platform type2
        md_force_platform = btk.btkMetaData('FORCE_PLATFORM')
        btk.btkMetaDataCreateChild(md_force_platform, "USED", int(forcePlateNumber))# a
        btk.btkMetaDataCreateChild(md_force_platform, "ZERO", [1,0]) #
        btk.btkMetaDataCreateChild(md_force_platform, "TYPE", btk.btkDoubleArray(forcePlateNumber, 2)) #btk.btkDoubleArray(forcePlateNumber, 2))# add a child

        self.m_acq.GetMetaData().AppendChild(md_force_platform)


        origins = []
        for nexusForcePlate in self.m_nexusForcePlates:
            origins.append(-1.0*nexusForcePlate.getLocalOrigin())

        md_origin = btk.btkMetaData('ORIGIN')
        md_origin.SetInfo(btk.btkMetaDataInfo([3,int(forcePlateNumber)], np.concatenate(origins)))
        md_force_platform.AppendChild(md_origin)


        corners = []
        for nexusForcePlate in self.m_nexusForcePlates:
            corners.append(nexusForcePlate.getCorners().T.flatten())

        md_corners = btk.btkMetaData('CORNERS')
        md_corners.SetInfo(btk.btkMetaDataInfo([3,4,int(forcePlateNumber)], np.concatenate(corners)))
        md_force_platform.AppendChild(md_corners)


        md_channel = btk.btkMetaData('CHANNEL')
        md_channel.SetInfo(btk.btkMetaDataInfo([6,int(forcePlateNumber)], np.arange(1,int(forcePlateNumber)*6+1)))
        md_force_platform.AppendChild(md_channel)


    def appendModelOutputs(self):

        modelOutputNames = NEXUS.GetModelOutputNames(self.m_subject)

        if modelOutputNames!=[]:
            for modelOutputName in modelOutputNames:
                data, E = NEXUS.GetModelOutput(self.m_subject,modelOutputName)

                type = NEXUS.GetModelOutputDetails(self.m_subject,modelOutputName)[0]

                E = np.asarray(E).astype("float")-1
                values =np.array([np.asarray(data[0]),np.asarray(data[1]),np.asarray(data[2])]).T

                if values.shape[0]<self.m_lastFrame:
                    values_cut = values
                    E_cut = E
                else:
                    values_cut = values[(self.m_firstFrame-1):self.m_lastFrame,:]
                    E_cut = E[(self.m_firstFrame-1):self.m_lastFrame]

                if type == "Angles":
                    btkTools.smartAppendPoint(self.m_acq,str(modelOutputName),values_cut, PointType=btk.btkPoint.Angle,desc="",
                                              residuals=E_cut)
                elif type == "Forces":
                    btkTools.smartAppendPoint(self.m_acq,str(modelOutputName),values_cut, PointType=btk.btkPoint.Force,desc="",
                                              residuals=E_cut)
                elif type == "Moments":
                    btkTools.smartAppendPoint(self.m_acq,str(modelOutputName),values_cut, PointType=btk.btkPoint.Moment,desc="",
                                              residuals=E_cut)
                elif type == "Powers":
                    btkTools.smartAppendPoint(self.m_acq,str(modelOutputName),values_cut, PointType=btk.btkPoint.Power,desc="",
                                              residuals=E_cut)
                elif type == "Modeled Markers":
                    btkTools.smartAppendPoint(self.m_acq,str(modelOutputName),values_cut, PointType=btk.btkPoint.Marker,desc="",
                                              residuals=E_cut)
                else:
                    logging.warning("[pyCGM2] : Model Output (%s) from Nexus not added to the btk acquisition"%(modelOutputName))



    def build(self):
        self.appendEvents()
        self.appendMarkers()
        if self.m_nexusForcePlates !=[]: self.appendForcePlates()
        if self.m_nexusAnalogDevices !=[]:self.appendAnalogs()
        self.appendModelOutputs()


        return self.m_acq

    def exportC3d(self,filenameNoExt=None):

        if filenameNoExt is None:
            btkTools.smartWriter(self.m_acq,str(self.m_dataPath+self.m_filenameNoExt+".c3d"))
        else:
            btkTools.smartWriter(self.m_acq,str(self.m_dataPath+filenameNoExt+".c3d"))


class NexusConstructTrialFilter(object):
    def __init__(self,dataPath,filenameNoExt,subject):

        """
        """
        self.m_dataPath = dataPath
        self.m_filenameNoExt = filenameNoExt
        self.m_subject = subject

        self.m_framerate = NEXUS.GetFrameRate()
        #self.m_frames = NEXUS.GetTrialRange()[1]
        self.m_rangeROI = NEXUS.GetTrialRegionOfInterest()
        self.m_trialRange = NEXUS.GetTrialRange()

        self.m_firstFrame = self.m_rangeROI[0]
        self.m_lastFrame = self.m_rangeROI[1]
        self.m_frames = self.m_lastFrame-(self.m_firstFrame-1)



        deviceIDs = NEXUS.GetDeviceIDs()
        self.m_analogFrameRate = NEXUS.GetDeviceDetails(1)[2] if ( len(deviceIDs) > 0 ) else self.m_framerate


        self.m_numberAnalogSamplePerFrame = int(self.m_analogFrameRate/self.m_framerate)
        self.m_analogFrameNumber = self.m_frames * self.m_numberAnalogSamplePerFrame


        self.m_nexusForcePlates = list()
        self.m_nexusAnalogDevices = list()


        if( len(deviceIDs) > 0 ):
            for deviceID in deviceIDs:
                if NEXUS.GetDeviceDetails( deviceID )[1] == "ForcePlate":
                    self.m_nexusForcePlates.append( Devices.ForcePlate(deviceID))
                else:
                    self.m_nexusAnalogDevices.append(Devices.AnalogDevice(deviceID))

        self.m_root = ma.Node('root')
        self.m_trial = ma.Trial("NexusTrial",self.m_root)


    def appendEvents(self):

        eventType = "Foot Strike"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())


        eventType = "Foot Off"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())

        eventType = "Foot Strike"
        eventContext = "Right"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())

        eventType = "Foot Off"
        eventContext = "Right"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())

        eventType = "Event"
        eventContext = "General"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())

        eventType = "Left-FP"
        eventContext = "General"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())

        eventType = "Right-FP"
        eventContext = "General"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())


        eventType = "Start"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())

        eventType = "End"
        eventContext = "Left"
        if NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0] != []:
            for frame in NEXUS.GetEvents(self.m_subject,eventContext,eventType)[0]:
                time = (frame-1)/self.m_framerate
                ev = ma.Event(eventType,time,eventContext,str(self.m_subject),self.m_trial.events())



    def appendMarkers(self):


        markersLoaded = NEXUS.GetMarkerNames(self.m_subject)
        markers =[]
        for i in range(0,len(markersLoaded)):
            data = NEXUS.GetTrajectory(self.m_subject,markersLoaded[i])
            if data != ([],[],[],[]):
                markers.append(markersLoaded[i])


        for marker in markers:
            rawDataX, rawDataY, rawDataZ, E = NEXUS.GetTrajectory(self.m_subject,marker)

            E = np.asarray(E).astype("float")-1
            values =np.array([np.asarray(rawDataX),np.asarray(rawDataY),np.asarray(rawDataZ)]).T

            if values.shape[0]<self.m_lastFrame:
                values_cut = values
                E_cut = E
            else:
                values_cut = values[(self.m_firstFrame-1):self.m_lastFrame,:]
                E_cut = E[(self.m_firstFrame-1):self.m_lastFrame]

            data = np.zeros((values_cut.shape[0],4))
            data[:,0:3] = values_cut
            data[:,3] = E_cut

            if self.m_firstFrame ==1:
                time_init = 0.0
            else:
                time_init = self.m_firstFrame/self.m_framerate


            ts = ma.TimeSequence(str(marker),4,data.shape[0],self.m_framerate,time_init,ma.TimeSequence.Type_Marker,"mm", self.m_trial.timeSequences())
            ts.setData(data)







    def appendAnalogs(self):
        for nexusAnalogDevice in self.m_nexusAnalogDevices:

            channels = nexusAnalogDevice.getChannels()
            for channel in channels:

                data = channel.getValues()[(self.m_firstFrame-1)*self.m_numberAnalogSamplePerFrame:self.m_lastFrame*self.m_numberAnalogSamplePerFrame]

                if self.m_firstFrame ==1:
                    time_init = 0.0
                else:
                    time_init = self.m_firstFrame/self.m_analogFrameRate


                ts = ma.TimeSequence(str(channel.getLabel()),1,data.shape[0],self.m_analogFrameRate,time_init,ma.TimeSequence.Type_Analog,"V", 1.0,0.0,[-10.0,10.0], self.m_trial.timeSequences())
                ts.setData(data.reshape((data.shape[0],1)))
                ts.setDescription(channel.getDescription())



    def appendForcePlates(self):

        pass


    def appendModelOutputs(self):

        modelOutputNames = NEXUS.GetModelOutputNames(self.m_subject)

        if modelOutputNames!=[]:
            for modelOutputName in modelOutputNames:
                data, E = NEXUS.GetModelOutput(self.m_subject,modelOutputName)

                type = NEXUS.GetModelOutputDetails(self.m_subject,modelOutputName)[0]

                E = np.asarray(E).astype("float")-1
                values =np.array([np.asarray(data[0]),np.asarray(data[1]),np.asarray(data[2])]).T

                if values.shape[0]<self.m_lastFrame:
                    values_cut = values
                    E_cut = E
                else:
                    values_cut = values[(self.m_firstFrame-1):self.m_lastFrame,:]
                    E_cut = E[(self.m_firstFrame-1):self.m_lastFrame]

                data = np.zeros((values_cut.shape[0],4))
                data[:,0:3] = values_cut
                data[:,3] = E_cut

                if self.m_firstFrame ==1:
                    time_init = 0.0
                else:
                    time_init = self.m_firstFrame/self.m_framerate

                if type == "Angles":
                    ts = ma.TimeSequence(str(modelOutputName),4,values_cut.shape[0],self.m_framerate,time_init,ma.TimeSequence.Type_Angle,"Deg", self.m_trial.timeSequences())
                    ts.setData(data)

                elif type == "Forces":
                    ts = ma.TimeSequence(str(modelOutputName),4,values_cut.shape[0],self.m_framerate,time_init,ma.TimeSequence.Type_Force,"N.Kg-1", self.m_trial.timeSequences())
                    ts.setData(data)

                elif type == "Moments":
                    ts = ma.TimeSequence(str(modelOutputName),4,values_cut.shape[0],self.m_framerate,time_init,ma.TimeSequence.Type_Moment,"Nmm.Kg-1", self.m_trial.timeSequences())
                    ts.setData(data)

                elif type == "Powers":
                    ts = ma.TimeSequence(str(modelOutputName),4,values_cut.shape[0],self.m_framerate,time_init,ma.TimeSequence.Type_Power,"Watt.Kg-1", self.m_trial.timeSequences())
                    ts.setData(data)

                elif type == "Modeled Markers":
                    ts = ma.TimeSequence(str(modelOutputName),4,values_cut.shape[0],self.m_framerate,time_init,ma.TimeSequence.Type_Marker,"mm", self.m_trial.timeSequences())
                    ts.setData(data)
                else:
                    logging.warning("[pyCGM2] : Model Output (%s) from Nexus not added to the btk acquisition"%(modelOutputName))


    def build(self):
        self.appendEvents()
        self.appendMarkers()
        if self.m_nexusForcePlates !=[]: self.appendForcePlates()
        if self.m_nexusAnalogDevices !=[]:self.appendAnalogs()
        self.appendModelOutputs()

        trialTools.sortedEvents(self.m_trial)

        return self.m_trial

    def exportC3d(self,filenameNoExt=None):


        if filenameNoExt is None:
            ma.io.write(self.m_root,str(self.m_dataPath+self.m_filenameNoExt+".c3d"))
        else:
            ma.io.write(self.m_root,str(self.m_dataPath+filenameNoExt+".c3d"))
