import os
import sys
import math

#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Win32')
#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Python')

from ViconNexus import *
from ViconUtils import *

class DetectFootEvents:
  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  FindEvents will look for foot strike and foot off events for
  a loaded subject
  This function will look for events in the valid updatable frame
  range for the loaded trial
  This function will look for events for all defined force plates
  
  Input
      vicon           = instance of a Vicon sdk object
      subject         = name of the subject
      forceThresh     = force threshold
      leftAntMarker   = name of the Left Anterior Marker
      leftPostMarker  = name of the Left Posterior Marker
      rightAntMarker  = name of the Right Anterior Marker
      rightPostMarker = name of the Right Posterior Marker
  
  Usage Example: 
  
      vicon = ViconNexus()
      vicon.ClearAllEvents()
      eventDetector = DetectFootEvents()
      eventDetector.FindEvents(vicon, "Colin", 20.0, "LTOE", "LANK", "RTOE", "RANK")
  
  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  def FindEvents(self, vicon, subject, forceThresh, leftAntMarker, leftPostMarker, rightAntMarker, rightPostMarker):
    self.MarkerNames = [leftAntMarker, leftPostMarker, rightAntMarker, rightPostMarker]

    # validate the input data
    rate = vicon.GetFrameRate()
    if rate < 1:
      print 'Invalid frame rate for the loaded trial'
      sys.exit(1)

    subjects = vicon.GetSubjectNames()
    if subject in subjects:
      DefinedMarkers = vicon.GetMarkerNames(subject)
      for i in range(4): 
        if not self.MarkerNames[i] in DefinedMarkers: 
          print 'Invalid marker name ' + self.MarkerNames[i]

      # retrieve the marker data that we are going to need
      FrameCount = vicon.GetFrameCount()
      self.MarkerDataExists = [[False]*FrameCount for x in xrange(4)]
      self.MarkerData = [[[0]*FrameCount]*3]*4
      for i in range(4):
        #print str(i)
        #print vicon.GetTrajectory(subject,self.MarkerNames[i])[0]
        #self.MarkerData[i][0] = vicon.GetTrajectory(subject,self.MarkerNames[i])[0]
        (self.MarkerData[i][0], self.MarkerData[i][1], self.MarkerData[i][2], self.MarkerDataExists[i]) = vicon.GetTrajectory(subject,self.MarkerNames[i])

      # loop through the force plates looking for events
      deviceIDs = vicon.GetDeviceIDs()
      for deviceID in deviceIDs:
        type = vicon.GetDeviceDetails(deviceID)[1]
        if type =='ForcePlate': 
          self.ProcessForcePlate(vicon,subject,deviceID,forceThresh)
    else:
      print 'Invalid subject name ' + subject
      sys.exit(1)
      
  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  FindEvents helper function
  Look for events for a specific force plate.
  
  Input
      vicon           = instance of a Vicon sdk object
      subject         = name of the subject
      deviceID        = unique deviceID identifying the force plate
      forceThresh     = force threshold
  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

  def ProcessForcePlate(self,vicon,subject,deviceID,forceThresh):
    # retrieve the force plate data
    deviceRate = vicon.GetDeviceDetails(deviceID)[2]
    rate = vicon.GetFrameRate()
    SamplesPerFrame = 1
    if deviceRate > rate:
      SamplesPerFrame = deviceRate / rate

    FrameCount = vicon.GetFrameCount()
    self.ForcePlateData = [[0]*(FrameCount * SamplesPerFrame) for y in range(3)]
            
    # DeviceOutput = 'Force', channels 'Fx', 'Fy' and 'Fz'
    deviceOutputID = vicon.GetDeviceOutputIDFromName(deviceID,'Force')
    channelID = vicon.GetDeviceChannelIDFromName(deviceID, deviceOutputID, 'Fx')
    self.ForcePlateData[0] = vicon.GetDeviceChannelGlobal( deviceID, deviceOutputID, channelID )[0]
    ready = vicon.GetDeviceChannelGlobal( deviceID, deviceOutputID, channelID )[1]
    channelID = vicon.GetDeviceChannelIDFromName(deviceID, deviceOutputID, 'Fy')
    self.ForcePlateData[1] = vicon.GetDeviceChannelGlobal( deviceID, deviceOutputID, channelID )[0]
    channelID = vicon.GetDeviceChannelIDFromName(deviceID, deviceOutputID, 'Fz')
    self.ForcePlateData[2] = vicon.GetDeviceChannelGlobal( deviceID, deviceOutputID, channelID )[0]

    # DeviceOutput = 'CoP', channels 'Cx', 'Cy' and 'Cz'
    self.CoPData = [[0]*(FrameCount * SamplesPerFrame) for y in range(3)]
    deviceOutputID = vicon.GetDeviceOutputIDFromName(deviceID,'CoP')
    channelID = vicon.GetDeviceChannelIDFromName(deviceID, deviceOutputID, 'Cx')
    self.CoPData[0] = vicon.GetDeviceChannel( deviceID, deviceOutputID, channelID )[0]
    channelID = vicon.GetDeviceChannelIDFromName(deviceID, deviceOutputID, 'Cy')
    self.CoPData[1] = vicon.GetDeviceChannel( deviceID, deviceOutputID, channelID )[0]
    channelID = vicon.GetDeviceChannelIDFromName(deviceID, deviceOutputID, 'Cz')
    self.CoPData[2] = vicon.GetDeviceChannel( deviceID, deviceOutputID, channelID )[0]

    # now look for the events based on the threshold
    if ready == True:
      bForceOn = False
      self.LastFrameAboveThreshold = -1
            
      (startFrame, endFrame) = vicon.GetTrialRange()
      for i in range(startFrame,endFrame+1):
        for j in range(1, SamplesPerFrame+1):
          SampleIndex = ((i-1) * SamplesPerFrame) + (j-1)
          x = self.ForcePlateData[0][SampleIndex]
          y = self.ForcePlateData[1][SampleIndex]
          z = self.ForcePlateData[2][SampleIndex]
          force = self.norm(x,y,z)             
          if force >= forceThresh and self.ForcePlateData[2][SampleIndex] < 0:
            self.LastFrameAboveThreshold = i
            if bForceOn == False:
              # Foot Strike
              self.HandleEvent( vicon, subject, i, (j-1) * (1.0/deviceRate), 'Foot Strike', deviceID, SamplesPerFrame ) 
              bForceOn = True
          else:
            if bForceOn and force < forceThresh:
              # Foot Off
              self.HandleEvent( vicon, subject, i, (j-1) * (1.0/deviceRate), 'Foot Off', deviceID, SamplesPerFrame)
              bForceOn = False


  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  FindEvents helper function
  Determine the proper context (left or right) and creates an event in the application
  
  Input
  vicon           = instance of a Vicon sdk object
  subject         = name of the subject
  frame           = frame number where the event occurs
  offset          = offset to add to the frame if the event occurred between frame boundaries
  event           = type of event, Foot Strike or Foot Off   
  deviceID        = unique deviceID identifying the force plate
  SamplesPerFrame = number of force plate samples that correspond to each data frame
  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  def HandleEvent(self,vicon,subject,frame,offset,event,deviceID,SamplesPerFrame):
    context = self.DetermineRightLeft(vicon,deviceID,frame,SamplesPerFrame)
    print context + ' ' + event + ' event at frame ' + str(frame) + ' offset ' + str(offset)
            
    # do not add a duplicate event if this one already exists
    (eventFrames, eventOffsets) = vicon.GetEvents(subject, context, event )
    exists = False
    for i in range (len(eventFrames)):
      if eventFrames[i] == frame:
        if abs(eventOffsets[i] - offset) < 0.001:
          exists = True
    if exists:
      print ' Events already exists'
    else:
      vicon.CreateAnEvent( subject, context, event, frame, offset )


  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  FindEvents helper function
  Determine whether or not the left or right markers are
  closest to the plate (point = either centre, or the CoP)
  
  Input
  vicon           = instance of a Vicon sdk object
  deviceID        = unique deviceID identifying the force plate
  frame           = frame number where the event occurs
  SamplesPerFrame = number of force plate samples that correspond to each data frame
  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  def DetermineRightLeft(self,vicon,deviceID,frame,SamplesPerFrame):
    context = 'General'
            
    # Calculate a reference point, start with the centre of the force plate
    ForcePlate = vicon.GetDeviceDetails(deviceID)[4]
   
    utils = ViconUtils()

    # reference point is the center of the plate
    UpperBounds = ForcePlate.UpperBounds
    UpperBoundsX = UpperBounds[0]
    UpperBoundsY = UpperBounds[1]
    UpperBoundsZ = UpperBounds[2]
    LowerBounds = ForcePlate.LowerBounds
    LowerBoundsX = LowerBounds[0]
    LowerBoundsY = LowerBounds[1]
    LowerBoundsZ = LowerBounds[2]
    RefPointX = (UpperBoundsX + LowerBoundsX) / 2
    RefPointY = (UpperBoundsY + LowerBoundsY) / 2
    RefPointZ = (UpperBoundsZ + LowerBoundsZ) / 2
    RefPoint = [RefPointX, RefPointY, RefPointZ]
    
    if (frame - self.LastFrameAboveThreshold) <= 1:
      # However, if the last sample that had an above threshold reading
      # is either the current sample or the one before, use the position
      # of the FPs centre of pressure as the reference point
      #CoP = [0]*3 
      SampleIndex = ((self.LastFrameAboveThreshold-1) * SamplesPerFrame) + 1
      CoPx = 0
      CoPy = 0
      CoPz = 0
      for i in range(SamplesPerFrame):
        x = self.CoPData[0][SampleIndex]
        y = self.CoPData[1][SampleIndex]
        z = self.CoPData[2][SampleIndex]
        CoPx = CoPx + x
        CoPy = CoPy + y
        CoPz = CoPz + z
        SampleIndex = SampleIndex + 1
        
      RefPoint = (CoPx / SamplesPerFrame, CoPy / SamplesPerFrame, CoPz / SamplesPerFrame)
      # unit conversion
      RefPoint = (RefPoint[0] / 1000.0, RefPoint[1] / 1000.0, RefPoint[2] / 1000.0)
      
            
    # Globalise the reference point
    RefPoint = utils.Globalise(RefPoint, ForcePlate.WorldR, ForcePlate.WorldT)
            
    # Find the marker that is closest to the plate 
    MinDist = -1.0
    ClosestMarkerIndex = 0
            
    for i in range(4):
      if self.MarkerDataExists[i][frame]:
        # Test if the marker is within the bounds of the plate and,
        # if so, see if it is the closest. 
        MarkerPosx = self.MarkerData[i][0][frame]
        MarkerPosy = self.MarkerData[i][1][frame]
        MarkerPosz = self.MarkerData[i][2][frame]
        MarkerPos = [MarkerPosx, MarkerPosy, MarkerPosz]
        MarkerPosLocal = utils.Localise(MarkerPos, ForcePlate.WorldR, ForcePlate.WorldT )

        if (ForcePlate.LowerBounds[0] < MarkerPosLocal[0]) and (MarkerPosLocal[0] < ForcePlate.UpperBounds[0]):  
          if (ForcePlate.LowerBounds[1] < MarkerPosLocal[1]) and (MarkerPosLocal[1] < ForcePlate.UpperBounds[1]):
            if MarkerPosLocal[2] > 0:
              # candidate for closest marker
              xDiff = MarkerPosx - RefPoint[0]
              yDiff = MarkerPosy - RefPoint[1]
              zDiff = MarkerPosz - RefPoint[2]
              dist = self.norm(xDiff, yDiff, zDiff)
              if (MinDist < 0.0) or (dist < MinDist):
                MinDist = dist
                ClosestMarkerIndex =  i
                
            
    # the closest marker determines the context, if we didn't determine
    # a closes marker then we return 'General' as the context
    if ( ClosestMarkerIndex == 0 ) or ( ClosestMarkerIndex == 1 ):
      context = 'Left'
    else:
      if ( ClosestMarkerIndex == 2 ) or ( ClosestMarkerIndex == 3 ):
        context = 'Right'

    return context

  def norm(self, x, y, z):
    return math.sqrt( x*x + y*y + z*z )

if __name__ == "__main__":    
    vicon = ViconNexus()
    eventDetector = DetectFootEvents()
    eventDetector.FindEvents(vicon, "Colin", 20.0, "LTOE", "LANK", "RTOE", "RANK")     
