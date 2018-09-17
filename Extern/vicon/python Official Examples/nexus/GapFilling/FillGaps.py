import os
import sys

#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Win32')
#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Python')

from ViconNexus import *
from ProcessGaps import *

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 FillGaps will fill small gaps within a trajectory.
 This function does not process missing data at the trajectory endpoints

 function ProcessGaps must be accessible to run this script

 Input
     vicon    = instance of a Vicon sdk object
     Subjects = array of strings, List of subjects to process
     Markers  = array of strings, List of markers to gap fill
     MaxGap   = integer number, largest gap size that you want to fill

 Usage Example: Fill gaps for LTOE and RTOE for all subjects 

    subjects = vicon.GetSubjectNames()
    markers = {'LTOE','RTOE'}
    FillGaps(vicon, subjects, markers, 4)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def FillGaps(vicon, Subjects, Markers, MaxGap):
  for Subject in Subjects:
    for Marker in Markers:
      # Get Trajectory for the trajectory from the host application
      (trajX, trajY, trajZ, trajExists) = vicon.GetTrajectory(Subject,Marker)

      # Get Frame Count for current Trial
      frameCount = vicon.GetFrameCount()
        
      # Get the updateable trial frame range
      (startFrame, endFrame ) = vicon.GetTrialRange()

      # Make sure that we have gaps that need filling
      HaveGaps = False
      idx = startFrame
      print 'idx: ' +str(idx)
      while not HaveGaps and idx <= endFrame:
        if trajExists[idx] != True:
          HaveGaps = True
        idx = idx + 1

      if HaveGaps:
        # we dont interpolate endpoints so find the first frame of
        # existing data to define our processing range
        firstFrame = startFrame
        for n in range(frameCount):
          if trajExists[n] == True:  
            firstFrame = n
            break

        # now find the last frame of our processing range
        lastFrame = endFrame
        for n in reversed(range(endFrame)):
          if trajExists[n] == True: 
            lastFrame = n
            break

        # List of frame numbers corresponding to existing data
        ExistingDataFrameNumbers = []
        for index, val in enumerate(trajExists):
          if val == True:
            ExistingDataFrameNumbers.append(index)

        # Determine the size of the gaps
        gapSize = []
        gapLocation = []
        for n in range(len(ExistingDataFrameNumbers)-1):
          gapFrame = 2*[0]
          if abs(ExistingDataFrameNumbers[n]-ExistingDataFrameNumbers[n+1]) !=1:
            x = abs(ExistingDataFrameNumbers[n]-ExistingDataFrameNumbers[n+1]) - 1
            gapSize.append(x) # Size of each gap
            gapFrame = (ExistingDataFrameNumbers[n], ExistingDataFrameNumbers[n+1])
            # Location of gaps. first and last frame before/after gap
            gapLocation.append(gapFrame)

        # LargeGaps is the list of indicies in the gapSize array 
        # corresponding to the large gaps
        LargeGaps = []
        for index, val in enumerate(gapSize):
          if val > MaxGap:
            LargeGaps.append(index)

        # SmallGaps is the list of indicies in the gapSize array 
        # corresponding to the small gaps
        SmallGaps = []
        for index, val in enumerate(gapSize):
          if val <= MaxGap:
            SmallGaps.append(index)
       
        # process the trajectory in sections, jumping over the large
        # gaps
        currentIdx = 0
        firstToProcess = firstFrame
        lastToProcess = lastFrame
        
        while currentIdx < len(LargeGaps):
          # calculate the last frame number of our processing section
          lastToProcess = gapLocation[LargeGaps[currentIdx]][0]
          
          # fill all the gaps in the section
          for location in gapLocation:
            if location[0] > firstToProcess and location[0] < lastToProcess:
              (X, Y, Z, E) = ProcessGaps(location[0], location[1], trajX, trajY, trajZ)
              trajX[location[0]+1:location[1]] = X
              trajY[location[0]+1:location[1]] = Y
              trajZ[location[0]+1:location[1]] = Z
              trajExists[location[0]+1:location[1]] = E 
              
          # calculate the frame range of the next section
          # assuming that the next section goes to the end
          firstToProcess = gapLocation[LargeGaps[currentIdx]][1]
          lastToProcess = lastFrame
                
          currentIdx = currentIdx + 1
        
        # now process the final section (also handles the case of no
        # large gaps)
        for location in gapLocation:
          if location[0] > firstToProcess and location[0] < lastToProcess:
            (X, Y, Z, E) = ProcessGaps(location[0], location[1], trajX, trajY, trajZ)
            trajX[location[0]+1:location[1]] = X
            trajY[location[0]+1:location[1]] = Y
            trajZ[location[0]+1:location[1]] = Z
            trajExists[location[0]+1:location[1]] = E

        # Update Trajectory with gap filled data
        vicon.SetTrajectory(Subject, Marker, trajX, trajY, trajZ, trajExists)

if __name__ == "__main__":
    vicon = ViconNexus()
    subjects = vicon.GetSubjectNames()
    markers = ['LTOE','RTOE']
    FillGaps(vicon, subjects, markers, 4)
          
