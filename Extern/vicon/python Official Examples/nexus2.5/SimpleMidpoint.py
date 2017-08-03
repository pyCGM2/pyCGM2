import os
import sys

#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Win32')
#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Python')

from ViconNexus import *

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 SimpleMidpoint will create a midpoint between 2 existing markers
 for a loaded subject. The midpoint marker is created as a Modeled Marker

 Input
     vicon    = instance of a Vicon sdk object
     subject  = name of the subject
     marker1  = name of the first marker to be used to create the midpoint
     marker2  = name of the second marker to be used to create the midpoint
     name     = name of the midpoint modeled marker to create

 Usage Example: 

    vicon = ViconNexus();
    SimpleMidpoint(vicon, 'Colin', 'RKNE', 'RANK', 'MyMidpoint')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def SimpleMidpoint(vicon, subject, marker1, marker2, name):
  subjects = vicon.GetSubjectNames()
  # Validate the input data
  if subject in subjects:
    markers = vicon.GetMarkerNames(subject)
    if marker1 in markers :
      if marker2 in markers:
        # Create a modeled marker in the model outputs which will allow data
        # storage as well as workspace visualization
        vicon.CreateModeledMarker(subject, name)

        # Get the input data
        (X1, Y1, Z1, E1) = vicon.GetTrajectory(subject, marker1)
        (X2, Y2, Z2, E2) = vicon.GetTrajectory(subject, marker2)

        # Calculate the output data
        (data, exists) = vicon.GetModelOutput(subject, name)
        framecount = vicon.GetFrameCount()
        for i in range(framecount):
          if E1[i] and E2[i]:
            exists[i] = True
            data[0][i] = (X1[i]+X2[i])/2
            data[1][i] = (Y1[i]+Y2[i])/2
            data[2][i] = (Z1[i]+Z2[i])/2
          else:
            exists[i] = False
        # Update the model output in the application
        vicon.SetModelOutput( subject, name, data, exists )
      else:
        print "Error: Invalid marker name: " + marker2
        sys.exit(1)
    else:
      print "Error: Invalid marker name: " + marker1
      sys.exit(1)
  else:
    print "Error: Invalid Subject Name: " + subject
    sys.exit(1)

    
if __name__ == "__main__":    
    vicon = ViconNexus()
    SimpleMidpoint (vicon, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]) 
  


