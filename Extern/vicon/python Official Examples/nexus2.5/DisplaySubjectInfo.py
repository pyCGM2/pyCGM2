import os
import sys

#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Win32')
#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Python')

from ViconNexus import *

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 DisplaySubjectInfo will display general information about a loaded subject.

 Input
     vicon    = instance of a Vicon sdk object
     subject  = name of the subject

 Usage Example: 

    vicon = ViconNexus()
    DisplaySubjectInfo(vicon, 'Colin')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def DisplaySubjectInfo(vicon,subject):
  # Markers
  Markers = vicon.GetMarkerNames(subject)
  if len(Markers) > 0:
    print 'Markers:'
    for Marker in Markers:
      print Marker

  # Segments
  Segments = vicon.GetSegmentNames(subject)
  if len(Segments) > 0:
    print 'Segments:'
    for Segment in Segments:
      markers = vicon.GetSegmentDetails(subject, Segment)[-1]
      theSegment = Segment + ' ['
      for j, marker in enumerate(markers):
        if j==0:
          theSegment = theSegment + marker
        else: 
          theSegment = theSegment + ', ' + marker
      print theSegment + ']'

  # Joints
  Joints = vicon.GetJointNames(subject)
  if len(Joints) > 0:
    print 'Joints:'
    for Joint in Joints:
      (parent, child) = vicon.GetJointDetails(subject, Joint)
      print Joint + ' [' + parent + ' - ' + child + ']'

  # Subject Parameters
  Parameters = vicon.GetSubjectParamNames(subject)
  if len(Parameters) > 0:
    print 'Subject Parameters:'
    for param in Parameters:
      value = vicon.GetSubjectParamDetails( subject, param)[0]
      unit = vicon.GetSubjectParamDetails( subject, param)[1]
      print param + ' = ' + str(value) + ' ' + unit
      
if __name__ == "__main__":
    vicon = ViconNexus()
    DisplaySubjectInfo(vicon, sys.argv[1])
  
