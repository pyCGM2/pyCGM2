import os
import sys

#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Win32')
#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Python')

from ViconNexus import *

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 DisplayModelOutputInfo will display general information about the model
 outputs that are defined for a loaded subject.

 Input
     vicon    = instance of a Vicon sdk object
     subject  = name of the subject

 Usage Example: 

    vicon = ViconNexus();
    DisplayModelOutputInfo(vicon, 'Colin')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def DisplayModelOutputInfo(vicon,subject):
  ModelOutputs = vicon.GetModelOutputNames(subject)
  if len(ModelOutputs) > 0:
    print 'Model Outputs:'

    GroupNames = []
    for ModelOutput in ModelOutputs:
      group = vicon.GetModelOutputDetails(subject, ModelOutput)[0]
      if not (group in GroupNames):
        GroupNames.append(group)

    GroupNames.sort()

    for groupName in GroupNames:
      print groupName
      ModelOutputNames = []
      for ModelOutput in ModelOutputs:
        group = vicon.GetModelOutputDetails(subject, ModelOutput)[0]
        if groupName == group:
          ModelOutputNames.append(ModelOutput)

      ModelOutputNames.sort()

      for ModelOutputName in ModelOutputNames:
        components = vicon.GetModelOutputDetails(subject, ModelOutputName)[1]
        types = vicon.GetModelOutputDetails(subject, ModelOutputName)[2]
        theOutput = ModelOutputName + ' ['
        for k, component in enumerate(components):
          if k == 0:
            theOutput = theOutput + component + '(' + types[k] + ')'
          else:
            theOutput = theOutput + ', ' + component + '(' + types[k] + ')'
        print theOutput + ']'
  else:
    print 'No defined model outputs for subject ' + subject
    
if __name__ == "__main__":
    vicon = ViconNexus();
    DisplayModelOutputInfo(vicon, sys.argv[1])  
