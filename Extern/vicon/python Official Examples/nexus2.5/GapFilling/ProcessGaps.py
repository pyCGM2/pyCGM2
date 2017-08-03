import os
import sys

#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Win32')
#sys.path.append( 'C:/Program Files (x86)/Vicon/Nexus2.1/SDK/Python')

from ViconNexus import *

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 ProcessGaps is a function that fills small gaps linear interpolation.  

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def ProcessGaps(startGapFrame, endGapFrame, trajX, trajY, trajZ):
  
  # data values for existing data
  dataX1 = trajX[startGapFrame]
  dataY1 = trajY[startGapFrame]
  dataZ1 = trajZ[startGapFrame]
  dataX2 = trajX[endGapFrame]
  dataY2 = trajY[endGapFrame]
  dataZ2 = trajZ[endGapFrame]

  # fill the gaps within the XYZ vectors
  X = []
  Y = []
  Z = []
  for frameNum in range(startGapFrame+1, endGapFrame):
    x = dataX1 + (dataX2 - dataX1)*(frameNum - startGapFrame)/(endGapFrame - startGapFrame)
    X.append(x)
    y = dataY1 + (dataY2 - dataY1)*(frameNum - startGapFrame)/(endGapFrame - startGapFrame)
    Y.append(y)
    z = dataZ1 + (dataZ2 - dataZ1)*(frameNum - startGapFrame)/(endGapFrame - startGapFrame)
    Z.append(z)
    E = (endGapFrame - startGapFrame -1) * [True]
      
  return (X,Y,Z,E)

