import numpy as np
import pyCGM2
from pyCGM2.Tools import btkTools

from pyCGM2.Model.Models import singleBody


DATA_PATH = "C:\\YOURPATH\\"

acqStatic = btkTools.smartReader(DATA_PATH + "static.c3d")
acqDyn = btkTools.smartReader(DATA_PATH + "motion 01.c3d")

targetMarker = "pointToreconstruct"

trackingMarkers = ["rigidO","rigidX","rigidY"]

calibrationMarkers = ["rigidO","rigidX","rigidY","rigidO"] # 4 markers to define the primary Axis  (marker2-marker1), the secondary axis (marker3-marker1), and the Origin (marker4)

rigidBody = singleBody.SingleBody(calibrationMarkers+[targetMarker], trackingMarkers)
rigidBody.calibrate(acqStatic) 
rigidBody.fit(acqDyn)

# Use 1 - add the target marker to the c3d
traj = rigidBody.getTrajectory(targetMarker)
btkTools.smartAppendPoint(acqDyn,targetMarker,traj)

# Use 1 - add a global position as node and add its trajectory  to the c3d
globalpos = np.array([300,400,800])
rigidBody.addNode("new",globalpos,positionType="Global")
traj = rigidBody.getTrajectory("new")

btkTools.smartAppendPoint(acqDyn,"new",traj)

# Use 2 - add a local position ( ie in the local coordinate system of the single body) as node and add its trajectory  to the c3d
localpos = np.array([0.5,2,4])
rigidBody.addNode("new2",localpos,positionType="Local")
traj = rigidBody.getTrajectory("new2")
btkTools.smartAppendPoint(acqDyn,"new2",traj)


btkTools.smartWriter(acqDyn,"new.c3d") 