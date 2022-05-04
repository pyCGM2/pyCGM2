# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Model/Opensim
#APIDOC["Draft"]=False
#--end--

"""
This module contains  convenient functions and classes for working with opensim.
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER
import numpy as np

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    try:
        import opensim
    except:
        LOGGER.logger.error("[pyCGM2] : opensim not find on your system. Install it for working with the API")

from bs4 import BeautifulSoup





R_OSIM_CGM = {"Pelvis" : np.array([[1,0,0],[0,0,1],[0,-1,0]]) ,
              "Left Thigh" : np.array([[1,0,0],[0,0,1],[0,-1,0]]),
              "Left Shank" : np.array([[1,0,0],[0,0,1],[0,-1,0]]),
              "Right Thigh" : np.array([[1,0,0],[0,0,1],[0,-1,0]]),
              "Right Shank" : np.array([[1,0,0],[0,0,1],[0,-1,0]]),
              "Left Foot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
              "Right Foot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
              "Right HindFoot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
              "Right ForeFoot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
              "Left HindFoot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
              "Left ForeFoot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
              "Right Hindfoot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
              "Right Forefoot" : np.array([[0,0,-1],[1,0,0],[0,-1,0]])

              }




def globalTransformationLabToOsim(acq,R_LAB_OSIM):
    """transform all markers into the the opensim global coordinate system

    Args:
        acq (Btk.Acquisition): an acquisition
        R_LAB_OSIM (array(3,3)): rotation matrix for passing from the lab to opensim coordinate sytem

    """

    points = acq.GetPoints()
    for it in btk.Iterate(points):
        if it.GetType() == btk.btkPoint.Marker:
            values = np.zeros(it.GetValues().shape)
            for i in range(0,it.GetValues().shape[0]):
                values[i,:] = np.dot(R_LAB_OSIM,it.GetValues()[i,:])
            it.SetValues(values)

def smartTrcExport(acq,filenameNoExt):
    """save as a trc file

    Args:
        acq (Btk.Acquisition): an acquisition
        filenameNoExt (str): filename without extension

    """
    writerDyn = btk.btkAcquisitionFileWriter()
    writerDyn.SetInput(acq)
    writerDyn.SetFilename(filenameNoExt + ".trc")
    writerDyn.Update()



def sto2pointValues(storageObject,label,R_LAB_OSIM):
    """transform an element of a storage instance into the Laboratory coordinate system.

    Args:
        storageObject (opensim.Storage): an opensim storage instance.
        label (str): label to extract from the storage instance
        R_LAB_OSIM (array(3,3)): rotation matrix for passing from the lab to opensim coordinate sytem

    """

    # storageObject = opensim.Storage(stoFilename)
    index_x =storageObject.getStateIndex(label+"_tx")
    index_y =storageObject.getStateIndex(label+"_ty")
    index_z =storageObject.getStateIndex(label+"_tz")

    array_x=opensim.ArrayDouble()
    storageObject.getDataColumn(index_x,array_x)

    array_y=opensim.ArrayDouble()
    storageObject.getDataColumn(index_y,array_y)

    array_z=opensim.ArrayDouble()
    storageObject.getDataColumn(index_z,array_z)

    n= array_x.getSize()
    pointValues = np.zeros((n,3))
    for i in range(0,n):
        pointValues[i,0] = array_x.getitem(i)
        pointValues[i,1] = array_y.getitem(i)
        pointValues[i,2] = array_z.getitem(i)


    for i in range(0,n):
        pointValues[i,:] = np.dot(R_LAB_OSIM.T,pointValues[i,:])*1000.0


    return pointValues


def mot2pointValues(motFilename,labels,orientation =[1,1,1]):
    """convert  an element of an opensim mot file into a btk-point values
    (ie array(n,3) with n: the number of frames)

    Args:
        motFilename (str): full filename of the mot file
        label (str): label to convert
        orientation (list): scales to apply before returning final values. Default is [1,1,1]
    """
    storageObject = opensim.Storage(motFilename)

    index_x =storageObject.getStateIndex(labels[0])
    index_y =storageObject.getStateIndex(labels[1])
    index_z =storageObject.getStateIndex(labels[2])

    array_x=opensim.ArrayDouble()
    storageObject.getDataColumn(index_x,array_x)

    array_y=opensim.ArrayDouble()
    storageObject.getDataColumn(index_y,array_y)

    array_z=opensim.ArrayDouble()
    storageObject.getDataColumn(index_z,array_z)

    n= array_x.getSize()
    pointValues = np.zeros((n,3))
    for i in range(0,n):
        pointValues[i,0] = orientation[0]*array_x.getitem(i)
        pointValues[i,1] = orientation[1]*array_y.getitem(i)
        pointValues[i,2] = orientation[2]*array_z.getitem(i)


    return pointValues

def setGlobalTransormation_lab_osim(axis,forwardProgression):
    """return the rotation matrix expressing the opensim global coordinate system
    to the laboratory coordinate system

    Args:
        axis (str): defined progression axis  (ie X or Y ).
        forwardProgression (bool):  forward progression flag.
        (if True, the movement progression is along the positive value of
        your progression axis )

    """

    if axis =="X":
        if forwardProgression:
            R_LAB_OSIM=np.array([[1,0,0],[0,0,1],[0,-1,0]])
        else:
            R_LAB_OSIM=np.array([[-1,0,0],[0,0,1],[0,1,0]])

    elif axis =="Y":
        if forwardProgression:
            R_LAB_OSIM=np.array([[0,1,0],[0,0,1],[1,0,0]])
        else:
            R_LAB_OSIM=np.array([[0,-1,0],[0,0,1],[-1,0,0]])

    else:
        raise Exception("[pyCGM2] - Global Referential not configured yet")


    return R_LAB_OSIM


class opensimModel(object):


    def __init__(self,osimFile,cgmModel,):
        self.m_osimFile = osimFile


        self.m_model = opensim.Model(osimFile)
        self.m_cgmModel = cgmModel
        self.m_markers= list()


        self.m_model.initSystem()
        self.m_myState = self.m_model.initSystem()




    def addMarkerSet(self,markersetFile):
        markerSet= opensim.MarkerSet(markersetFile)
        self.m_model.updateMarkerSet(markerSet)

    def setOsimJoinCentres(self,R_OSIM_CGM, jointLabelInOsim, parentSegmentLabel,childSegmentLabel, nodeLabel, toMeter = 1000.0):

        locationInParent =  self.m_cgmModel.getSegment(parentSegmentLabel).anatomicalFrame.static.getNode_byLabel(nodeLabel).m_local # TODO : verifier que node exist
        locationInChild =  self.m_cgmModel.getSegment(childSegmentLabel).anatomicalFrame.static.getNode_byLabel(nodeLabel).m_local # TODO : verifier que node exist

        positionParent =   np.dot(R_OSIM_CGM[parentSegmentLabel],locationInParent)
        positionChild =   np.dot(R_OSIM_CGM[childSegmentLabel],locationInChild)

        osimJointInParent = self.m_model.getJointSet().get(jointLabelInOsim).get_frames(0).get_translation()
        osimJointInParent.set(0,positionParent[0]/toMeter)
        osimJointInParent.set(1,positionParent[1]/toMeter)
        osimJointInParent.set(2,positionParent[2]/toMeter)

        osimJointInChild = self.m_model.getJointSet().get(jointLabelInOsim).get_frames(1).get_translation()
        osimJointInChild.set(0,positionChild[0]/toMeter)
        osimJointInChild.set(1,positionChild[1]/toMeter)
        osimJointInChild.set(2,positionChild[2]/toMeter)

        LOGGER.logger.debug( "osim joint centres %s modified"%(jointLabelInOsim))



    def addmarkerFromModel(self, label, osimBodyName = "", modelSegmentLabel = "", rotation_osim_model=np.eye(3), toMeter=1000.0):

        if (osimBodyName != ""  and  modelSegmentLabel != "") :
            localPos = np.dot(rotation_osim_model,
                               self.m_cgmModel.getSegment(modelSegmentLabel).anatomicalFrame.static.getNode_byLabel(label).m_local) # TODO : verifier que node exist

            m = opensim.Marker()
            m.setOffset(opensim.Vec3(localPos[0]/toMeter,localPos[1]/toMeter,localPos[2]/toMeter))
            m.setName(label)
            m.setBodyName(osimBodyName)
            m.setFixed(False)

            self.m_markers.append(m)

        LOGGER.logger.debug( "marker (%s) added"%(label))


    def createMarkerSet(self):

        for i in range(0,len(self.m_markers)):
            self.m_model.getMarkerSet().set(i,self.m_markers[i]) # FIXME  : this line cause a memory issue : it would be better to use add_maker but i have an error with const double[3] when i specified the offset


    def updateMarkerInMarkerSet(self, label, modelSegmentLabel = "", rotation_osim_model=np.eye(3), toMeter=1000.0):

       #if label =="LTOE": pdb.set_trace()
       markers = opensim.ArrayStr()
       self.m_model.getMarkerSet().getMarkerNames(markers)

       index = markers.findIndex(label)
       if index != -1:
           LOGGER.logger.debug( "marker (%s) found"%(label))
           localPos = np.dot(rotation_osim_model,
                                  self.m_cgmModel.getSegment(modelSegmentLabel).anatomicalFrame.static.getNode_byLabel(label).m_local) # TODO : check node exist

           self.m_model.updMarkerSet().get(label).get_location().set(0,localPos[0]/toMeter)
           self.m_model.updMarkerSet().get(label).get_location().set(1,localPos[1]/toMeter)
           self.m_model.updMarkerSet().get(label).get_location().set(2,localPos[2]/toMeter)

       else:
           raise Exception ("[pyCGM2] marker (%s) is not within the markerset"%label)





class opensimKinematicFitting(object):

    def __init__(self,osimModel,ikToolFiles):
        self.m_model = osimModel

        ikTool =  opensim.InverseKinematicsTool(ikToolFiles)
        self.m_ikMarkers = list()

        self.m_ikTool = ikTool

        #opensim.InverseKinematicsTool()
        # pr= self.m_ikTool.getPropertyByName("report_marker_locations")
        # opensim.PropertyHelper().setValueBool(True,pr)

    # def setAccuracy(self,value):
        # pr= self.m_ikTool.getPropertyByName("accuracy")
        # opensim.PropertyHelper().setValueDouble(value,pr)
    #
    # def setResultsDirectory(self,path):
    #     self.m_ikTool.setResultsDir(path)
        # pr= self.m_ikTool.getPropertyByName("results_directory")
        # opensim.PropertyHelper().setValueString(path,pr)

    def addIkMarkerTask(self,label,weight=100):
        markerTask = opensim.IKMarkerTask()
        markerTask.setName(label)
        markerTask.setWeight(weight)
        self.m_ikMarkers.append(markerTask)

    def createIKTaskSet(self):
        for i in range(0,len(self.m_ikMarkers)):
            self.m_ikTool.getIKTaskSet().set(int(i),self.m_ikMarkers[i]) #FIXME memory issue


    def config(self, R_LAB_OSIM,acq, acqFileName):

        # steps 1 : motion data processing
        globalTransformationLabToOsim(acq,R_LAB_OSIM) # global Transformation lab to OSIM

        filenameNoExt = acqFileName[:-4]
        smartTrcExport(acq,filenameNoExt) # trc export

        # steps 2 : config ikTool
        self.m_ikTool.setModel(self.m_model)
        self.m_ikTool.setMarkerDataFileName( filenameNoExt.replace("\\","/")  +".trc")
        self.m_ikTool.setOutputMotionFileName(filenameNoExt.replace("\\","/")  +".mot")

        #  set times ( FIXME - I had surprise with set method, i prefer to handle the xmlnode directly)
        # prTime= self.m_ikTool.getPropertyByName("time_range")

        # opensim.PropertyHelper().appendValueDouble(0.0, prTime)
        # endTime = (acq.GetLastFrame() - acq.GetFirstFrame())/acq.GetPointFrequency()
        # opensim.PropertyHelper().appendValueDouble(endTime, prTime)

        # doesn t work
        # markerData = opensim.MarkerData(filenameNoExt+".trc")
        # self.m_ikTool.setStartTime(markerData.getStartFrameTime()) # time not frame !
        # self.m_ikTool.setEndTime(markerData.getLastFrameTime())


    def updateIKTask(self,label,weight):
        ts =self.m_ikTool.getIKTaskSet()
        index = ts.getIndex(label)
        if index !=-1 :
            if weight != 0:
                ts.get(label).setApply(True)
                ts.get(label).setWeight(weight)
            else:
                ts.get(label).setApply(False)
        else:
            raise Exception("[[pyCGM2]] the label (%s) doesn t exist "%(label))

    def run(self):

        # self.m_ikTool.printToXML(os.getcwd()+"\\IK_SETUP-pyCGM2.xml")
        self.m_ikTool.run()
