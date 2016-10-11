# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:00:12 2016

@author: aaa34169
"""
import opensim 
import numpy as np
import btk
import pdb

def globalTransformationLabToOsim(acq,R_LAB_OSIM):

    points = acq.GetPoints()
    for it in btk.Iterate(points):
        if it.GetType() == btk.btkPoint.Marker:
            print "marker : " + str(it.GetLabel()) 
            values = np.zeros(it.GetValues().shape)            
            for i in range(0,it.GetValues().shape[0]):            
                values[i,:] = np.dot(R_LAB_OSIM,it.GetValues()[i,:])
            it.SetValues(values)
             
def smartTrcExport(acq,filenameNoExt):
    writerDyn = btk.btkAcquisitionFileWriter()
    writerDyn.SetInput(acq)
    writerDyn.SetFilename(str(filenameNoExt + ".trc"))
    writerDyn.Update() 



def sto2pointValues(stoFilename,label,R_LAB_OSIM):
    storageObject = opensim.Storage(stoFilename)
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
    print axis
    print "forward"
    print forwardProgression
    """ Todo : incomplet, il faut traiter tous les cas """
    if axis =="X":
        if forwardProgression:
            R_LAB_OSIM=np.array([[1,0,0],[0,0,1],[0,-1,0]])
        else:
            R_LAB_OSIM=np.array([[-1,0,0],[0,0,1],[0,1,0]])

    return R_LAB_OSIM


class opensimModel(object):
    

    def __init__(self,osimFile,cgmModel):
        """Constructor 
        """
        self.m_osimFile = osimFile
        
        
        self.m_model = opensim.Model(str(osimFile))
        self.m_cgmModel = cgmModel
        self.m_markers= list()


        self.m_model.initSystem()
        self.m_myState = self.m_model.initSystem()

    def setOsimJoinCentres(self,R_OSIM_CGM, jointLabelInOsim, parentSegmentLabel,childSegmentLabel, nodeLabel, toMeter = 1000.0 , verbose = False):

        locationInParent =  self.m_cgmModel.getSegment(parentSegmentLabel).anatomicalFrame.static.getNode_byLabel(nodeLabel).m_local # TODO : verifier que node exist
        locationInChild =  self.m_cgmModel.getSegment(childSegmentLabel).anatomicalFrame.static.getNode_byLabel(nodeLabel).m_local # TODO : verifier que node exist
            
        positionParent =   np.dot(R_OSIM_CGM[parentSegmentLabel],locationInParent)
        positionChild =   np.dot(R_OSIM_CGM[childSegmentLabel],locationInChild)
        
        osimJointInParent = self.m_model.getJointSet().get(jointLabelInOsim).get_location_in_parent()
        osimJointInParent.set(0,positionParent[0]/toMeter)
        osimJointInParent.set(1,positionParent[1]/toMeter)
        osimJointInParent.set(2,positionParent[2]/toMeter)
        
        osimJointInParent = self.m_model.getJointSet().get(jointLabelInOsim).get_location()
        osimJointInParent.set(0,positionChild[0]/toMeter)
        osimJointInParent.set(1,positionChild[1]/toMeter)
        osimJointInParent.set(2,positionChild[2]/toMeter)
        
        if verbose:  print " osim joint centres %s modified" %(jointLabelInOsim)


#    def setOsimJoinCentres(self,R_OSIM_CGM, localLHJC ,localRHJC ,localLKJC ,localRKJC ,localLAJC ,localRAJC, virtualForeFoot):
#        
#        LHJC =   np.dot(R_OSIM_CGM["Pelvis"],localLHJC)
#        RHJC =   np.dot(R_OSIM_CGM["Pelvis"],localRHJC)
#        
#        LKJC =   np.dot(R_OSIM_CGM["Left Thigh"],localLKJC)
#        RKJC =   np.dot(R_OSIM_CGM["Right Thigh"],localRKJC)
#        
#        LAJC =   np.dot(R_OSIM_CGM["Left Shank"],localLAJC)
#        RAJC =   np.dot(R_OSIM_CGM["Right Shank"],localRAJC)
#        
#
#        RCUN =   np.dot(R_OSIM_CGM["Right Hindfoot"],virtualForeFoot)
#
#
#        lhjc = self.m_model.getJointSet().get("hip_l").get_location_in_parent()
#        lhjc.set(0,LHJC[0]/1000.0)
#        lhjc.set(1,LHJC[1]/1000.0)
#        lhjc.set(2,LHJC[2]/1000.0)
#    
#        rhjc = self.m_model.getJointSet().get("hip_r").get_location_in_parent()
#        rhjc.set(0,RHJC[0]/1000.0) 
#        rhjc.set(1,RHJC[1]/1000.0)
#        rhjc.set(2,RHJC[2]/1000.0)
#    
#        # modif KJC
#        lkjc = self.m_model.getJointSet().get("knee_l").get_location_in_parent()
#        lkjc.set(0,LKJC[0]/1000.0)
#        lkjc.set(1,LKJC[1]/1000.0)
#        lkjc.set(2,LKJC[2]/1000.0)
#    
#        rkjc = self.m_model.getJointSet().get("knee_r").get_location_in_parent()
#        rkjc.set(0,RKJC[0]/1000.0)
#        rkjc.set(1,RKJC[1]/1000.0)
#        rkjc.set(2,RKJC[2]/1000.0)
#    
#        # modif AJC
#        lajc = self.m_model.getJointSet().get("ankle_l").get_location_in_parent()
#        lajc.set(0,LAJC[0]/1000.0)
#        lajc.set(1,LAJC[1]/1000.0)
#        lajc.set(2,LAJC[2]/1000.0)
#    
#        rajc = self.m_model.getJointSet().get("ankle_r").get_location_in_parent()
#        rajc.set(0,RAJC[0]/1000.0) 
#        rajc.set(1,RAJC[1]/1000.0)
#        rajc.set(2,RAJC[2]/1000.0)
#        
#
#
#        rvff = self.m_model.getJointSet().get("mtp_r").get_location_in_parent()
#        rvff.set(0,RCUN[0]/1000.0) 
#        rvff.set(1,RCUN[1]/1000.0)
#        rvff.set(2,RCUN[2]/1000.0)
        
        

    def addmarkerFromModel(self, label, osimBodyName = "", modelSegmentLabel = "", rotation_osim_model=np.eye(3), toMeter=1000.0,verbose = False):
        
        if (osimBodyName != ""  and  modelSegmentLabel != "") :  
            localPos = np.dot(rotation_osim_model,
                               self.m_cgmModel.getSegment(modelSegmentLabel).anatomicalFrame.static.getNode_byLabel(label).m_local) # TODO : verifier que node exist
             
            m = opensim.Marker()
            m.setOffset(opensim.Vec3(localPos[0]/toMeter,localPos[1]/toMeter,localPos[2]/toMeter))
            m.setName(label)
            m.setBodyName(osimBodyName)
            m.setFixed(False)
             
            self.m_markers.append(m)
             
        if verbose : print "marker (" + label + ") added"              
             
         
    def createMarkerSet(self):
        
        for i in range(0,len(self.m_markers)):
            self.m_model.getMarkerSet().set(i,self.m_markers[i]) # FIXME  : this line cause a memory issue : it would be better to use add_maker but i have an error with const double[3] when i specified the offset 
            
        
    def updateMarkerInMarkerSet(self, label, modelSegmentLabel = "", rotation_osim_model=np.eye(3), toMeter=1000.0,verbose = False):
       
       
       markers = opensim.ArrayStr()
       self.m_model.getMarkerSet().getMarkerNames(markers)

       index = markers.findIndex(label)
       if index != -1:
           if verbose : print " le marker (%s) trouve " %(label)
           localPos = np.dot(rotation_osim_model,
                                  self.m_cgmModel.getSegment(modelSegmentLabel).anatomicalFrame.static.getNode_byLabel(label).m_local) # TODO : verifier que node exist
           
           self.m_model.updMarkerSet().get(label).getOffset().set(0,localPos[0]/toMeter)
           self.m_model.updMarkerSet().get(label).getOffset().set(1,localPos[1]/toMeter)
           self.m_model.updMarkerSet().get(label).getOffset().set(2,localPos[2]/toMeter)
           
       else:          
           if verbose : print " le marker (%s) n est pas dans le markerset " %(label)
                    
        
        
        
class opensimKinematicFitting(object):

    #def __init__(self,osimModel,ikTool):    
    def __init__(self,osimModel,ikToolFiles):
        """Constructor 
        """
        self.m_model = osimModel

        ikTool =  opensim.InverseKinematicsTool(ikToolFiles)         
        self.m_ikMarkers = list()

        self.m_ikTool = ikTool#opensim.InverseKinematicsTool()
        pr= self.m_ikTool.getPropertyByName("report_marker_locations")
        opensim.PropertyHelper().setValueBool(True,pr)
        
        self.setAccuracy(0.00000001)

    def setAccuracy(self,value):
        pr= self.m_ikTool.getPropertyByName("accuracy")
        opensim.PropertyHelper().setValueDouble(value,pr)        
        
        

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
        self.m_ikTool.setMarkerDataFileName(filenameNoExt+".trc")
        self.m_ikTool.setOutputMotionFileName(filenameNoExt+".mot")
 
        markerData = opensim.MarkerData(filenameNoExt+".trc")
        self.m_ikTool.setStartTime(markerData.getStartFrameTime()) # time not frame !
        self.m_ikTool.setEndTime(markerData.getLastFrameTime())     

    def updateIKTask(self,label,weight):
        ts =self.m_ikTool.getIKTaskSet()
        index = ts.getIndex(label)
        if index !=-1 : 
            ts.get(label).setWeight(weight)
        else:
            raise Exception("[pycga-osim] the label (%s) doesn t exist ")

    def run(self):
        
        self.m_ikTool.run()
        


        
        