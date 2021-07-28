# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from bs4 import BeautifulSoup

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Processing import progressionFrame

from pyCGM2.Model.Opensim import opensimInterfaceFilters

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim



class opensimInterfaceHighLevelInverseKinematicsProcedure(object):
    def __init__(self,DATA_PATH, scaleOsim,ikToolsTemplate):

        self.m_DATA_PATH = DATA_PATH

        self.m_osimModel = scaleOsim


        self.m_ikToolFile = DATA_PATH + "/IKTool-setup.xml" #"/IKTool-setup.xml"
        self.xml = opensimInterfaceFilters.opensimXmlInterface(ikToolsTemplate,self.m_ikToolFile)


    def preProcess(self, acq, m_dynamicFileNoExt):
        self.m_dynamicFileNoExt =m_dynamicFileNoExt

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        forwardProgression = pff.outputs["forwardProgression"]

        self.m_acq0 = acq
        self.m_acqMotion_forIK = btk.btkAcquisition.Clone(acq)

        R_LAB_OSIM = opensimTools.setGlobalTransormation_lab_osim(progressionAxis,forwardProgression)
        opensimTools.globalTransformationLabToOsim(self.m_acqMotion_forIK,R_LAB_OSIM)
        opensimTools.smartTrcExport(self.m_acqMotion_forIK,self.m_DATA_PATH +  self.m_dynamicFileNoExt[:-4])

        self.m_R_LAB_OSIM = R_LAB_OSIM
        self.m_markerFile = self.m_DATA_PATH +  self.m_dynamicFileNoExt[:-4]+".trc"


    def setAccuracy(self,value):
        self.xml.set_one("accuracy",str(value))

    def setWeights(self,weights_dict):
        self.m_weights = weights_dict


    def setTimeRange(self,beginFrame=None,lastFrame=None):

        ff = self.m_acqMotion_forIK.GetFirstFrame()
        freq = self.m_acqMotion_forIK.GetPointFrequency()
        beginTime = 0.0 if beginFrame is None else (beginFrame-ff)/freq
        endTime = (self.m_acqMotion_forIK.GetLastFrame() - ff)/freq  if lastFrame is  None else (lastFrame-ff)/freq
        text = str(beginTime) + " " + str(endTime)
        self.xml.set_one("time_range",text)

        self.m_frameRange = [int((beginTime*freq)+ff),int((endTime*freq)+ff)]

    def finalize(self):
        acqMotionFinal = btk.btkAcquisition.Clone(self.m_acq0)
        storageObject = opensim.Storage(self.m_DATA_PATH + "_ik_model_marker_locations.sto")
        for marker in self.m_weights.keys():
            if self.m_weights[marker] != 0:
                values =opensimTools.sto2pointValues(storageObject,marker,self.m_R_LAB_OSIM)
                btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" )
                modelled = acqMotionFinal.GetPoint(marker).GetValues()
                ff = acqMotionFinal.GetFirstFrame()
                modelled[self.m_frameRange[0]-ff:self.m_frameRange[1]-ff+1,:] = values
                btkTools.smartAppendPoint(acqMotionFinal,marker, modelled, desc= "kinematic fitting" ) # new acq with marker overwrited

        self.m_acqMotionFinal = acqMotionFinal


    def run(self):

        if os.path.isfile(self.m_DATA_PATH +"_ik_model_marker_locations.sto"):
            os.remove(self.m_DATA_PATH +"_ik_model_marker_locations.sto")


        markerFileShort = self.m_markerFile[self.m_markerFile.rfind("\\")+1:]
        self.xml.set_one("marker_file", markerFileShort)
        self.xml.set_one("output_motion_file", markerFileShort[:-4]+".mot")

        for marker in self.m_weights.keys():
            self.xml.set_inList_fromAttr("IKMarkerTask","weight","name",marker,str(self.m_weights[marker]))


        self.xml.update()

        ikTool = opensim.InverseKinematicsTool(self.m_ikToolFile)
        ikTool.setModel(self.m_osimModel)
        ikTool.run()

        self.finalize()

        for it in self.m_osimModel.getMuscleList():
            import ipdb; ipdb.set_trace()


# NOT WORK : need opensim4.2 and bug fix of property
class opensimInterfaceLowLevelInverseKinematicsProcedure(object):
    def __init__(self,DATA_PATH, scaleOsim,ikToolsTemplate):
        pass

# class opensimInterfaceLowLevelInverseKinematicsProcedure(object):
#
#
#     def __init__(self,DATA_PATH, scaleOsim,ikTools, acq, gaitFilename ):
#
#         self.m_DATA_PATH = DATA_PATH
#
#         self.m_osimModel = scaleOsim
#
#         self.m_ikscale_tool = opensim.InverseKinematicsTool(ikTools)
#
#         self.m_acqMotion = acq
#         self.m_dynamicFile = gaitFilename
#
#     def setWeights(self,weights_dict):
#         self.m_weights = weights_dict
#
#
#     def setTimeRange(self,beginFrame=None,lastFrame=None):
#         ff = self.m_acqMotion.GetFirstFrame()
#         freq = self.m_acqMotion.GetPointFrequency()
#         beginTime = 0.0 if beginFrame is None else (beginFrame-ff)/freq
#         endTime = (self.m_acqMotion.GetLastFrame() - ff)/freq  if lastFrame is  None else (lastFrame-ff)/freq
#
#         self.m_ikscale_tool.setStartTime(beginTime)
#         self.m_ikscale_tool.setEndTime(endTime - 1e-2)
#         # self.m_ikscale_tool.printToXML("verif")
#
#         m = opensim.MarkerData((self.m_DATA_PATH+ self.m_dynamicFile[:-4]  +".trc").replace("\\","/"))
#         initial_time = m.getStartFrameTime()
#         final_time = m.getLastFrameTime()
#         self.m_ikscale_tool.setStartTime(initial_time)
#         self.m_ikscale_tool.setEndTime(final_time)
#         self.m_ikscale_tool.printToXML("verif")
#
# # end = m.getLastFrameTime() - 1e-2
#
#         import ipdb; ipdb.set_trace()
#         self.m_frameRange = [int((beginTime*freq)+ff),int((endTime*freq)+ff)]
#
#
#     def updateIKTask(self,label,weight):
#         ts =self.m_ikscale_tool.getIKTaskSet()
#         index = ts.getIndex(label)
#         if index !=-1 :
#             if weight != 0:
#                 ts.get(label).setApply(True)
#                 ts.get(label).setWeight(weight)
#             else:
#                 ts.get(label).setApply(False)
#         else:
#             raise Exception("[[pyCGM2]] the label (%s) doesn t exist "%(label))
#
#
#     def run(self):
#
#         # --- configuration and run IK
#         if os.path.isfile(self.m_DATA_PATH +"_ik_model_marker_locations.sto"):
#             os.remove(self.m_DATA_PATH +"_ik_model_marker_locations.sto")
#
#
#         self.m_ikscale_tool.setModel(self.m_osimModel)
#         self.m_ikscale_tool.setMarkerDataFileName( (self.m_DATA_PATH+ self.m_dynamicFile[:-4]  +".trc").replace("\\","/"))
#         self.m_ikscale_tool.setOutputMotionFileName((self.m_DATA_PATH+ self.m_dynamicFile[:-4]  +".mot").replace("\\","/"))
#         self.m_ikscale_tool.setResultsDir(self.m_DATA_PATH.replace("\\","/")[:-1])
#
#
#
#         pfp = progressionFrame.PelvisProgressionFrameProcedure()
#         pff = progressionFrame.ProgressionFrameFilter(self.m_acqMotion,pfp)
#         pff.compute()
#         progressionAxis = pff.outputs["progressionAxis"]
#         forwardProgression = pff.outputs["forwardProgression"]
#
#
#         acqMotion_forIK = btk.btkAcquisition.Clone(self.m_acqMotion)
#         R_LAB_OSIM = opensimTools.setGlobalTransormation_lab_osim(progressionAxis,forwardProgression)
#         opensimTools.globalTransformationLabToOsim(acqMotion_forIK,R_LAB_OSIM)
#         opensimTools.smartTrcExport(acqMotion_forIK,self.m_DATA_PATH +  self.m_dynamicFile[:-4])
#
#         self.setTimeRange()
#
#
#         for marker in self.m_weights.keys():
#             self.updateIKTask(marker,self.m_weights[marker])
#
#         self.m_ikscale_tool.printToXML(self.m_DATA_PATH+"verifIKTools.xml")
#         self.m_ikscale_tool.run()
#
#         acqMotionFinal = btk.btkAcquisition.Clone(self.m_acqMotion)
#         storageObject = opensim.Storage(self.m_DATA_PATH + "_ik_model_marker_locations.sto")
#         for marker in self.m_weights.keys():
#             if self.m_weights[marker] != 0:
#                 values =opensimTools.sto2pointValues(storageObject,marker,R_LAB_OSIM)
#                 btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" )
#
#                 modelled = acqMotionFinal.GetPoint(marker).GetValues()
#                 ff = acqMotionFinal.GetFirstFrame()
#                 modelled[self.m_frameRange[0]-ff:self.m_frameRange[1]-ff+1,:] = values
#                 btkTools.smartAppendPoint(acqMotionFinal,marker, modelled, desc= "kinematic fitting" ) # new acq with marker overwrited
#
#         btkTools.smartWriter(acqMotionFinal,self.m_DATA_PATH+"verifGait1.c3d")
#         return acqMotionFinal
