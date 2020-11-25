# -*- coding: utf-8 -*-
import numpy as np

from pyCGM2.Tools import  btkTools
import logging

class ProgressionFrameFilter(object):
    """
    """

    def __init__(self, acq,progressionProcedure):

        self.m_procedure = progressionProcedure
        self.m_acq = acq

        self.outputs = {"progressionAxis": None, "forwardProgression": None, "globalFrame": None}


    def compute(self):
        progressionAxis,forwardProgression,globalFrame= self.m_procedure.compute(self.m_acq)

        self.outputs["progressionAxis"] = progressionAxis
        self.outputs["forwardProgression"] = forwardProgression
        self.outputs["globalFrame"] = globalFrame

class PointProgressionFrameProcedure(object):

    def __init__(self,marker="LHEE"):
        self.m_marker=marker

        self.__threshold = 800


    def compute(self,acq):

        if not btkTools.isPointExist(acq,self.m_marker):
            raise Exception( "[pyCGM2] : origin point doesnt exist")

        f,ff,lf = btkTools.findValidFrames(acq,[self.m_marker])

        values = acq.GetPoint(self.m_marker).GetValues()[ff:lf,0:3]

        MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
        absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

        ind = np.argmax(absMaxValues)
        diff = MaxValues[ind]

        if ind ==0 :
            progressionAxis = "X"
            lateralAxis = "Y"
        else:
            progressionAxis = "Y"
            lateralAxis = "X"

        forwardProgression = True if diff>0 else False

        globalFrame = (progressionAxis+lateralAxis+"Z")

        logging.debug("Progression axis : %s"%(progressionAxis))
        logging.debug("forwardProgression : %s"%(forwardProgression))
        logging.debug("globalFrame : %s"%(globalFrame))

        return   progressionAxis,forwardProgression,globalFrame

class PelvisProgressionFrameProcedure(object):


    def __init__(self,marker="LASI",frontMarkers = ["LASI","RASI"], backMarkers =  ["LPSI","RPSI"]):

        self.m_marker=marker

        self.m_frontmarkers = frontMarkers
        self.m_backmarkers = backMarkers

        self.__threshold = 800


    def compute(self,acq):

        if not btkTools.isPointExist(acq,self.m_marker):
            raise Exception( "[pyCGM2] : marker %s doesn't exist"%(self.m_marker))

        # find valid frames and get the first one
        flag,vff,vlf = btkTools.findValidFrames(acq,[self.m_marker])

        values = acq.GetPoint(self.m_marker).GetValues()[vff:vlf,:]

        MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
        absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

        ind = np.argmax(absMaxValues)

        if absMaxValues[ind] > self.__threshold:
            logging.debug("progression axis detected from displacement of the marker (%s)"%(self.m_marker))

            diff = MaxValues[ind]

            if ind ==0 :
                progressionAxis = "X"
                lateralAxis = "Y"
            else:
                progressionAxis = "Y"
                lateralAxis = "X"

            forwardProgression = True if diff>0 else False

            globalFrame = (progressionAxis+lateralAxis+"Z")


        else:
            logging.debug("progression axis detected from pelvis longitudinal axis")

            for marker in self.m_frontmarkers+self.m_backmarkers:
                if not btkTools.isPointExist(acq,marker):
                    raise Exception( "[pyCGM2] : marker %s doesn't exist"%(marker))

            # find valid frames and get the first one
            flag,vff,vlf = btkTools.findValidFrames(acq,self.m_frontmarkers+self.m_backmarkers)
            index = vff


            # barycentres
            values = np.zeros((acq.GetPointFrameNumber(),3))
            count = 0
            for marker in self.m_frontmarkers:
                values = values + acq.GetPoint(marker).GetValues()
                count +=1
            frontValues = values / count


            values = np.zeros((acq.GetPointFrameNumber(),3))
            count = 0
            for marker in self.m_backmarkers:
                values = values + acq.GetPoint(marker).GetValues()
                count +=1
            backValues = values / count


            # axes
            back = backValues[index,:]
            front = frontValues[index,:]
            front[2] = back[2] # allow to avoid detecting Z axis if pelvs anterior axis point dowwnward

            z=np.array([0,0,1])

            a1=(front-back)
            a1=a1/np.linalg.norm(a1)

            a2=np.cross(a1,z)
            a2=a2/np.linalg.norm(a2)

            globalAxes = {"X" : np.array([1,0,0]), "Y" : np.array([0,1,0]), "Z" : np.array([0,0,1])}

            # longitudinal axis
            tmp=[]
            for axis in globalAxes.keys():
                res = np.dot(a1,globalAxes[axis])
                tmp.append(res)
            maxIndex = np.argmax(np.abs(tmp))
            progressionAxis =  list(globalAxes.keys())[maxIndex]
            forwardProgression = True if tmp[maxIndex]>0 else False
            # lateral axis
            tmp=[]
            for axis in globalAxes.keys():
                res = np.dot(a2,globalAxes[axis])
                tmp.append(res)
            maxIndex = np.argmax(np.abs(tmp))
            lateralAxis =  list(globalAxes.keys())[maxIndex]

            # global frame
            if "X" not in (progressionAxis+lateralAxis):
                globalFrame = (progressionAxis+lateralAxis+"X")
            if "Y" not in (progressionAxis+lateralAxis):
                globalFrame = (progressionAxis+lateralAxis+"Y")
            if "Z" not in (progressionAxis+lateralAxis):
                globalFrame = (progressionAxis+lateralAxis+"Z")



        logging.info("Progression axis : %s"%(progressionAxis))
        logging.info("forwardProgression : %s"%((forwardProgression)))
        logging.info("globalFrame : %s"%((globalFrame)))


        return   progressionAxis,forwardProgression,globalFrame


class ThoraxProgressionFrameProcedure(object):


    def __init__(self,marker="CLAV",frontMarkers = ["CLAV"], backMarkers =  ["C7"]):

        self.m_marker=marker

        self.m_frontmarkers = frontMarkers
        self.m_backmarkers = backMarkers

        self.__threshold = 800


    def compute(self,acq):

        if not btkTools.isPointExist(acq,self.m_marker):
            raise Exception( "[pyCGM2] : marker %s doesn't exist"%(m_marker))

        # find valid frames and get the first one
        flag,vff,vlf = btkTools.findValidFrames(acq,[self.m_marker])

        values = acq.GetPoint(self.m_marker).GetValues()[vff:vlf,:]

        MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
        absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

        ind = np.argmax(absMaxValues)

        if absMaxValues[ind] > self.__threshold:
            logging.debug("progression axis detected from displacement of the marker (%s)"%(self.m_marker))
            diff = MaxValues[ind]

            if ind ==0 :
                progressionAxis = "X"
                lateralAxis = "Y"
            else:
                progressionAxis = "Y"
                lateralAxis = "X"

            forwardProgression = True if diff>0 else False

            globalFrame = (progressionAxis+lateralAxis+"Z")


        else:
            logging.debug("progression axis detected from pelvis longitudinal axis")
            for marker in self.m_frontmarkers+self.m_backmarkers:
                if not btkTools.isPointExist(acq,marker):
                    raise Exception( "[pyCGM2] : marker %s doesn't exist"%(marker))

            # find valid frames and get the first one
            flag,vff,vlf = btkTools.findValidFrames(acq,self.m_frontmarkers+self.m_backmarkers)
            index = vff


            # barycentres
            values = np.zeros((acq.GetPointFrameNumber(),3))
            count = 0
            for marker in self.m_frontmarkers:
                values = values + acq.GetPoint(marker).GetValues()
                count +=1
            frontValues = values / count


            values = np.zeros((acq.GetPointFrameNumber(),3))
            count = 0
            for marker in self.m_backmarkers:
                values = values + acq.GetPoint(marker).GetValues()
                count +=1
            backValues = values / count


            # axes
            back = backValues[index,:]
            front = frontValues[index,:]
            z=np.array([0,0,1])

            a1=(front-back)
            a1=a1/np.linalg.norm(a1)

            a2=np.cross(a1,z)
            a2=a2/np.linalg.norm(a2)

            globalAxes = {"X" : np.array([1,0,0]), "Y" : np.array([0,1,0]), "Z" : np.array([0,0,1])}

            # longitudinal axis
            tmp=[]
            for axis in globalAxes.keys():
                res = np.dot(a1,globalAxes[axis])
                tmp.append(res)
            maxIndex = np.argmax(np.abs(tmp))
            progressionAxis =  list(globalAxes.keys())[maxIndex]
            forwardProgression = True if tmp[maxIndex]>0 else False

            # lateral axis
            tmp=[]
            for axis in globalAxes.keys():
                res = np.dot(a2,globalAxes[axis])
                tmp.append(res)
            maxIndex = np.argmax(np.abs(tmp))
            lateralAxis =  list(globalAxes.keys())[maxIndex]

            # global frame
            if "X" not in (progressionAxis+lateralAxis):
                globalFrame = (progressionAxis+lateralAxis+"X")
            if "Y" not in (progressionAxis+lateralAxis):
                globalFrame = (progressionAxis+lateralAxis+"Y")
            if "Z" not in (progressionAxis+lateralAxis):
                globalFrame = (progressionAxis+lateralAxis+"Z")



        logging.info("Progression axis : %s"%(progressionAxis))
        logging.info("forwardProgression : %s"%((forwardProgression)))
        logging.info("globalFrame : %s"%((globalFrame)))


        return   progressionAxis,forwardProgression,globalFrame
