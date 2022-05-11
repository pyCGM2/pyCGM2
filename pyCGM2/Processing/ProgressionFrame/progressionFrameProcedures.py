# -*- coding: utf-8 -*-
import numpy as np

from pyCGM2.Tools import  btkTools
import pyCGM2; LOGGER = pyCGM2.LOGGER


class ProgressionFrameProcedure(object):
    def __init__(self):
        pass

class PointProgressionFrameProcedure(ProgressionFrameProcedure):
    """detect the progression from the trajectory of a single marker

    Args:
        marker (str,Optional[LHEE]): marker label

    """

    def __init__(self,marker="LHEE"):
        super(PointProgressionFrameProcedure, self).__init__()
        self.m_marker=marker

        self.__threshold = 800


    def compute(self,acq):

        if not btkTools.isPointExist(acq,self.m_marker):
            raise Exception( "[pyCGM2] : origin point doesnt exist")

        vff,vlf = btkTools.getFrameBoundaries(acq,[self.m_marker])
        ff = acq.GetFirstFrame()

        values = acq.GetPoint(self.m_marker).GetValues()[vff-ff:vlf-ff+1,0:3]

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

        LOGGER.logger.debug("Progression axis : %s"%(progressionAxis))
        LOGGER.logger.debug("forwardProgression : %s"%(forwardProgression))
        LOGGER.logger.debug("globalFrame : %s"%(globalFrame))

        return   progressionAxis,forwardProgression,globalFrame

class PelvisProgressionFrameProcedure(ProgressionFrameProcedure):
    """detect the progression from the trajectory of pelvic markers

    Args:
        marker (str,Optional[LASI]): marker label
        frontMarkers (list,Optional["LASI","RASI"]): anterior pelvic marker labels
        backMarkers (list,Optional["LPSI","RPSI"]): posterior pelvic markers labels

    """


    def __init__(self,marker="LASI",frontMarkers = ["LASI","RASI"], backMarkers =  ["LPSI","RPSI"]):
        super(PelvisProgressionFrameProcedure, self).__init__()

        self.m_marker=marker

        self.m_frontmarkers = frontMarkers
        self.m_backmarkers = backMarkers

        self.__threshold = 800


    def compute(self,acq):

        if not btkTools.isPointExist(acq,self.m_marker):
            raise Exception( "[pyCGM2] : marker %s doesn't exist"%(self.m_marker))

        # find valid frames and get the first one
        vff,vlf = btkTools.getFrameBoundaries(acq,[self.m_marker])
        ff = acq.GetFirstFrame()

        values = acq.GetPoint(self.m_marker).GetValues()[vff-ff:vlf-ff+1,:]
        MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
        absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

        ind = np.argmax(absMaxValues)

        if absMaxValues[ind] > self.__threshold:
            LOGGER.logger.debug("progression axis detected from displacement of the marker (%s)"%(self.m_marker))

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
            LOGGER.logger.debug("progression axis detected from pelvis longitudinal axis")

            for marker in self.m_frontmarkers+self.m_backmarkers:
                if not btkTools.isPointExist(acq,marker):
                    raise Exception( "[pyCGM2] : marker %s doesn't exist"%(marker))

            # find valid frames and get the first one
            vff,vlf = btkTools.getFrameBoundaries(acq,self.m_frontmarkers+self.m_backmarkers)
            ff = acq.GetFirstFrame()
            index = vff-ff


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



        LOGGER.logger.info("Progression axis : %s"%(progressionAxis))
        LOGGER.logger.info("forwardProgression : %s"%((forwardProgression)))
        LOGGER.logger.info("globalFrame : %s"%((globalFrame)))


        return   progressionAxis,forwardProgression,globalFrame


class ThoraxProgressionFrameProcedure(ProgressionFrameProcedure):
    """detect the progression from the trajectory of thoracic markers

    Args:
        marker (str,Optional[CLAV]): marker label
        frontMarkers (list,Optional["CLAV"]): anterior pelvic marker labels
        backMarkers (list,Optional["C7"]): posterior pelvic markers labels

    """


    def __init__(self,marker="CLAV",frontMarkers = ["CLAV"], backMarkers =  ["C7"]):
        super(ThoraxProgressionFrameProcedure, self).__init__()
        self.m_marker=marker

        self.m_frontmarkers = frontMarkers
        self.m_backmarkers = backMarkers

        self.__threshold = 800


    def compute(self,acq):

        if not btkTools.isPointExist(acq,self.m_marker):
            raise Exception( "[pyCGM2] : marker %s doesn't exist"%(self.m_marker))

        # find valid frames and get the first one
        vff,vlf = btkTools.getFrameBoundaries(acq,[self.m_marker])
        ff = acq.GetFirstFrame()
        values = acq.GetPoint(self.m_marker).GetValues()[vff-ff:vlf-ff+1,:]

        MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
        absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

        ind = np.argmax(absMaxValues)

        if absMaxValues[ind] > self.__threshold:
            LOGGER.logger.debug("progression axis detected from displacement of the marker (%s)"%(self.m_marker))
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
            LOGGER.logger.debug("progression axis detected from pelvis longitudinal axis")
            for marker in self.m_frontmarkers+self.m_backmarkers:
                if not btkTools.isPointExist(acq,marker):
                    raise Exception( "[pyCGM2] : marker %s doesn't exist"%(marker))

            # find valid frames and get the first one
            vff,vlf = btkTools.getFrameBoundaries(acq,self.m_frontmarkers+self.m_backmarkers)
            ff = acq.GetFirstFrame()
            index = vff-ff


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



        LOGGER.logger.info("Progression axis : %s"%(progressionAxis))
        LOGGER.logger.info("forwardProgression : %s"%((forwardProgression)))
        LOGGER.logger.info("globalFrame : %s"%((globalFrame)))


        return   progressionAxis,forwardProgression,globalFrame
