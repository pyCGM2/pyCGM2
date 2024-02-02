"""
This module provides procedures for detecting the progression frame in gait analysis 
using different anatomical markers or marker sets. These procedures are crucial for 
accurately determining the orientation and movement direction during gait analysis, 
which is essential for proper kinematic assessment.
"""
import numpy as np
import btk

from pyCGM2.Tools import  btkTools

import pyCGM2; LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional,Union,Any

class ProgressionFrameProcedure(object):
    """
    Base class for progression frame detection procedures.

    This class provides a common interface for different procedures to determine the progression 
    frame in gait analysis. It is intended to be subclassed by specific implementations using 
    various anatomical markers or sets.
    """
    def __init__(self):
        pass

class PointProgressionFrameProcedure(ProgressionFrameProcedure):
    """
    Detects the progression frame using the trajectory of a single marker.

    This procedure analyzes the movement of a single specified marker to determine the 
    progression axis and direction during gait analysis.

    Args:
        marker (str, optional): The marker label used for detection. Defaults to "LHEE".
    """

    def __init__(self,marker:str="LHEE"):
        super(PointProgressionFrameProcedure, self).__init__()
        self.m_marker=marker

        self.__threshold = 800


    def compute(self,acq:btk.btkAcquisition):
        """
        Computes the progression frame based on the specified marker's trajectory.

        Args:
            acq (btk.btkAcquisition): The acquisition containing gait data.

        Returns:
            Tuple[str, bool, str]: A tuple containing the progression axis, forward progression flag, and global frame.
        """

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
    """Detects progression frame using the trajectory of pelvic markers.

    This procedure utilizes the movement of anterior and posterior pelvic markers to 
    ascertain the progression axis and direction.

    Args:
        marker (str): Primary marker label for detection. Defaults to "LASI".
        frontMarkers (List[str]): List of anterior pelvic marker labels. Defaults to ["LASI", "RASI"].
        backMarkers (List[str]): List of posterior pelvic markers labels. Defaults to ["LPSI", "RPSI"].
    """


    def __init__(self,marker:str="LASI",frontMarkers:List[str] = ["LASI","RASI"], backMarkers:List[str] =  ["LPSI","RPSI"]):
        super(PelvisProgressionFrameProcedure, self).__init__()

        self.m_marker=marker

        self.m_frontmarkers = frontMarkers
        self.m_backmarkers = backMarkers

        self.__threshold = 800


    def compute(self,acq:btk.btkAcquisition):
        """
        Computes the progression frame based on the pelvic markers' trajectory.

        Args:
            acq (btk.btkAcquisition): The acquisition containing gait data.

        Returns:
            Tuple[str, bool, str]: A tuple containing the progression axis, forward progression flag, and global frame.
        """

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
        LOGGER.logger.debug("globalFrame : %s"%((globalFrame)))


        return   progressionAxis,forwardProgression,globalFrame


class ThoraxProgressionFrameProcedure(ProgressionFrameProcedure):
    """Detects progression frame using the trajectory of thoracic markers.

    This procedure analyzes the movement of anterior and posterior thoracic markers to determine 
    the progression axis and direction during gait analysis.

    Args:
        marker (str): Primary marker label for detection. Defaults to "CLAV".
        frontMarkers (List[str]): List of anterior pelvic marker labels. Defaults to ["CLAV"].
        backMarkers (List[str]): List of posterior pelvic markers labels. Defaults to ["C7"].
    """


    def __init__(self,marker:str="CLAV",frontMarkers:List[str] = ["CLAV"], backMarkers:List[str] =  ["C7"]):
        super(ThoraxProgressionFrameProcedure, self).__init__()
        self.m_marker=marker

        self.m_frontmarkers = frontMarkers
        self.m_backmarkers = backMarkers

        self.__threshold = 800


    def compute(self,acq:btk.btkAcquisition):
        """
        Computes the progression frame based on the thorax markers' trajectory.

        Args:
            acq (btk.btkAcquisition): The acquisition containing gait data.

        Returns:
            Tuple[str, bool, str]: A tuple containing the progression axis, forward progression flag, and global frame.
        """

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
        LOGGER.logger.debug("globalFrame : %s"%((globalFrame)))


        return   progressionAxis,forwardProgression,globalFrame
