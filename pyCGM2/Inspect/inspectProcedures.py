import logging
import pyCGM2
from pyCGM2 import btk
from pyCGM2.Tools import btkTools
from pyCGM2.Math import derivation,geometry
from pyCGM2.Signal import detect_peaks
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from pyCGM2.Model import model, modelDecorator, frame, motion
from pyCGM2.Model.CGM2 import cgm, cgm2

from pyCGM2.Utils import utils


class GaitEventQualityProcedure(object):
    def __init__(self,acq):
        self.acq = acq

    def check(self):

        events = btkTools.sortedEvents(self.acq)

        if events != []:

            events_L = list()
            events_R = list()
            for ev in events:
                if ev.GetContext() == "Left":
                    events_L.append(ev)
                if ev.GetContext() == "Right":
                    events_R.append(ev)

            if events_L!=[]:
                init = events_L[0].GetLabel()
                if len(events_L)>1:
                    for i in range(1,len(events_L)):
                        label = events_L[i].GetLabel()
                        if label == init:
                            logging.error("Wrong Left Event - two consecutive (%s) detected at frane (%i)"%(str(label),events_L[i].GetFrame()) )
                        init = label
                else:
                    logging.warning("Only one left events")

            if events_R!=[]:
                init = events_R[0].GetLabel()
                if len(events_R)>1:
                    for i in range(1,len(events_R)):
                        label = events_R[i].GetLabel()
                        if label == init:
                            logging.error("Wrong Right Event - two consecutive (%s) detected at frane (%i)"%(str(label),events_R[i].GetFrame()) )
                        init = label
                else:
                    logging.warning("Only one right events ")




class AnthropometricDataQualityProcedure(object):
    def __init__(self,mp):
        self.mp = mp

    def check(self):
        """
        TODO :
        - use relation between variable ( width/height)
        - use marker measurement
        """

        if self.mp["RightLegLength"] < 500: logging.warning("Right Leg Lenth < 500 mm")
        if self.mp["LeftLegLength"] < 500: logging.warning("Left Leg Lenth < 500 mm")
        if self.mp["RightKneeWidth"] < self.mp["RightAnkleWidth"]: logging.error("Right ankle width > knee width ")
        if self.mp["LeftKneeWidth"] < self.mp["LeftAnkleWidth"]: logging.error("Right ankle width > knee width ")
        if self.mp["RightKneeWidth"] > self.mp["RightLegLength"]: logging.error("Right knee width > leg length ")
        if self.mp["LeftKneeWidth"] > self.mp["LeftLegLength"]: logging.error("Left knee width > leg length ")


        if not utils.isInRange(self.mp["RightKneeWidth"],
            self.mp["LeftKneeWidth"]-0.3*self.mp["LeftKneeWidth"],
            self.mp["LeftKneeWidth"]+0.3*self.mp["LeftKneeWidth"]):
             logging.warning("Knee widths differed by more than 30%")

        if not utils.isInRange(self.mp["RightAnkleWidth"],
            self.mp["LeftAnkleWidth"]-0.3*self.mp["LeftAnkleWidth"],
            self.mp["LeftAnkleWidth"]+0.3*self.mp["LeftAnkleWidth"]):
             logging.warning("Ankle widths differed by more than 30%")


        if not utils.isInRange(self.mp["RightLegLength"],
            self.mp["LeftLegLength"]-0.3*self.mp["LeftLegLength"],
            self.mp["LeftLegLength"]+0.3*self.mp["LeftLegLength"]):
             logging.warning("Leg lengths differed by more than 30%")



class GapQualityProcedure(object):
    def __init__(self,acq,markers=None):
        self.acq = acq

        self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)

    def check(self):

        frameNumber = self.acq.GetPointFrameNumber()

        for marker in self.markers:
            gapCount = list()
            previousValue = 0
            count =0
            for i in range(0,frameNumber):
                val = self.acq.GetPoint(marker).GetResidual(i)
                if val <0 : count+=1
                if previousValue<0  and  val==0.0:
                    gapCount.append(count)
                    count = 0
                previousValue = val

            gapNumber = len(gapCount)
            if gapNumber!=0: maxGap = max(gapCount)

            if gapNumber!=0:
                logging.warning("marker [%s] - number of gap [%i] and max gap [%i]"%(marker,gapNumber,maxGap))






class SwappingMarkerQualityProcedure(object):
    def __init__(self,acq,markers=None):
        self.acq = acq

        self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)
        self.epsilon = 50.0

    def check(self):

        frameNumber = self.acq.GetPointFrameNumber()
        freq = self.acq.GetPointFrequency()

        for marker in self.markers:


            values = self.acq.GetPoint(marker).GetValues()
            valueDer = derivation.firstOrderFiniteDifference(values,freq)
            norms = np.linalg.norm(values,axis =1)


            for i in range(1,frameNumber-1):
                residual = self.acq.GetPoint(marker).GetResidual(i)
                value_minus1 = norms[i-1]
                value = norms[i]
                value_plus1 = norms[i+1]

                if residual>=0.0:
                    if np.abs(value-value_plus1) >self.epsilon :#10.0*(np.abs(value-value_minus1)):
                        logging.warning("marker [%s] - swapped at frame [%i] "%(marker,i))


class MarkerQualityProcedure(object):
    """
    TODO :
    - check medial markers if exist
    """


    def __init__(self,acq,markers = None):
        self.acq = acq
        self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)
    def check(self):

        frameNumber = self.acq.GetPointFrameNumber()


        LASI_values = self.acq.GetPoint("LASI").GetValues()
        RASI_values = self.acq.GetPoint("RASI").GetValues()
        LPSI_values = self.acq.GetPoint("LPSI").GetValues()
        RPSI_values = self.acq.GetPoint("RPSI").GetValues()
        sacrum_values=(self.acq.GetPoint("LPSI").GetValues() + self.acq.GetPoint("RPSI").GetValues()) / 2.0
        midAsis_values=(self.acq.GetPoint("LASI").GetValues() + self.acq.GetPoint("RASI").GetValues()) / 2.0


        projectedLASI = np.array([LASI_values[:,0],LASI_values[:,1],np.zeros((frameNumber))]).T
        projectedRASI = np.array([RASI_values[:,0],RASI_values[:,1],np.zeros((frameNumber))]).T
        projectedLPSI = np.array([LPSI_values[:,0],LPSI_values[:,1],np.zeros((frameNumber))]).T
        projectedRPSI = np.array([RPSI_values[:,0],RPSI_values[:,1],np.zeros((frameNumber))]).T


        for i  in range(0,frameNumber):
            verts = [
                projectedLASI[i,0:2], # left, bottom
                projectedRASI[i,0:2], # left, top
                projectedRPSI[i,0:2], # right, top
                projectedLPSI[i,0:2], # right, bottom
                projectedLASI[i,0:2], # right, top
                ]

            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.CLOSEPOLY,
                     ]

            path = Path(verts, codes)

            intersection = geometry.LineLineIntersect(projectedLASI[i,:],projectedLPSI[i,:],projectedRASI[i,:],projectedRPSI[i,:])


            if path.contains_point(intersection[0]):
                logging.error("wrong Labelling of pelvic markers at frame [%i]"%(i))
            else:
                # check marker side
                pt1=RASI_values[i,:]
                pt2=LASI_values[i,:]
                pt3=sacrum_values[i,:]
                ptOrigin=midAsis_values[i,:]

                a1=(pt2-pt1)
                a1=np.divide(a1,np.linalg.norm(a1))
                v=(pt3-pt1)
                v=np.divide(v,np.linalg.norm(v))
                a2=np.cross(a1,v)
                a2=np.divide(a2,np.linalg.norm(a2))

                x,y,z,R=frame.setFrameData(a1,a2,"YZX")

                csFrame=frame.Frame()
                csFrame.setRotation(R)
                csFrame.setTranslation(ptOrigin)

                for marker in self.markers:
                    local = np.dot(csFrame.getRotation().T,self.acq.GetPoint(marker).GetValues()[i,:]-csFrame.getTranslation())

                    if marker[0] == "L" and local[1]<0: logging.error("check location of the marker [%s] at frame [%i]"%(marker,i))
                    if marker[0] == "R" and local[1]>0: logging.error("check location of the marker [%s] at frame [%i]"%(marker,i))


class ForcePlateQualityProcedure(object):
    def __init__(self,acq):
        self.acq = acq

    def check(self):
        # TODO :  - saturation and foot asignment
        pass

class EMGQualityProcedure(object):
    def __init__(self,acq, analogLabels ):
        self.acq = acq
        self.analogLabels =  analogLabels

    def check(self):
        # TODO :  - saturation and foot asignment
        pass
