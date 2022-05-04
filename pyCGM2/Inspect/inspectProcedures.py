# -*- coding: utf-8 -*-
"""
Obsolete module : work with anomaly and inspector modules instead
"""

import numpy as np
from matplotlib.path import Path
import pyCGM2; LOGGER = pyCGM2.LOGGER

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

from pyCGM2.Tools import btkTools
from pyCGM2.Math import geometry

from pyCGM2.Model import   frame
from pyCGM2.Utils import utils
from pyCGM2.Signal import detect_changes


class GaitEventQualityProcedure(object):
    def __init__(self,acq, title = None):

        self.acq = acq
        self.state = True
        self.exceptionMode = False

        self.title = "Gait events" if title is None else title

    def check(self):

        events = self.acq.GetEvents()

        if events != []:

            events_L = list()
            events_R = list()
            for ev in btk.Iterate(events):
                if ev.GetContext() == "Left":
                    events_L.append(ev)
                if ev.GetContext() == "Right":
                    events_R.append(ev)


            if events_L!=[] and events_R!=[]:
                labels = [it.GetLabel() for it in btk.Iterate(events) if it.GetLabel() in ["Foot Strike","Foot Off"]]
                frames = [it.GetFrame() for it in btk.Iterate(events) if it.GetLabel() in ["Foot Strike","Foot Off"]]

                init = labels[0]
                for i in range(1,len(labels)):
                    label = labels[i]
                    frame = frames[i]
                    if label == init:
                        LOGGER.logger.error("[pyCGM2-Checking] two consecutive (%s) detected at frame (%i)"%((label),frame))
                        if self.exceptionMode:
                            raise Exception("[pyCGM2-Checking]  two consecutive (%s) detected at frame (%i)"%((label),frame))

                        self.state = False
                    init = label

            if events_L!=[]:
                init = events_L[0].GetLabel()
                if len(events_L)>1:
                    for i in range(1,len(events_L)):
                        label = events_L[i].GetLabel()
                        if label == init:
                            LOGGER.logger.error("[pyCGM2-Checking]  Wrong Left Event - two consecutive (%s) detected at frane (%i)"%((label),events_L[i].GetFrame()) )
                            if self.exceptionMode:
                                raise Exception("[pyCGM2-Checking]  Wrong Left Event - two consecutive (%s) detected at frane (%i)"%((label),events_L[i].GetFrame()))

                            self.state = False
                        init = label
                else:
                    LOGGER.logger.warning("Only one left events")
                    self.state = False

            if events_R!=[]:
                init = events_R[0].GetLabel()
                if len(events_R)>1:
                    for i in range(1,len(events_R)):
                        label = events_R[i].GetLabel()
                        if label == init:
                            LOGGER.logger.error("[pyCGM2-Checking] Wrong Right Event - two consecutive (%s) detected at frane (%i)"%((label),events_R[i].GetFrame()) )
                            if self.exceptionMode:
                                raise Exception("[pyCGM2-Checking] Wrong Right Event - two consecutive (%s) detected at frane (%i)"%((label),events_R[i].GetFrame()) )
                            self.state = False
                        init = label
                else:
                    LOGGER.logger.warning("Only one right events ")
                    self.state = False
        else:
            LOGGER.logger.error("[pyCGM2-Checking] No events are in trial")
            if self.exceptionMode:
                raise Exception("[pyCGM2-Checking]  No events are in the trials")
            self.state = False





class AnthropometricDataQualityProcedure(object):
    def __init__(self,mp,title=None):
        self.mp = mp
        self.state = True
        self.exceptionMode = False

        self.title = "CGM anthropometric parameters" if title is None else title

    def check(self):
        """
        TODO :
        - use relation between variable ( width/height)
        - use marker measurement
        """

        if self.mp["RightLegLength"] < 500: LOGGER.logger.warning("[pyCGM2-Checking] Right Leg Lenth < 500 mm");self.state = False
        if self.mp["LeftLegLength"] < 500: LOGGER.logger.warning("[pyCGM2-Checking] Left Leg Lenth < 500 mm");self.state = False
        if self.mp["RightKneeWidth"] < self.mp["RightAnkleWidth"]: LOGGER.logger.error("[pyCGM2-Checking] Right ankle width > knee width ");self.state = False
        if self.mp["LeftKneeWidth"] < self.mp["LeftAnkleWidth"]: LOGGER.logger.error("[pyCGM2-Checking] Right ankle width > knee width ");self.state = False
        if self.mp["RightKneeWidth"] > self.mp["RightLegLength"]: LOGGER.logger.error("[pyCGM2-Checking]  Right knee width > leg length ");self.state = False
        if self.mp["LeftKneeWidth"] > self.mp["LeftLegLength"]: LOGGER.logger.error(" [pyCGM2-Checking] Left knee width > leg length ");self.state = False


        if not utils.isInRange(self.mp["RightKneeWidth"],
            self.mp["LeftKneeWidth"]-0.3*self.mp["LeftKneeWidth"],
            self.mp["LeftKneeWidth"]+0.3*self.mp["LeftKneeWidth"]):
            LOGGER.logger.warning("[pyCGM2-Checking] Knee widths differed by more than 30%")
            self.state = False

        if not utils.isInRange(self.mp["RightAnkleWidth"],
            self.mp["LeftAnkleWidth"]-0.3*self.mp["LeftAnkleWidth"],
            self.mp["LeftAnkleWidth"]+0.3*self.mp["LeftAnkleWidth"]):
            LOGGER.logger.warning("[pyCGM2-Checking] Ankle widths differed by more than 30%")
            self.state = False

        if not utils.isInRange(self.mp["RightLegLength"],
            self.mp["LeftLegLength"]-0.3*self.mp["LeftLegLength"],
            self.mp["LeftLegLength"]+0.3*self.mp["LeftLegLength"]):
            LOGGER.logger.warning("[pyCGM2-Checking] Leg lengths differed by more than 30%")
            self.state = False

class MarkerPresenceQualityProcedure(object):

    def __init__(self,acq,markers=None,title=None):
        self.acq = acq
        self.exceptionMode = False
        self.title = "Marker presence" if title is None else title

        self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)

        self.state = True
        self.markersIn = []

    def check(self):

        frameNumber = self.acq.GetPointFrameNumber()

        count = 0
        for marker in self.markers:
            try:
                self.acq.GetPoint(marker)
            except RuntimeError:
                LOGGER.logger.warning("[pyCGM2-Checking]  marker [%s] - not exist in the acquisition"%(marker))
            else:
                self.markersIn.append(marker)
                count +=1

        if not count == len(self.markers):
            self.state = False

class GapQualityProcedure(object):

    def __init__(self,acq,markers=None,title=None):
        self.acq = acq
        self.exceptionMode = False

        self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)
        self.state = True

        self.title = "Gap" if title is None else title

    def check(self):

        frameNumber = self.acq.GetPointFrameNumber()

        for marker in self.markers:
            gapCount = list()
            previousValue = 0
            count =0
            try:
                self.acq.GetPoint(marker)
            except RuntimeError:
                LOGGER.logger.warning("[pyCGM2-Checking] marker [%s] - not exist in the acquisition"%(marker))
            else:
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
                    LOGGER.logger.warning("[pyCGM2-Checking] marker [%s] - number of gap [%i] and max gap [%i]"%(marker,gapNumber,maxGap))
                    self.state = False






class SwappingMarkerQualityProcedure(object):
    def __init__(self,acq,markers=None,title=None,plot=False):
        self.acq = acq
        self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)
        self.exceptionMode = False
        self.title = "Swapping marker" if title is None else title

        self.state = True
        self.plot = plot



    def check(self):

        ff = self.acq.GetFirstFrame()
        frameNumber = self.acq.GetPointFrameNumber()
        freq = self.acq.GetPointFrequency()

        for marker in self.markers:
            markerList = list(self.markers)
            markerList.remove(marker)
            nearest,dist = btkTools.findNearestMarker(self.acq,1,marker,markerNames = markerList)
            values = self.acq.GetPoint(marker).GetValues()
            norms = np.linalg.norm(values,axis =1)

            ta, tai, taf, amp = detect_changes.detect_cusum(norms, dist, dist/2.0, True, self.plot)

            if ta.size != 0:
                for index in tai:
                    residual = self.acq.GetPoint(marker).GetResidual(int(index))
                    if residual>=0.0:
                        frame = index+ff
                        LOGGER.logger.warning("[pyCGM2-Checking] marker [%s] - swapped at frame [%i] (nearest marker= %s - dist=%.2f) "%(marker,frame,nearest,dist))


class MarkerPositionQualityProcedure(object):
    """
    TODO :
    - check medial markers if exist
    """


    def __init__(self,acq,markers = None, title = None):
        self.acq = acq
        self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)
        self.exceptionMode = False
        self.state = True

        self.title = "Marker position" if title is None else title

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
                LOGGER.logger.error("[pyCGM2-Checking] wrong Labelling of pelvic markers at frame [%i]"%(i))
                if self.exceptionMode:
                    raise Exception("[pyCGM2-Checking] wrong Labelling of pelvic markers at frame [%i]"%(i))

                self.state = False
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

                csFrame_L=frame.Frame()
                csFrame_L.setRotation(R)
                csFrame_L.setTranslation(RASI_values[i,:])

                csFrame_R=frame.Frame()
                csFrame_R.setRotation(R)
                csFrame_R.setTranslation(LASI_values[i,:])


                for marker in self.markers:
                    residual = self.acq.GetPoint(marker).GetResidual(i)

                    if marker[0] == "L":
                        local = np.dot(csFrame_L.getRotation().T,self.acq.GetPoint(marker).GetValues()[i,:]-csFrame_L.getTranslation())
                    if marker[0] == "R":
                        local = np.dot(csFrame_R.getRotation().T,self.acq.GetPoint(marker).GetValues()[i,:]-csFrame_R.getTranslation())
                    if residual >0.0:
                        if marker[0] == "L" and local[1]<0:
                            LOGGER.logger.error("[pyCGM2-Checking] check location of the marker [%s] at frame [%i]"%(marker,i))
                            self.state = False
                            if self.exceptionMode:
                                raise Exception("[pyCGM2-Checking] check location of the marker [%s] at frame [%i]"%(marker,i))

                        if marker[0] == "R" and local[1]>0:
                            LOGGER.logger.error("[pyCGM2-Checking] check location of the marker [%s] at frame [%i]"%(marker,i))
                            self.state = False
                            if self.exceptionMode:
                                raise Exception("[pyCGM2-Checking] check location of the marker [%s] at frame [%i]"%(marker,i))
                                self.state = False


class ForcePlateQualityProcedure(object):

    def __init__(self,acq,title=None):
        self.acq = acq
        self.exceptionMode = False
        self.state = True

        self.title = "Force Plate" if title is None else title

    def check(self):
        # TODO :  - saturation and foot asignment
        pass

class EMGQualityProcedure(object):
    def __init__(self,acq, analogLabels,title=None):
        self.acq = acq
        self.exceptionMode = False
        self.analogLabels =  analogLabels
        self.state = True

        self.title = "EMG" if title is None else title
    def check(self):
        # TODO :  - saturation and foot asignment
        pass
