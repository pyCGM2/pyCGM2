# -*- coding: utf-8 -*-

import numpy as np
import logging

import pyCGM2

from pyCGM2 import btk

import cgm

import pyCGM2.Model.frame as cfr
import pyCGM2.Model.motion as cmot
import pyCGM2.Math.euler as ceuler

import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import geometry
from pyCGM2.Tools import  btkTools

markerDiameter=14.0 # TODO ou mettre ca
basePlate = 2.0




class CGM2ModelInf(cgm.CGM1LowerLimbs):
    """ implementation of the cgm2


    """


    def __init__(self):
        """Constructor

           - Run configuration internally
           - Initialize deviation data

        """
        super(CGM2ModelInf, self).__init__()

        self.decoratedModel = False

        #self.__configure()




    def __repr__(self):
        return "cgm2"

    def configure(self):
        # todo create a Foot segment
        self.addSegment("Pelvis", 0,pyCGM2Enums.SegmentSide.Central,["LASI","RASI","LPSI","RPSI"], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,pyCGM2Enums.SegmentSide.Left,["LKNE","LTHI"], tracking_markers = ["LKNE","LTHI"])
        self.addSegment("Right Thigh",4,pyCGM2Enums.SegmentSide.Right,["RKNE","RTHI","RTHIAP","RTHIAD"], tracking_markers = ["RKNE","RTHI","RTHIAP","RTHIAD"])
        self.addSegment("Left Shank",2,pyCGM2Enums.SegmentSide.Left,["LANK","LTIB"], tracking_markers = ["LANK","LTIB"])
        self.addSegment("Right Shank",5,pyCGM2Enums.SegmentSide.Right,["RANK","RTIB","RSHN","RTIAP"], tracking_markers = ["RANK","RTIB","RSHN","RTIAP"])
        self.addSegment("Right Hindfoot",6,pyCGM2Enums.SegmentSide.Right,["RHEE","RCUN","RANK"], tracking_markers = ["RHEE","RCUN"])
        self.addSegment("Right Forefoot",7,pyCGM2Enums.SegmentSide.Right,["RD1M","RD5M","RTOE"], tracking_markers = ["RD1M","RD5M"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])


        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ")
        self.addJoint("LKnee","Left Thigh", "Left Shank","YXZ")

        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ")
        self.addJoint("RKnee","Right Thigh", "Right Shank","YXZ")
        self.addJoint("RAnkle","Right Shank", "Right Hindfoot","YXZ")
        self.addJoint("RForeFoot","Right Hindfoot", "Right Forefoot","YXZ")


    def calibrationProcedure(self):

        """ calibration procedure of the cgm1

        .. note : call from staticCalibration procedure

        .. warning : output TWO dictionary. One for Referentials. One for Anatomical frame

        .. todo :: Include Foot


        """
        dictRef={}
        dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
        dictRef["Left Thigh"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LKNE","LHJC","LTHI","LKNE"]} }
        dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RKNE","RHJC","RTHI","RKNE"]} }
        dictRef["Left Shank"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LANK","LKJC","LTIB","LANK"]} }
        dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RANK","RKJC","RTIB","RANK"]} }

        # left Foot ( nothing yet)
        # right foot
        dictRef["Right Hindfoot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RCUN","RAJC",None,"RAJC"]} }
        dictRef["Right Forefoot"]={"TF" : {'sequence':"ZXY", 'labels':    ["RTOE","RvCUN","RD5M","RTOE"]} }

        dictRefAnatomical={}
        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]} # normaly : midHJC
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]} # origin = Proximal ( differ from native)
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]}
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]}
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}

        # left Foot ( nothing yet)
        # right foot
        dictRefAnatomical["Right Hindfoot"]= {'sequence':"ZXiY", 'labels':   ["RCUN","RHEE",None,"RAJC"]}
        dictRefAnatomical["Right Forefoot"]= {'sequence':"ZYX", 'labels':   ["RvTOE","RvCUN",None,"RvTOE"]} # look out : use virtual Point

        return dictRef,dictRefAnatomical


    def calibrate(self,aquiStatic, dictRef, dictAnatomic,  options=None):
        """ static calibration

        :Parameters:

           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `options` (kwargs) - use to pass option altering the standard construction


        .. todo:: shrink and clone the aquisition to seleted frames
        """
        logging.info("=====================================================")
        logging.info("===================CGM CALIBRATION===================")
        logging.info("=====================================================")

        ff=aquiStatic.GetFirstFrame()
        lf=aquiStatic.GetLastFrame()
        frameInit=ff-ff
        frameEnd=lf-ff+1

        if not self.decoratedModel:
            logging.warning(" Native CGM")
            if not btkTools.isPointExist(aquiStatic,"LKNE"):
                btkTools.smartAppendPoint(aquiStatic,"LKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))
            if not btkTools.isPointExist(aquiStatic,"RKNE"):
                btkTools.smartAppendPoint(aquiStatic,"RKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))

        else:
            logging.warning(" Decorated CGM")


        # --- PELVIS - THIGH - SHANK
        #-------------------------------------

        # calibration of technical Referentials
        logging.info(" --- Pelvis - TF calibration ---")
        logging.info(" -------------------------------")
        self._pelvis_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options) # from CGM1

        logging.info(" --- Left Thigh - TF calibration ---")
        logging.info(" -----------------------------------")
        self._left_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options) # from CGM1
        logging.info(" --- Right Thigh - TF calibration ---")
        logging.info(" ------------------------------------")
        self._right_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options) # from CGM1
        logging.info(" --- Left Shank - TF calibration ---")
        logging.info(" -----------------------------------")
        self._left_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options) # from CGM1
        logging.info(" --- Right Shank - TF calibration ---")
        logging.info(" ------------------------------------")
        self._right_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options) # from CGM1

        # calibration of anatomical Referentials
        logging.info(" --- Pelvis - AF calibration ---")
        logging.info(" -------------------------------")
        self._pelvis_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Pelvis","Pelvis",referential = "Anatomical"  ) # from CGM1
        logging.info(" --- Left Thigh - AF calibration ---")
        logging.info(" -----------------------------------")
        self._left_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Left Thigh","LThigh",referential = "Anatomical"  ) # from CGM1
        logging.info(" --- Right Thigh - AF calibration ---")
        logging.info(" ------------------------------------")
        self._right_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Thigh","RThigh",referential = "Anatomical"  )  # from CGM1
        logging.info(" --- Left Shank - AF calibration ---")
        logging.info(" -----------------------------------")
        self._left_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Left Shank","LShank",referential = "Anatomical"  ) # from CGM1
        logging.info(" --- Right Shank - AF calibration ---")
        logging.info(" ------------------------------------")
        self._right_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Shank","RShank",referential = "Anatomical"  ) # from CGM1

        #offsets
        logging.info(" --- Compute Offsets ---")
        logging.info(" -----------------------")
        self.getThighOffset(side="left")
        self.getThighOffset(side="right")

        self.getShankOffsets(side="both")# compute TibialRotation and Shank offset
        self.getAbdAddAnkleJointOffset(side="both")


        # ---  FOOT segment
        # ---------------
        # need anatomical flexion axis of the shank.


        # --- hind foot
        # --------------

        logging.info(" --- Right Hind Foot  - TF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightHindFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        #self.displayStaticCoordinateSystem( aquiStatic, "Right Hindfoot","RHindFootUncorrected",referential = "technic"  )

        logging.info(" --- Right Hind Foot  - AF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightHindFoot_anatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Hindfoot","RHindFoot",referential = "Anatomical"  )

        logging.info(" --- Hind foot Offset---")
        logging.info(" -----------------------")
        self.getHindFootOffset(side = "both")

        # --- fore foot
        # ----------------
        logging.info(" --- Right Fore Foot  - TF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightForeFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Forefoot","RTechnicForeFoot",referential = "Technical"  )

        logging.info(" --- Right Fore Foot  - AF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightForeFoot_anatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Forefoot","RForeFoot",referential = "Anatomical"  )



        btkTools.smartWriter(aquiStatic, "tmp-static.c3d")


    def _rightHindFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        nFrames = aquiStatic.GetPointFrameNumber()

        seg=self.getSegment("Right Hindfoot")

        # ---  additional markers and Update of the marker segment list


        # new markers
        # mid point NAV and RP5M
        valMidFoot=(aquiStatic.GetPoint("RNAV").GetValues() + aquiStatic.GetPoint("RP5M").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"RMidFoot",valMidFoot,desc="")

        # virtual CUN
        cun =  aquiStatic.GetPoint("RCUN").GetValues()
        valuesVirtualCun = np.zeros((nFrames,3))
        for i in range(0,nFrames):
            valuesVirtualCun[i,:] = np.array([cun[i,0], cun[i,1], cun[i,2]-self.mp["rightToeOffset"]])

        btkTools.smartAppendPoint(aquiStatic,"RvCUN",valuesVirtualCun,desc="cun Registrate")

        # update marker list
        seg.addMarkerLabel("RAJC")         # required markers
        seg.addMarkerLabel("RMidFoot")
        seg.addMarkerLabel("RvCUN")


        # --- technical frame selection and construction
        tf=seg.getReferential("TF")


        pt1=aquiStatic.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictRef["Right Hindfoot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)


        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Hindfoot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

    def _rightForeFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        nFrames = aquiStatic.GetPointFrameNumber()


        seg=self.getSegment("Right Forefoot")

        # ---  additional markers and Update of the marker segment list

        # new markers ( RvTOE - RvD5M)
        toe =  aquiStatic.GetPoint("RTOE").GetValues()
        d5 =  aquiStatic.GetPoint("RD5M").GetValues()

        valuesVirtualToe = np.zeros((nFrames,3))
        valuesVirtualD5 = np.zeros((nFrames,3))
        for i in range(0,nFrames):
            valuesVirtualToe[i,:] = np.array([toe[i,0], toe[i,1], toe[i,2]-self.mp["rightToeOffset"] ])#valuesVirtualCun[i,2]])#
            valuesVirtualD5 [i,:]= np.array([d5[i,0], d5[i,1], valuesVirtualToe[i,2]])

        btkTools.smartAppendPoint(aquiStatic,"RvTOE",valuesVirtualToe,desc="virtual")
        btkTools.smartAppendPoint(aquiStatic,"RvD5M",valuesVirtualD5,desc="virtual-flat ")

        # update marker list
        seg.addMarkerLabel("RMidFoot")
        seg.addMarkerLabel("RvCUN")
        seg.addMarkerLabel("RvTOE")
        seg.addMarkerLabel("RvD5M")

        # --- technical frame selection and construction
        tf=seg.getReferential("TF")


        pt1=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # D5
        pt2=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # Toe

        if dictRef["Right Forefoot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
           v= 100 * np.array([0.0, 0.0, 1.0])

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Forefoot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")


    def _rightHindFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd, options = None):

        seg=self.getSegment("Right Hindfoot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Right Hindfoot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("rightFlatHindFoot" in options.keys() and options["rightFlatHindFoot"]):
            logging.warning("option (rightFlatHindFoot) enable")
            #pt2[2] = pt1[2]
            pt1[2] = pt2[2]
        else:
            logging.warning("option (rightFlatHindFoot) disable")


        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)


        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Right Hindfoot"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)



        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")

        # actual Relative Rotation
        trueRelativeMatrixAnatomic = np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())

        # native CGM relative rotation
        y,x,z = ceuler.euler_yxz(trueRelativeMatrixAnatomic)
        rotX =np.array([[1,0,0],
                        [0,np.cos(x),-np.sin(x)],
                         [0,np.sin(x),np.cos(x)]])
        rotY =np.array([[np.cos(y),0,np.sin(y)],
                        [0,1,0],
                         [-np.sin(y),0,np.cos(y)]])

        relativeMatrixAnatomic = np.dot(rotY,rotX)
        tf.setRelativeMatrixAnatomic( relativeMatrixAnatomic)


        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")



    def _rightForeFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):


        seg=self.getSegment("Right Forefoot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Forefoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Forefoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Right Forefoot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Forefoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v= 100 * np.array([0.0, 0.0, 1.0])

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Forefoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Right Forefoot"]['sequence'])


        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic (np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")

    #---- Offsets -------
    def getHindFootOffset(self, side = "both"):


        if side == "both" or side == "right" :
            R = self.getSegment("Right Hindfoot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)

            print "right hindfoot Static Plantar Flexion"
            self.mp_computed["rightStaticPlantarFlexion"] = np.rad2deg(y)
            logging.debug(" rightStaticPlantarFlexion => %s " % str(self.mp_computed["rightStaticPlantarFlexion"]))

            print "right hindFoot Static Rotation Offset"
            self.mp_computed["rightStaticRotOff"] = np.rad2deg(x)
            logging.debug(" rightStaticRotOff => %s " % str(self.mp_computed["rightStaticRotOff"]))

    # ----- Motion --------------
    def computeMotion(self,aqui, dictRef,dictAnat, motionMethod,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        logging.info("=====================================================")
        logging.info("===================  CGM MOTION   ===================")
        logging.info("=====================================================")

        if motionMethod == pyCGM2Enums.motionMethod.Determinist:
            logging.info("--- Native motion process ---")

            logging.info(" - Pelvis - motion -")
            logging.info(" -------------------")
            self._pelvis_motion(aqui, dictRef, dictAnat)
            logging.info(" - Left Thigh - motion -")
            logging.info(" -----------------------")
            self._left_thigh_motion(aqui, dictRef, dictAnat, options=options)
            logging.info(" - Right Thigh - motion -")
            logging.info(" ------------------------")
            self._right_thigh_motion(aqui, dictRef, dictAnat, options=options)
            logging.info(" - Left Shank - motion -")
            logging.info(" -----------------------")
            self._left_shank_motion(aqui, dictRef, dictAnat, options=options)
            logging.info(" - Right Shank - motion -")
            logging.info(" ------------------------")
            self._right_shank_motion(aqui, dictRef, dictAnat, options=options)
            logging.info(" - Right Hindfoot - motion -")
            logging.info(" ---------------------------")
            self._right_hindFoot_motion(aqui, dictRef, dictAnat, options=options)
            logging.info(" - Right Forefoot - motion -")
            logging.info(" ---------------------------")
            self._right_foreFoot_motion(aqui, dictRef, dictAnat, options=options)

        if motionMethod == pyCGM2Enums.motionMethod.Sodervisk:

            logging.warning("--- Sodervisk motion process ---")

            self._pelvis_motion_optimize(aqui, dictRef, motionMethod)
            self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))

            self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Thigh",originLabel = str(dictAnat["Left Thigh"]['labels'][3]))

            self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Thigh",originLabel = str(dictAnat["Right Thigh"]['labels'][3]))

            self._left_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Shank",originLabel = str(dictAnat["Left Shank"]['labels'][3]))

            self._right_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Shank",originLabel = str(dictAnat["Right Shank"]['labels'][3]))

            # hindFoot
            self._rightHindFoot_motion_optimize(aqui, dictRef,dictAnat,motionMethod)


            # foreFoot
            self._rightForeFoot_motion_optimize(aqui, dictRef,dictAnat,motionMethod)



        logging.info("--- Display Coordinate system ---")
        logging.info(" --------------------------------")
        self.displayMotionCoordinateSystem( aqui,  "Pelvis" , "Pelvis" )
        self.displayMotionCoordinateSystem( aqui,  "Left Thigh" , "LThigh" )
        self.displayMotionCoordinateSystem( aqui,  "Right Thigh" , "RThigh" )
        self.displayMotionCoordinateSystem( aqui,  "Left Shank" , "LShank" )
        self.displayMotionCoordinateSystem( aqui,  "Right Shank" , "RShank" )
        self.displayMotionCoordinateSystem( aqui,  "Right Hindfoot" , "RHindFoot" )
        self.displayMotionCoordinateSystem( aqui,  "Right Forefoot" , "RForeFoot" )

        btkTools.smartWriter(aqui, "tmp-dyn.c3d")

    # ----- native motion ------
    def _right_hindFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Right Hindfoot")



        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][0])).GetValues()[i,:] #cun
            pt2=aqui.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc

            if dictRef["Right Hindfoot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Right Shank").anatomicalFrame.motion[i].m_axisY



            ptOrigin=aqui.GetPoint(str(dictRef["Right Hindfoot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)

            v=v/np.linalg.norm(v)

            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Hindfoot"]["TF"]['sequence'])
            frame=cfr.Frame()

            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(frame)

        # --- RvCUN
        btkTools.smartAppendPoint(aqui,"RvCUN",seg.getReferential("TF").getNodeTrajectory("RvCUN"),desc="from hindFoot" )

        # --- motion of the technical referential
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Hindfoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame()
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)


    def _right_foreFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Right Forefoot")


        # --- motion of the technical referential

        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        #computation
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][2])).GetValues()[i,:]

            if dictRef["Right Forefoot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Right Shank").anatomicalFrame.motion[i].m_axisY

            ptOrigin=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)

            v=(pt3-pt1)
            v=v/np.linalg.norm(v)

            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Forefoot"]["TF"]['sequence'])
            frame=cfr.Frame()

            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(frame)


        # --- motion of new markers
        btkTools.smartAppendPoint(aqui,"RvTOE",seg.getReferential("TF").getNodeTrajectory("RvTOE") )
        btkTools.smartAppendPoint(aqui,"RvD5M",seg.getReferential("TF").getNodeTrajectory("RvD5M") )

        # --- motion of the anatomical referential

        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Forefoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame()
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    # ----- least-square Segmental motion ------
    def _rightHindFoot_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):

        seg=self.getSegment("Right Hindfoot")

        #  --- add RAJC if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "RAJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RAJC")
                    logging.debug("RAJC added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == pyCGM2Enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm= cmot.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                frame=cfr.Frame()
                frame.setRotation(R)
                frame.setTranslation(tOri)
                frame.m_axisX=R[:,0]
                frame.m_axisY=R[:,1]
                frame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(frame)

        # --- RvCUN
        btkTools.smartAppendPoint(aqui,"RvCUN",seg.getReferential("TF").getNodeTrajectory("RvCUN"),desc="opt" )


        # --- Motion of the Anatomical  frame
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Hindfoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame()
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    def _rightForeFoot_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):

        seg=self.getSegment("Right Forefoot")

        #  --- add RvCUN if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "RvCUN" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RvCUN")
                    logging.debug("RvCUN added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == pyCGM2Enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=cmot.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                frame=cfr.Frame()
                frame.setRotation(R)
                frame.setTranslation(tOri)
                frame.m_axisX=R[:,0]
                frame.m_axisY=R[:,1]
                frame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(frame)

        # --- motion of new markers
        # NA

        # ---- motion of anatomical Frame
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Forefoot"]['labels'][3])).GetValues()[i,:]
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame()
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    # ---- finalize methods ------
    # finalizeAbsoluteAngles => no need overloading
    # finalizeKinetics => no need overload

    def finalizeJCS(self,jointLabel,jointValues):
        """ TODO  class method ?

        """

        values = np.zeros((jointValues.shape))


        if jointLabel == "LHip" :  #LHPA=<-1(LHPA),-2(LHPA),-3(LHPA)> {*flexion, adduction, int. rot.			*}
            values[:,0] = - np.rad2deg(  jointValues[:,0])
            values[:,1] = - np.rad2deg(  jointValues[:,1])
            values[:,2] = - np.rad2deg(  jointValues[:,2])

        elif jointLabel == "LKnee" : # LKNA=<1(LKNA),-2(LKNA),-3(LKNA)-$LTibialTorsion>  {*flexion, varus, int. rot.		*}
            values[:,0] = np.rad2deg(  jointValues[:,0])
            values[:,1] = -np.rad2deg(  jointValues[:,1])
            values[:,2] = -np.rad2deg(  jointValues[:,2])

        elif jointLabel == "RHip" :  # RHPA=<-1(RHPA),2(RHPA),3(RHPA)>   {*flexion, adduction, int. rot.			*}
            values[:,0] = - np.rad2deg(  jointValues[:,0])
            values[:,1] =  np.rad2deg(  jointValues[:,1])
            values[:,2] =  np.rad2deg(  jointValues[:,2])

        elif jointLabel == "RKnee" : #  RKNA=<1(RKNA),2(RKNA),3(RKNA)-$RTibialTorsion>    {*flexion, varus, int. rot.		*}
            values[:,0] = np.rad2deg(  jointValues[:,0])
            values[:,1] = np.rad2deg(  jointValues[:,1])
            values[:,2] = np.rad2deg(  jointValues[:,2])

        elif jointLabel == "LAnkle":
            values[:,0] = -1.0* np.rad2deg(  jointValues[:,0]  + np.radians(90))
            values[:,1] = -1.0*np.rad2deg(  jointValues[:,2])
            values[:,2] =  -1.0*np.rad2deg(  jointValues[:,1])

        elif jointLabel == "RAnkle":
            values[:,0] = -1.0* np.rad2deg(  jointValues[:,0]  + np.radians(90))
            values[:,1] = np.rad2deg(  jointValues[:,2])
            values[:,2] =  np.rad2deg(  jointValues[:,1])


        elif jointLabel == "RForeFoot":
            values[:,0] = -1.0 * np.rad2deg(  jointValues[:,0])
            values[:,1] = np.rad2deg(  jointValues[:,1])
            values[:,2] = np.rad2deg(  jointValues[:,2])

        else:
            values[:,0] = np.rad2deg(  jointValues[:,0])
            values[:,1] = np.rad2deg(  jointValues[:,1])
            values[:,2] = np.rad2deg(  jointValues[:,2])

        return values

    # --- opensim --------
    def opensimTrackingMarkers(self):

        out={}
        for segIt in self.m_segmentCollection:
            out[segIt.name] = segIt.m_tracking_markers

        return out

    def opensimGeometry(self):
        """
        TODO require : joint name from opensim -> find alternative

        rather a class method than a method instance
        """

        out={}
        out["hip_r"]= {"joint label":"RHJC", "proximal segment label":"Pelvis", "distal segment label":"Right Thigh" }
        out["knee_r"]= {"joint label":"RKJC", "proximal segment label":"Right Thigh", "distal segment label":"Right Shank" }
        out["ankle_r"]= {"joint label":"RAJC", "proximal segment label":"Right Shank", "distal segment label":"Right Hindfoot" }
        out["mtp_r"]= {"joint label":"RvCUN", "proximal segment label":"Right Hindfoot", "distal segment label":"Right Forefoot" }


        out["hip_l"]= {"joint label":"LHJC", "proximal segment label":"Pelvis", "distal segment label":"Left Thigh" }
        out["knee_l"]= {"joint label":"LKJC", "proximal segment label":"Left Thigh", "distal segment label":"Left Shank" }
        #out["ankle_l"]= {"joint label":"LAJC", "proximal segment label":"Left Shank", "distal segment label":"Left Hindfoot" }
        #out["mtp_l"]= {"joint label":"LvCUN", "proximal segment label":"Left Hindfoot", "distal segment label":"Left Forefoot" }

        return out

    def opensimIkTask(self,expert = False):
        out={}
        out={"LASI":100,
             "RASI":100,
             "LPSI":100,
             "RPSI":100,
             "RTHI":100,
             "RTHIAP":100,
             "RTHIAD":100,
             "RKNE":100,
             "RTIB":100,
             "RTIAP":100,
             "RSHN":100,
             "RANK":100,
             "RHEE":100,
             "RCUN":100,
             "RD1M":100,
             "RD5M":100}

        return out
