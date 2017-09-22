# -*- coding: utf-8 -*-
import numpy as np
import logging
import pdb
import copy

import btk

import model as cmb
import modelDecorator as cmd
import frame as cfr
import motion as cmot
import euler as ceuler

import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import geometry
from pyCGM2.Tools import  btkTools
from pyCGM2.Nexus import nexusTools

import cgm


class CGM2_1LowerLimbs(cgm.CGM1LowerLimbs):
    """

    """

    #nativeCgm1 = True

    MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

    def __init__(self):
        super(CGM2_1LowerLimbs, self).__init__()
        self.decoratedModel = False

        self.version = "CGM2.1"

    def __repr__(self):
        return "LowerLimb CGM2.1"

class CGM2_2LowerLimbs(cgm.CGM1LowerLimbs):
    """

    """

    MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

    def __init__(self):
        super(CGM2_2LowerLimbs, self).__init__()
        self.decoratedModel = False

        self.version = "CGM2.2"

    def __repr__(self):
        return "LowerLimb CGM2.2"


class CGM2_3LowerLimbs(cgm.CGM1LowerLimbs):
    """
        implementation of the cgm2.3 skin marker added
    """

    MARKERS = ["LASI", "RASI","RPSI", "LPSI",
               "LTHI","LKNE","LTHIAP","LTHIAD",
               "LTIB","LANK","LTIBAP","LTIBAD",
               "LHEE","LTOE",
               "RTHI","RKNE","RTHIAP","RTHIAD",
               "RTIB","RANK","RTIBAP","RTIBAD",
               "RHEE","RTOE"]

    def __init__(self):
        """Constructor

           - Run configuration internally
           - Initialize deviation data

        """
        super(CGM2_3LowerLimbs, self).__init__()

        self.decoratedModel = False


        self.version = "CGM2.3"

        #self.__configure()


    def __repr__(self):
        return "LowerLimb CGM2.3"

    def configure(self):
        self.addSegment("Pelvis",0,pyCGM2Enums.SegmentSide.Central,calibration_markers=[], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,pyCGM2Enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LKNE","LTHI","LTHIAP","LTHIAD"])
        self.addSegment("Right Thigh",4,pyCGM2Enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RKNE","RTHI","RTHIAP","RTHIAD"])
        self.addSegment("Left Shank",2,pyCGM2Enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LANK","LTIB","LTIBAP","LTIBAD"])
        self.addSegment("Left Shank Proximal",7,pyCGM2Enums.SegmentSide.Left) # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Right Shank",5,pyCGM2Enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RANK","RTIB","RTIBAP","RTIBAD"])
        self.addSegment("Right Shank Proximal",8,pyCGM2Enums.SegmentSide.Right)        # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Left Foot",3,pyCGM2Enums.SegmentSide.Left,calibration_markers=["LAJC"], tracking_markers = ["LHEE","LTOE"] )
        self.addSegment("Right Foot",6,pyCGM2Enums.SegmentSide.Right,calibration_markers=["RAJC"], tracking_markers = ["RHEE","RTOE"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])

        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ")
        self.addJoint("LKnee","Left Thigh", "Left Shank","YXZ")
        #self.addJoint("LKneeAngles_cgm","Left Thigh", "Left Shank","YXZ")
        self.addJoint("LAnkle","Left Shank", "Left Foot","YXZ")
        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ")
        self.addJoint("RKnee","Right Thigh", "Right Shank","YXZ")
        self.addJoint("RAnkle","Right Shank", "Right Foot","YXZ")


    def calibrationProcedure(self):
        """
            Define the calibration Procedure

            :Return:
                - `dictRef` (dict) - dictionnary reporting markers and sequence use for building Technical coordinate system
                - `dictRefAnatomical` (dict) - dictionnary reporting markers and sequence use for building Anatomical coordinate system
        """

        dictRef={}
        dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
        dictRef["Left Thigh"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LKNE","LHJC","LTHI","LKNE"]} }
        dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RKNE","RHJC","RTHI","RKNE"]} }
        dictRef["Left Shank"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LANK","LKJC","LTIB","LANK"]} }
        dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RANK","RKJC","RTIB","RANK"]} }

        dictRef["Left Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LTOE","LAJC",None,"LAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis
        dictRef["Right Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RTOE","RAJC",None,"RAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis


        dictRefAnatomical={}
        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]} # normaly : midHJC
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]} # origin = Proximal ( differ from native)
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]}
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]}
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}

        dictRefAnatomical["Left Foot"]={'sequence':"ZXiY", 'labels':  ["LTOE","LHEE",None,"LAJC"]}    # corrected foot
        dictRefAnatomical["Right Foot"]={'sequence':"ZXiY", 'labels':  ["RTOE","RHEE",None,"RAJC"]}    # corrected foot


        return dictRef,dictRefAnatomical


    def calibrate(self,aquiStatic, dictRef, dictAnatomic,  options=None): #

        super(CGM2_3LowerLimbs, self).calibrate(aquiStatic, dictRef, dictAnatomic,  options=options)


    # --- motion --------




#    def computeMotion(self,aqui, dictRef,dictAnat, motionMethod,options=None ):
#
#        super(CGM2_3LowerLimbs, self).computeMotion(aqui, dictRef,dictAnat, motionMethod, options=options)



    # --- opensim --------
    def opensimTrackingMarkers(self):


        out={}
        for segIt in self.m_segmentCollection:
            if "Proximal" not in segIt.name:
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
        out["ankle_r"]= {"joint label":"RAJC", "proximal segment label":"Right Shank", "distal segment label":"Right Foot" }
        #out["mtp_r"]=


        out["hip_l"]= {"joint label":"LHJC", "proximal segment label":"Pelvis", "distal segment label":"Left Thigh" }
        out["knee_l"]= {"joint label":"LKJC", "proximal segment label":"Left Thigh", "distal segment label":"Left Shank" }
        out["ankle_l"]= {"joint label":"LAJC", "proximal segment label":"Left Shank", "distal segment label":"Left Foot" }
        #out["mtp_l"]=

        return out

    def opensimIkTask(self,expert=False):
        out={}

        if expert:
          out={"LASI":0,
                 "LASI_posAnt":100,
                 "LASI_medLat":100,
                 "LASI_supInf":100,
                 "RASI":0,
                 "RASI_posAnt":100,
                 "RASI_medLat":100,
                 "RASI_supInf":100,
                 "LPSI":0,
                 "LPSI_posAnt":100,
                 "LPSI_medLat":100,
                 "LPSI_supInf":100,
                 "RPSI":0,
                 "RPSI_posAnt":100,
                 "RPSI_medLat":100,
                 "RPSI_supInf":100,

                 "RTHI":0,
                 "RTHI_posAnt":100,
                 "RTHI_medLat":100,
                 "RTHI_proDis":100,
                 "RKNE":0,
                 "RKNE_posAnt":100,
                 "RKNE_medLat":100,
                 "RKNE_proDis":100,
                 "RTIB":0,
                 "RTIB_posAnt":100,
                 "RTIB_medLat":100,
                 "RTIB_proDis":100,
                 "RANK":0,
                 "RANK_posAnt":100,
                 "RANK_medLat":100,
                 "RANK_proDis":100,
                 "RHEE":0,
                 "RHEE_supInf":100,
                 "RHEE_medLat":100,
                 "RHEE_proDis":100,
                 "RTOE":0,
                 "RTOE_supInf":100,
                 "RTOE_medLat":100,
                 "RTOE_proDis":100,

                 "LTHI":0,
                 "LTHI_posAnt":100,
                 "LTHI_medLat":100,
                 "LTHI_proDis":100,
                 "LKNE":0,
                 "LKNE_posAnt":100,
                 "LKNE_medLat":100,
                 "LKNE_proDis":100,
                 "LTIB":0,
                 "LTIB_posAnt":100,
                 "LTIB_medLat":100,
                 "LTIB_proDis":100,
                 "LANK":0,
                 "LANK_posAnt":100,
                 "LANK_medLat":100,
                 "LANK_proDis":100,
                 "LHEE":0,
                 "LHEE_supInf":100,
                 "LHEE_medLat":100,
                 "LHEE_proDis":100,
                 "LTOE":0,
                 "LTOE_supInf":100,
                 "LTOE_medLat":100,
                 "LTOE_proDis":100,

                 "LTHIAP":0,
                 "LTHIAP_posAnt":100,
                 "LTHIAP_medLat":100,
                 "LTHIAP_proDis":100,
                 "LTHIAD":0,
                 "LTHIAD_posAnt":100,
                 "LTHIAD_medLat":100,
                 "LTHIAD_proDis":100,
                 "RTHIAP":0,
                 "RTHIAP_posAnt":100,
                 "RTHIAP_medLat":100,
                 "RTHIAP_proDis":100,
                 "RTHIAD":0,
                 "RTHIAD_posAnt":100,
                 "RTHIAD_medLat":100,
                 "RTHIAD_proDis":100,
                 "LTIBAP":0,
                 "LTIBAP_posAnt":100,
                 "LTIBAP_medLat":100,
                 "LTIBAP_proDis":100,
                 "LTIBAD":0,
                 "LTIBAD_posAnt":100,
                 "LTIBAD_medLat":100,
                 "LTIBAD_proDis":100,
                 "RTIBAP":0,
                 "RTIBAP_posAnt":100,
                 "RTIBAP_medLat":100,
                 "RTIBAP_proDis":100,
                 "RTIBAD":0,
                 "RTIBAD_posAnt":100,
                 "RTIBAD_medLat":100,
                 "RTIBAD_proDis":100,

                 "LTHLD":0,
                 "LTHLD_posAnt":0,
                 "LTHLD_medLat":0,
                 "LTHLD_proDis":0,
                 "LPAT":0,
                 "LPAT_posAnt":0,
                 "LPAT_medLat":0,
                 "LPAT_proDis":0,
                 "RTHLD":0,
                 "RTHLD_posAnt":0,
                 "RTHLD_medLat":0,
                 "RTHLD_proDis":0,
                 "RPAT":0,
                 "RPAT_posAnt":0,
                 "RPAT_medLat":0,
                 "RPAT_proDis":0,

                 }
        else:
            out={"LASI":100,
                 "RASI":100,
                 "LPSI":100,
                 "RPSI":100,
                 "RTHI":100,
                 "RKNE":100,
                 "RTHIAP":100,
                 "RTHIAD":100,
                 "RTIB":100,
                 "RANK":100,
                 "RTIBAP":100,
                 "RTIBAD":100,
                 "RHEE":100,
                 "RTOE":100,
                 "LTHI":100,
                 "LKNE":100,
                 "LTHIAP":100,
                 "LTHIAD":100,
                 "LTIB":100,
                 "LANK":100,
                 "LTIBAP":100,
                 "LTIBAD":100,
                 "LHEE":100,
                 "LTOE":100,
                 "RTHLD":0,
                 "RPAT":0,
                 "LTHLD":0,
                 "LPAT":0
                 }

        return out

class CGM2_4LowerLimbs(CGM2_3LowerLimbs):
    MARKERS = ["LASI", "RASI","RPSI", "LPSI",
               "LTHI","LKNE","LTHIAP","LTHIAD",
               "LTIB","LANK","LTIBAP","LTIBAD",
               "LHEE","LTOE","LCUN","LD1M","LD5M",
               "RTHI","RKNE","RTHIAP","RTHIAD",
               "RTIB","RANK","RTIBAP","RTIBAD",
               "RHEE","RTOE","RCUN","RD1M","RD5M"]

    ANALYSIS_KINEMATIC_LABELS_DICT ={ 'Left': ["LHipAngles","LKneeAngles","LAnkleAngles","LFootProgressAngles","LPelvisAngles","LForeFoot"],
                       'Right': ["RHipAngles","RKneeAngles","RAnkleAngles","RFootProgressAngles","RPelvisAngles","LForeFoot"]}

    ANALYSIS_KINETIC_LABELS_DICT ={ 'Left': ["LHipMoment","LKneeMoment","LAnkleMoment","LHipPower","LKneePower","LAnklePower"],
                          'Right': ["RHipMoment","RKneeMoment","RAnkleMoment","RHipPower","RKneePower","RAnklePower"]}

    def __init__(self):
        """Constructor

           - Run configuration internally
           - Initialize deviation data

        """
        super(CGM2_4LowerLimbs, self).__init__()

        self.decoratedModel = False

        self.version = "CGM2.4"

        #self.__configure()

    def __repr__(self):
        return "LowerLimb CGM2.4"

    def configure(self):
        self.addSegment("Pelvis",0,pyCGM2Enums.SegmentSide.Central,["LASI","RASI","LPSI","RPSI"], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,pyCGM2Enums.SegmentSide.Left,["LKNE","LTHI","LTHIAP","LTHIAD"], tracking_markers = ["LKNE","LTHI","LTHIAP","LTHIAD"])
        self.addSegment("Right Thigh",4,pyCGM2Enums.SegmentSide.Right,["RKNE","RTHI","RTHIAP","RTHIAD"], tracking_markers = ["RKNE","RTHI","RTHIAP","RTHIAD"])
        self.addSegment("Left Shank",2,pyCGM2Enums.SegmentSide.Left,["LANK","LTIB","LTIBAP","LTIBAD"], tracking_markers = ["LANK","LTIB","LTIBAP","LTIBAD"])
        self.addSegment("Left Shank Proximal",7,pyCGM2Enums.SegmentSide.Left) # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Right Shank",5,pyCGM2Enums.SegmentSide.Right,["RANK","RTIB","RTIBAP","RTIBAD"], tracking_markers = ["RANK","RTIB","RTIBAP","RTIBAD"])
        self.addSegment("Right Shank Proximal",8,pyCGM2Enums.SegmentSide.Right)        # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Left HindFoot",6,pyCGM2Enums.SegmentSide.Left,["LHEE","LCUN","LANK"], tracking_markers = ["LHEE","LCUN"])
        self.addSegment("Left ForeFoot",7,pyCGM2Enums.SegmentSide.Left,["LD1M","LD5M","LTOE"], tracking_markers = ["LD1M","LD5M","LTOE"])
        self.addSegment("Right HindFoot",6,pyCGM2Enums.SegmentSide.Right,["RHEE","RCUN","RANK"], tracking_markers = ["RHEE","RCUN"])
        self.addSegment("Right ForeFoot",7,pyCGM2Enums.SegmentSide.Right,["RD1M","RD5M","RTOE"], tracking_markers = ["RD1M","RD5M","RTOE"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])

        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ")
        self.addJoint("LKnee","Left Thigh", "Left Shank","YXZ")
        self.addJoint("LAnkle","Left Shank", "Left HindFoot","YXZ")
        self.addJoint("LForeFoot","Left HindFoot", "Left ForeFoot","YXZ")

        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ")
        self.addJoint("RKnee","Right Thigh", "Right Shank","YXZ")
        self.addJoint("RAnkle","Right Shank", "Right HindFoot","YXZ")
        self.addJoint("RForeFoot","Right HindFoot", "Right ForeFoot","YXZ")


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

        # left Foot
        dictRef["Left HindFoot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LCUN","LAJC",None,"LAJC"]} }
        dictRef["Left ForeFoot"]={"TF" : {'sequence':"ZXY", 'labels':    ["LTOE","LvCUN","LD5M","LTOE"]} }

        # right foot
        dictRef["Right HindFoot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RCUN","RAJC",None,"RAJC"]} }
        dictRef["Right ForeFoot"]={"TF" : {'sequence':"ZXY", 'labels':    ["RTOE","RvCUN","RD5M","RTOE"]} }

        dictRefAnatomical={}
        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]} # normaly : midHJC
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]} # origin = Proximal ( differ from native)
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]}
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]}
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}

        # left Foot ( nothing yet)
        dictRefAnatomical["Left HindFoot"]= {'sequence':"ZXiY", 'labels':   ["LCUN","LHEE",None,"LAJC"]}
        dictRefAnatomical["Left ForeFoot"]= {'sequence':"ZYX", 'labels':   ["LvTOE","LvCUN",None,"LvTOE"]} # look out : use virtual Point

        # right foot
        dictRefAnatomical["Right HindFoot"]= {'sequence':"ZXiY", 'labels':   ["RCUN","RHEE",None,"RAJC"]}
        dictRefAnatomical["Right ForeFoot"]= {'sequence':"ZYX", 'labels':   ["RvTOE","RvCUN",None,"RvTOE"]} # look out : use virtual Point

        return dictRef,dictRefAnatomical

    def calibrate(self,aquiStatic, dictRef, dictAnatomic,  options=None):
        """
            Perform full CGM1 calibration.

            :Parameters:
               - `aquiStatic` (btkAcquisition) - btkAcquisition instance from a static c3d
               - `dictRef` (dict) - dictionnary reporting markers and sequence use for building **Technical** coordinate system
               - `dictAnatomic` (dict) - dictionnary reporting markers and sequence use for building **Anatomical**  coordinate system
               - `options` (dict) - use to pass options, like options altering the standard segment construction.

            .. note:: This method constructs technical and anatomical frane sucessively.

            .. warning : Foot Calibration need attention. Indeed, its technical coordinate system builder requires the anatomical coordinate system of the shank

        """
        #TODO : to input Frane init and Frame end manually

        logging.debug("=====================================================")
        logging.debug("===================CGM CALIBRATION===================")
        logging.debug("=====================================================")

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

        # ---- Pelvis-THIGH-SHANK CALIBRATION
        #-------------------------------------
        # calibration of technical Referentials
        logging.debug(" --- Pelvis - TF calibration ---")
        logging.debug(" -------------------------------")
        self._pelvis_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)

        logging.debug(" --- Left Thigh- TF calibration ---")
        logging.debug(" ----------------------------------")
        self._left_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        logging.debug(" --- Right Thigh - TF calibration ---")
        logging.debug(" ------------------------------------")
        self._right_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        logging.debug(" --- Left Shank - TF calibration ---")
        logging.debug(" -----------------------------------")
        self._left_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)


        logging.debug(" --- Richt Shank - TF calibration ---")
        logging.debug(" ------------------------------------")
        self._right_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)



        # calibration of anatomical Referentials
        logging.debug(" --- Pelvis - AF calibration ---")
        logging.debug(" -------------------------------")
        self._pelvis_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Pelvis","Pelvis",referential = "Anatomic"  )

        logging.debug(" --- Left Thigh - AF calibration ---")
        logging.debug(" -----------------------------------")
        self._left_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
             self.displayStaticCoordinateSystem( aquiStatic, "Left Thigh","LThigh",referential = "Anatomic"  )


        logging.debug(" --- Right Thigh - AF calibration ---")
        logging.debug(" ------------------------------------")
        self._right_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
             self.displayStaticCoordinateSystem( aquiStatic, "Right Thigh","RThigh",referential = "Anatomic"  )


        logging.debug(" --- Thigh Offsets ---")
        logging.debug(" --------------------")


        logging.debug(" ------Left-------")
        if self.mp.has_key("LeftThighRotation") and self.mp["LeftThighRotation"] != 0:
            self.mp_computed["LeftThighRotationOffset"]= -self.mp["LeftThighRotation"]

        else:
            self.getThighOffset(side="left")

        # if SARA axis
        if "enableLongitudinalRotation" in options.keys():
            if self.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("KJC_SaraAxis"):
                logging.debug("SARA axis found from the left thigh")

                self.getAngleOffsetFromFunctionalSaraAxis("left")

                self._rotateAnatomicalFrame("Left Thigh",self.mp_computed["LeftKneeFuncCalibrationOffset"],
                                            aquiStatic, dictAnatomic,frameInit,frameEnd)

        # if dynaKad offset
        if self.mp_computed.has_key("LeftKneeDynaKadOffset") and self.mp_computed["LeftKneeDynaKadOffset"] != 0:
            logging.debug("left DynaKad offset found. Anatomical referential rotated from dynaKad offset")
            self._rotateAnatomicalFrame("Left Thigh",self.mp_computed["LeftKneeDynaKadOffset"],
                                            aquiStatic, dictAnatomic,frameInit,frameEnd)


        logging.debug(" ------Right-------")
        if self.mp.has_key("RightThighRotation") and self.mp["RightThighRotation"] != 0:
            self.mp_computed["RightThighRotationOffset"]= self.mp["RightThighRotation"]
        else:
            self.getThighOffset(side="right")

        if "enableLongitudinalRotation" in options.keys():
            if self.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("KJC_SaraAxis"):
                logging.debug("SARA axis found from the Right thigh")

                self.getAngleOffsetFromFunctionalSaraAxis("right")

                self._rotateAnatomicalFrame("Right Thigh",self.mp_computed["RightKneeFuncCalibrationOffset"],
                                            aquiStatic, dictAnatomic,frameInit,frameEnd)


        # if dynaKad offset
        if self.mp_computed.has_key("RightKneeDynaKadOffset") and self.mp_computed["RightKneeDynaKadOffset"] != 0:
            logging.debug("Right DynaKad offset found. Anatomical referential rotated from dynaKad offset")
            self._rotateAnatomicalFrame("Right Thigh",self.mp_computed["RightKneeDynaKadOffset"],
                                            aquiStatic, dictAnatomic,frameInit,frameEnd)



        logging.debug(" --- Left Shank - AF calibration ---")
        logging.debug(" -------------------------------")
        self._left_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Left Shank","LShank",referential = "Anatomic"  )


        logging.debug(" --- Right Shank - AF calibration ---")
        logging.debug(" -------------------------------")
        self._right_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Right Shank","RShank",referential = "Anatomic"  )


        logging.debug(" ---Shank  Offsets ---")
        logging.debug(" ---------------------")

        # shakRotation
        if self.mp.has_key("LeftShankRotation") and self.mp["LeftShankRotation"] != 0:
            self.mp_computed["LeftShankRotationOffset"]= -self.mp["LeftShankRotation"]
        else:
            self.getShankOffsets(side="left")

        if self.mp.has_key("RightShankRotation") and self.mp["RightShankRotation"] != 0:
            self.mp_computed["RightShankRotationOffset"]= self.mp["RightShankRotation"]
        else:
            self.getShankOffsets(side="right")

        # tibial Torsion

        if self.mp.has_key("LeftTibialTorsion") and self.mp["LeftTibialTorsion"] != 0: #   - check if TibialTorsion whithin main mp
            self.mp_computed["LeftTibialTorsionOffset"]= -self.mp["LeftTibialTorsion"]
            self.m_useLeftTibialTorsion=True
        else:
            if self.m_useLeftTibialTorsion: # if useTibialTorsion flag enable from a decorator
                self.getTibialTorsionOffset(side="left")
            else:
                self.mp_computed["LeftTibialTorsionOffset"]= 0

        #   right
        if self.mp.has_key("RightTibialTorsion") and self.mp["RightTibialTorsion"] != 0:
            self.mp_computed["RightTibialTorsionOffset"]= self.mp["RightTibialTorsion"]
            self.m_useRightTibialTorsion=True
        else:
            if self.m_useRightTibialTorsion:
                self.getTibialTorsionOffset(side="right")
            else:
                self.mp_computed["RightTibialTorsionOffset"]= 0


        # AbdAdd offset
        self.getAbdAddAnkleJointOffset(side="left")
        self.getAbdAddAnkleJointOffset(side="right")


        logging.debug(" --- Left Shank Proximal- AF calibration ---")
        logging.debug(" -------------------------------------------")
        #   shank Prox ( copy )
        self.updateSegmentFromCopy("Left Shank Proximal", self.getSegment("Left Shank")) # look out . I copied the shank instance and rename it
        self._left_shankProximal_AnatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Left Shank Proximal","LShankProx",referential = "Anatomic"  )

        logging.debug(" --- Right Shank Proximal- AF calibration ---")
        logging.debug(" --------------------------------------------")
        self.updateSegmentFromCopy("Right Shank Proximal", self.getSegment("Right Shank"))
        self._right_shankProximal_AnatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Right Shank Proximal","RShankProx",referential = "Anatomic"  )

        # ---  FOOT segments
        # ---------------
        # need anatomical flexion axis of the shank.


        logging.info(" --- Left Hind Foot  - TF calibration ---")
        logging.info(" -----------------------------------------")
        self._leftHindFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Left HindFoot","LHindFootUncorrected",referential = "technic"  )

        logging.info(" --- Left Hind Foot  - AF calibration ---")
        logging.info(" -----------------------------------------")
        self._leftHindFoot_anatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Left HindFoot","LHindFoot",referential = "Anatomical"  )

        logging.info(" --- Hind foot Offset---")
        logging.info(" -----------------------")
        self.getHindFootOffset(side = "Left")

        # --- fore foot
        # ----------------
        logging.info(" --- Left Fore Foot  - TF calibration ---")
        logging.info(" -----------------------------------------")
        self._leftForeFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Left ForeFoot","LTechnicForeFoot",referential = "Technical"  )

        logging.info(" --- Left Fore Foot  - AF calibration ---")
        logging.info(" -----------------------------------------")
        self._leftForeFoot_anatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Left ForeFoot","LForeFoot",referential = "Anatomical"  )


        logging.info(" --- Right Hind Foot  - TF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightHindFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Right HindFoot","RHindFootUncorrected",referential = "technic"  )

        logging.info(" --- Right Hind Foot  - AF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightHindFoot_anatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Right HindFoot","RHindFoot",referential = "Anatomical"  )

        logging.info(" --- Hind foot Offset---")
        logging.info(" -----------------------")
        self.getHindFootOffset(side = "Right")

        # --- fore foot
        # ----------------
        logging.info(" --- Right Fore Foot  - TF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightForeFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Right ForeFoot","RTechnicForeFoot",referential = "Technical"  )

        logging.info(" --- Right Fore Foot  - AF calibration ---")
        logging.info(" -----------------------------------------")
        self._rightForeFoot_anatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        if "useDisplayPyCGM2_CoordinateSystem" in options.keys():
            self.displayStaticCoordinateSystem( aquiStatic, "Right ForeFoot","RForeFoot",referential = "Anatomical"  )


    # ---- Technical Referential Calibration
    def _leftHindFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        nFrames = aquiStatic.GetPointFrameNumber()

        seg=self.getSegment("Left HindFoot")

        # ---  additional markers and Update of the marker segment list

        # new markers

        # virtual CUN
        cun =  aquiStatic.GetPoint("LCUN").GetValues()
        valuesVirtualCun = np.zeros((nFrames,3))
        for i in range(0,nFrames):
            valuesVirtualCun[i,:] = np.array([cun[i,0], cun[i,1], cun[i,2]-self.mp["LeftToeOffset"]])

        btkTools.smartAppendPoint(aquiStatic,"LvCUN",valuesVirtualCun,desc="cun Registrate")

        # update marker list
        seg.addMarkerLabel("LAJC")         # required markers
        seg.addMarkerLabel("LvCUN")


        # --- technical frame selection and construction
        tf=seg.getReferential("TF")


        pt1=aquiStatic.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictRef["Left HindFoot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Left Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)


        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left HindFoot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

    def _rightHindFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        nFrames = aquiStatic.GetPointFrameNumber()

        seg=self.getSegment("Right HindFoot")

        # ---  additional markers and Update of the marker segment list

        # new markers

        # virtual CUN
        cun =  aquiStatic.GetPoint("RCUN").GetValues()
        valuesVirtualCun = np.zeros((nFrames,3))
        for i in range(0,nFrames):
            valuesVirtualCun[i,:] = np.array([cun[i,0], cun[i,1], cun[i,2]-self.mp["RightToeOffset"]])

        btkTools.smartAppendPoint(aquiStatic,"RvCUN",valuesVirtualCun,desc="cun Registrate")

        # update marker list
        seg.addMarkerLabel("RAJC")         # required markers
        seg.addMarkerLabel("RvCUN")


        # --- technical frame selection and construction
        tf=seg.getReferential("TF")


        pt1=aquiStatic.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictRef["Right HindFoot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)


        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right HindFoot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")



    def _leftForeFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        nFrames = aquiStatic.GetPointFrameNumber()


        seg=self.getSegment("Left ForeFoot")

        # ---  additional markers and Update of the marker segment list

        # new markers ( RvTOE - RvD5M)
        toe =  aquiStatic.GetPoint("LTOE").GetValues()
        d5 =  aquiStatic.GetPoint("LD5M").GetValues()

        valuesVirtualToe = np.zeros((nFrames,3))
        valuesVirtualD5 = np.zeros((nFrames,3))
        for i in range(0,nFrames):
            valuesVirtualToe[i,:] = np.array([toe[i,0], toe[i,1], toe[i,2]-self.mp["LeftToeOffset"] ])#valuesVirtualCun[i,2]])#
            valuesVirtualD5 [i,:]= np.array([d5[i,0], d5[i,1], valuesVirtualToe[i,2]])

        btkTools.smartAppendPoint(aquiStatic,"LvTOE",valuesVirtualToe,desc="virtual")
        btkTools.smartAppendPoint(aquiStatic,"LvD5M",valuesVirtualD5,desc="virtual-flat ")

        # update marker list
        seg.addMarkerLabel("LvCUN")
        seg.addMarkerLabel("LvTOE")
        seg.addMarkerLabel("LvD5M")

        # --- technical frame selection and construction
        tf=seg.getReferential("TF")


        pt1=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # D5
        pt2=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # Toe

        if dictRef["Left ForeFoot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
           v= 100 * np.array([0.0, 0.0, 1.0])

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left ForeFoot"]["TF"]['sequence'])

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


        seg=self.getSegment("Right ForeFoot")

        # ---  additional markers and Update of the marker segment list

        # new markers ( RvTOE - RvD5M)
        toe =  aquiStatic.GetPoint("RTOE").GetValues()
        d5 =  aquiStatic.GetPoint("RD5M").GetValues()

        valuesVirtualToe = np.zeros((nFrames,3))
        valuesVirtualD5 = np.zeros((nFrames,3))
        for i in range(0,nFrames):
            valuesVirtualToe[i,:] = np.array([toe[i,0], toe[i,1], toe[i,2]-self.mp["RightToeOffset"] ])#valuesVirtualCun[i,2]])#
            valuesVirtualD5 [i,:]= np.array([d5[i,0], d5[i,1], valuesVirtualToe[i,2]])

        btkTools.smartAppendPoint(aquiStatic,"RvTOE",valuesVirtualToe,desc="virtual")
        btkTools.smartAppendPoint(aquiStatic,"RvD5M",valuesVirtualD5,desc="virtual-flat ")

        # update marker list
        seg.addMarkerLabel("RvCUN")
        seg.addMarkerLabel("RvTOE")
        seg.addMarkerLabel("RvD5M")

        # --- technical frame selection and construction
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # D5
        pt2=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # Toe

        if dictRef["Right ForeFoot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
           v= 100 * np.array([0.0, 0.0, 1.0])

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right ForeFoot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")






    def _leftHindFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd, options = None):

        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0



        seg=self.getSegment("Left HindFoot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left HindFoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left HindFoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Left HindFoot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Left HindFoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Left Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left HindFoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("leftFlatHindFoot" in options.keys() and options["leftFlatHindFoot"]):
            logging.warning("option (leftFlatHindFoot) enable")
            #pt2[2] = pt1[2]
            pt1[2] = pt2[2]
        else:
            logging.warning("option (leftFlatHindFoot) disable")


        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)


        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Left HindFoot"]['sequence'])

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

        # add TOE
        toePosition=aquiStatic.GetPoint(str("LTOE")).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        seg.anatomicalFrame.static.addNode("LTOE",toePosition,positionType="Global")


        # --- compute amthropo
        # length
        toe = seg.anatomicalFrame.static.getNode_byLabel("LTOE").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel("LHEE").m_local
        seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel("LTOE").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel("LHEE").m_global
        footLongAxis = (toe-hee)/np.linalg.norm(toe-hee)

        com = hee + 0.5 * seg.m_bsp["length"] * footLongAxis

        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")

    def _rightHindFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd, options = None):

        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0


        seg=self.getSegment("Right HindFoot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right HindFoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right HindFoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Right HindFoot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right HindFoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right HindFoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

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

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Right HindFoot"]['sequence'])

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

# add TOE
        toePosition=aquiStatic.GetPoint(str("RTOE")).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        seg.anatomicalFrame.static.addNode("RTOE",toePosition,positionType="Global")


        # --- compute amthropo
        # length
        toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_local
        seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_global
        footLongAxis = (toe-hee)/np.linalg.norm(toe-hee)

        com = hee + 0.5 * seg.m_bsp["length"] * footLongAxis

        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")

    def _leftForeFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):


        seg=self.getSegment("Left ForeFoot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Left ForeFoot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v= 100 * np.array([0.0, 0.0, 1.0])

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Left ForeFoot"]['sequence'])


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


    def _rightForeFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):


        seg=self.getSegment("Right ForeFoot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Right ForeFoot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v= 100 * np.array([0.0, 0.0, 1.0])

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)

        v=v/np.linalg.norm(v)

        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Right ForeFoot"]['sequence'])


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
    def getHindFootOffset(self, side = "Both"):


        if side == "Both" or side == "Left" :
            R = self.getSegment("Left HindFoot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)


            self.mp_computed["LeftStaticPlantFlexOffset"] = np.rad2deg(y)
            logging.debug(" LeftStaticPlantFlexOffset => %s " % str(self.mp_computed["LeftStaticPlantFlexOffset"]))


            self.mp_computed["LeftStaticRotOffset"] = np.rad2deg(x)
            logging.debug(" LeftStaticRotOffset => %s " % str(self.mp_computed["LeftStaticRotOffset"]))

        if side == "Both" or side == "Right" :
            R = self.getSegment("Right HindFoot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)


            self.mp_computed["RightStaticPlantFlexOffset"] = np.rad2deg(y)
            logging.debug(" RightStaticPlantFlexOffset => %s " % str(self.mp_computed["RightStaticPlantFlexOffset"]))

            self.mp_computed["RightStaticRotOffset"] = np.rad2deg(x)
            logging.debug(" RightStaticRotOffset => %s " % str(self.mp_computed["RightStaticRotOffset"]))


    #---- Motion -------
    def computeOptimizedSegmentMotion(self,aqui,segments, dictRef,dictAnat,motionMethod ):
        """

        """

        # ---remove all  direction marker from tracking markers.
        for seg in self.m_segmentCollection:
            selectedTrackingMarkers=list()
            for marker in seg.m_tracking_markers:
                if marker in self.__class__.MARKERS : # get class variable MARKER even from child
                    selectedTrackingMarkers.append(marker)
            seg.m_tracking_markers= selectedTrackingMarkers


        logging.debug("--- Segmental Least-square motion process ---")
        if "Pelvis" in segments:
            self._pelvis_motion(aqui, dictRef, motionMethod)
            self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))

        if "Left Thigh" in segments:
            self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Left Thigh"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Left Thigh",originLabel = originAnatomicalFrame)
            else:
                logging.info("[pyCGM2] no motion of the left thigh Anatomical Referential. OriginLabel unknown")


        if "Right Thigh" in segments:
            self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Right Thigh"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Right Thigh",originLabel = originAnatomicalFrame)

        if "Left Shank" in segments:
            self._left_shank_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Left Shank"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Left Shank",originLabel = originAnatomicalFrame)

        if "Right Shank" in segments:
            self._right_shank_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Right Shank"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Right Shank",originLabel = originAnatomicalFrame)

        if "Left HindFoot" in segments:
            self._left_foot_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Left HindFoot"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Left HindFoot",originLabel = originAnatomicalFrame)

        if "Right HindFoot" in segments:
            self._right_foot_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Right HindFoot"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Right HindFoot",originLabel = originAnatomicalFrame)

        if "Left ForeFoot" in segments:
            self._left_foot_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Left ForeFoot"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Left ForeFoot",originLabel = originAnatomicalFrame)

        if "Right ForeFoot" in segments:
            self._right_foot_motion_optimize(aqui, dictRef,motionMethod)
            originAnatomicalFrame = str(dictAnat["Right ForeFoot"]['labels'][3])
            if btkTools.isPointExist(aqui,originAnatomicalFrame):
                self._anatomical_motion(aqui,"Right ForeFoot",originLabel = originAnatomicalFrame)




    def computeMotion(self,aqui, dictRef,dictAnat, motionMethod,options=None ):
        """
        Compute Motion of both **Technical and Anatomical** coordinate systems

        :Parameters:

           - `aqui` (btkAcquisition) - acquisition instance of a dynamic trial
           - `dictRef` (dict) - dictionnary reporting markers and sequence use for building Technical coordinate system
           - `dictAnat` (dict) - dictionnary reporting markers and sequence use for building Anatomical coordinate system
           - `motionMethod` (pyCGM2.enums) - method use to compute segment pose
           - `options` (dict) - dictionnary use to pass options

        """
        logging.debug("=====================================================")
        logging.debug("===================  CGM MOTION   ===================")
        logging.debug("=====================================================")

        pigStaticProcessing= True if "pigStatic" in options.keys() and options["pigStatic"] else False


        if motionMethod == pyCGM2Enums.motionMethod.Determinist: #cmf.motionMethod.Native:

            #if not pigStaticProcessing:
            logging.debug(" - Pelvis - motion -")
            logging.debug(" -------------------")
            self._pelvis_motion(aqui, dictRef, dictAnat)

            logging.debug(" - Left Thigh - motion -")
            logging.debug(" -----------------------")
            self._left_thigh_motion(aqui, dictRef, dictAnat,options=options)

            if "enableLongitudinalRotation" in options.keys() and options["enableLongitudinalRotation"]:
                if self.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("KJC_SaraAxis"):
                    logging.debug("SARA axis found from the left thigh")

                    self._rotate_anatomical_motion("Left Thigh",self.mp_computed["LeftKneeFuncCalibrationOffset"],
                                            aqui,options=options)

            # if dynaKad offset
            if self.mp_computed.has_key("LeftKneeDynaKadOffset") and self.mp_computed["LeftKneeDynaKadOffset"] != 0:
                logging.debug("Left DynaKad offset found. Anatomical referential rotated from dynaKad offset")
                self._rotate_anatomical_motion("Left Thigh",self.mp_computed["LeftKneeDynaKadOffset"],
                                            aqui,options=options)


            logging.debug(" - Right Thigh - motion -")
            logging.debug(" ------------------------")
            self._right_thigh_motion(aqui, dictRef, dictAnat,options=options)

            if "enableLongitudinalRotation" in options.keys() and options["enableLongitudinalRotation"]:
                if self.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("KJC_SaraAxis"):
                    logging.debug("SARA axis found from the Right thigh")

                    self._rotate_anatomical_motion("Right Thigh",self.mp_computed["RightKneeFuncCalibrationOffset"],
                                            aqui,options=options)

            # if dynaKad offset
            if self.mp_computed.has_key("RightKneeDynaKadOffset") and self.mp_computed["RightKneeDynaKadOffset"] != 0:
                logging.debug("Right DynaKad offset found. Anatomical referential rotated from dynaKad offset")
                self._rotate_anatomical_motion("Right Thigh",self.mp_computed["RightKneeDynaKadOffset"],
                                            aqui,options=options)


            logging.debug(" - Left Shank - motion -")
            logging.debug(" -----------------------")
            self._left_shank_motion(aqui, dictRef, dictAnat,options=options)
            logging.debug(" - Left Shank-proximal - motion -")
            logging.debug(" --------------------------------")
            self._left_shankProximal_motion(aqui,dictAnat,options=options)

            logging.debug(" - Right Shank - motion -")
            logging.debug(" ------------------------")
            self._right_shank_motion(aqui, dictRef, dictAnat,options=options)

            logging.debug(" - Right Shank-proximal - motion -")
            logging.debug(" ---------------------------------")
            self._right_shankProximal_motion(aqui,dictAnat,options=options)

            logging.info(" - Left HindFoot - motion -")
            logging.info(" ---------------------------")
            self._left_hindFoot_motion(aqui, dictRef, dictAnat, options=options)

            logging.info(" - Left ForeFoot - motion -")
            logging.info(" ---------------------------")
            self._left_foreFoot_motion(aqui, dictRef, dictAnat, options=options)


            logging.info(" - Right Hindfoot - motion -")
            logging.info(" ---------------------------")
            self._right_hindFoot_motion(aqui, dictRef, dictAnat, options=options)

            logging.info(" - Right ForeFoot - motion -")
            logging.info(" ---------------------------")
            self._right_foreFoot_motion(aqui, dictRef, dictAnat, options=options)



        if motionMethod == pyCGM2Enums.motionMethod.Sodervisk:

            # ---remove all  direction marker from tracking markers.
            for seg in self.m_segmentCollection:

                selectedTrackingMarkers=list()

                for marker in seg.m_tracking_markers:
                    if marker in self.__class__.MARKERS :
                        selectedTrackingMarkers.append(marker)

                seg.m_tracking_markers= selectedTrackingMarkers


            logging.debug("--- Segmental Least-square motion process ---")
            self._pelvis_motion_optimize(aqui, dictRef, motionMethod)
            self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))

            self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Thigh",originLabel = str(dictAnat["Left Thigh"]['labels'][3]))

            if "enableLongitudinalRotation" in options.keys() and options["enableLongitudinalRotation"]:
                if self.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("KJC_SaraAxis"):
                    logging.debug("SARA axis found from the left thigh")

                    self._rotate_anatomical_motion("Left Thigh",self.mp_computed["LeftKneeFuncCalibrationOffset"],
                                            aqui,options=options)

            # if dynaKad offset
            if self.mp_computed.has_key("LeftKneeDynaKadOffset") and self.mp_computed["LeftKneeDynaKadOffset"] != 0:
                logging.debug("Left DynaKad offset found. Anatomical referential rotated from dynaKad offset")
                self._rotate_anatomical_motion("Left Thigh",self.mp_computed["LeftKneeDynaKadOffset"],
                                            aqui,options=options)

            self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Thigh",originLabel = str(dictAnat["Right Thigh"]['labels'][3]))


            if "enableLongitudinalRotation" in options.keys() and options["enableLongitudinalRotation"]:
                if self.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("KJC_SaraAxis"):
                    logging.debug("SARA axis found from the Right thigh")

                    self._rotate_anatomical_motion("Right Thigh",self.mp_computed["RightKneeFuncCalibrationOffset"],
                                            aqui,options=options)

            # if dynaKad offset
            if self.mp_computed.has_key("RightKneeDynaKadOffset") and self.mp_computed["RightKneeDynaKadOffset"] != 0:
                logging.debug("Right DynaKad offset found. Anatomical referential rotated from dynaKad offset")
                self._rotate_anatomical_motion("Right Thigh",self.mp_computed["RightKneeDynaKadOffset"],
                                            aqui,options=options)


            self._left_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Shank",originLabel = str(dictAnat["Left Shank"]['labels'][3]))

            self._right_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Shank",originLabel = str(dictAnat["Right Shank"]['labels'][3]))

            # hindFoot
            self._leftHindFoot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left HindFoot",originLabel = str(dictAnat["Left HindFoot"]['labels'][3]))

            self._rightHindFoot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right HindFoot",originLabel = str(dictAnat["Right HindFoot"]['labels'][3]))

            # foreFoot
            self._leftForeFoot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left ForeFoot",originLabel = str(dictAnat["Left ForeFoot"]['labels'][3]))

            self._rightForeFoot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right ForeFoot",originLabel = str(dictAnat["Right ForeFoot"]['labels'][3]))





        logging.debug("--- Display Coordinate system ---")
        logging.debug(" --------------------------------")



        if not pigStaticProcessing:
            if "usePyCGM2_coordinateSystem" in options.keys() and options["usePyCGM2_coordinateSystem"]:
                self.displayMotionCoordinateSystem( aqui,  "Pelvis" , "Pelvis" )
                self.displayMotionCoordinateSystem( aqui,  "Left Thigh" , "LThigh" )
                self.displayMotionCoordinateSystem( aqui,  "Right Thigh" , "RThigh" )
                self.displayMotionCoordinateSystem( aqui,  "Left Shank" , "LShank" )
                self.displayMotionCoordinateSystem( aqui,  "Left Shank Proximal" , "LShankProx" )
                self.displayMotionCoordinateSystem( aqui,  "Right Shank" , "RShank" )
                self.displayMotionCoordinateSystem( aqui,  "Right Shank Proximal" , "RShankProx" )
                self.displayMotionCoordinateSystem( aqui,  "Left Foot" , "LFoot" )
                self.displayMotionCoordinateSystem( aqui,  "Right Foot" , "RFoot" )
                self.displayMotionCoordinateSystem( aqui,  "Left HindFoot" , "LHindFoot")
                self.displayMotionCoordinateSystem( aqui,  "Right HindFoot" , "RHindFoot")
                self.displayMotionCoordinateSystem( aqui,  "Left ForeFoot" , "LForeFoot")
                self.displayMotionCoordinateSystem( aqui,  "Right ForeFoot" , "RForeFoot")


            else:

                self.displayMotionViconCoordinateSystem(aqui,"Pelvis","PELO","PELA","PELL","PELP")
                self.displayMotionViconCoordinateSystem(aqui,"Left Thigh","LFEO","LFEA","LFEL","LFEP")
                self.displayMotionViconCoordinateSystem(aqui,"Right Thigh","RFEO","RFEA","RFEL","RFEP")
                self.displayMotionViconCoordinateSystem(aqui,"Left Shank","LTIO","LTIA","LTIL","LTIP")
                self.displayMotionViconCoordinateSystem(aqui,"Left Shank Proximal","LTPO","LTPA","LTPL","LTPP")
                self.displayMotionViconCoordinateSystem(aqui,"Right Shank","RTIO","RTIA","RTIL","RTIP")
                self.displayMotionViconCoordinateSystem(aqui,"Right Shank Proximal","RTPO","RTPA","RTPL","RTPP")
                self.displayMotionViconCoordinateSystem(aqui,"Left HindFoot","LFOO","LFOA","LFOL","LFOP")
                self.displayMotionViconCoordinateSystem(aqui,"Right HindFoot","RFOO","RFOA","RFOL","RFOP")
                self.displayMotionViconCoordinateSystem(aqui,"Left ForeFoot","LTOO","LTOA","LTOL","LTOP")
                self.displayMotionViconCoordinateSystem(aqui,"Right ForeFoot","RTOO","RTOA","RTOL","RTOP")


    # ----- native motion ------
    def _left_hindFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Left HindFoot")



        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][0])).GetValues()[i,:] #cun
            pt2=aqui.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc

            if dictRef["Left HindFoot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Left Shank").anatomicalFrame.motion[i].m_axisY



            ptOrigin=aqui.GetPoint(str(dictRef["Left HindFoot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)

            v=v/np.linalg.norm(v)

            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left HindFoot"]["TF"]['sequence'])

            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))

        # --- RvCUN
        btkTools.smartAppendPoint(aqui,"LvCUN",seg.getReferential("TF").getNodeTrajectory("LvCUN"),desc="from hindFoot" )

        # --- motion of the technical referential
        seg.anatomicalFrame.motion=[]
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left HindFoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)

            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(frame))


    def _left_foreFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Left ForeFoot")


        # --- motion of the technical referential

        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        #computation
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][2])).GetValues()[i,:]

            if dictRef["Left ForeFoot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Left Shank").anatomicalFrame.motion[i].m_axisY

            ptOrigin=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)

            v=(pt3-pt1)
            v=v/np.linalg.norm(v)

            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left ForeFoot"]["TF"]['sequence'])

            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))


        # --- motion of new markers
        btkTools.smartAppendPoint(aqui,"LvTOE",seg.getReferential("TF").getNodeTrajectory("LvTOE") )
        btkTools.smartAppendPoint(aqui,"LvD5M",seg.getReferential("TF").getNodeTrajectory("LvD5M") )

        # --- motion of the anatomical referential



        seg.anatomicalFrame.motion=[]
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left ForeFoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(frame))



    def _right_hindFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Right HindFoot")



        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][0])).GetValues()[i,:] #cun
            pt2=aqui.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc

            if dictRef["Right HindFoot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Right Shank").anatomicalFrame.motion[i].m_axisY



            ptOrigin=aqui.GetPoint(str(dictRef["Right HindFoot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)

            v=v/np.linalg.norm(v)

            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right HindFoot"]["TF"]['sequence'])

            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))

        # --- RvCUN
        btkTools.smartAppendPoint(aqui,"RvCUN",seg.getReferential("TF").getNodeTrajectory("RvCUN"),desc="from hindFoot" )

        # --- motion of the technical referential
        seg.anatomicalFrame.motion=[]
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right HindFoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(frame))


    def _right_foreFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Right ForeFoot")


        # --- motion of the technical referential

        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        #computation
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][2])).GetValues()[i,:]

            if dictRef["Right ForeFoot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Right Shank").anatomicalFrame.motion[i].m_axisY

            ptOrigin=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)

            v=(pt3-pt1)
            v=v/np.linalg.norm(v)

            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right ForeFoot"]["TF"]['sequence'])

            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))


        # --- motion of new markers
        btkTools.smartAppendPoint(aqui,"RvTOE",seg.getReferential("TF").getNodeTrajectory("RvTOE") )
        btkTools.smartAppendPoint(aqui,"RvD5M",seg.getReferential("TF").getNodeTrajectory("RvD5M") )

        # --- motion of the anatomical referential



        seg.anatomicalFrame.motion=[]
        frame=cfr.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right ForeFoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(frame))


    # ----- least-square Segmental motion ------
    def _leftHindFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Left HindFoot")

        #  --- add RAJC if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "LAJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LAJC")
                    logging.debug("LAJC added to tracking marker list")

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
        frame=cfr.Frame()
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


                frame.setRotation(R)
                frame.setTranslation(tOri)
                frame.m_axisX=R[:,0]
                frame.m_axisY=R[:,1]
                frame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))

        # --- vCUN and AJC
        btkTools.smartAppendPoint(aqui,"LAJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("LAJC"),desc="opt from hindfoot" )
        btkTools.smartAppendPoint(aqui,"LvCUN",seg.getReferential("TF").getNodeTrajectory("LvCUN"),desc="opt from hindfoot" )

        # remove AJC from list of tracking markers
        if "LAJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("LAJC")


    def _leftForeFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Left ForeFoot")

        #  --- add RvCUN if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "LvCUN" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LvCUN")
                    logging.debug("LvCUN added to tracking marker list")

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
        frame=cfr.Frame()
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

                frame.setRotation(R)
                frame.setTranslation(tOri)
                frame.m_axisX=R[:,0]
                frame.m_axisY=R[:,1]
                frame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))

        # --- motion of new markers
        # --- vCUN and AJC
        btkTools.smartAppendPoint(aqui,"LvCUN-ForeFoot",seg.getReferential("TF").getNodeTrajectory("LvCUN"),desc="opt from forefoot" )

    def _rightHindFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Right HindFoot")

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
        frame=cfr.Frame()
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

                frame.setRotation(R)
                frame.setTranslation(tOri)
                frame.m_axisX=R[:,0]
                frame.m_axisY=R[:,1]
                frame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))

        # --- vCUN and AJC
        btkTools.smartAppendPoint(aqui,"RAJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("RAJC"),desc="opt from hindfoot" )
        btkTools.smartAppendPoint(aqui,"RvCUN",seg.getReferential("TF").getNodeTrajectory("RvCUN"),desc="opt from hindfoot" )

        # remove AJC from list of tracking markers
        if "RAJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("RAJC")


    def _rightForeFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Right ForeFoot")

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
        frame=cfr.Frame()
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

                frame.setRotation(R)
                frame.setTranslation(tOri)
                frame.m_axisX=R[:,0]
                frame.m_axisY=R[:,1]
                frame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(frame))

        # --- motion of new markers
        # --- vCUN and AJC
        btkTools.smartAppendPoint(aqui,"RvCUN-ForeFoot",seg.getReferential("TF").getNodeTrajectory("RvCUN"),desc="opt from forefoot" )


    # ---- finalize methods ------
    def finalizeJCS(self,jointLabel,jointValues):
        """
            Finalize joint angles for clinical interpretation

            :Parameters:
               - `SegmentLabel` (str) - label of the segment
               - `jointValues` (numpy.array(:,3)) - angle values

        """
        values = np.zeros((jointValues.shape))

        if jointLabel not in  ["RForeFoot","LForeFoot"]:
            values = super(CGM2_4LowerLimbs, self).finalizeJCS(jointLabel, jointValues)
        elif jointLabel == "LForeFoot": # TODO : check
            values[:,0] = -1.0 * np.rad2deg(  jointValues[:,0])
            values[:,1] = -1.0*np.rad2deg(  jointValues[:,2])
            values[:,2] = -1.0*np.rad2deg(  jointValues[:,1])

        elif jointLabel == "RForeFoot": # TODO : check
            values[:,0] = -1.0 * np.rad2deg(  jointValues[:,0])
            values[:,1] = np.rad2deg(  jointValues[:,2])
            values[:,2] = np.rad2deg(  jointValues[:,1])




#        if jointLabel == "LHip" :  #LHPA=<-1(LHPA),-2(LHPA),-3(LHPA)> {*flexion, adduction, int. rot.			*}
#            values[:,0] = - np.rad2deg(  jointValues[:,0])
#            values[:,1] = - np.rad2deg(  jointValues[:,1])
#            values[:,2] = - np.rad2deg(  jointValues[:,2])
#
#        elif jointLabel == "LKnee" : # LKNA=<1(LKNA),-2(LKNA),-3(LKNA)-$LTibialTorsion>  {*flexion, varus, int. rot.		*}
#            values[:,0] = np.rad2deg(  jointValues[:,0])
#            values[:,1] = -np.rad2deg(  jointValues[:,1])
#            values[:,2] = -np.rad2deg(  jointValues[:,2])
#
#        elif jointLabel == "RHip" :  # RHPA=<-1(RHPA),2(RHPA),3(RHPA)>   {*flexion, adduction, int. rot.			*}
#            values[:,0] = - np.rad2deg(  jointValues[:,0])
#            values[:,1] =  np.rad2deg(  jointValues[:,1])
#            values[:,2] =  np.rad2deg(  jointValues[:,2])
#
#        elif jointLabel == "RKnee" : #  RKNA=<1(RKNA),2(RKNA),3(RKNA)-$RTibialTorsion>    {*flexion, varus, int. rot.		*}
#            values[:,0] = np.rad2deg(  jointValues[:,0])
#            values[:,1] = np.rad2deg(  jointValues[:,1])
#            values[:,2] = np.rad2deg(  jointValues[:,2])
#
#        elif jointLabel == "LAnkle":
#            values[:,0] = -1.0* np.rad2deg(  jointValues[:,0]  + np.radians(90))
#            values[:,1] = -1.0*np.rad2deg(  jointValues[:,2])
#            values[:,2] =  -1.0*np.rad2deg(  jointValues[:,1])
#
#        elif jointLabel == "RAnkle":
#            values[:,0] = -1.0* np.rad2deg(  jointValues[:,0]  + np.radians(90))
#            values[:,1] = np.rad2deg(  jointValues[:,2])
#            values[:,2] =  np.rad2deg(  jointValues[:,1])
#
#        elif jointLabel == "LForeFoot": # TODO : check
#            values[:,0] = -1.0 * np.rad2deg(  jointValues[:,0])
#            values[:,1] = -1.0*np.rad2deg(  jointValues[:,2])
#            values[:,2] = -1.0*np.rad2deg(  jointValues[:,1])
#
#        elif jointLabel == "RForeFoot": # TODO : check
#            values[:,0] = -1.0 * np.rad2deg(  jointValues[:,0])
#            values[:,1] = np.rad2deg(  jointValues[:,2])
#            values[:,2] = np.rad2deg(  jointValues[:,1])
#
#        else:
#            values[:,0] = np.rad2deg(  jointValues[:,0])
#            values[:,1] = np.rad2deg(  jointValues[:,1])
#            values[:,2] = np.rad2deg(  jointValues[:,2])

        return values

    # --- opensim --------
    def opensimGeometry(self):
        """
        TODO require : joint name from opensim -> find alternative

        rather a class method than a method instance
        """


        out={}
        out["hip_r"]= {"joint label":"RHJC", "proximal segment label":"Pelvis", "distal segment label":"Right Thigh" }
        out["knee_r"]= {"joint label":"RKJC", "proximal segment label":"Right Thigh", "distal segment label":"Right Shank" }
        out["ankle_r"]= {"joint label":"RAJC", "proximal segment label":"Right Shank", "distal segment label":"Right HindFoot" }
        out["mtp_r"]= {"joint label":"RvCUN", "proximal segment label":"Right HindFoot", "distal segment label":"Right ForeFoot" }


        out["hip_l"]= {"joint label":"LHJC", "proximal segment label":"Pelvis", "distal segment label":"Left Thigh" }
        out["knee_l"]= {"joint label":"LKJC", "proximal segment label":"Left Thigh", "distal segment label":"Left Shank" }
        out["ankle_l"]= {"joint label":"LAJC", "proximal segment label":"Left Shank", "distal segment label":"Left HindFoot" }
        out["mtp_l"]= {"joint label":"LvCUN", "proximal segment label":"Left HindFoot", "distal segment label":"Left ForeFoot" }

        return out

    def opensimIkTask(self,expert=False):
        out={}

        if expert:
          out={"LASI":0,
                 "LASI_posAnt":100,
                 "LASI_medLat":100,
                 "LASI_supInf":100,
                 "RASI":0,
                 "RASI_posAnt":100,
                 "RASI_medLat":100,
                 "RASI_supInf":100,
                 "LPSI":0,
                 "LPSI_posAnt":100,
                 "LPSI_medLat":100,
                 "LPSI_supInf":100,
                 "RPSI":0,
                 "RPSI_posAnt":100,
                 "RPSI_medLat":100,
                 "RPSI_supInf":100,

                 "RTHI":0,
                 "RTHI_posAnt":100,
                 "RTHI_medLat":100,
                 "RTHI_proDis":100,
                 "RKNE":0,
                 "RKNE_posAnt":100,
                 "RKNE_medLat":100,
                 "RKNE_proDis":100,
                 "RTIB":0,
                 "RTIB_posAnt":100,
                 "RTIB_medLat":100,
                 "RTIB_proDis":100,
                 "RANK":0,
                 "RANK_posAnt":100,
                 "RANK_medLat":100,
                 "RANK_proDis":100,
                 "RHEE":0,
                 "RHEE_supInf":100,
                 "RHEE_medLat":100,
                 "RHEE_proDis":100,
                 "RTOE":0,
                 "RTOE_supInf":100,
                 "RTOE_medLat":100,
                 "RTOE_proDis":100,

                 "RCUN":0,
                 "RCUN_supInf":100,
                 "RCUN_medLat":100,
                 "RCUN_proDis":100,

                 "RD1M":0,
                 "RD1M_supInf":100,
                 "RD1M_medLat":100,
                 "RD1M_proDis":100,

                 "RD5M":0,
                 "RD5M_supInf":100,
                 "RD5M_medLat":100,
                 "RD5M_proDis":100,


                 "LTHI":0,
                 "LTHI_posAnt":100,
                 "LTHI_medLat":100,
                 "LTHI_proDis":100,
                 "LKNE":0,
                 "LKNE_posAnt":100,
                 "LKNE_medLat":100,
                 "LKNE_proDis":100,
                 "LTIB":0,
                 "LTIB_posAnt":100,
                 "LTIB_medLat":100,
                 "LTIB_proDis":100,
                 "LANK":0,
                 "LANK_posAnt":100,
                 "LANK_medLat":100,
                 "LANK_proDis":100,
                 "LHEE":0,
                 "LHEE_supInf":100,
                 "LHEE_medLat":100,
                 "LHEE_proDis":100,
                 "LTOE":0,
                 "LTOE_supInf":100,
                 "LTOE_medLat":100,
                 "LTOE_proDis":100,

                 "LCUN":0,
                 "LCUN_supInf":100,
                 "LCUN_medLat":100,
                 "LCUN_proDis":100,

                 "LD1M":0,
                 "LD1M_supInf":100,
                 "LD1M_medLat":100,
                 "LD1M_proDis":100,

                 "LD5M":0,
                 "LD5M_supInf":100,
                 "LD5M_medLat":100,
                 "LD5M_proDis":100,


                 "LTHIAP":0,
                 "LTHIAP_posAnt":100,
                 "LTHIAP_medLat":100,
                 "LTHIAP_proDis":100,
                 "LTHIAD":0,
                 "LTHIAD_posAnt":100,
                 "LTHIAD_medLat":100,
                 "LTHIAD_proDis":100,
                 "RTHIAP":0,
                 "RTHIAP_posAnt":100,
                 "RTHIAP_medLat":100,
                 "RTHIAP_proDis":100,
                 "RTHIAD":0,
                 "RTHIAD_posAnt":100,
                 "RTHIAD_medLat":100,
                 "RTHIAD_proDis":100,
                 "LTIBAP":0,
                 "LTIBAP_posAnt":100,
                 "LTIBAP_medLat":100,
                 "LTIBAP_proDis":100,
                 "LTIBAD":0,
                 "LTIBAD_posAnt":100,
                 "LTIBAD_medLat":100,
                 "LTIBAD_proDis":100,
                 "RTIBAP":0,
                 "RTIBAP_posAnt":100,
                 "RTIBAP_medLat":100,
                 "RTIBAP_proDis":100,
                 "RTIBAD":0,
                 "RTIBAD_posAnt":100,
                 "RTIBAD_medLat":100,
                 "RTIBAD_proDis":100,

                 "LTHLD":0,
                 "LTHLD_posAnt":0,
                 "LTHLD_medLat":0,
                 "LTHLD_proDis":0,
                 "LPAT":0,
                 "LPAT_posAnt":0,
                 "LPAT_medLat":0,
                 "LPAT_proDis":0,
                 "RTHLD":0,
                 "RTHLD_posAnt":0,
                 "RTHLD_medLat":0,
                 "RTHLD_proDis":0,
                 "RPAT":0,
                 "RPAT_posAnt":0,
                 "RPAT_medLat":0,
                 "RPAT_proDis":0

                 }
        else:
            out={"LASI":100,
                 "RASI":100,
                 "LPSI":100,
                 "RPSI":100,
                 "RTHI":100,
                 "RKNE":100,
                 "RTHIAP":100,
                 "RTHIAD":100,
                 "RTIB":100,
                 "RANK":100,
                 "RTIBAP":100,
                 "RTIBAD":100,
                 "RHEE":100,
                 "RTOE":100,
                 "RCUN":100,
                 "RD1M":100,
                 "RD5M":100,
                 "LTHI":100,
                 "LKNE":100,
                 "LTHIAP":100,
                 "LTHIAD":100,
                 "LTIB":100,
                 "LANK":100,
                 "LTIBAP":100,
                 "LTIBAD":100,
                 "LHEE":100,
                 "LTOE":100,
                 "LCUN":100,
                 "LD1M":100,
                 "LD5M":100,
                 "RTHLD":0,
                 "RPAT":0,
                 "LTHLD":0,
                 "LPAT":0,

                 }

        return out

    # ----- vicon API -------
    def viconExport(self,NEXUS,acq,vskName,pointSuffix,staticProcessingFlag):
        """
            method exporting model outputs to Nexus UI

            This method exports :

                - joint centres as modelled-marker
                - angles
                - moment
                - force
                - power
                - bones


            :Parameters:
                - `NEXUS` () - Nexus environment
                - `vskName` (str) - name of the subject created in Nexus
                - `staticProcessingFlag` (bool`) : flag indicating only static model ouput will be export

        """

         # export JC
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LHJC", acq)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RHJC", acq)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LKJC", acq)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RKJC", acq)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LAJC", acq)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RAJC", acq)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LvCUN", acq)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RvCUN", acq)
        logging.debug("jc over")

        # export angles
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetType() == btk.btkPoint.Angle:
                if pointSuffix!="":
                    if pointSuffix in it.GetLabel():
                        nexusTools.appendAngleFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                else:
                    nexusTools.appendAngleFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)

        logging.debug("angles over")

        # bones
        # -------------
        nexusTools.appendBones(NEXUS,vskName,acq,"PEL", self.getSegment("Pelvis"),OriginValues = acq.GetPoint("midHJC").GetValues() )

        nexusTools.appendBones(NEXUS,vskName,acq,"LFE", self.getSegment("Left Thigh"),OriginValues = acq.GetPoint("LKJC").GetValues() )
        #nexusTools.appendBones(NEXUS,vskName,"LFEP", self.getSegment("Left Shank Proximal"),OriginValues = acq.GetPoint("LKJC").GetValues(),manualScale = 100 )
        nexusTools.appendBones(NEXUS,vskName,acq,"LTI", self.getSegment("Left Shank"),OriginValues = acq.GetPoint("LAJC").GetValues() )
        nexusTools.appendBones(NEXUS,vskName,acq,"LFO", self.getSegment("Left HindFoot"), OriginValues = acq.GetPoint("LHEE").GetValues() )
        nexusTools.appendBones(NEXUS,vskName,acq,"LTO", self.getSegment("Left ForeFoot"), OriginValues = acq.GetPoint("LvCUN").GetValues() )

        nexusTools.appendBones(NEXUS,vskName,acq,"RFE", self.getSegment("Right Thigh"),OriginValues = acq.GetPoint("RKJC").GetValues() )
        #nexusTools.appendBones(NEXUS,vskName,"RFEP", self.getSegment("Right Shank Proximal"),OriginValues = acq.GetPoint("RKJC").GetValues(),manualScale = 100 )
        nexusTools.appendBones(NEXUS,vskName,acq,"RTI", self.getSegment("Right Shank"),OriginValues = acq.GetPoint("RAJC").GetValues() )
        nexusTools.appendBones(NEXUS,vskName,acq,"RFO", self.getSegment("Right HindFoot") , OriginValues = acq.GetPoint("RHEE").GetValues()  )
        nexusTools.appendBones(NEXUS,vskName,acq,"RTO", self.getSegment("Right ForeFoot") ,  OriginValues = acq.GetPoint("RvCUN").GetValues())

        logging.debug("bones over")

        if not staticProcessingFlag:
            # export Force
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Force:
                    if pointSuffix!="":
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendForceFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendForceFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            logging.debug("force over")

            # export Moment
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Moment:
                    if pointSuffix!="":
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendMomentFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendMomentFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            logging.debug("Moment over")

            # export Moment
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Power:
                    if pointSuffix!="":
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendPowerFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendPowerFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            logging.debug("power over")
