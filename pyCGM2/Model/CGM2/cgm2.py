# -*- coding: utf-8 -*-
import numpy as np
import logging
import copy

try: 
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

from pyCGM2.Model.CGM2 import cgm

from pyCGM2 import enums
from  pyCGM2.Model import frame, motion, modelDecorator
from pyCGM2.Math import euler
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools


class CGM2_1(cgm.CGM1):
    """

    """

    #nativeCgm1 = True

    def __init__(self):
        super(CGM2_1, self).__init__()
        self.decoratedModel = False

        self.version = "CGM2.1"

    def __repr__(self):
        return "LowerLimb CGM2.1"

class CGM2_2(cgm.CGM1):
    """

    """


    def __init__(self):
        super(CGM2_2, self).__init__()
        self.decoratedModel = False

        self.version = "CGM2.2"

    def __repr__(self):
        return "LowerLimb CGM2.2"


class CGM2_3(cgm.CGM1):
    """
        implementation of the cgm2.3 skin marker added
    """

    LOWERLIMB_TRACKING_MARKERS= ["LASI", "RASI","RPSI", "LPSI",
               "LTHI","LKNE","LTHAP","LTHAD",
               "LTIB","LANK","LTIAP","LTIAD",
               "LHEE","LTOE",
               "RTHI","RKNE","RTHAP","RTHAD",
               "RTIB","RANK","RTIAP","RTIAD",
               "RHEE","RTOE"]

    def __init__(self):
        """Constructor

           - Run configuration internally
           - Initialize deviation data

        """
        super(CGM2_3, self).__init__()


        self.decoratedModel = False
        self.version = "CGM2.3"


        #self.__configure()


    def __repr__(self):
        return "LowerLimb CGM2.3"

    def _lowerLimbTrackingMarkers(self):
        return CGM2_3.LOWERLIMB_TRACKING_MARKERS


    def _lowerlimbConfigure(self):
        self.addSegment("Pelvis",0,enums.SegmentSide.Central,calibration_markers=[], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LKNE","LTHI","LTHAP","LTHAD"])
        self.addSegment("Right Thigh",4,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RKNE","RTHI","RTHAP","RTHAD"])
        self.addSegment("Left Shank",2,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LANK","LTIB","LTIAP","LTIAD"])
        self.addSegment("Left Shank Proximal",7,enums.SegmentSide.Left,cloneOf=True) # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Right Shank",5,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RANK","RTIB","RTIAP","RTIAD"])
        self.addSegment("Right Shank Proximal",8,enums.SegmentSide.Right,cloneOf=True)        # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Left Foot",3,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LHEE","LTOE"] )
        self.addSegment("Right Foot",6,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RHEE","RTOE"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])

        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ","LHJC")
        self.addJoint("LKnee","Left Thigh", "Left Shank","YXZ","LKJC")
        #self.addJoint("LKneeAngles_cgm","Left Thigh", "Left Shank","YXZ")
        self.addJoint("LAnkle","Left Shank", "Left Foot","YXZ","LAJC")
        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ","RHJC")
        self.addJoint("RKnee","Right Thigh", "Right Shank","YXZ","RKJC")
        self.addJoint("RAnkle","Right Shank", "Right Foot","YXZ","RAJC")

        # clinics
        self.setClinicalDescriptor("LHip",enums.DataType.Angle, [0,1,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LKnee",enums.DataType.Angle, [0,1,2],[+1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LAnkle",enums.DataType.Angle, [0,2,1],[-1.0,-1.0,-1.0], [ np.radians(90),0.0,0.0])
        self.setClinicalDescriptor("RHip",enums.DataType.Angle, [0,1,2],[-1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RKnee",enums.DataType.Angle, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RAnkle",enums.DataType.Angle, [0,2,1],[-1.0,+1.0,+1.0], [ np.radians(90),0.0,0.0])

        self.setClinicalDescriptor("LPelvis",enums.DataType.Angle,[0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RPelvis",enums.DataType.Angle,[0,1,2],[1.0,-1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Left Foot",enums.DataType.Angle,[0,2,1],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Right Foot",enums.DataType.Angle,[0,2,1],[1.0,-1.0,1.0], [0.0,0.0,0.0])

        # distal Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[-1.0,+1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        # proximal Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        # Global Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        # JCS Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)


        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)

        # JCS-dual Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)


        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)

    def _lowerLimbCalibrationProcedure(self,dictRef,dictRefAnatomical):
        dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
        dictRef["Left Thigh"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LKNE","LTHAP","LTHI","LKNE"]} }
        dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RKNE","RTHAP","RTHI","RKNE"]} }
        dictRef["Left Shank"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LANK","LTIAP","LTIB","LANK"]} }
        dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RANK","RTIAP","RTIB","RANK"]} }

        dictRef["Left Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LTOE","LAJC",None,"LAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis
        dictRef["Right Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RTOE","RAJC",None,"RAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis


        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]} # normaly : midHJC
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]} # origin = Proximal ( differ from native)
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]}
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]}
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}

        dictRefAnatomical["Left Foot"]={'sequence':"ZXiY", 'labels':  ["LTOE","LHEE",None,"LAJC"]}    # corrected foot
        dictRefAnatomical["Right Foot"]={'sequence':"ZXiY", 'labels':  ["RTOE","RHEE",None,"RAJC"]}    # corrected foot


    def _lowerLimbCoordinateSystemDefinitions(self):
        self.setCoordinateSystemDefinition( "Pelvis", "PELVIS", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Thigh", "LFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Thigh", "RFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Shank", "LTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Shank", "RTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Foot", "LFOOT", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Foot", "RFOOT", "Anatomic")

    def calibrate(self,aquiStatic, dictRef, dictAnatomic,  options=None): #

        super(CGM2_3, self).calibrate(aquiStatic, dictRef, dictAnatomic,  options=options)


    def opensimIkTask(self):
        out={}

        if self.staExpert:
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

                 "LTHAP":0,
                 "LTHAP_posAnt":100,
                 "LTHAP_medLat":100,
                 "LTHAP_proDis":100,
                 "LTHAD":0,
                 "LTHAD_posAnt":100,
                 "LTHAD_medLat":100,
                 "LTHAD_proDis":100,
                 "RTHAP":0,
                 "RTHAP_posAnt":100,
                 "RTHAP_medLat":100,
                 "RTHAP_proDis":100,
                 "RTHAD":0,
                 "RTHAD_posAnt":100,
                 "RTHAD_medLat":100,
                 "RTHAD_proDis":100,
                 "LTIAP":0,
                 "LTIAP_posAnt":100,
                 "LTIAP_medLat":100,
                 "LTIAP_proDis":100,
                 "LTIAD":0,
                 "LTIAD_posAnt":100,
                 "LTIAD_medLat":100,
                 "LTIAD_proDis":100,
                 "RTIAP":0,
                 "RTIAP_posAnt":100,
                 "RTIAP_medLat":100,
                 "RTIAP_proDis":100,
                 "RTIAD":0,
                 "RTIAD_posAnt":100,
                 "RTIAD_medLat":100,
                 "RTIAD_proDis":100,

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
                 "RTHAP":100,
                 "RTHAD":100,
                 "RTIB":100,
                 "RANK":100,
                 "RTIAP":100,
                 "RTIAD":100,
                 "RHEE":100,
                 "RTOE":100,
                 "LTHI":100,
                 "LKNE":100,
                 "LTHAP":100,
                 "LTHAD":100,
                 "LTIB":100,
                 "LANK":100,
                 "LTIAP":100,
                 "LTIAD":100,
                 "LHEE":100,
                 "LTOE":100,
                 "RTHLD":0,
                 "RPAT":0,
                 "LTHLD":0,
                 "LPAT":0
                 }

        return out

class CGM2_4(CGM2_3):

    ANALYSIS_KINEMATIC_LABELS_DICT ={ 'Left': ["LHipAngles","LKneeAngles","LAnkleAngles","LFootProgressAngles","LPelvisAngles","LForeFoot"],
                       'Right': ["RHipAngles","RKneeAngles","RAnkleAngles","RFootProgressAngles","RPelvisAngles","LForeFoot"]}

    ANALYSIS_KINETIC_LABELS_DICT ={ 'Left': ["LHipMoment","LKneeMoment","LAnkleMoment","LHipPower","LKneePower","LAnklePower"],
                          'Right': ["RHipMoment","RKneeMoment","RAnkleMoment","RHipPower","RKneePower","RAnklePower"]}

    LOWERLIMB_TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI",
               "LTHI","LKNE","LTHAP","LTHAD",
               "LTIB","LANK","LTIAP","LTIAD",
               "LHEE","LTOE","LFMH","LVMH",
               "RTHI","RKNE","RTHAP","RTHAD",
               "RTIB","RANK","RTIAP","RTIAD",
               "RHEE","RTOE","RFMH","RVMH"]

    LOWERLIMB_SEGMENTS = cgm.CGM1.LOWERLIMB_SEGMENTS + ["Left ForeFoot","Right ForeFoot"]

    LOWERLIMB_JOINTS = cgm.CGM1.LOWERLIMB_JOINTS + ["LForeFoot", "RForeFoot"]



    def __init__(self):
        """Constructor

           - Run configuration internally
           - Initialize deviation data

        """
        super(CGM2_4, self).__init__()

        self.decoratedModel = False

        self.version = "CGM2.4"

        #self.__configure()

    def __repr__(self):
        return "LowerLimb CGM2.4"

    def _lowerLimbTrackingMarkers(self):
        return CGM2_4.LOWERLIMB_TRACKING_MARKERS

    def _lowerlimbConfigure(self):
        self.addSegment("Pelvis",0,enums.SegmentSide.Central,calibration_markers=[], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LKNE","LTHI","LTHAP","LTHAD"])
        self.addSegment("Right Thigh",4,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RKNE","RTHI","RTHAP","RTHAD"])
        self.addSegment("Left Shank",2,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LANK","LTIB","LTIAP","LTIAD"])
        self.addSegment("Left Shank Proximal",7,enums.SegmentSide.Left,cloneOf=True) # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Right Shank",5,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RANK","RTIB","RTIAP","RTIAD"])
        self.addSegment("Right Shank Proximal",8,enums.SegmentSide.Right,cloneOf=True)        # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Left Foot",6,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LHEE","LTOE"])
        self.addSegment("Left ForeFoot",7,enums.SegmentSide.Left,calibration_markers=["LSMH"], tracking_markers = ["LFMH","LVMH"])
        self.addSegment("Right Foot",6,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RHEE","RTOE"])
        self.addSegment("Right ForeFoot",7,enums.SegmentSide.Right,calibration_markers=["RSMH"], tracking_markers = ["RFMH","RVMH"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])

        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ","LHJC")
        self.addJoint("LKnee","Left Thigh", "Left Shank","YXZ","LKJC")
        self.addJoint("LAnkle","Left Shank", "Left Foot","YXZ","LAJC")
        self.addJoint("LForeFoot","Left Foot", "Left ForeFoot","YXZ","LFJC")

        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ","RHJC")
        self.addJoint("RKnee","Right Thigh", "Right Shank","YXZ","RKJC")
        self.addJoint("RAnkle","Right Shank", "Right Foot","YXZ","RAJC")
        self.addJoint("RForeFoot","Right Foot", "Right ForeFoot","YXZ","RFJC")

        # clinics
        self.setClinicalDescriptor("LHip",enums.DataType.Angle, [0,1,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LKnee",enums.DataType.Angle, [0,1,2],[+1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LAnkle",enums.DataType.Angle, [0,2,1],[-1.0,-1.0,-1.0], [ np.radians(90),0.0,0.0])
        self.setClinicalDescriptor("RHip",enums.DataType.Angle, [0,1,2],[-1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RKnee",enums.DataType.Angle, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RAnkle",enums.DataType.Angle, [0,2,1],[-1.0,+1.0,+1.0], [ np.radians(90),0.0,0.0])

        self.setClinicalDescriptor("LForeFoot",enums.DataType.Angle, [0,2,1],[-1.0,-1.0,-1.0], [0.0,0.0,0.0]) # TODO check
        self.setClinicalDescriptor("RForeFoot",enums.DataType.Angle, [0,2,1],[-1.0,+1.0,+1.0], [ 0.0,0.0,0.0])


        self.setClinicalDescriptor("LPelvis",enums.DataType.Angle,[0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RPelvis",enums.DataType.Angle,[0,1,2],[1.0,-1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Left Foot",enums.DataType.Angle,[0,2,1],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Right Foot",enums.DataType.Angle,[0,2,1],[1.0,-1.0,1.0], [0.0,0.0,0.0])

        # distal Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[-1.0,+1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        # proximal Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        # Global Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        # JCS Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)


        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)

        # JCS-dual Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)


        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)


    def _lowerLimbCalibrationProcedure(self,dictRef,dictRefAnatomical):

        dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
        dictRef["Left Thigh"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LKNE","LTHAP","LTHI","LKNE"]} }
        dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RKNE","RTHAP","RTHI","RKNE"]} }
        dictRef["Left Shank"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LANK","LTIAP","LTIB","LANK"]} }
        dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RANK","RTIAP","RTIB","RANK"]} }

        # left Foot
        dictRef["Left Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LTOE","LAJC",None,"LAJC"]} }
        dictRef["Left ForeFoot"]={"TF" : {'sequence':"ZXiY", 'labels':    ["LFMH","LTOE","LVMH","LTOE"]} }

        # right foot
        dictRef["Right Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RTOE","RAJC",None,"RAJC"]} }
        dictRef["Right ForeFoot"]={"TF" : {'sequence':"ZXY", 'labels':    ["RFMH","RTOE","RVMH","RTOE"]} }


        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]}
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]}
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]}
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]}
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}

        # left Foot ( nothing yet)
        dictRefAnatomical["Left Foot"]= {'sequence':"ZXiY", 'labels':   ["LTOE","LHEE",None,"LAJC"]}
        dictRefAnatomical["Left ForeFoot"]= {'sequence':"ZXiY", 'labels':   ["LvSMH","LFJC","LVMH","LvSMH"]}

        # right foot
        dictRefAnatomical["Right Foot"]= {'sequence':"ZXiY", 'labels':   ["RTOE","RHEE",None,"RAJC"]}
        dictRefAnatomical["Right ForeFoot"]= {'sequence':"ZXY", 'labels':   ["RvSMH","RFJC","RVMH","RvSMH"]} # look out : use virtual Point


    def _lowerLimbCoordinateSystemDefinitions(self):
        self.setCoordinateSystemDefinition( "Pelvis", "PELVIS", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Thigh", "LFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Thigh", "RFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Shank", "LTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Shank", "RTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Foot", "LFOOT", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Foot", "RFOOT", "Anatomic")
        self.setCoordinateSystemDefinition( "Left ForeFoot", "LFOREFOOT", "Anatomic")
        self.setCoordinateSystemDefinition( "Right ForeFoot", "RFOREFOOT", "Anatomic")


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

        if self.m_bodypart !=enums.BodyPart.UpperLimb:
            if not self.decoratedModel:
                logging.debug(" Native CGM")
                if not btkTools.isPointExist(aquiStatic,"LKNE"):
                    btkTools.smartAppendPoint(aquiStatic,"LKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))
                if not btkTools.isPointExist(aquiStatic,"RKNE"):
                    btkTools.smartAppendPoint(aquiStatic,"RKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))

            else:
                logging.debug(" Decorated CGM")

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

            logging.debug(" --- Left Thigh - AF calibration ---")
            logging.debug(" -----------------------------------")
            self._left_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

            logging.debug(" --- Right Thigh - AF calibration ---")
            logging.debug(" ------------------------------------")
            self._right_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

            logging.debug(" --- Thigh Offsets ---")
            logging.debug(" --------------------")


            logging.debug(" ------Left-------")
            if self.mp.has_key("LeftThighRotation") and self.mp["LeftThighRotation"] != 0:
                self.mp_computed["LeftThighRotationOffset"]= -self.mp["LeftThighRotation"]

            else:
                self.getThighOffset(side="left")

            # management of Functional method
            if self.mp_computed["LeftKneeFuncCalibrationOffset"] != 0:
                offset = self.mp_computed["LeftKneeFuncCalibrationOffset"]
                # SARA
                if self.checkCalibrationProperty("LeftFuncKneeMethod","SARA"):
                    logging.debug("Left knee functional calibration : SARA ")
                # 2DOF
                elif self.checkCalibrationProperty("LeftFuncKneeMethod","2DOF"):
                    logging.debug("Left knee functional calibration : 2Dof ")
                self._rotateAnatomicalFrame("Left Thigh",offset,
                                            aquiStatic, dictAnatomic,frameInit,frameEnd)


            logging.debug(" ------Right-------")
            if self.mp.has_key("RightThighRotation") and self.mp["RightThighRotation"] != 0:
                self.mp_computed["RightThighRotationOffset"]= self.mp["RightThighRotation"]
            else:
                self.getThighOffset(side="right")

            # management of Functional method
            if self.mp_computed["RightKneeFuncCalibrationOffset"] != 0:
                offset = self.mp_computed["RightKneeFuncCalibrationOffset"]
                # SARA
                if self.checkCalibrationProperty("RightFuncKneeMethod","SARA"):
                    logging.debug("Left knee functional calibration : SARA ")
                # 2DOF
                elif self.checkCalibrationProperty("RightFuncKneeMethod","2DOF"):
                    logging.debug("Left knee functional calibration : 2Dof ")
                self._rotateAnatomicalFrame("Right Thigh",offset,
                                            aquiStatic, dictAnatomic,frameInit,frameEnd)



            logging.debug(" --- Left Shank - AF calibration ---")
            logging.debug(" -------------------------------")
            self._left_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)


            logging.debug(" --- Right Shank - AF calibration ---")
            logging.debug(" -------------------------------")
            self._right_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

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

            logging.debug(" --- Right Shank Proximal- AF calibration ---")
            logging.debug(" --------------------------------------------")
            self.updateSegmentFromCopy("Right Shank Proximal", self.getSegment("Right Shank"))
            self._right_shankProximal_AnatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame

            # ---  FOOT segments ----
            # -----------------------
            # need anatomical flexion axis of the shank.


            logging.debug(" --- Left Hind Foot  - TF calibration ---")
            logging.debug(" -----------------------------------------")
            self._leftHindFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)

            logging.debug(" --- Left Hind Foot  - AF calibration ---")
            logging.debug(" -----------------------------------------")
            self._leftHindFoot_anatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options)

            logging.debug(" --- Hind foot Offset---")
            logging.debug(" -----------------------")
            self.getHindFootOffset(side = "Left")

            logging.debug(" --- Left Fore Foot  - TF calibration ---")
            logging.debug(" -----------------------------------------")
            self._leftForeFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)


            logging.debug(" --- Left Fore Foot  - AF calibration ---")
            logging.debug(" -----------------------------------------")
            self._leftForeFoot_anatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)


            logging.debug(" --- Right Hind Foot  - TF calibration ---")
            logging.debug(" -----------------------------------------")
            self._rightHindFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)

            logging.debug(" --- Right Hind Foot  - AF calibration ---")
            logging.debug(" -----------------------------------------")
            self._rightHindFoot_anatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options)

            logging.debug(" --- Hind foot Offset---")
            logging.debug(" -----------------------")
            self.getHindFootOffset(side = "Right")

            # --- fore foot
            # ----------------
            logging.debug(" --- Right Fore Foot  - TF calibration ---")
            logging.debug(" -----------------------------------------")
            self._rightForeFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)

            logging.debug(" --- Right Fore Foot  - AF calibration ---")
            logging.debug(" -----------------------------------------")
            self._rightForeFoot_anatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)

        #------_upperLimb
        if self.m_bodypart == enums.BodyPart.LowerLimbTrunk:
            self._torso_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
            self._torso_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        if self.m_bodypart == enums.BodyPart.UpperLimb or self.m_bodypart == enums.BodyPart.FullBody:

            self._torso_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
            self._torso_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

            self._head_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
            self._head_AnatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

            self._clavicle_calibrate("Left",aquiStatic,dictRef,frameInit,frameEnd,options=options)
            self._clavicle_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)

            self._constructArmVirtualMarkers("Left", aquiStatic)

            self._upperArm_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd, options=options)
            self._foreArm_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd, options=options)

            self._upperArm_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)
            self._foreArm_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)

            self._hand_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd, options=options)
            self._hand_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)

            self._clavicle_calibrate("Right",aquiStatic,dictRef,frameInit,frameEnd,options=options)
            self._clavicle_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)

            self._constructArmVirtualMarkers("Right", aquiStatic)

            self._upperArm_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd, options=options)
            self._foreArm_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd, options=options)

            self._upperArm_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)
            self._foreArm_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)

            self._hand_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd, options=options)
            self._hand_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)



    # ---- Technical Referential Calibration
    def _leftHindFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            logging.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        seg=self.getSegment("Left Foot")

        # --- technical frame selection and construction ["LTOE","LAJC",None,"LAJC"]} }
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictRef["Left Foot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Left Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)


        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1 = np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2 = np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # ajc from prox
        node_prox = self.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC")
        tf.static.addNode("LAJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        if tf.static.isNodeExist("LFJC"):
            nodeFJC = tf.static.getNode_byLabel("LFJC")
        else:
            fjc = modelDecorator.footJointCentreFromMet(aquiStatic,"left",frameInit,frameEnd, markerDiameter = markerDiameter, offset = 0)
            tf.static.addNode("LFJC",fjc,positionType="Global",desc = "MET")
            nodeFJC = tf.static.getNode_byLabel("LFJC")

        btkTools.smartAppendPoint(aquiStatic,"LFJC",
            nodeFJC.m_global* np.ones((pfn,3)),
            desc=nodeFJC.m_desc)


    def _leftForeFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        pfn = aquiStatic.GetPointFrameNumber()


        seg=self.getSegment("Left ForeFoot")

        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #LMidMET
        pt2=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #LTOE = CUN
        pt3=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #LVMH

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v = (pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left ForeFoot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_tracking_markers: #["LFMH","LVMH","LSMH"]
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # fjc from prox
        node_prox = self.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LFJC")
        tf.static.addNode("LFJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        # virtual LvSMH
        local = tf.static.getNode_byLabel("LSMH").getLocal()
        projSMH  = np.array([ 0.0,local[1],local[2] ] )
        tf.static.addNode("LvSMH",projSMH,positionType="Local",desc = "proj (TOE-5-1) ")

        global_projSMH = tf.static.getGlobalPosition("LvSMH")
        btkTools.smartAppendPoint(aquiStatic,"LvSMH",
        global_projSMH*np.ones((pfn,3)),desc="proj (TOE-5-1)")


    def _leftHindFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd, options = None):
        """
        idem foot of cgm1
        """

        if "markerDiameter" in options.keys():
            logging.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        seg=self.getSegment("Left Foot")

        # --- Construction of the anatomical Referential ["LTOE","LHEE",None,"LAJC"ZXiY]
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Left Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Left Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("leftFlatFoot" in options.keys() and options["leftFlatFoot"]):
            logging.debug("option (leftFlatFoot) enable")
            #pt2[2] = pt1[2]
            pt1[2] = pt2[2]
        else:
            logging.debug("option (leftFlatFoot) disable")


        a1=(pt2-pt1)
        a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Left Foot"]['sequence'])

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
        y,x,z = euler.euler_yxz(trueRelativeMatrixAnatomic)
        rotX =np.array([[1,0,0],
                        [0,np.cos(x),-np.sin(x)],
                         [0,np.sin(x),np.cos(x)]])
        rotY =np.array([[np.cos(y),0,np.sin(y)],
                        [0,1,0],
                         [-np.sin(y),0,np.cos(y)]])

        relativeMatrixAnatomic = np.dot(rotY,rotX)
        tf.setRelativeMatrixAnatomic( relativeMatrixAnatomic)

        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # --- compute amthropo
        toePosition=aquiStatic.GetPoint(str("LSMH")).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        seg.anatomicalFrame.static.addNode("LSMH",toePosition,positionType="Global")

        # length
        toe = seg.anatomicalFrame.static.getNode_byLabel("LSMH").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel("LHEE").m_local
        seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel("LSMH").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel("LHEE").m_global
        footLongAxis = (toe-hee)/np.linalg.norm(toe-hee)

        com = hee + 0.5 * seg.m_bsp["length"] * footLongAxis

        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")


    def _leftForeFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd,options = None):

        seg=self.getSegment("Left ForeFoot")

        # --- Construction of the anatomical Referential ["LvSMH","LFJC",None,"LvSMH", ZYX]
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("leftFlatFoot" in options.keys() and options["leftFlatFoot"]):
            sequence = "ZYX"
            v= 100 * np.array([0.0, 0.0, 1.0])
        else:
            sequence = dictAnatomic["Left ForeFoot"]['sequence']
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left ForeFoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,sequence)

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic (np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())


    def _rightHindFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            logging.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        seg=self.getSegment("Right Foot")

        # --- technical frame selection and construction ["RTOE","RAJC",None,"RAJC"]} }
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictRef["Right Foot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)


        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # ajc
        node_prox = self.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC")
        tf.static.addNode("RAJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        if tf.static.isNodeExist("RFJC"):
            nodeFJC = tf.static.getNode_byLabel("RFJC")
        else:
            fjc = modelDecorator.footJointCentreFromMet(aquiStatic,"right",frameInit,frameEnd, markerDiameter = markerDiameter, offset = 0)
            tf.static.addNode("RFJC",fjc,positionType="Global",desc = "MET")
            nodeFJC = tf.static.getNode_byLabel("RFJC")

        btkTools.smartAppendPoint(aquiStatic,"RFJC",
            nodeFJC.m_global* np.ones((pfn,3)),
            desc=nodeFJC.m_desc)



    def _rightForeFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        pfn = aquiStatic.GetPointFrameNumber()

        seg=self.getSegment("Right ForeFoot")

        # --- technical frame selection and construction ["RMidMET","RTOE","RVMH","RTOE"]
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #RMidMET
        pt2=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #RTOE
        pt3=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #RVMH

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right ForeFoot"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_tracking_markers: #["RFMH","RVMH","RSMH"]
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # fjc from prox
        node_prox = self.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RFJC")
        tf.static.addNode("RFJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        # virtual RvSMH
        local = tf.static.getNode_byLabel("RSMH").getLocal()
        projSMH  = np.array([ 0.0,local[1],local[2] ] )
        tf.static.addNode("RvSMH",projSMH,positionType="Local",desc = "proj (TOE-5-1) ")

        global_projSMH = tf.static.getGlobalPosition("RvSMH")
        btkTools.smartAppendPoint(aquiStatic,"RvSMH",
        global_projSMH*np.ones((pfn,3)),desc="proj (TOE-5-1)")

    def _rightHindFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd, options = None):

        if "markerDiameter" in options.keys():
            logging.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0


        seg=self.getSegment("Right Foot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if dictAnatomic["Right Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
            logging.debug("option (rightFlatFoot) enable")
            #pt2[2] = pt1[2]
            pt1[2] = pt2[2]
        else:
            logging.debug("option (rightFlatFoot) disable")


        a1=(pt2-pt1)
        a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Right Foot"]['sequence'])

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
        y,x,z = euler.euler_yxz(trueRelativeMatrixAnatomic)
        rotX =np.array([[1,0,0],
                        [0,np.cos(x),-np.sin(x)],
                         [0,np.sin(x),np.cos(x)]])
        rotY =np.array([[np.cos(y),0,np.sin(y)],
                        [0,1,0],
                         [-np.sin(y),0,np.cos(y)]])

        relativeMatrixAnatomic = np.dot(rotY,rotX)
        tf.setRelativeMatrixAnatomic( relativeMatrixAnatomic)


        # --- node manager
        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())


        # --- compute amthropo
        toePosition=aquiStatic.GetPoint(str("RSMH")).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        seg.anatomicalFrame.static.addNode("RSMH",toePosition,positionType="Global")

        # length
        toe = seg.anatomicalFrame.static.getNode_byLabel("RSMH").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_local
        seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel("RSMH").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_global
        footLongAxis = (toe-hee)/np.linalg.norm(toe-hee)

        com = hee + 0.5 * seg.m_bsp["length"] * footLongAxis

        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")




    def _rightForeFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd,options = None):


        seg=self.getSegment("Right ForeFoot")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
            sequence = "ZYX"
            v= 100 * np.array([0.0, 0.0, 1.0])
        else:
            sequence = dictAnatomic["Right ForeFoot"]['sequence']
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right ForeFoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,sequence)


        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic (np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())


    #---- Offsets -------
    def getHindFootOffset(self, side = "Both"):


        if side == "Both" or side == "Left" :
            R = self.getSegment("Left Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = euler.euler_yxz(R)


            self.mp_computed["LeftStaticPlantFlexOffset"] = np.rad2deg(y)
            logging.debug(" LeftStaticPlantFlexOffset => %s " % str(self.mp_computed["LeftStaticPlantFlexOffset"]))


            self.mp_computed["LeftStaticRotOffset"] = np.rad2deg(x)
            logging.debug(" LeftStaticRotOffset => %s " % str(self.mp_computed["LeftStaticRotOffset"]))

        if side == "Both" or side == "Right" :
            R = self.getSegment("Right Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = euler.euler_yxz(R)


            self.mp_computed["RightStaticPlantFlexOffset"] = np.rad2deg(y)
            logging.debug(" RightStaticPlantFlexOffset => %s " % str(self.mp_computed["RightStaticPlantFlexOffset"]))

            self.mp_computed["RightStaticRotOffset"] = np.rad2deg(x)
            logging.debug(" RightStaticRotOffset => %s " % str(self.mp_computed["RightStaticRotOffset"]))


    #---- Motion -------
    def computeOptimizedSegmentMotion(self,aqui,segments, dictRef,dictAnat,motionMethod,options):
        """

        """
        # ---remove all  direction marker from tracking markers.
        if self.staExpert:
            for seg in self.m_segmentCollection:
                selectedTrackingMarkers=list()
                for marker in seg.m_tracking_markers:
                    if marker in self.__class__.TRACKING_MARKERS : # get class variable MARKER even from child
                        selectedTrackingMarkers.append(marker)
                seg.m_tracking_markers= selectedTrackingMarkers


        logging.debug("--- Segmental Least-square motion process ---")
        if "Pelvis" in segments:
            self._pelvis_motion_optimize(aqui, dictRef, motionMethod)
            self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))

        if "Left Thigh" in segments:
            self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Thigh",originLabel = "LKJC")


        if "Right Thigh" in segments:
            self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Thigh",originLabel = "RKJC")


        if "Left Shank" in segments:
            self._left_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Shank",originLabel = "LAJC")

        if "Right Shank" in segments:
            self._right_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Shank",originLabel = "RAJC")

        if "Left Foot" in segments:
            self._left_foot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Foot",originLabel = "LHEE")

        if "Right Foot" in segments:
            self._right_foot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Foot",originLabel = "RHEE")

        if "Left ForeFoot" in segments:
            self._left_foot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left ForeFoot",originLabel = "LFJC")

        if "Right ForeFoot" in segments:
            self._right_foot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right ForeFoot",originLabel = "RFJC")


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


        if motionMethod == enums.motionMethod.Determinist: #cmf.motionMethod.Native:

            if self.m_bodypart != enums.BodyPart.UpperLimb:
                #if not pigStaticProcessing:
                logging.debug(" - Pelvis - motion -")
                logging.debug(" -------------------")
                self._pelvis_motion(aqui, dictRef, dictAnat)

                logging.debug(" - Left Thigh - motion -")
                logging.debug(" -----------------------")
                self._left_thigh_motion(aqui, dictRef, dictAnat,options=options)

                # if rotation offset from knee functional calibration methods
                if self.mp_computed["LeftKneeFuncCalibrationOffset"]:
                    offset = self.mp_computed["LeftKneeFuncCalibrationOffset"]
                    self._rotate_anatomical_motion("Left Thigh",offset,
                                            aqui,options=options)


                logging.debug(" - Right Thigh - motion -")
                logging.debug(" ------------------------")
                self._right_thigh_motion(aqui, dictRef, dictAnat,options=options)

                if  self.mp_computed["RightKneeFuncCalibrationOffset"]:
                    offset = self.mp_computed["RightKneeFuncCalibrationOffset"]
                    self._rotate_anatomical_motion("Right Thigh",offset,
                                            aqui,options=options)


                logging.debug(" - Left Shank - motion -")
                logging.debug(" -----------------------")
                self._left_shank_motion(aqui, dictRef, dictAnat,options=options)

                logging.debug(" - Right Shank - motion -")
                logging.debug(" ------------------------")
                self._right_shank_motion(aqui, dictRef, dictAnat,options=options)


                logging.debug(" - Left HindFoot - motion -")
                logging.debug(" ---------------------------")
                self._left_hindFoot_motion(aqui, dictRef, dictAnat, options=options)

                logging.debug(" - Left ForeFoot - motion -")
                logging.debug(" ---------------------------")
                self._left_foreFoot_motion(aqui, dictRef, dictAnat, options=options)


                logging.debug(" - Right Hindfoot - motion -")
                logging.debug(" ---------------------------")
                self._right_hindFoot_motion(aqui, dictRef, dictAnat, options=options)

                logging.debug(" - Right ForeFoot - motion -")
                logging.debug(" ---------------------------")
                self._right_foreFoot_motion(aqui, dictRef, dictAnat, options=options)

            if self.m_bodypart == enums.BodyPart.LowerLimbTrunk:
                self._thorax_motion(aqui, dictRef,dictAnat,options=options)

            if self.m_bodypart == enums.BodyPart.UpperLimb or self.m_bodypart == enums.BodyPart.FullBody:
                self._thorax_motion(aqui, dictRef,dictAnat,options=options)
                self._head_motion(aqui, dictRef,dictAnat,options=options)

                self._clavicle_motion("Left",aqui, dictRef,dictAnat,options=options)
                self._constructArmVirtualMarkers("Left", aqui)
                self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
                self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
                self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
                self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
                self._hand_motion("Left",aqui, dictRef,dictAnat,options=options)

                self._clavicle_motion("Right",aqui, dictRef,dictAnat,options=options)
                self._constructArmVirtualMarkers("Right", aqui)
                self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
                self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
                self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
                self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
                self._hand_motion("Right",aqui, dictRef,dictAnat,options=options)

        if motionMethod == enums.motionMethod.Sodervisk:

            # ---remove all  direction marker from tracking markers.
            if self.staExpert:
                for seg in self.m_segmentCollection:
                    selectedTrackingMarkers=list()
                    for marker in seg.m_tracking_markers:
                        if marker in self.__class__.TRACKING_MARKERS : # get class variable MARKER even from child
                            selectedTrackingMarkers.append(marker)
                    seg.m_tracking_markers= selectedTrackingMarkers

            if self.m_bodypart != enums.BodyPart.UpperLimb:
                logging.debug("--- Segmental Least-square motion process ---")
                self._pelvis_motion_optimize(aqui, dictRef, motionMethod)
                self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))

                TopLumbar5=np.zeros((aqui.GetPointFrameNumber(),3))

                for i in range(0,aqui.GetPointFrameNumber()):
                    lhjc = aqui.GetPoint("LHJC").GetValues()[i,:]
                    rhjc =  aqui.GetPoint("RHJC").GetValues()[i,:]
                    pelvisScale = np.linalg.norm(lhjc-rhjc)
                    offset = (lhjc+rhjc)/2.0
                    R = self.getSegment("Pelvis").anatomicalFrame.motion[i].getRotation()
                    TopLumbar5[i,:] = offset +  np.dot(R,(np.array([ 0, 0, 0.925]))* pelvisScale)

                self._TopLumbar5 = TopLumbar5

                self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
                self._anatomical_motion(aqui,"Left Thigh",originLabel = str(dictAnat["Left Thigh"]['labels'][3]))

                self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
                self._anatomical_motion(aqui,"Right Thigh",originLabel = str(dictAnat["Right Thigh"]['labels'][3]))

                self._left_shank_motion_optimize(aqui, dictRef,motionMethod)
                self._anatomical_motion(aqui,"Left Shank",originLabel = str(dictAnat["Left Shank"]['labels'][3]))


                self._right_shank_motion_optimize(aqui, dictRef,motionMethod)
                self._anatomical_motion(aqui,"Right Shank",originLabel = str(dictAnat["Right Shank"]['labels'][3]))

                # hindFoot ( because of singularities AJC- TOE and HEE align)
                self._left_hindFoot_motion(aqui, dictRef, dictAnat, options=options)
                self._right_hindFoot_motion(aqui, dictRef, dictAnat, options=options)

                #self._leftHindFoot_motion_optimize(aqui, dictRef,motionMethod)
                #self._anatomical_motion(aqui,"Left Foot",originLabel = str(dictAnat["Left Foot"]['labels'][3]))
                #self._rightHindFoot_motion_optimize(aqui, dictRef,motionMethod)
                #self._anatomical_motion(aqui,"Right Foot",originLabel = str(dictAnat["Right Foot"]['labels'][3]))

                # foreFoot (more robust than Sodervisk)
                self._left_foreFoot_motion(aqui, dictRef, dictAnat, options=options)
                self._right_foreFoot_motion(aqui, dictRef,dictAnat, options=options)

                #self._leftForeFoot_motion(aqui, dictRef,motionMethod)
                #self._anatomical_motion(aqui,"Left ForeFoot",originLabel = str(dictAnat["Left ForeFoot"]['labels'][3]))


                #self._rightForeFoot_motion(aqui, dictRef,motionMethod)
                #self._anatomical_motion(aqui,"Right ForeFoot",originLabel = str(dictAnat["Right ForeFoot"]['labels'][3]))
            if self.m_bodypart == enums.BodyPart.LowerLimbTrunk:
                self._thorax_motion(aqui, dictRef,dictAnat,options=options)

            if self.m_bodypart == enums.BodyPart.UpperLimb or self.m_bodypart == enums.BodyPart.FullBody:
                self._thorax_motion(aqui, dictRef,dictAnat,options=options)
                self._head_motion(aqui, dictRef,dictAnat,options=options)

                self._clavicle_motion("Left",aqui, dictRef,dictAnat,options=options)
                self._constructArmVirtualMarkers("Left", aqui)
                self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
                self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
                self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
                self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
                self._hand_motion("Left",aqui, dictRef,dictAnat,options=options)

                self._clavicle_motion("Right",aqui, dictRef,dictAnat,options=options)
                self._constructArmVirtualMarkers("Right", aqui)
                self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
                self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
                self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
                self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
                self._hand_motion("Right",aqui, dictRef,dictAnat,options=options)

    # ----- native motion ------
    def _left_hindFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Left Foot")



        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][1])).GetValues()[i,:]

            if dictRef["Left Foot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Left Shank").anatomicalFrame.motion[i].m_axisY



            ptOrigin=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- FJC
        # btkTools.smartAppendPoint(aqui,"LFJC",seg.getReferential("TF").getNodeTrajectory("LFJC"),desc="from hindFoot" ) # put in ForefootMotion
        btkTools.smartAppendPoint(aqui,"LFJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("LFJC"),desc="from hindFoot" )
        btkTools.smartAppendPoint(aqui,"LFJC",seg.getReferential("TF").getNodeTrajectory("LFJC"),desc="from hindFoot" )

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Foot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    def _left_foreFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """

        btkTools.smartAppendPoint(aqui,"LFJC",self.getSegment("Left Foot").getReferential("TF").getNodeTrajectory("LFJC"),desc="from hindFoot" )

        seg=self.getSegment("Left ForeFoot")


        # --- motion of the technical referential

        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        #computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][2])).GetValues()[i,:]

            ptOrigin=aqui.GetPoint(str(dictRef["Left ForeFoot"]["TF"]['labels'][3])).GetValues()[i,:]

            a1=(pt2-pt1)
            a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left ForeFoot"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- motion of new markers
        btkTools.smartAppendPoint(aqui,"LvSMH",seg.getReferential("TF").getNodeTrajectory("LvSMH") )

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left ForeFoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))



    def _right_hindFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Right Foot")



        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][0])).GetValues()[i,:] #cun
            pt2=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc

            if dictRef["Right Foot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][2])).GetValues()[i,:] # not used
                v=(pt3-pt1)
            else:
                v=self.getSegment("Right Shank").anatomicalFrame.motion[i].m_axisY



            ptOrigin=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- RvTOE
        btkTools.smartAppendPoint(aqui,"RFJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("RFJC"),desc="from hindFoot" )
        btkTools.smartAppendPoint(aqui,"RFJC",seg.getReferential("TF").getNodeTrajectory("RFJC"),desc="from hindFoot" )

        # --- motion of the technical referential
        seg.anatomicalFrame.motion=[]
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Foot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    def _right_foreFoot_motion(self,aqui, dictRef,dictAnat,options=None):
        """
        :Parameters:

           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """

        btkTools.smartAppendPoint(aqui,"RFJC",self.getSegment("Right Foot").getReferential("TF").getNodeTrajectory("RFJC"),desc="from hindFoot" )

        seg=self.getSegment("Right ForeFoot")


        # --- motion of the technical referential

        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        #computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][2])).GetValues()[i,:]

            ptOrigin=aqui.GetPoint(str(dictRef["Right ForeFoot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1= np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right ForeFoot"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- motion of new markers
        btkTools.smartAppendPoint(aqui,"RvSMH",seg.getReferential("TF").getNodeTrajectory("RvSMH") )

        # --- motion of the anatomical referential

        seg.anatomicalFrame.motion=[]
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right ForeFoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    # ----- least-square Segmental motion ------
    def _leftHindFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Left Foot")

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
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm= motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt


                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- vTOE and AJC
        btkTools.smartAppendPoint(aqui,"LAJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("LAJC"),desc="opt from hindfoot" )
        btkTools.smartAppendPoint(aqui,"LFJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("LFJC"),desc="opt from hindfoot" )
        btkTools.smartAppendPoint(aqui,"LFJC",seg.getReferential("TF").getNodeTrajectory("LFJC"),desc="from hindFoot" )

        # remove AJC from list of tracking markers
        if "LAJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("LAJC")


    def _leftForeFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Left ForeFoot")

        #  --- add RvTOE if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "LFJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LFJC")
                    logging.debug("LFJC added to tracking marker list")

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
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- motion of new markers
        # --- LvSMH
        desc = seg.getReferential('TF').static.getNode_byLabel("LvSMH").m_desc
        values_vSMHnode=seg.getReferential('TF').getNodeTrajectory("LvSMH")
        btkTools.smartAppendPoint(aqui,"LvSMH",values_vSMHnode, desc=str("opt-"+desc))

        btkTools.smartAppendPoint(aqui,"LFJC_ForeFoot",seg.getReferential("TF").getNodeTrajectory("LFJC"),desc="opt from forefoot" )

        # remove FJC from list of tracking markers
        if "LFJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("LFJC")

    def _rightHindFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Right Foot")

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
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm= motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- vTOE and AJC
        btkTools.smartAppendPoint(aqui,"RAJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("RAJC"),desc="opt from hindfoot" )
        btkTools.smartAppendPoint(aqui,"RFJC-HindFoot",seg.getReferential("TF").getNodeTrajectory("RFJC"),desc="opt from hindfoot" )
        btkTools.smartAppendPoint(aqui,"RFJC",seg.getReferential("TF").getNodeTrajectory("RFJC"),desc="from hindFoot" )

        # remove AJC from list of tracking markers
        if "RAJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("RAJC")


    def _rightForeFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Right ForeFoot")

        #  --- add RvTOE if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "RFJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RFJC")
                    logging.debug("RFJC added to tracking marker list")

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
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- motion of new markers
        # --- LvSMH
        desc = seg.getReferential('TF').static.getNode_byLabel("RvSMH").m_desc
        values_vSMHnode=seg.getReferential('TF').getNodeTrajectory("RvSMH")
        btkTools.smartAppendPoint(aqui,"RvSMH",values_vSMHnode, desc=str("opt-"+desc))

        btkTools.smartAppendPoint(aqui,"RFJC_ForeFoot",seg.getReferential("TF").getNodeTrajectory("RFJC"),desc="opt from forefoot" )

        # remove FJC from list of tracking markers
        if "RFJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("RFJC")

    # --- opensim --------
    def opensimGeometry(self):
        """
        TODO require : joint name from opensim -> find alternative

        rather a class method than a method instance
        """


        out={}
        out["hip_r"]= {"joint label":"RHJC", "proximal segment label":"Pelvis", "distal segment label":"Right Thigh" }
        out["knee_r"]= {"joint label":"RKJC", "proximal segment label":"Right Thigh", "distal segment label":"Right Shank" }
        out["ankle_r"]= {"joint label":"RAJC", "proximal segment label":"Right Shank", "distal segment label":"Right Foot" }
        out["mtp_r"]= {"joint label":"RFJC", "proximal segment label":"Right Foot", "distal segment label":"Right ForeFoot" }


        out["hip_l"]= {"joint label":"LHJC", "proximal segment label":"Pelvis", "distal segment label":"Left Thigh" }
        out["knee_l"]= {"joint label":"LKJC", "proximal segment label":"Left Thigh", "distal segment label":"Left Shank" }
        out["ankle_l"]= {"joint label":"LAJC", "proximal segment label":"Left Shank", "distal segment label":"Left Foot" }
        out["mtp_l"]= {"joint label":"LFJC", "proximal segment label":"Left Foot", "distal segment label":"Left ForeFoot" }

        return out

    def opensimIkTask(self):
        out={}

        if self.staExpert:
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
                 "RSMH":0,
                 "RSMH_supInf":100,
                 "RSMH_medLat":100,
                 "RSMH_proDis":100,

                 "RTOE":0,
                 "RTOE_supInf":100,
                 "RTOE_medLat":100,
                 "RTOE_proDis":100,

                 "RFMH":0,
                 "RFMH_supInf":100,
                 "RFMH_medLat":100,
                 "RFMH_proDis":100,

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

                 "LFMH":0,
                 "LFMH_supInf":100,
                 "LFMH_medLat":100,
                 "LFMH_proDis":100,

                 "LVMH":0,
                 "LVMH_supInf":100,
                 "LVMH_medLat":100,
                 "LVMH_proDis":100,


                 "LTHAP":0,
                 "LTHAP_posAnt":100,
                 "LTHAP_medLat":100,
                 "LTHAP_proDis":100,
                 "LTHAD":0,
                 "LTHAD_posAnt":100,
                 "LTHAD_medLat":100,
                 "LTHAD_proDis":100,
                 "RTHAP":0,
                 "RTHAP_posAnt":100,
                 "RTHAP_medLat":100,
                 "RTHAP_proDis":100,
                 "RTHAD":0,
                 "RTHAD_posAnt":100,
                 "RTHAD_medLat":100,
                 "RTHAD_proDis":100,
                 "LTIAP":0,
                 "LTIAP_posAnt":100,
                 "LTIAP_medLat":100,
                 "LTIAP_proDis":100,
                 "LTIAD":0,
                 "LTIAD_posAnt":100,
                 "LTIAD_medLat":100,
                 "LTIAD_proDis":100,
                 "RTIAP":0,
                 "RTIAP_posAnt":100,
                 "RTIAP_medLat":100,
                 "RTIAP_proDis":100,
                 "RTIAD":0,
                 "RTIAD_posAnt":100,
                 "RTIAD_medLat":100,
                 "RTIAD_proDis":100,



                 "LSMH":0,
                 "LSMH_supInf":0,
                 "LSMH_medLat":0,
                 "LSMH_proDis":0,

                 "RVMH":0,
                 "RVMH_supInf":100,
                 "RVMH_medLat":100,
                 "RVMH_proDis":100,

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
                 "RTHAP":100,
                 "RTHAD":100,
                 "RTIB":100,
                 "RANK":100,
                 "RTIAP":100,
                 "RTIAD":100,
                 "RHEE":100,

                 "RTOE":100,
                 "RFMH":100,
                 "RVMH":100,
                 "LTHI":100,
                 "LKNE":100,
                 "LTHAP":100,
                 "LTHAD":100,
                 "LTIB":100,
                 "LANK":100,
                 "LTIAP":100,
                 "LTIAD":100,
                 "LHEE":100,

                 "LTOE":100,
                 "LFMH":100,
                 "LVMH":100,

                 "RSMH":0,
                 "RTHLD":0,
                 "RPAT":0,
                 "LSMH":0,
                 "LTHLD":0,
                 "LPAT":0,

                 }

        return out



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

        pointSuffix  =  pointSuffix if pointSuffix is not None else ""

        if staticProcessingFlag:
            if self.checkCalibrationProperty("LeftKAD",True):
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LKNE", acq)
            if self.checkCalibrationProperty("RightKAD",True):
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RKNE", acq)

        # export JC
        if self.m_bodypart != enums.BodyPart.UpperLimb:
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LHJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RHJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LKJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RKJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LAJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RAJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LFJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RFJC", acq,suffix=pointSuffix)

        if self.m_bodypart == enums.BodyPart.LowerLimbTrunk:
            pass

        if self.m_bodypart == enums.BodyPart.UpperLimb or self.m_bodypart == enums.BodyPart.FullBody:
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LSJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RSJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LEJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"REJC", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LHO", acq,suffix=pointSuffix)
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RHO", acq,suffix=pointSuffix)

            logging.debug("jc over")


        # export angles
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetType() == btk.btkPoint.Angle:
                if pointSuffix is not None:
                    if pointSuffix in it.GetLabel():
                        nexusTools.appendAngleFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                else:
                    nexusTools.appendAngleFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)

        logging.debug("angles over")

        # bones
        # -------------
        if self.m_bodypart != enums.BodyPart.UpperLimb:
            nexusTools.appendBones(NEXUS,vskName,acq,"PELVIS", self.getSegment("Pelvis"),OriginValues = acq.GetPoint("midHJC").GetValues() ,suffix=pointSuffix)

            nexusTools.appendBones(NEXUS,vskName,acq,"LFEMUR", self.getSegment("Left Thigh"),OriginValues = acq.GetPoint("LKJC").GetValues() ,suffix=pointSuffix)
            #nexusTools.appendBones(NEXUS,vskName,"LFEP", self.getSegment("Left Shank Proximal"),OriginValues = acq.GetPoint("LKJC").GetValues(),manualScale = 100 )
            nexusTools.appendBones(NEXUS,vskName,acq,"LTIBIA", self.getSegment("Left Shank"),OriginValues = acq.GetPoint("LAJC").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"LFOOT", self.getSegment("Left Foot"), OriginValues = acq.GetPoint("LHEE").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"LTOES", self.getSegment("Left ForeFoot"), OriginValues = acq.GetPoint("LFJC").GetValues() ,suffix=pointSuffix)


            nexusTools.appendBones(NEXUS,vskName,acq,"RFEMUR", self.getSegment("Right Thigh"),OriginValues = acq.GetPoint("RKJC").GetValues() ,suffix=pointSuffix)
            #nexusTools.appendBones(NEXUS,vskName,"RFEP", self.getSegment("Right Shank Proximal"),OriginValues = acq.GetPoint("RKJC").GetValues(),manualScale = 100 )
            nexusTools.appendBones(NEXUS,vskName,acq,"RTIBIA", self.getSegment("Right Shank"),OriginValues = acq.GetPoint("RAJC").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"RFOOT", self.getSegment("Right Foot") , OriginValues = acq.GetPoint("RHEE").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"RTOES", self.getSegment("Right ForeFoot") ,  OriginValues = acq.GetPoint("RFJC").GetValues(),suffix=pointSuffix)

        if self.m_bodypart == enums.BodyPart.LowerLimbTrunk :
            nexusTools.appendBones(NEXUS,vskName,acq,"THORAX", self.getSegment("Thorax"),OriginValues = acq.GetPoint("OT").GetValues() ,suffix=pointSuffix)

        if self.m_bodypart == enums.BodyPart.UpperLimb or self.m_bodypart == enums.BodyPart.FullBody:
            nexusTools.appendBones(NEXUS,vskName,acq,"THORAX", self.getSegment("Thorax"),OriginValues = acq.GetPoint("OT").GetValues() ,suffix=pointSuffix)

            nexusTools.appendBones(NEXUS,vskName,acq,"LUPPERARM", self.getSegment("Left UpperArm"),OriginValues = acq.GetPoint("LEJC").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"LFOREARM", self.getSegment("Left ForeArm"),OriginValues = acq.GetPoint("LWJC").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"LHAND", self.getSegment("Left Hand"),OriginValues = acq.GetPoint("LHO").GetValues() ,suffix=pointSuffix)

            nexusTools.appendBones(NEXUS,vskName,acq,"RUPPERARM", self.getSegment("Right UpperArm"),OriginValues = acq.GetPoint("REJC").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"RFOREARM", self.getSegment("Right ForeArm"),OriginValues = acq.GetPoint("RWJC").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"RHAND", self.getSegment("Right Hand"),OriginValues = acq.GetPoint("RHO").GetValues() ,suffix=pointSuffix)
            nexusTools.appendBones(NEXUS,vskName,acq,"HEAD", self.getSegment("Head"),OriginValues = acq.GetPoint("HC").GetValues() ,suffix=pointSuffix)
        logging.debug("bones over")

        if not staticProcessingFlag:
            # export Force
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Force:
                    if pointSuffix is not None:
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendForceFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendForceFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            logging.debug("force over")

            # export Moment
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Moment:
                    if pointSuffix is not None:
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendMomentFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendMomentFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            logging.debug("Moment over")

            # export Power
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Power:
                    if pointSuffix is not None:
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendPowerFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendPowerFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            logging.debug("power over")

        # centre of mass
        centreOfMassLabel  = "CentreOfMass" + pointSuffix if pointSuffix is not None else "CentreOfMass"
        if self.m_centreOfMass is not None:
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,str(centreOfMassLabel), acq)


class CGM2_5(CGM2_4):
    """

    """

    def __init__(self):
        super(CGM2_5, self).__init__()
        self.decoratedModel = False
        self.version = "CGM2.5"

    def __repr__(self):
        return "LowerLimb CGM2.5"
