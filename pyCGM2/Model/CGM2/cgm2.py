# -*- coding: utf-8 -*-
import numpy as np
import logging
import pdb


import btk

import model as cmb
import modelDecorator as cmd
import frame as cfr
import motion as cmot
import euler as ceuler

import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import geometry
from pyCGM2.Tools import  btkTools,nexusTools
import cgm


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
        
        #self.__configure()
        
        
    def __repr__(self):
        return "cgm2-3"

    def configure(self):
        self.addSegment("Pelvis",0,pyCGM2Enums.SegmentSide.Central,["LASI","RASI","LPSI","RPSI"], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,pyCGM2Enums.SegmentSide.Left,["LKNE","LTHI","LTHIAP","LTHIAD"], tracking_markers = ["LKNE","LTHI","LTHIAP","LTHIAD"])
        self.addSegment("Right Thigh",4,pyCGM2Enums.SegmentSide.Right,["RKNE","RTHI","RTHIAP","RTHIAD"], tracking_markers = ["RKNE","RTHI","RTHIAP","RTHIAD"])
        self.addSegment("Left Shank",2,pyCGM2Enums.SegmentSide.Left,["LANK","LTIB","LTIBAP","LTIBAD"], tracking_markers = ["LANK","LTIB","LTIBAP","LTIBAD"])
        self.addSegment("Left Shank Proximal",7,pyCGM2Enums.SegmentSide.Left) # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Right Shank",5,pyCGM2Enums.SegmentSide.Right,["RANK","RTIB","RTIBAP","RTIBAD"], tracking_markers = ["RANK","RTIB","RTIBAP","RTIBAD"])
        self.addSegment("Right Shank Proximal",8,pyCGM2Enums.SegmentSide.Right)        # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Left Foot",3,pyCGM2Enums.SegmentSide.Left,["LAJC","LHEE","LTOE"], tracking_markers = ["LHEE","LTOE"] )
        self.addSegment("Right Foot",6,pyCGM2Enums.SegmentSide.Right,["RAJC","RHEE","RTOE"], tracking_markers = ["RHEE","RTOE"])

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
#
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
          pass      
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
                 "RTHL":0,
                 "RTHLD":0,
                 "RPAT":0,
                 "RTIBL":0,
                 "LTHL":0,
                 "LTHLD":0,
                 "LPAT":0,
                 "LTIBL":0
                 } 
        
        return out        