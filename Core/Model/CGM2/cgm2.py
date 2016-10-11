# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 10:59:01 2015
@author: fleboeuf
"""


#---------------------------------------------
import pdb

import numpy as np
import logging


import cgm 
import modelDecorator as cmd
import frame as cfr
import motion as cmot
import euler as ceuler

import pyCGM2.Core.enums as pyCGM2Enums
from pyCGM2.Core.Math import geometry
from pyCGM2.Core.Tools import  btkTools

markerDiameter=14.0 # TODO ou mettre ca
basePlate = 2.0





class CGM2ModelInf(cgm.CGM):
    """ implementation of the cgm2
    
    
    """

    nativeCgm1 = True        
    
    KINEMATIC_LABELS ={ 'Left': ["LPelvisAngles","LHipAngles","LKneeAngles","LAnkleAngles","LFootProgressAngles"],
                        'Right': ["RPelvisAngles","RHipAngles","RKneeAngles","RAnkleAngles","RFootProgressAngles"],
                        'All' : ["LPelvisAngles","LHipAngles","LKneeAngles","LAnkleAngles","LFootProgressAngles",
                                 "RPelvisAngles","RHipAngles","RKneeAngles","RAnkleAngles","RFootProgressAngles"]
                        }

    KINETIC_LABELS ={ 'Left': ["LHipMoment","LKneeMoment","LAnkleMoment", "LHipPower","LKneePower","LAnklePower"],
                        'Right': ["RHipMoment","RKneeMoment","RAnkleMoment", "RHipPower","RKneePower","RAnklePower"],
                        'All' : ["LHipMoment","LKneeMoment","LAnkleMoment", "LHipPower","LKneePower","LAnklePower",
                                 "RHipMoment","RKneeMoment","RAnkleMoment", "RHipPower","RKneePower","RAnklePower"]
                        }

                                 
    KINETIC_MOMENT_LABELS ={ 'Left': ["LHipMoment","LKneeMoment","LAnkleMoment"],
                                  'Right': ["RHipMoment","RKneeMoment","RAnkleMoment"]}
    KINETIC_POWER_LABELS ={ 'Left': ["LHipPower","LKneePower","LAnklePower"],
                                  'Right':   ["RHipPower","RKneePower","RAnklePower"]}

    AXIS_FOR_GPS={"Pelvis":[0,1,2],"Hip":[0,1,2],"Knee":[0],"Ankle":[0],"Foot":[2]}    
    
    
    
    def __init__(self):
        """Constructor 
        
           - Run configuration internally
           - Initialize deviation data  

        """
        super(CGM2ModelInf, self).__init__()

        self.decoratedModel = False
        
        self.__configure()
        
        


    def __repr__(self):
        return "cgm2"

    def __configure(self):
        # todo create a Foot segment
        self.addSegment("Pelvis", 0,pyCGM2Enums.SegmentSide.Central,["LASI","RASI","LPSI","RPSI"], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,pyCGM2Enums.SegmentSide.Left,["LKNE","LTHI"], tracking_markers = ["LHJC","LKNE","LTHI"])
        self.addSegment("Right Thigh",4,pyCGM2Enums.SegmentSide.Right,["RKNE","RTHI","RTHIAP","RTHIAD"], tracking_markers = ["RHJC","RKNE","RTHI","RTHIAP","RTHIAD"])
        self.addSegment("Left Shank",2,pyCGM2Enums.SegmentSide.Left,["LANK","LTIB"], tracking_markers = ["LKJC","LANK","LTIB"])
        self.addSegment("Right Shank",5,pyCGM2Enums.SegmentSide.Right,["RANK","RTIB","RSHN","RTIAP"], tracking_markers = ["RKJC","RANK","RTIB","RSHN","RTIAP"])
        self.addSegment("Right Hindfoot",6,pyCGM2Enums.SegmentSide.Right,["RHEE","RCUN","RANK"], tracking_markers = ["RHEE","RCUN","RAJC"])
        self.addSegment("Right Forefoot",7,pyCGM2Enums.SegmentSide.Right,["RD1M","RD5M","RCUN","RTOE"], tracking_markers = ["RD1M","RD5M","RTOE"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])


        self.addJoint("LHipAngles_cgm","Pelvis", "Left Thigh","YXZ")
        self.addJoint("LKneeAngles_cgm","Left Thigh", "Left Shank","YXZ")
        
        self.addJoint("RHipAngles_cgm","Pelvis", "Right Thigh","YXZ")
        self.addJoint("RKneeAngles_cgm","Right Thigh", "Right Shank","YXZ")

        self.addJoint("RAnkleAngles_cgm","Right Shank", "Right Hindfoot","YXZ")
        self.addJoint("RForeFootAngles_cgm","Right Hindfoot", "Right Forefoot","YXZ") # rename Joint label

        self.mp_computed["leftThighOffset"] = 0.0
        self.mp_computed["rightThighOffset"] = 0.0
        self.mp_computed["leftShankOffset"] = 0.0
        self.mp_computed["rightShankOffset"] = 0.0
        self.mp_computed["leftTibialTorsion"] = 0.0
        self.mp_computed["rightTibialTorsion"] = 0.0

          
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
                        
        dictRef["Right Hindfoot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RCUN","RAJC",None,"RAJC"]} }  #{"TF" : {'sequence':"ZYiX", 'labels':   ["RCUN","RAJC","RHEE","RAJC"]} }
        dictRef["Right Forefoot"]={"TF" : {'sequence':"ZXY", 'labels':   ["RTOE","RCUN","RD5M","RCUN"]} }   
        
        dictRefAnatomical={}
        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]} # normaly : midHJC
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]} # origin = Proximal ( differ from native)
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]} 
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]} 
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}         
        
        # left Foot ( nothing yet)
        dictRefAnatomical["Right Hindfoot"]= {'sequence':"ZXiY", 'labels':   ["RCUN","RHEE",None,"RAJC"]}   #  {'sequence':"ZYX", 'labels':   ["RCUN","RHEE","RAJC","RAJC"
        #dictRefAnatomical["Right Forefoot"]= {'sequence':"ZXY", 'labels':   ["RvTOE","RvCUN","RvD5M","RvTOE"]} # look out : use virtual Point 

        dictRefAnatomical["Right Forefoot"]= {'sequence':"ZYX", 'labels':   ["RvTOE","RvCUN",None,"RvTOE"]} # look out : use virtual Point 
        
        return dictRef,dictRefAnatomical


    def finalizeJCS(self,jointLabel,jointValues):
        """ TODO  class method ? 

        """ 
             
        values = np.zeros((jointValues.shape))        

        
        if jointLabel == "LHipAngles_cgm" :  #LHPA=<-1(LHPA),-2(LHPA),-3(LHPA)> {*flexion, adduction, int. rot.			*}       
            values[:,0] = - np.rad2deg(  jointValues[:,0])     
            values[:,1] = - np.rad2deg(  jointValues[:,1])
            values[:,2] = - np.rad2deg(  jointValues[:,2])

        elif jointLabel == "LKneeAngles_cgm" : # LKNA=<1(LKNA),-2(LKNA),-3(LKNA)-$LTibialTorsion>  {*flexion, varus, int. rot.		*}       
            values[:,0] = np.rad2deg(  jointValues[:,0])     
            values[:,1] = -np.rad2deg(  jointValues[:,1])
            values[:,2] = -np.rad2deg(  jointValues[:,2])

        elif jointLabel == "RHipAngles_cgm" :  # RHPA=<-1(RHPA),2(RHPA),3(RHPA)>   {*flexion, adduction, int. rot.			*}
            values[:,0] = - np.rad2deg(  jointValues[:,0])     
            values[:,1] =  np.rad2deg(  jointValues[:,1])
            values[:,2] =  np.rad2deg(  jointValues[:,2])

        elif jointLabel == "RKneeAngles_cgm" : #  RKNA=<1(RKNA),2(RKNA),3(RKNA)-$RTibialTorsion>    {*flexion, varus, int. rot.		*}  
            values[:,0] = np.rad2deg(  jointValues[:,0])     
            values[:,1] = np.rad2deg(  jointValues[:,1])
            values[:,2] = np.rad2deg(  jointValues[:,2])

        elif jointLabel == "LAnkleAngles_cgm":
            values[:,0] = -1.0* np.rad2deg(  jointValues[:,0]  + np.radians(90))
            values[:,1] = -1.0*np.rad2deg(  jointValues[:,2])
            values[:,2] =  -1.0*np.rad2deg(  jointValues[:,1])            

        elif jointLabel == "RAnkleAngles_cgm":
            values[:,0] = -1.0* np.rad2deg(  jointValues[:,0]  + np.radians(90))
            values[:,1] = np.rad2deg(  jointValues[:,2])
            values[:,2] =  np.rad2deg(  jointValues[:,1]) 

                  
        elif jointLabel == "RForeFootAngles_cgm":
            values[:,0] = -1.0 * np.rad2deg(  jointValues[:,0])     
            values[:,1] = np.rad2deg(  jointValues[:,1])
            values[:,2] = np.rad2deg(  jointValues[:,2])

        else:
            values[:,0] = np.rad2deg(  jointValues[:,0])     
            values[:,1] = np.rad2deg(  jointValues[:,1])
            values[:,2] = np.rad2deg(  jointValues[:,2])
            
        return values

    def calibrate(self,aquiStatic, dictRef, dictAnatomic,  options=None):
        """ static calibration 
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `options` (kwargs) - use to pass option altering the standard construction


        .. todo:: shrink and clone the aquisition to seleted frames           
        """        
        
        ff=aquiStatic.GetFirstFrame() 
        lf=aquiStatic.GetLastFrame()
        frameInit=ff-ff  
        frameEnd=lf-ff+1
        nFrames = aquiStatic.GetPointFrameNumber()         
        
        if self.nativeCgm1:
            print "native CGM1"
            if not btkTools.isPointExist(aquiStatic,"LKNE"):
                btkTools.smartAppendPoint(aquiStatic,"LKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))
            if not btkTools.isPointExist(aquiStatic,"RKNE"):
                btkTools.smartAppendPoint(aquiStatic,"RKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))                
        else:
            print "cgm1 decorated"
        
        
        # --- PELVIS - THIGH - SHANK        
        # calibration of technical calibration                
        self._pelvis_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        self._left_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self._right_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self._left_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self._right_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        # calibration of anatomical frame
        self._pelvis_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd) 
        self.displayStaticCoordinateSystem( aquiStatic, "Pelvis","Pelvis",referential = "Anatomical"  )
        # Thigh
        self._left_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)                
        self._right_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Left Thigh","LThigh",referential = "Anatomical"  )
        self.displayStaticCoordinateSystem( aquiStatic, "Right Thigh","RThigh",referential = "Anatomical"  ) 

        self.getThighOffset(side="left")
        self.getThighOffset(side="right")

        # shank
        self._left_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)        
        self._right_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)    
        self.displayStaticCoordinateSystem( aquiStatic, "Left Shank","LShank",referential = "Anatomical"  )
        self.displayStaticCoordinateSystem( aquiStatic, "Right Shank","RShank",referential = "Anatomical"  )
        self.getShankOffsets(side="both")# compute TibialRotation and Shank offset
        self.getAbdAddAnkleJointOffset(side="both")


        # ---  FOOT segment 
        # ---------------
        # need anaotonical flexion axis of the shank. 
        
        
        # --- hind foot 
        # --------------
        # new points required
        
        #   virtual Point between nav et P5    
        valMidFoot=(aquiStatic.GetPoint("RNAV").GetValues() + aquiStatic.GetPoint("RP5M").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aquiStatic,"RMidFoot",valMidFoot,desc="")
        self.getSegment("Right Hindfoot").addMarkerLabel("RMidFoot")

        #   registrration of CUN with Toe Offset
        cun =  aquiStatic.GetPoint("RCUN").GetValues()
        valuesVirtualCun = np.zeros((nFrames,3))
        for i in range(0,nFrames):
            valuesVirtualCun[i,:] = np.array([cun[i,0], cun[i,1], cun[i,2]-self.mp["rightToeOffset"]])

        btkTools.smartAppendPoint(aquiStatic,"RvCUN",valuesVirtualCun,desc="cun Registrate")
        self.getSegment("Right Hindfoot").addMarkerLabel("RvCUN")        
       
        # calibrations   
        self._rightHindFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options) 
        #self.displayStaticCoordinateSystem( aquiStatic, "Right Hindfoot","RHindFootUncorrected",referential = "technic"  )         
        self._rightHindFoot_anatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Hindfoot","RHindFoot",referential = "Anatomical"  ) 
        
        
        self.getHindFootOffset(side = "both")
        
        # --- fore foot 
        # ----------------
        # new Points required
        
        # method 1
        # method : Projection of TOE on the plan CUN-D-D5
#        valueProj = np.zeros((aquiStatic.GetPointFrameNumber(),3))
#        for i in range(frameInit,frameEnd):
#            D5= aquiStatic.GetPoint("RD5M").GetValues()[i,:]
#            D1= aquiStatic.GetPoint("RD1M").GetValues()[i,:]
#            Or= aquiStatic.GetPoint("RCUN").GetValues()[i,:]
#            toe = aquiStatic.GetPoint("RTOE").GetValues()[i,:]
#
#            valueProj[i,:] = cg.pointProjectionOnPlane(toe,Or,D5,D1)        
#
#        btkTools.smartAppendPoint(aquiStatic,"RvTOE",valueProj,desc="projectPlan")
        
        
        
        # toe offset registration
        
        toe =  aquiStatic.GetPoint("RTOE").GetValues()
        d5 =  aquiStatic.GetPoint("RD5M").GetValues()
    
        valuesVirtualToe = np.zeros((nFrames,3))
        valuesVirtualD5 = np.zeros((nFrames,3))        
        #new = np.zeros((nFrames,3))
        for i in range(0,nFrames):        
            valuesVirtualToe[i,:] = np.array([toe[i,0], toe[i,1], toe[i,2]-self.mp["rightToeOffset"] ])#valuesVirtualCun[i,2]])#
            valuesVirtualD5 [i,:]= np.array([d5[i,0], d5[i,1], valuesVirtualToe[i,2]])
            #new[i,:] = np.array([1736.0, 44.77, 103.9])

        btkTools.smartAppendPoint(aquiStatic,"RvTOE",valuesVirtualToe,desc="virtual")
        btkTools.smartAppendPoint(aquiStatic,"RvD5M",valuesVirtualD5,desc="virtual-flat ")
        #btkTools.smartAppendPoint(aquiStatic,"new",new,desc="virtual-flat ")


        
        print "pt 1RvTOE"
        print aquiStatic.GetPoint("RvTOE").GetValues().mean(axis= 0)
        
        print "Pt2 - RvCUN"
        print aquiStatic.GetPoint("RvCUN").GetValues().mean(axis= 0)


        print "Pt3-RvD5M"
        print aquiStatic.GetPoint("RvD5M").GetValues().mean(axis= 0)

        # new points added
        self.getSegment("Right Forefoot").addMarkerLabel("RMidFoot")
        self.getSegment("Right Forefoot").addMarkerLabel("RvCUN")  
        self.getSegment("Right Forefoot").addMarkerLabel("RvTOE")
        self.getSegment("Right Forefoot").addMarkerLabel("RvD5M")
        
        
        self._rightForeFoot_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        self._rightForeFoot_anatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Forefoot","RForeFoot",referential = "Anatomical"  ) 
        self.displayStaticCoordinateSystem( aquiStatic, "Right Forefoot","RTechnicForeFoot",referential = "Technical"  ) 


        
        val =  valuesVirtualToe +  100 * np.array([0.0, -1.0, 0.0 ]) 
        btkTools.smartAppendPoint(aquiStatic,"ToeHor",val)
        
        valuesVirtualD5 - valuesVirtualToe

        btkTools.smartWriter(aquiStatic, "tmp-static.c3d")

        
    def getThighOffset(self,side= "both"):

        if side == "both" or side=="left":
            # Left --------        
            kneeFlexionAxis=    np.dot(self.getSegment("Left Thigh").anatomicalFrame.static.getRotation().T, 
                                           self.getSegment("Left Thigh").anatomicalFrame.static.m_axisY)
            
            
            proj_kneeFlexionAxis = np.array([ kneeFlexionAxis[0],
                                   kneeFlexionAxis[1],
                                     0]) 
            v_kneeFlexionAxis = proj_kneeFlexionAxis/np.linalg.norm(proj_kneeFlexionAxis)  
    
            thiLocal = self.getSegment("Left Thigh").anatomicalFrame.static.getNode_byLabel("LTHI").m_local
                       
            proj_thi = np.array([ thiLocal[0],
                                   thiLocal[1],
                                     0])                                 
            v_thi = proj_thi/np.linalg.norm(proj_thi)
            
            angle=geometry.angleFrom2Vectors(v_kneeFlexionAxis, v_thi, self.getSegment("Left Thigh").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)    
#            angle=np.sign(np.cross(v_kneeFlexionAxis,v_thi)[2]) * geometry.angleFrom2Vectors(v_kneeFlexionAxis, v_thi)*360.0/(2.0*np.pi)
            self.mp_computed["leftThighOffset"]= -angle # angle needed : Thi toward knee flexion
            print "left Thigh Offset"
            print self.mp_computed["leftThighOffset"]  


        if side == "both" or side=="right":
            # Left --------        
            kneeFlexionAxis=    np.dot(self.getSegment("Right Thigh").anatomicalFrame.static.getRotation().T, 
                                           self.getSegment("Right Thigh").anatomicalFrame.static.m_axisY)
            
            
            proj_kneeFlexionAxis = np.array([ kneeFlexionAxis[0],
                                   kneeFlexionAxis[1],
                                     0]) 
            v_kneeFlexionAxis = proj_kneeFlexionAxis/np.linalg.norm(proj_kneeFlexionAxis)  
    
    
            thiLocal = self.getSegment("Right Thigh").anatomicalFrame.static.getNode_byLabel("RTHI").m_local
            proj_thi = np.array([ thiLocal[0],
                                   thiLocal[1],
                                     0])                                 
            v_thi = proj_thi/np.linalg.norm(proj_thi)
            
            v_kneeFlexionAxis_opp = geometry.oppositeVector(v_kneeFlexionAxis)         
            
            angle=geometry.angleFrom2Vectors(v_kneeFlexionAxis_opp, v_thi,self.getSegment("Right Thigh").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)
            #angle=np.sign(np.cross(v_kneeFlexionAxis_opp,v_thi)[2]) * geometry.angleFrom2Vectors(v_kneeFlexionAxis_opp,v_thi)*360.0/(2.0*np.pi)
            self.mp_computed["rightThighOffset"]=-angle # angle needed : Thi toward knee flexion
            print "right Thigh Offset"
            print self.mp_computed["rightThighOffset"]  

                                 
              
        
        
        
        
        
    
        
        

    def  getShankOffsets(self, side = "both"):
        
        if side == "both" or side == "left" :
            #"****** right angle beetween anatomical axis **********"            
            kneeFlexionAxis=    np.dot(self.getSegment("Left Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Left Thigh").anatomicalFrame.static.m_axisY)
        
        
            proj_kneeFlexionAxis = np.array([ kneeFlexionAxis[0],
                               kneeFlexionAxis[1],
                                 0]) 
            
            v_kneeFlexionAxis= proj_kneeFlexionAxis/np.linalg.norm(proj_kneeFlexionAxis)
            
            ankleFlexionAxis=    np.dot(self.getSegment("Left Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Left Shank").anatomicalFrame.static.m_axisY)
        
        
            proj_ankleFlexionAxis = np.array([ ankleFlexionAxis[0],
                               ankleFlexionAxis[1],
                                 0]) 
                                 
            v_ankleFlexionAxis = proj_ankleFlexionAxis/np.linalg.norm(proj_ankleFlexionAxis)
        
        
            angle= geometry.angleFrom2Vectors(v_kneeFlexionAxis,v_ankleFlexionAxis,self.getSegment("Left Shank").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)
            self.mp_computed["leftTibialTorsion"] = angle 
            print "left tibial torsion"
            print self.mp_computed["leftTibialTorsion"]             
            

            #"****** left angle beetween tib and flexion axis **********"    
            tibLocal = self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LTIB").m_local
            proj_tib = np.array([ tibLocal[0],
                               tibLocal[1],
                                 0])                                 
            v_tib = proj_tib/np.linalg.norm(proj_tib)
        
            angle=geometry.angleFrom2Vectors(v_ankleFlexionAxis,v_tib,self.getSegment("Left Shank").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)
            self.mp_computed["leftShankOffset"]= -angle
            print "left shank offset"
            print self.mp_computed["leftShankOffset"]         
        
        
    
            
            #"****** left angle beetween ank and flexion axis **********"        
            ANK =  self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LANK").m_local                    
            v_ank = ANK/np.linalg.norm(ANK)            
            angle = geometry.angleFrom2Vectors(v_ankleFlexionAxis,v_ank,self.getSegment("Left Shank").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)

            self.mp_computed["leftProjectionAngle_AnkleFlexion_LateralAnkle"] = angle
            print "left projection offset"
            print self.mp_computed["leftProjectionAngle_AnkleFlexion_LateralAnkle"]

        
        #        print "****** right angle beetween ank and knee flexion axis **********"                  
        #        angle= np.arccos(np.dot(v_kneeFlexionAxis,v_ank))
        #        print 180-np.sign(np.cross(v_kneeFlexionAxis,v_ank)[2])*angle*360.0/(2.0*np.pi)
        #
        #        print "****** right angle beetween ank and tib **********"                  
        #        angle= np.arccos(np.dot(v_tib,v_ank))
        #        print np.sign(np.cross(v_tib,v_ank)[2])*angle*360.0/(2.0*np.pi)        
        
        
        if side == "both" or side == "right" :
            #"****** right angle beetween anatomical axis **********"            
            kneeFlexionAxis=    np.dot(self.getSegment("Right Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Right Thigh").anatomicalFrame.static.m_axisY)
        
        
            proj_kneeFlexionAxis = np.array([ kneeFlexionAxis[0],
                               kneeFlexionAxis[1],
                                 0]) 
            
            v_kneeFlexionAxis= proj_kneeFlexionAxis/np.linalg.norm(proj_kneeFlexionAxis)
            
            ankleFlexionAxis=    np.dot(self.getSegment("Right Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Right Shank").anatomicalFrame.static.m_axisY)
        
        
            proj_ankleFlexionAxis = np.array([ ankleFlexionAxis[0],
                               ankleFlexionAxis[1],
                                 0]) 
                                 
            v_ankleFlexionAxis = proj_ankleFlexionAxis/np.linalg.norm(proj_ankleFlexionAxis)
        
        
            angle= geometry.angleFrom2Vectors(v_kneeFlexionAxis,v_ankleFlexionAxis,self.getSegment("Right Shank").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)
            self.mp_computed["rightTibialTorsion"] = angle 
            print "right tibial torsion"
            print self.mp_computed["rightTibialTorsion"]        
        
        
            #"****** right angle beetween tib and flexion axis **********"    
            tibLocal = self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RTIB").m_local
            proj_tib = np.array([ tibLocal[0],
                               tibLocal[1],
                                 0])
                                 
            v_tib = proj_tib/np.linalg.norm(proj_tib)
            v_ankleFlexionAxis_opp = geometry.oppositeVector(v_ankleFlexionAxis) 

        
            angle = geometry.angleFrom2Vectors(v_ankleFlexionAxis_opp,v_tib,self.getSegment("Right Shank").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)
            self.mp_computed["rightShankOffset"]= -angle
            print "right shank offset"
            print self.mp_computed["rightShankOffset"]     
            
            
            #"****** right angle beetween ank and flexion axis **********"        
            ANK =  self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RANK").m_local                    
            v_ank = ANK/np.linalg.norm(ANK)              
           
          
           
            angle = geometry.angleFrom2Vectors(v_ankleFlexionAxis_opp,v_ank,self.getSegment("Right Shank").anatomicalFrame.static.m_axisZ)*360.0/(2.0*np.pi)

            self.mp_computed["rightProjectionAngle_AnkleFlexion_LateralAnkle"] = angle
            print "right projection offset"
            print self.mp_computed["rightProjectionAngle_AnkleFlexion_LateralAnkle"]
        

    def getAbdAddAnkleJointOffset(self,side="both"):

        if side == "both" or side == "left" :        
        
            ankleFlexionAxis=    np.dot(self.getSegment("Left Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Left Shank").anatomicalFrame.static.m_axisY)
        

                                 
            v_ankleFlexionAxis = ankleFlexionAxis/np.linalg.norm(ankleFlexionAxis)
            
            ANK =  self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LANK").m_local - \
                   self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LAJC").m_local              
            v_ank = ANK/np.linalg.norm(ANK)
            
            print "left AnkleAbAdd"
            angle = geometry.angleFrom2Vectors(v_ankleFlexionAxis,v_ank,self.getSegment("Left Shank").anatomicalFrame.static.m_axisX)*360.0/(2.0*np.pi)
            self.mp_computed["leftAJCAbAdOffset"] = angle
            print angle
            

        if side == "both" or side == "right" : 
            ankleFlexionAxis=    np.dot(self.getSegment("Right Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Right Shank").anatomicalFrame.static.m_axisY)
        
                                 
            v_ankleFlexionAxis = ankleFlexionAxis/np.linalg.norm(ankleFlexionAxis)
            
            v_ankleFlexionAxis_opp = geometry.oppositeVector(v_ankleFlexionAxis)
            ANK =  self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RANK").m_local - \
                   self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RAJC").m_local                     
            v_ank = ANK/np.linalg.norm(ANK)
           
            print "right AnkleAbAdd"
            angle = geometry.angleFrom2Vectors(v_ankleFlexionAxis_opp,v_ank,self.getSegment("Right Shank").anatomicalFrame.static.m_axisX)*360.0/(2.0*np.pi)
            self.mp_computed["rightAJCAbAdOffset"] = angle
            print angle

    def getFootOffset(self, side = "both"):
        
        if side == "both" or side == "left" :      
            R = self.getSegment("Left Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)
            
            print "left StaticPlantarFlexion"
            self.mp_computed["leftStaticPlantarFlexion"] = np.rad2deg(y)
            print np.rad2deg(y)
            
            print "left Static Rotation Offset"
            self.mp_computed["leftStaticRotOff"] = np.rad2deg(y)
            print np.rad2deg(x)
        
        if side == "both" or side == "right" :      
            R = self.getSegment("Right Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)    
            
            print "right StaticPlantarFlexion"
            self.mp_computed["rightStaticPlantarFlexion"] = np.rad2deg(y)
            print np.rad2deg(y)
            
            print "left Static Rotation Offset"
            self.mp_computed["rightStaticRotOff"] = np.rad2deg(y)
            print np.rad2deg(x)
        
    def getHindFootOffset(self, side = "both"):
        
        
        if side == "both" or side == "right" :      
            R = self.getSegment("Right Hindfoot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)    
            
            print "right hindfoot Static Plantar Flexion"
            self.mp_computed["rightStaticPlantarFlexion"] = np.rad2deg(y)
            print np.rad2deg(y)
            
            print "left hindFoot Static Rotation Offset"
            self.mp_computed["rightStaticRotOff"] = np.rad2deg(y)
            print np.rad2deg(x)


    
    def _pelvis_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):    
        """ pelvis
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        .. note:: 
        


        """             
        print "-- PELVIS -- "
        seg=self.getSegment("Pelvis")


        # ---  additional markers and Update of the marker segment list

        valSACR=(aquiStatic.GetPoint("LPSI").GetValues() + aquiStatic.GetPoint("RPSI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aquiStatic,"SACR",valSACR,desc="")        

        valMidAsis=(aquiStatic.GetPoint("LASI").GetValues() + aquiStatic.GetPoint("RASI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aquiStatic,"midASIS",valMidAsis,desc="")        

            
        if self.mp.has_key("pelvisDepth"):
            self.mp_computed["pelvisDepth"] = self.mp["pelvisDepth"]
        else:
            print "Pelvis Depth computed and added to model parameters"
            self.mp_computed["pelvisDepth"] = np.linalg.norm( valMidAsis.mean(axis=0)-valSACR.mean(axis=0)) - 2.0* (markerDiameter/2.0) -2.0* (basePlate/2.0)


        if self.mp.has_key("asisDistance"):
            self.mp_computed["asisDistance"] = self.mp["asisDistance"]
        else:
            print "asisDistance computed and added to model parameters"
            self.mp_computed["asisDistance"] = np.linalg.norm( aquiStatic.GetPoint("LASI").GetValues().mean(axis=0) - aquiStatic.GetPoint("RASI").GetValues().mean(axis=0))


        # virtual-technical markers add to segment
        seg.addMarkerLabel("SACR")
        seg.addMarkerLabel("midASIS")
        
  

        # --- technical frame selection and construction  
              
        tf=seg.getReferential("TF")
        
        
        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Pelvis"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


        # --- Hip Joint centers location 

        # anthropometric parameter computed

        self.mp_computed['leftAsisTrocanterDistance'] = 0.1288*self.mp['leftLegLength']-48.56
        self.mp_computed['rightAsisTrocanterDistance'] = 0.1288*self.mp['rightLegLength']-48.56
        self.mp_computed['meanlegLength'] = np.mean( [self.mp['leftLegLength'],self.mp['rightLegLength'] ])

#        self.mp_computed = { 'leftAsisTrocanterDistance' : 0.1288*self.mp['leftLegLength']-48.56,
#                      'rightAsisTrocanterDistance' : 0.1288*self.mp['rightLegLength']-48.56,
#                      'meanlegLength' : np.mean( [self.mp['leftLegLength'],self.mp['rightLegLength'] ]) }


        # local Position of the hip joint centers
        LHJC_loc,RHJC_loc= cgm.CGM.hipJointCenters(self.mp,self.mp_computed,markerDiameter)



        # --- nodes manager
        tf.static.addNode("LHJC_cgm1",LHJC_loc,positionType="Local")
        tf.static.addNode("RHJC_cgm1",RHJC_loc,positionType="Local")


        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")



        # --- Btk Points and decorator manager
        if self.nativeCgm1:
            print "-- cgm1 native -- "
            print "   -> btkpoints LHJC and RHJC append with description cgm1-- "
            val = tf.static.getNode_byLabel("LHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
            btkTools.smartAppendPoint(aquiStatic,"LHJC",val, desc="cgm1") 

            # construction of the btkPoint label (RLHJC)
            val = tf.static.getNode_byLabel("RHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
            btkTools.smartAppendPoint(aquiStatic,"RHJC",val, desc="cgm1")
        else:
            print "-- cgm1 decorated -- "
            
            print "   -> btkpoints LHJC_cgm1 and RHJC_cgm1 append -- "
            val = tf.static.getNode_byLabel("LHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LHJC_cgm1",val,desc="")        
                        
            val = tf.static.getNode_byLabel("RHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
            btkTools.smartAppendPoint(aquiStatic,"RHJC_cgm1",val,desc="")
            
            if "useLeftHJCnode" in options.keys():
                print "   ->  hjc left option found "

                nodeLabel = options["useLeftHJCnode"]
                desc = cmd.setDescription(nodeLabel)
                
                
                # set difference hipModify-and cgm1

                # construction of the btkPoint label (LHJC) 
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(aquiStatic,"LHJC",val,desc=desc)
            else:
                print "   -> hjc left option not found - lhjc comes from cgm1 "
                val = tf.static.getNode_byLabel("LHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
                btkTools.smartAppendPoint(aquiStatic,"LHJC",val,desc="cgm1")                
                
            
            if "useRightHJCnode" in options.keys():
                print "   ->  hjc right option found "                  

                nodeLabel = options["useRightHJCnode"]
                desc = cmd.setDescription(nodeLabel)

 
                # set difference hipModify-and cgm1

                
                # construction of the btkPoint label (RHJC)                
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(aquiStatic,"RHJC",val,desc=desc) 
            else:
                print "   -> hjc right option not found - rhjc comes from cgm1 "
                val = tf.static.getNode_byLabel("RHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
                btkTools.smartAppendPoint(aquiStatic,"RHJC",val,desc="cgm1")                  
        
        # final block (other points and final node HJC)
        # builing of mid HJC and final node HJC coming from option block
        val=(aquiStatic.GetPoint("LHJC").GetValues() + aquiStatic.GetPoint("RHJC").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aquiStatic,"midHJC",val,desc="")        

        final_LHJC = aquiStatic.GetPoint("LHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("LHJC",final_LHJC,positionType="Global")

        final_RHJC = aquiStatic.GetPoint("RHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("RHJC",final_RHJC,positionType="Global")

        seg.addMarkerLabel("LHJC")   
        seg.addMarkerLabel("RHJC")   


    def _left_thigh_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        """ 
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        .. note:: After segmental frame construction, we call chord method for locating knee joint center
        

        """ 

        print " --- LEFT THIGH ---"
        seg = self.getSegment("Left Thigh")


        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("LHJC")        


        
        # --- technical frame selection and construction  
        
        tf=seg.getReferential("TF")       
        
        pt1=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Thigh"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)



        # --- knee Joint centers location from chord method
        LKJC = cgm.CGM.chord( (self.mp["leftKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftThighOffset"] ) 
      
        # --- node manager
        tf.static.addNode("LKJC_chord",LKJC,positionType="Global")
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")



        # --- Btk Points and decorator manager
        if self.nativeCgm1:
            print "-- cgm1 native -- "
            print "   -> btkpoint LKJC append with description cgm1-- "
            val = tf.static.getNode_byLabel("LKJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LKJC",val,desc="cgm1")
        else:
            print "-- cgm1 decorated -- "
            
            print "   -> btkpoints LKJC_chord  appends -- "
        
            val = LKJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LKJC_chord",val,desc="")
            
        

            if "useLeftKJCnode" in options.keys():
                nodeLabel = options["useLeftKJCnode"]
                desc = cmd.setDescription(nodeLabel)
                
                # construction of the btkPoint label (LKJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"LKJC",val,desc=desc)
            else:
                 btkTools.smartAppendPoint(aquiStatic,"LKJC",val,desc="chord")
           

        # final block
        final_LKJC = aquiStatic.GetPoint("LKJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("LKJC",final_LKJC,positionType="Global")
        seg.addMarkerLabel("LKJC")

    def _right_thigh_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        """ 
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """ 
        
        print " --- RIGHT THIGH ---"
        seg = self.getSegment("Right Thigh")


        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("RHJC")        


        
        # --- technical frame selection and construction  
        
        tf=seg.getReferential("TF")       
        
        pt1=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Thigh"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)



        # --- knee Joint centers location

        RKJC = cgm.CGM.chord( (self.mp["rightKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3,beta=self.mp_computed["rightThighOffset"] ) # could consider decorqted LHJC

      
        # --- node manager
        tf.static.addNode("RKJC_chord",RKJC,positionType="Global")
      
      
      
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")



        # --- Btk Points and decorator manager
        if self.nativeCgm1:
            print "-- cgm1 native -- "
            print "   -> btkpoint RKJC append with description cgm1-- "
            val = tf.static.getNode_byLabel("RKJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RKJC",val,desc="cgm1")
        else:
            print "-- cgm1 decorated -- "
                          
            print "   -> btkpoints LKJC_chord  appends -- "
        
            val = RKJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RKJC_chord",val,desc="")

            if "useRightKJCnode" in options.keys():

                nodeLabel = options["useRightKJCnode"]
                desc = cmd.setDescription(nodeLabel)
    
                                                          
                # construction of the btkPoint label (LKJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"RKJC",val,desc=desc)
            else:
                 btkTools.smartAppendPoint(aquiStatic,"RKJC",val,desc="chord")
           

        # final block
        final_RKJC = aquiStatic.GetPoint("RKJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("RKJC",final_RKJC,positionType="Global")
        seg.addMarkerLabel("RKJC")
        
        
        
        
      
        
        
        
        
        



    def _left_shank_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        """ 
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """ 
        
        print " --- LEFT SHANK ---"
        seg = self.getSegment("Left Shank")
        
        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("LKJC")


        # --- technical frame selection and construction
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Shank"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


        # --- ankle Joint centers location
#        if self.mp["leftThighOffset"]!=0:
#            
#                pt3= self.getSegment("Left Thigh").static.
            
            
        LAJC = cgm.CGM.chord( (self.mp["leftAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftShankOffset"] )
        


        # --- node manager
        tf.static.addNode("LAJC_chord",LAJC,positionType="Global")
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")



        # --- Btk Points and decorator manager
        if self.nativeCgm1:
            print "-- cgm1 native -- "
            print "   -> btkpoint LAJC append with description cgm1-- "
            val = tf.static.getNode_byLabel("LAJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LAJC",val,desc="cgm1")
        else:
            print "-- cgm1 decorated -- "
            print "   -> btkpoints LAJC_chord  appends -- "
        
            val = LAJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LAJC_chord",val,desc="")

            if "useLeftAJCnode" in options.keys():
                nodeLabel = options["useLeftAJCnode"]
                desc = cmd.setDescription(nodeLabel)

                                                           
                # construction of the btkPoint label (LAJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"LAJC",val,desc=desc)
            else:
                 btkTools.smartAppendPoint(aquiStatic,"LAJC",val,desc="chord")
           

        # final block
        final_LAJC = aquiStatic.GetPoint("LAJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("LAJC",final_LAJC,positionType="Global")
        seg.addMarkerLabel("LAJC")


    def _right_shank_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        """ 
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """ 
        
        print "--- RIGHT SHANK ----"
        
        seg = self.getSegment("Right Shank")

        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("RKJC")

        # --- technical frame selection and construction
        tf=seg.getReferential("TF")
        
        pt1=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Shank"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- ankle Joint centers location

        RAJC = cgm.CGM.chord( (self.mp["rightAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["rightShankOffset"] )

        # --- node manager
        tf.static.addNode("RAJC_chord",RAJC,positionType="Global")

        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")

       #Btk Points and decorator manager
        if self.nativeCgm1:
            print "-- cgm1 native -- "
            print "   -> btkpoint RAJC append with description cgm1-- "
            val = tf.static.getNode_byLabel("RAJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RAJC",val,desc="cgm1")
        else:
            print "-- cgm1 decorated -- "
            print "   -> btkpoints RAJC_chord  appends -- "
        
            val = RAJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RAJC_chord",val,desc="")

            if "useRightAJCnode" in options.keys():

                nodeLabel = options["useRightAJCnode"]
                desc = cmd.setDescription(nodeLabel)

                                                           
                # construction of the btkPoint label (RAJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"RAJC",val,desc=desc)
            else:
                 btkTools.smartAppendPoint(aquiStatic,"RAJC",val,desc="chord")
           

        # final block
        final_RAJC = aquiStatic.GetPoint("RAJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("RAJC",final_RAJC,positionType="Global")
        seg.addMarkerLabel("RAJC")

    


    def _rightHindFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):    

        seg=self.getSegment("Right Hindfoot")
        seg.addMarkerLabel("RAJC")
        
        tf=seg.getReferential("TF")
                
        #   referential construction
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

        # node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")

    def _rightForeFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):    

        # calibration technique  
        print "-- Right Forefoot -- "
        seg=self.getSegment("Right Forefoot")
        tf=seg.getReferential("TF")
        
        
        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # toe
        pt2=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # cun
        pt3=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # M5
    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Forefoot"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")
   
    def _pelvis_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
        
            1. add content to seg.anatomicalFrame regarding the dictionnary dictAnatomic 
            2. add rletaive matrix to express Antomical Frame from a Referential.
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Pelvis")
        
        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Pelvis"]['sequence'])          
           
        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)


        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))
                
        # node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")
            
    def _left_thigh_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
        
            1. add content to seg.anatomicalFrame regarding the dictionnary dictAnatomic 
            2. add rletaive matrix to express Antomical Frame from a Referential.
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Left Thigh")
        
        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Left Thigh"]['sequence'])          
           
        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)


        # rigid matrix between Technical and anatonical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        
        # node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")



    def _right_thigh_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
        
            1. add content to seg.anatomicalFrame regarding the dictionnary dictAnatomic 
            2. add rletaive matrix to express Antomical Frame from a Referential.
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Right Thigh")
        
        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Right Thigh"]['sequence'])          
           
        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)


        tf=seg.getReferential("TF")
        
        tf.setRelativeMatrixAnatomic (np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")


    def _left_shank_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
        
            1. add content to seg.anatomicalFrame regarding the dictionnary dictAnatomic 
            2. add rletaive matrix to express Antomical Frame from a Referential.
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Left Shank")
        
        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Left Shank"]['sequence'])          
           
        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)


        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic (np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")


    def _right_shank_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
        
            1. add content to seg.anatomicalFrame regarding the dictionnary dictAnatomic 
            2. add rletaive matrix to express Antomical Frame from a Referential.
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Right Shank")
        
        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=(pt3-pt1)
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Right Shank"]['sequence'])          
           
        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())) 

        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")


    def _rightHindFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd, options = None):    
 
        seg=self.getSegment("Right Hindfoot")
              
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)                       
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        if dictAnatomic["Right Hindfoot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # not used
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)         

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("rightFlatHindFoot" in options.keys() and options["rightFlatHindFoot"]):
            logging.info("rightFlatHindFoot option enable")
            #pt2[2] = pt1[2]   
            pt1[2] = pt2[2]
        else:
            logging.info("rightFlatHindFoot option disable")

       
#        if ("rightHindFootFlat" in options.keys() and options["rightHindFootFlat"]):
#   
#
#           print " Hind Flat Foot -- > True" 
#            valCUN = aquiStatic.GetPoint("RCUN").GetValues()
#            valCUN[:,2] = aquiStatic.GetPoint("RAJC").GetValues()[:,2]
#     
#            pt1 =  valCUN[frameInit:frameEnd,:].mean(axis=0)
#            
#            btkTools.smartAppendPoint(aquiStatic,"cun-AJC",valCUN)
#        else:
#            print " no HindFoot Flat" 
#            pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)                
##        if self.modelOptions["rightHindFootFlat"] :
##            print " Hind Flat Foot -- > True" s
##            valCUN = aquiStatic.GetPoint("RCUN").GetValues()
##            valCUN[:,2] = aquiStatic.GetPoint("RAJC").GetValues()[:,2]
##         
##            pt1 =  valCUN[frameInit:frameEnd,:].mean(axis=0)
##        else:
#           
#        
#    
#        #   referential construction
#        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#    
#        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Hindfoot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        
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


        #tf=seg.getReferential("TF")
        #tf.setRelativeMatrixAnatomic (np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        tf=seg.getReferential("TF")
        trueRelativeMatrixAnatomic = np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())
        
        y,x,z = ceuler.euler_yxz(trueRelativeMatrixAnatomic)
        rotX =np.array([[1,0,0],
                        [0,np.cos(x),-np.sin(x)],
                         [0,np.sin(x),np.cos(x)]]) 
        rotY =np.array([[np.cos(y),0,np.sin(y)],
                        [0,1,0],
                         [-np.sin(y),0,np.cos(y)]]) 

        
        relativeMatrixAnatomic = np.dot(rotY,rotX)
          
        tf.setRelativeMatrixAnatomic( relativeMatrixAnatomic) 


        # node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")



    def _rightForeFoot_anatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
                
          
        seg=self.getSegment("Right Forefoot")
        
        #   referential construction

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

        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic (np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")        


    def displayStaticCoordinateSystem(self,aquiStatic,  segmentLabel, targetPointLabel, referential = "Anatomical" ):
        seg=self.getSegment(segmentLabel)
        if referential == "Anatomical":
            ref =seg.anatomicalFrame
            desc = "anatomical"
        else:
            ref = seg.getReferential("TF")
            desc = referential
       
        val =  np.dot(ref.static.getRotation() , np.array([100.0,0.0,0.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_X",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc=desc)
        val =  np.dot(ref.static.getRotation() , np.array([0.0,100.0,0.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_Y",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc=desc)
        val =  np.dot(ref.static.getRotation() , np.array([0.0,0,100.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_Z",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc=desc)


                


    # ----- Motion --------------
    def computeMotion(self,aqui, dictRef,dictAnat, motionMethod,options=None):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `motionMethod` (Enum motionMethod) - method use to optimize pose         

        """         

        if motionMethod == pyCGM2Enums.motionMethod.Native:
            self._pelvis_motion(aqui, dictRef, dictAnat)
            self._left_thigh_motion(aqui, dictRef, dictAnat)
            self._right_thigh_motion(aqui, dictRef, dictAnat)
 
            self._left_shank_motion(aqui, dictRef, dictAnat)        
            self._right_shank_motion(aqui, dictRef, dictAnat)
            
            self._right_hindFoot_motion(aqui, dictRef, dictAnat)
            self._right_foreFoot_motion(aqui, dictRef, dictAnat)
        
        if motionMethod == pyCGM2Enums.motionMethod.Sodervisk:
            self._pelvis_motion_optimize(aqui, dictRef,dictAnat,motionMethod)
            self._left_thigh_motion_optimize(aqui, dictRef,dictAnat,motionMethod)
            self._right_thigh_motion_optimize(aqui, dictRef,dictAnat,motionMethod)
            self._left_shank_motion_optimize(aqui, dictRef,dictAnat,motionMethod)        
            self._right_shank_motion_optimize(aqui, dictRef,dictAnat,motionMethod)
            self._rightHindFoot_motion_optimize(aqui, dictRef,dictAnat,motionMethod)
            self._rightForeFoot_motion_optimize(aqui, dictRef,dictAnat,motionMethod)
            

        self.displayMotionCoordinateSystem( aqui,  "Pelvis" , "Pelvis" )
        self.displayMotionCoordinateSystem( aqui,  "Left Thigh" , "LThigh" )
        self.displayMotionCoordinateSystem( aqui,  "Right Thigh" , "RThigh" )
        self.displayMotionCoordinateSystem( aqui,  "Left Shank" , "LShank" )
        self.displayMotionCoordinateSystem( aqui,  "Right Shank" , "RShank" )
        self.displayMotionCoordinateSystem( aqui,  "Right Hindfoot" , "RHindFoot" )
        self.displayMotionCoordinateSystem( aqui,  "Right Forefoot" , "RForeFoot" )
        
        btkTools.smartWriter(aqui, "tmp-dyn.c3d")

    def _pelvis_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `motionMethod` (Enum motionMethod) 

        """ 

        seg=self.getSegment("Pelvis")
        # reinit Technical Frame Motion (USEFUL if you work with several aquisitions)
        seg.getReferential("TF").motion =[]


        # Build necessary markers
        #---------------------------
        val=(aqui.GetPoint("LPSI").GetValues() + aqui.GetPoint("RPSI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aqui,"SACR",val, desc="")
         
        val=(aqui.GetPoint("LASI").GetValues() + aqui.GetPoint("RASI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aqui,"midASIS",val, desc="")


        # Technical Frame
        #----------------------
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Pelvis"]["TF"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(frame)

        # --- get nodes ( HJC)
        values_LHJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        values_RHJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")

        # --- Create or append btkPoint
        btkTools.smartAppendPoint(aqui,"LHJC",values_LHJCnode, desc="cgm1")
        btkTools.smartAppendPoint(aqui,"RHJC",values_RHJCnode, desc="cgm1")


        #  Anatomical Frame
        #-----------------------
#        seg.anatomicalFrame.motion=[]
#        # virtual marker needed
#        val=(aqui.GetPoint("LHJC").GetValues() + aqui.GetPoint("RHJC").GetValues()) / 2.0        
#        btkTools.smartAppendPoint(aqui,"midHJC",val,desc="")
#
#
#        for i in range(0,aqui.GetPointFrameNumber()):
#
#            pt1=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][0])).GetValues()[i,:]
#            pt2=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][1])).GetValues()[i,:]
#            pt3=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][2])).GetValues()[i,:]
#            ptOrigin=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][3])).GetValues()[i,:]             
#         
#         
#            a1=(pt2-pt1)
#            a1=a1/np.linalg.norm(a1)
#                     
#            v=(pt3-pt1)
#            v=v/np.linalg.norm(v)
#                    
#            a2=np.cross(a1,v)
#            a2=a2/np.linalg.norm(a2)
#
#            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Pelvis"]['sequence']) 
#            frame=cfr.Frame()                
#                
#            frame.m_axisX=x
#            frame.m_axisY=y
#            frame.m_axisZ=z
#            frame.setRotation(R)
#            frame.setTranslation(ptOrigin)
#
#            seg.anatomicalFrame.addMotionFrame(frame)

        # new method
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)


    def _pelvis_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `motionMethod` (Enum motionMethod) 


        .. note :: 
           

        """

        seg=self.getSegment("Pelvis")
        # reinit Technical Frame Motion (USEFUL if you work with several aquisitions)
        seg.getReferential("TF").motion =[]


        # check presence of markers in the acquisition
        if seg.m_tracking_markers != []:
            for label in seg.m_tracking_markers:
                if not btkTools.isPointExist(aqui,label):
                    raise Exception("[pycga] Pre-anatomical  calibration checking : Pelvis point %s doesn't exist"% label )

            print "OK (pelvis)==> all tracking markers are in the acquisition"


        # --- Motion of the Technical frame
        #-------------------------------------
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

        # --- get nodes ( HJC)
        values_LHJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        values_RHJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")

        
        # --- Create or append btkPoint
        btkTools.smartAppendPoint(aqui,"LHJC",values_LHJCnode, desc="opt")
        btkTools.smartAppendPoint(aqui,"RHJC",values_RHJCnode, desc="opt")


        # --- Motion of the Anatomical frame
        #-------------------------------------
        seg.anatomicalFrame.motion=[]
        
        # need midASIS
        val=(aqui.GetPoint("LASI").GetValues() + aqui.GetPoint("RASI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aqui,"midASIS",val, desc="")
        
        
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    def _left_thigh_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Left Thigh")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

                   
        #  Technical Frame
        #-----------------------
        LKJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))                   
                   
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Thigh"]["TF"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(frame)

            LKJCvalues[i,:] = cgm.CGM.chord( (self.mp["leftKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftThighOffset"] )
            
            
        # point build    
        btkTools.smartAppendPoint(aqui,"LKJC",LKJCvalues, desc="chord")

        #  Anatomical Frame
        #-----------------------
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)


#        seg.anatomicalFrame.motion=[]
#        for i in range(0,aqui.GetPointFrameNumber()):
#
#            pt1=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][0])).GetValues()[i,:]
#            pt2=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][1])).GetValues()[i,:]
#            pt3=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][2])).GetValues()[i,:]
#            ptOrigin=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][3])).GetValues()[i,:]             
#         
#         
#            a1=(pt2-pt1)
#            a1=a1/np.linalg.norm(a1)
#                     
#            v=(pt3-pt1)
#            v=v/np.linalg.norm(v)
#                    
#            a2=np.cross(a1,v)
#            a2=a2/np.linalg.norm(a2)
#
#            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Left Thigh"]['sequence']) 
#            frame=cfr.Frame()                
#                
#            frame.m_axisX=x
#            frame.m_axisY=y
#            frame.m_axisZ=z
#            frame.setRotation(R)
#            frame.setTranslation(ptOrigin)
#
#            seg.anatomicalFrame.addMotionFrame(frame)



    def _left_thigh_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Left Thigh")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

       
        
        # check presence of markers in the acquisition
        if seg.m_tracking_markers != []:
            for label in seg.m_tracking_markers:
                if not btkTools.isPointExist(aqui,label):
                    raise Exception("[pycga] Pre-anatomical  calibration checking : Left Thigh point %s doesn't exist"% label )

            print "OK (left thigh) => all tracking markers are in the acquisition"

        # --- Motion of the Technical frame
        #-------------------------------------
        # part 1: get back static global position ( look ou i use nodes)
                        
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0            
            for label in seg.m_tracking_markers: # recupere les tracki
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

        # --- get nodes ( HJC)
        values_LKJCnode=seg.getReferential('TF').getNodeTrajectory("LKJC")
       
        # create or append btkPoint
        btkTools.smartAppendPoint(aqui,"LKJC",values_LKJCnode, desc="opt")

        # --- Motion of the Anatomical frame
        #-------------------------------------
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame) 


    def _right_thigh_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Right Thigh")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

                   
        #  Technical Frame
        #-----------------------
        RKJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))                   
                   
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Thigh"]["TF"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(frame)

            RKJCvalues[i,:] = cgm.CGM.chord( (self.mp["rightKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["rightThighOffset"] )
            
            
        # point build    
        btkTools.smartAppendPoint(aqui,"RKJC",RKJCvalues, desc="chord")

        #  Anatomical Frame
        #-----------------------
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)
#        seg.anatomicalFrame.motion=[]
#        for i in range(0,aqui.GetPointFrameNumber()):
#
#            pt1=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][0])).GetValues()[i,:]
#            pt2=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][1])).GetValues()[i,:]
#            pt3=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][2])).GetValues()[i,:]
#            ptOrigin=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][3])).GetValues()[i,:]             
#         
#         
#            a1=(pt2-pt1)
#            a1=a1/np.linalg.norm(a1)
#                     
#            v=(pt3-pt1)
#            v=v/np.linalg.norm(v)
#                    
#            a2=np.cross(a1,v)
#            a2=a2/np.linalg.norm(a2)
#
#            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Right Thigh"]['sequence']) 
#            frame=cfr.Frame()                
#                
#            frame.m_axisX=x
#            frame.m_axisY=y
#            frame.m_axisZ=z
#            frame.setRotation(R)
#            frame.setTranslation(ptOrigin)
#
#            seg.anatomicalFrame.addMotionFrame(frame)


        

    def _right_thigh_motion_optimize(self,aqui, dictRef, dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Right Thigh")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

       
        
        # check presence of markers in the acquisition
        if seg.m_tracking_markers != []:
            for label in seg.m_tracking_markers:
                if not btkTools.isPointExist(aqui,label):
                    raise Exception("[pycga] Pre-anatomical  calibration checking : Left Thigh point %s doesn't exist"% label )

            print "OK (left thigh) => all tracking markers are in the acquisition"



        # --- Motion of the technical frame
        #---------------------------------------
        # part 1: get back static global position ( look ou i use nodes)
                        
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

        # --- get nodes ( HJC)
        values_RKJCnode=seg.getReferential('TF').getNodeTrajectory("RKJC")
       
        # create or append btkPoint
        btkTools.smartAppendPoint(aqui,"RKJC",values_RKJCnode, desc="opt")


        # --- Motion of the Anatomical frame
        #-------------------------------------
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)


    def _left_shank_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Left Shank")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

                   
        #  Anatomical Frame
        #-----------------------
        LAJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))                   
                   
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Shank"]["TF"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(frame)




            LAJCvalues[i,:] = cgm.CGM.chord( (self.mp["leftAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftShankOffset"] )
                
            # update of the AJC location with rotation around abdAddAxis 
            LAJCvalues[i,:] = self._rotateAjc(LAJCvalues[i,:],pt2,pt1,-self.mp_computed["leftAJCAbAdOffset"])
            

        # point build
        if self.mp_computed["leftAJCAbAdOffset"] > 0.01:
            desc="chord+AbAdRot"
        else:
            desc="chord"
            
        btkTools.smartAppendPoint(aqui,"LAJC",LAJCvalues, desc=desc)

        #  Anatomical Frame
        #-----------------------
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

#        seg.anatomicalFrame.motion=[]
        
#        for i in range(0,aqui.GetPointFrameNumber()):
#
#            pt1=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][0])).GetValues()[i,:]
#            pt2=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][1])).GetValues()[i,:]
#            pt3=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][2])).GetValues()[i,:]
#            ptOrigin=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3])).GetValues()[i,:]             
#         
#         
#            a1=(pt2-pt1)
#            a1=a1/np.linalg.norm(a1)
#                     
#            v=(pt3-pt1)
#            v=v/np.linalg.norm(v)
#                    
#            a2=np.cross(a1,v)
#            a2=a2/np.linalg.norm(a2)
#
#            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Left Shank"]['sequence']) 
#            frame=cfr.Frame()                
#                
#            frame.m_axisX=x
#            frame.m_axisY=y
#            frame.m_axisZ=z
#            frame.setRotation(R)
#            frame.setTranslation(ptOrigin)
#
#            seg.anatomicalFrame.addMotionFrame(frame)




    def _left_shank_motion_optimize(self,aqui, dictRef,dictAnat,  motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Left Shank")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

       
        
        # check presence of markers in the acquisition
        if seg.m_tracking_markers != []:
            for label in seg.m_tracking_markers:
                if not btkTools.isPointExist(aqui,label):
                    raise Exception("[pycga] Pre-anatomical  calibration checking : Left Shank point %s doesn't exist"% label )

            print "OK (left Shank) => all tracking markers are in the acquisition"

        # --- Motion of the technical frame
        #--------------------------------------

        # part 1: get back static global position ( look ou i use nodes)
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

        # --- get nodes ( AJC)
        values_LAJCnode=seg.getReferential('TF').getNodeTrajectory("LAJC")
       
        # create or append btkPoint
        btkTools.smartAppendPoint(aqui,"LAJC",values_LAJCnode, desc="opt")

        #  Anatomical Frame
        #-----------------------
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    def _rotateAjc(self,ajc,kjc,ank, offset):

        
        a1=(kjc-ajc)
        a1=a1/np.linalg.norm(a1)
        
        v=(ank-ajc)
        v=v/np.linalg.norm(v)
        
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)
        
        x,y,z,R=cfr.setFrameData(a1,a2,"ZXY") 
        frame=cfr.Frame()                
                
        frame.m_axisX=x
        frame.m_axisY=y
        frame.m_axisZ=z
        frame.setRotation(R)
        frame.setTranslation(ank)

        loc=np.dot(R.T,ajc-ank)

        abAdangle = offset*(2*np.pi/360.0)
       
        rotAbdAdd = np.array([[1, 0, 0],[0, np.cos(abAdangle), -1.0*np.sin(abAdangle)], [0, np.sin(abAdangle), np.cos(abAdangle) ]])

        finalRot= np.dot(R,rotAbdAdd)
        
        return  np.dot(finalRot,loc)+ank



    

    def _right_shank_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Right Shank")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

                   
        #  Technical Frame
        #-----------------------
        RAJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))                   
                   
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][0])).GetValues()[i,:] #ank
            pt2=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][1])).GetValues()[i,:] #kjc
            pt3=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][2])).GetValues()[i,:] #tib
            ptOrigin=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Shank"]["TF"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)
            
            seg.getReferential("TF").addMotionFrame(frame)

            # ajc position from chord modified by shank offset 
            RAJCvalues[i,:] = cgm.CGM.chord( (self.mp["rightAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["rightShankOffset"] )

            # update of the AJC location with rotation around abdAddAxis 
            RAJCvalues[i,:] = self._rotateAjc(RAJCvalues[i,:],pt2,pt1,   self.mp_computed["rightAJCAbAdOffset"])

        # point build
        if self.mp_computed["rightAJCAbAdOffset"] >0.01:
            desc="chord+AbAdRot"
        else:
            desc="chord"
            
        btkTools.smartAppendPoint(aqui,"RAJC",RAJCvalues, desc=desc)
        

        #  Anatomical Frame
        #-----------------------

        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)



#        seg.anatomicalFrame.motion=[]
#        for i in range(0,aqui.GetPointFrameNumber()):
#
#            pt1=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][0])).GetValues()[i,:] # ajc
#            pt2=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][1])).GetValues()[i,:] # kjc
#            pt3=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][2])).GetValues()[i,:] # ANK
#            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:]             
#         
#         
#            a1=(pt2-pt1)
#            a1=a1/np.linalg.norm(a1)
#                     
#            v=(pt3-pt1)
#            v=v/np.linalg.norm(v)
#                    
#            a2=np.cross(a1,v)
#            a2=a2/np.linalg.norm(a2)
#
#            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Right Shank"]['sequence']) 
#            frame=cfr.Frame()                
#                
#            frame.m_axisX=x
#            frame.m_axisY=y
#            frame.m_axisZ=z
#            frame.setRotation(R)
#            frame.setTranslation(ptOrigin)
#
#            seg.anatomicalFrame.addMotionFrame(frame)





    def _right_shank_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Right Shank")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

       
        
        # check presence of markers in the acquisition
        if seg.m_tracking_markers != []:
            for label in seg.m_tracking_markers:
                if not btkTools.isPointExist(aqui,label):
                    raise Exception("[pycga] Pre-anatomical  calibration checking : Left Thigh point %s doesn't exist"% label )

            print "OK (right Shank) => all tracking markers are in the acquisition"



        # --- Motion of the technical frame
        #-------------------------------------
        # part 1: get back static global position ( look ou i use nodes)
                        
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

        # --- get nodes 
        values_RAJCnode=seg.getReferential('TF').getNodeTrajectory("RAJC")
       
        # create or append btkPoint
        btkTools.smartAppendPoint(aqui,"RAJC",values_RAJCnode, desc="opt")


        #  Anatomical Frame
        #-----------------------

        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    

    def _right_hindFoot_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Right Hindfoot")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

                   
        #  Technical Frame
        #-----------------------
                   
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
            
        # anatomical axis
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Hindfoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)


    def _right_foreFoot_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
        """
        seg=self.getSegment("Right Forefoot")
        # reinit Technical Frame Motion ()
        seg.getReferential("TF").motion =[]

                   
        #  Technical Frame
        #-----------------------
                   
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][0])).GetValues()[i,:] 
            pt2=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][1])).GetValues()[i,:] 
            pt3=aqui.GetPoint(str(dictRef["Right Forefoot"]["TF"]['labels'][2])).GetValues()[i,:] 
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
            
        
        # ajout du point RvCUN
        btkTools.smartAppendPoint(aqui,"RvCUN",seg.getReferential("TF").getNodeTrajectory("RvCUN") )
        btkTools.smartAppendPoint(aqui,"RvTOE",seg.getReferential("TF").getNodeTrajectory("RvTOE") )
        btkTools.smartAppendPoint(aqui,"RvD5M",seg.getReferential("TF").getNodeTrajectory("RvD5M") ) 
        
        # anatomical Frame
        # ------------------    
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Forefoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)       

    def _rightHindFoot_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):
    
        seg=self.getSegment("Right Hindfoot")
        # reinit Technical Frame Motion (USEFUL if you work with several aquisitions)
        seg.getReferential("TF").motion =[]


        # check presence of markers in the acquisition
        if seg.m_tracking_markers != []:
            for label in seg.m_tracking_markers:
                if not btkTools.isPointExist(aqui,label):
                    raise Exception("[pycga] Pre-anatomical  calibration checking : Right Hindfoot point %s doesn't exist"% label )

            print "OK (Right Hindfoot)==> all tracking markers are in the acquisition"


        # --- Motion of the Technical frame
        #-------------------------------------
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

        # anatomical Frame
        # ------------------    
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
        # reinit Technical Frame Motion (USEFUL if you work with several aquisitions)
        seg.getReferential("TF").motion =[]

        # check presence of markers in the acquisition
        if seg.m_tracking_markers != []:
            for label in seg.m_tracking_markers:
                if not btkTools.isPointExist(aqui,label):
                    raise Exception("[pycga] Pre-anatomical  calibration checking : Right Hindfoot point %s doesn't exist"% label )

            print "OK (Right Forefoot)==> all tracking markers are in the acquisition"


        # --- Motion of the Technical frame
        #-------------------------------------
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

        # anatomical Frame
        # ------------------    
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Forefoot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)




    def displayMotionCoordinateSystem(self,acqui,  segmentLabel, targetPointLabel, referential = "Anatomical" ):
        seg=self.getSegment(segmentLabel)
        valX=np.zeros((acqui.GetPointFrameNumber(),3))
        valY=np.zeros((acqui.GetPointFrameNumber(),3))
        valZ=np.zeros((acqui.GetPointFrameNumber(),3))        
        
        
        if referential == "Anatomical":
            ref =seg.anatomicalFrame
            desc = "anatomical"
        else:
            ref = seg.getReferential("TF")
            desc = "technical"
            
        for i in range(0,acqui.GetPointFrameNumber()):    
            valX[i,:]= np.dot(ref.motion[i].getRotation() , np.array([100.0,0.0,0.0])) + ref.motion[i].getTranslation()
            valY[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,100.0,0.0])) + ref.motion[i].getTranslation()
            valZ[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,0.0,100.0])) + ref.motion[i].getTranslation()       
       
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_X",valX,desc=desc)
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_Y",valY,desc=desc)
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_Z",valZ,desc=desc)

		
   













        
        

#class Cgm15ModelInf(PigModelInf):
#    def __init__(self):
#        """
#        .. Note:: cgm1.5 inherits PigModelInf !
#
#        .. warning : by the way, PigModelInf is configure firstly   
#        
#        """
#        super(Cgm15ModelInf, self).__init__()
#        self._updateMarkers()
#
#    def __repr__(self):
#        return "cgm1.5"    
#
#    
#    def _updateMarkers(self):
#        
#        pdb.set_trace()
#        # Left thigh
#        self.getSegment("Left Thigh").m_tracking_markers.remove('LHJC') # remove is a standard method list
#        leftThighMarkerLabels = ["LTHAD","LTHL","LTHLD"]
#        for label in leftThighMarkerLabels:
#            self.getSegment("Left Thigh").addMarkerLabel(label)
#            self.getSegment("Left Thigh").addTrackingMarkerLabel(label)
#            
#            
#        # Left shank
#        self.getSegment("Left Shank").m_tracking_markers.remove('LKJC') # remove is a standard method list
#        leftShankMarkerLabels = ["LTIAD","LTIAP","LTIBL"]
#        for label in leftShankMarkerLabels:
#            self.getSegment("Left Shank").addMarkerLabel(label)
#            self.getSegment("Left Shank").addTrackingMarkerLabel(label)
#
#    
#
#
#    def calibrationProcedure(self):
#        """ overloading of the calibration procedure. 
#        
#        we add new Frames on both Thigh and Shank
#        
#        """
#
#        dictRef={}
#        dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
#        dictRef["Left Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["LKNE","LHJC","LTHI","LKNE"]},
#                              "TF_anaCalib" : {'sequence':"ZXY", 'labels':   ["LKNE","LTHI","LTHAD","LKNE"]} }
#        
#        dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RKNE","RHJC","RTHI","RKNE"]} }
#
#        
#        dictRef["Left Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["LANK","LKJC","LTIB","LANK"]},
#                              "TF_anaCalib" : {'sequence':"ZXY", 'labels':   ["LANK","LTIAP","LTIAD","LANK"]} }
#        dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RANK","RKJC","RTIB","RANK"]} }
#        
#        
#        dictRefAnatomical={}
#        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midHJC"]} 
#        dictRefAnatomical["Left Thigh"]= {'sequence':"YZX", 'labels':  ["LKJC","LHJC","LTHI","LKJC"]} 
#        dictRefAnatomical["Right Thigh"]= {'sequence':"YZX", 'labels': ["RKJC","RHJC","RTHI","RKJC"]} 
#        dictRefAnatomical["Left Shank"]={'sequence':"ZXY", 'labels':   ["LAJC","LKJC","LTIB","LAJC"]} 
#        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RTIB","RAJC"]}         
#        
#        return dictRef,dictRefAnatomical        
#        
#        
#        
#    def calibrate(self,aquiStatic, dictRef,dictAnatomic,options=None):
#        """ Overloading of the method calibrate (static calibration) 
#        
#        :Parameters:
#        
#           - `aquiStatic` (btkAcquisition) - acquisition
#           - `dictRef` (dict) - dictionray explaining how to construct technical referentials
#           - `dictAnatomic` (dict) - dictionray explaining how to construct the anatomical Frame
#           - `options` (kwargs) - use to pass option altering the standard construction
#
#        .. note:: Notice that  the pelvis calibration method is not overloaded
#
#        .. todo:: shrink and clone the aquisition to seleted frames   
#        """
#
#        
#        ff=aquiStatic.GetFirstFrame()        
#        frameInit=0 #frameInit-ff  
#        frameEnd=1 #frameEnd-ff+1         
#
#        # TF calibration from CGM1
#        print "------ CALIBRATION DU CGM1 ------"
#        self._pelvis_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
#        self._left_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
#        self._right_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
#        self._left_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
#        self._right_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
#
#        # specific TF calibration from CGM15
#        print "------ CALIBRATION DU CGM1.5 ------"
#        self._left_thigh_functionalFrame_calibrate(aquiStatic, dictRef,frameInit,frameEnd, options=options)
#        self._left_shank_functionalFrame_calibrate(aquiStatic, dictRef,frameInit,frameEnd, options=options)
#
#
#        writer = btk.btkAcquisitionFileWriter()
#        writer.SetInput(aquiStatic)
#        writer.SetFilename("C:\\Users\\AAA34169\\Documents\\Programming\\API\\pyCGA-DATA\\tmp-static.c3d")
#        writer.Update()
#        
#       
#    def _left_thigh_functionalFrame_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):    
#        """ 
#        :Parameters:
#        
#           - `aquiStatic` (btkAcquisition) - acquisition
#           - `dictRef` (dict) - instance of Model
#           - `frameInit` (int) - starting frame
#           - `frameEnd` (int) - starting frame           
#           - `options` (kwargs) - use to pass option altering the standard construction
#
#        .. todo:: we can recast this method as generic method by adding the frame Label as input argument.
#           is it worth ?
#
#        """ 
#        
#        seg=self.getSegment("Left Thigh")
#        
#
#
#        # --- technical frame selection and construction  
#              
#        tf=seg.getReferential("TF_anaCalib") # TODO a jouter exception si trouve pas la clé TF_anaCalib
#        
#        #   referential construction
#        pt1=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        pt2=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        pt3=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#    
#        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        
#        a1=(pt2-pt1)
#        a1=a1/np.linalg.norm(a1)
#                    
#        v=(pt3-pt1)
#        v=v/np.linalg.norm(v)
#                    
#        a2=np.cross(a1,v)
#        a2=a2/np.linalg.norm(a2)
#
#        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Thigh"]["TF_anaCalib"]['sequence'])          
#           
#        tf.static.m_axisX=x
#        tf.static.m_axisY=y
#        tf.static.m_axisZ=z
#        tf.static.setRotation(R)
#        tf.static.setTranslation(ptOrigin)
#
#
#        # --- node manager
#        
#        for label in seg.m_markerLabels: #LHJC is in the list
#            print label
#            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
#            tf.static.addNode(label,globalPosition,positionType="Global")
#
#
#    def _left_shank_functionalFrame_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):    
#        """ 
#        :Parameters:
#        
#           - `aquiStatic` (btkAcquisition) - acquisition
#           - `dictRef` (dict) - instance of Model
#           - `frameInit` (int) - starting frame
#           - `frameEnd` (int) - starting frame           
#           - `options` (kwargs) - use to pass option altering the standard construction
#
#        """        
#        
#        seg=self.getSegment("Left Shank")
# 
#
#        # --- technical frame selection and construction  
#              
#        tf=seg.getReferential("TF_anaCalib")
#        
#        #   referential construction
#        pt1=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        pt2=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        pt3=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#    
#        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
#        
#        a1=(pt2-pt1)
#        a1=a1/np.linalg.norm(a1)
#                    
#        v=(pt3-pt1)
#        v=v/np.linalg.norm(v)
#                    
#        a2=np.cross(a1,v)
#        a2=a2/np.linalg.norm(a2)
#
#        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Shank"]["TF_anaCalib"]['sequence'])          
#           
#        tf.static.m_axisX=x
#        tf.static.m_axisY=y
#        tf.static.m_axisZ=z
#        tf.static.setRotation(R)
#        tf.static.setTranslation(ptOrigin)
#
#        # --- node manager
#        for label in seg.m_markerLabels:
#            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
#            tf.static.addNode(label,globalPosition,positionType="Global")
#        
#    # ---- Motion --------    
#    def computeMotion(self,aqui, dictRef, motionMethod,goal):
#        """ 
#        :Parameters:
#        
#           - `aqui` (btkAcquisition) - acquisition
#           - `dictRef` (dict) - instance of Model
#           - `motionMethod` (Enum motionMethod) - method use to optimize pose
#           - `goal` (Enum motionGoal) - goal of the acquisition           
#
#        """   
#        
#        if goal == pyCGM2Enums.motionGoal.Full:
#            # TF motion from CGM1
#            self._pelvis_motion(aqui, dictRef, motionMethod)
#            self._left_thigh_motion(aqui, dictRef, motionMethod)
#            self._right_thigh_motion(aqui, dictRef, motionMethod)
#            self._left_shank_motion(aqui, dictRef, motionMethod)        
#            self._right_shank_motion(aqui, dictRef, motionMethod)
#            
#            # TF motion from CGM1.5
#            self._left_thigh_functionalFrame_motion(aqui, dictRef, motionMethod)
#            self._left_shank_functionalFrame_motion(aqui, dictRef, motionMethod)
#
#        if goal == pyCGM2Enums.motionGoal.LeftFunctionalKneeAxis:
#            self._pelvis_motion(aqui, dictRef, motionMethod)
#            self._left_thigh_functionalFrame_motion(aqui, dictRef, motionMethod)
#            self._left_shank_functionalFrame_motion(aqui, dictRef, motionMethod)
#
#        writer = btk.btkAcquisitionFileWriter()
#        writer.SetInput(aqui)
#        writer.SetFilename("C:\\Users\\AAA34169\\Documents\\Programming\\API\\pyCGA-DATA\\tmp-Dyn.c3d")
#        writer.Update()
#
#    def _left_thigh_functionalFrame_motion(self,aqui, dictRef, motionMethod):
#        """ 
#        :Parameters:
#        
#           - `aqui` (btkAcquisition) - acquisition
#           - `dictRef` (dict) - instance of Model
#           - `frameInit` (int) - starting frame
#           - `motionMethod` (Enum motionMethod) - method use to optimize pose
#
#        """
#        seg=self.getSegment("Left Thigh")
#        
#        pdb.set_trace()
#        
#        # check presence of markers in the acquisition
#        if seg.m_tracking_markers != []:
#            for label in seg.m_tracking_markers:
#                if not btkTools.isPointExist(aqui,label):
#                    raise Exception("[pycga] pre calibration of functional Frame : Left Thigh point %s doesn't exist"% label )
#
#            print "OK (left thigh) => all tracking markers are in the acquisition"
#
#        else:
#            print "tracking marker =  marker defined in the calibration procedure !! "        
#        
#        
#        # --- Motion of the technical frame
#        
#        
#        # part 1: get back static global position ( look ou i use nodes)
#        if motionMethod == pyCGM2Enums.motionMethod.Sodervisk :
#                       
#            if seg.m_tracking_markers != []: # work with tracking markers
#                staticPos = np.zeros((len(seg.m_tracking_markers),3))
#                i=0            
#                for label in seg.m_tracking_markers:
#                    staticPos[i,:] = seg.getReferential("TF_anaCalib").static.getNode_byLabel(label).m_global
#                    i+=1                
#            else:
#                staticPos = np.zeros((3,3))
#                i=0            
#                for label in dictRef["Left Thigh"]["TF_anaCalib"]['labels'][0:3]:
#                    staticPos[i,:] = seg.getReferential("TF_anaCalib").static.getNode_byLabel(label).m_global
#                    i+=1
#                    
#        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
#        for i in range(0,aqui.GetPointFrameNumber()):
#
#            if seg.m_tracking_markers != []: # work with traking markers 
#                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use 
#                k=0            
#                for label in seg.m_tracking_markers:
#                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
#                    k+=1 
#            else:
#                dynPos = np.zeros((3,3)) # array ( line = global poistion of a marker)
#                k=0            
#                for label in dictRef["Left Thigh"]["TF_anaCalib"]['labels'][0:3]: 
#                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
#                    k+=1
#            
#        
#            if motionMethod == pyCGM2Enums.motionMethod.None :
#                pt1=aqui.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][0])).GetValues()[i,:]
#                pt2=aqui.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][1])).GetValues()[i,:]
#                pt3=aqui.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][2])).GetValues()[i,:]
#                ptOrigin=aqui.GetPoint(str(dictRef["Left Thigh"]["TF_anaCalib"]['labels'][3])).GetValues()[i,:]             
#             
#             
#                a1=(pt2-pt1)
#                a1=a1/np.linalg.norm(a1)
#                         
#                v=(pt3-pt1)
#                v=v/np.linalg.norm(v)
#                        
#                a2=np.cross(a1,v)
#                a2=a2/np.linalg.norm(a2)
#
#                x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Thigh"]["TF_anaCalib"]['sequence']) 
#                frame=cfr.Frame()                
#                    
#                frame.m_axisX=x
#                frame.m_axisY=y
#                frame.m_axisZ=z
#                frame.setRotation(R)
#                frame.setTranslation(ptOrigin)
#        
#        
#        
#            elif motionMethod == pyCGM2Enums.motionMethod.Sodervisk :
#                Ropt, Lopt, RMSE, Am, Bm=cmot.segmentalLeastSquare(staticPos, 
#                                                              dynPos)
#                R=np.dot(Ropt,seg.getReferential("TF_anaCalib").static.getRotation())    
#                tOri=np.dot(Ropt,seg.getReferential("TF_anaCalib").static.getTranslation())+Lopt                                        
#                            
#                frame=cfr.Frame()
#                frame.setRotation(R)
#                frame.setTranslation(tOri)
#                frame.m_axisX=R[:,0]
#                frame.m_axisY=R[:,1]
#                frame.m_axisZ=R[:,2]
#                
#            seg.getReferential("TF_anaCalib").addMotionFrame(frame)
#
#
#    def _left_shank_functionalFrame_motion(self,aqui, dictRef, motionMethod):
#        """ 
#        :Parameters:
#        
#           - `aqui` (btkAcquisition) - acquisition
#           - `dictRef` (dict) - instance of Model
#           - `frameInit` (int) - starting frame
#           - `motionMethod` (Enum motionMethod) - method use to optimize pose
#
#        """
#        seg=self.getSegment("Left Shank")
#        
#         # check presence of markers in the acquisition
#        if seg.m_tracking_markers != []:
#            for label in seg.m_tracking_markers:
#                if not btkTools.isPointExist(aqui,label):
#                    raise Exception("[pycga] Pre-anatomical  calibration checking : Left Shank point %s doesn't exist"% label )
#
#            print "OK (left Shank) => all tracking markers are in the acquisition"
#
#        else:
#            print "tracking marker =  marker defined in the calibration procedure !! "        
#        
#        
#        # --- Motion of the technical frame
#        
#        
#        # part 1: get back static global position ( look ou i use nodes)
#        if motionMethod == pyCGM2Enums.motionMethod.Sodervisk :
#                       
#            if seg.m_tracking_markers != []: # work with tracking markers
#                staticPos = np.zeros((len(seg.m_tracking_markers),3))
#                i=0            
#                for label in seg.m_tracking_markers:
#                    staticPos[i,:] = seg.getReferential("TF_anaCalib").static.getNode_byLabel(label).m_global
#                    i+=1                
#            else:
#                staticPos = np.zeros((3,3))
#                i=0            
#                for label in dictRef["Left Shank"]["TF_anaCalib"]['labels'][0:3]:
#                    staticPos[i,:] = seg.getReferential("TF_anaCalib").static.getNode_byLabel(label).m_global
#                    i+=1
#                    
#        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
#        for i in range(0,aqui.GetPointFrameNumber()):
#
#            if seg.m_tracking_markers != []: # work with traking markers 
#                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use 
#                k=0            
#                for label in seg.m_tracking_markers:
#                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
#                    k+=1 
#            else:
#                dynPos = np.zeros((3,3)) # array ( line = global poistion of a marker)
#                k=0            
#                for label in dictRef["Left Shank"]["TF_anaCalib"]['labels'][0:3]: 
#                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
#                    k+=1
#            
#        
#            if motionMethod == pyCGM2Enums.motionMethod.None :
#                pt1=aqui.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][0])).GetValues()[i,:]
#                pt2=aqui.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][1])).GetValues()[i,:]
#                pt3=aqui.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][2])).GetValues()[i,:]
#                ptOrigin=aqui.GetPoint(str(dictRef["Left Shank"]["TF_anaCalib"]['labels'][3])).GetValues()[i,:]             
#             
#             
#                a1=(pt2-pt1)
#                a1=a1/np.linalg.norm(a1)
#                         
#                v=(pt3-pt1)
#                v=v/np.linalg.norm(v)
#                        
#                a2=np.cross(a1,v)
#                a2=a2/np.linalg.norm(a2)
#
#                x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Shank"]["TF_anaCalib"]['sequence']) 
#                frame=cfr.Frame()                
#                    
#                frame.m_axisX=x
#                frame.m_axisY=y
#                frame.m_axisZ=z
#                frame.setRotation(R)
#                frame.setTranslation(ptOrigin)
#        
#        
#        
#            elif motionMethod == pyCGM2Enums.motionMethod.Sodervisk :
#                Ropt, Lopt, RMSE, Am, Bm=cmot.segmentalLeastSquare(staticPos, 
#                                                              dynPos)
#                R=np.dot(Ropt,seg.getReferential("TF_anaCalib").static.getRotation())    
#                tOri=np.dot(Ropt,seg.getReferential("TF_anaCalib").static.getTranslation())+Lopt                                        
#                            
#                frame=cfr.Frame()
#                frame.setRotation(R)
#                frame.setTranslation(tOri)
#                frame.m_axisX=R[:,0]
#                frame.m_axisY=R[:,1]
#                frame.m_axisZ=R[:,2]
#                
#            seg.getReferential("TF_anaCalib").addMotionFrame(frame)
# 