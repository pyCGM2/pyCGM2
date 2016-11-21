# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 10:59:01 2015
@author: fleboeuf
"""


import numpy as np
import logging
import pdb

import model as cmb
import modelDecorator as cmd
import frame as cfr
import motion as cmot
import euler as ceuler

import pyCGM2.Core.enums as pyCGM2Enums
from pyCGM2.Core.Math import geometry
from pyCGM2.Core.Tools import  btkTools



class CGM(cmb.Model):

    def __init__(self):
        """Constructor 
        
          - Run configuration internally
          - Initialize deviation data  

        """
        super(CGM, self).__init__()    
    
    @classmethod
    def hipJointCenters(cls,mp_input,mp_computed,markerDiameter):
        """Class method : hip joint center regression of the CGM1 
        
        .. todo : do i build distinct anthropo dictionnary instead on one dict ?       
        
        """        
        
        C=mp_computed["meanlegLength"] * 0.115 - 15.3
        
        HJCx_L= C * np.cos(0.5) * np.sin(0.314) - (mp_computed["leftAsisTrocanterDistance"] + markerDiameter/2) * np.cos(0.314)
        HJCy_L=-1*(C * np.sin(0.5) - (mp_computed["asisDistance"] / 2.0))              
        HJCz_L= - C * np.cos(0.5) * np.cos(0.314) - (mp_computed["leftAsisTrocanterDistance"] + markerDiameter/2) * np.sin(0.314)

        HJC_L=np.array([HJCx_L,HJCy_L,HJCz_L])

        HJCx_R= C * np.cos(0.5) * np.sin(0.314) - (mp_computed["rightAsisTrocanterDistance"] + markerDiameter/2) * np.cos(0.314)
        HJCy_R=+1*(C * np.sin(0.5) - (mp_computed["asisDistance"] / 2.0))              
        HJCz_R= -C * np.cos(0.5) * np.cos(0.314) - (mp_computed["rightAsisTrocanterDistance"] + markerDiameter/2) * np.sin(0.314)

        HJC_R=np.array([HJCx_R,HJCy_R,HJCz_R])
        
        return HJC_L,HJC_R
    
    @classmethod
    def chord (cls,offset,I,J,K,beta=0.0):
        """Class method : chord and chord modified from Morgan
        
        if modified :
        A=J
        B=I
        C=K
        
        % Find the KJC (P) from the HJC (A), THI or KAX markers (C)
        % and KNE marker (B)
        % if Beta is provide, the plan projection of PC and PB equal Beta
        
        
        """        
 
        if beta == 0.0:
            y=(J-I)/np.linalg.norm(J-I)
            x=np.cross(y,K-I)
            x=(x)/np.linalg.norm(x)
            z=np.cross(x,y)
    
            matR=np.array([x,y,z]).T
            ori=(J+I)/2.0
            
            d=np.linalg.norm(I-J)
            theta=np.arcsin(offset/d)*2.0
            v_r=np.array([0, -d/2, 0])
            
            rot=np.array([[1,0,0],[0,np.cos(theta),-1.0*np.sin(theta)],[0,np.sin(theta),np.cos(theta)] ])
            
           
            return np.dot(np.dot(matR,rot),v_r)+ori    

        else:
 
            A=J
            B=I
            C=K
            L=offset
        
        
            eps = 0.00000001
                    
        
            AB = np.linalg.norm(A-B) 
            alpha = np.arcsin(L/AB)
            AO = np.sqrt(AB*AB-L*L*(1+np.cos(alpha)*np.cos(alpha)))
            
            # chord avec beta nul
            #P = chord(L,B,A,C,beta=0.0) # attention ma methode . attention au arg input
            
            y=(J-I)/np.linalg.norm(J-I)
            x=np.cross(y,K-I)
            x=(x)/np.linalg.norm(x)
            z=np.cross(x,y)
    
            matR=np.array([x,y,z]).T
            ori=(J+I)/2.0
            
            d=np.linalg.norm(I-J)
            theta=np.arcsin(offset/d)*2.0
            v_r=np.array([0, -d/2, 0])
            
            rot=np.array([[1,0,0],[0,np.cos(theta),-1.0*np.sin(theta)],[0,np.sin(theta),np.cos(theta)] ])
            
           
            P= np.dot(np.dot(matR,rot),v_r)+ori 
            # fin chord 0
                 
        
            Salpha = 0            
            diffBeta = np.abs(beta)
            alphaincr = beta # in degree
                        
        
            # define P research circle in T plan
            n = (A-B)/AB
            O = A - np.dot(n, AO)
            r = L*np.cos(alpha) #OK
        
        
            # build segment
            #T = BuildSegment(O,n,P-O,'zyx');
            Z=n/np.linalg.norm(n)
            Y=np.cross(Z,P-O)/np.linalg.norm(np.cross(Z,P-O))
            X=np.cross(Y,Z)/np.linalg.norm(np.cross(Y,Z))
            Origin= O
            
            # erreur ici, il manque les norm
            T=np.array([[ X[0],Y[0],Z[0],Origin[0] ],
                        [ X[1],Y[1],Z[1],Origin[1] ],
                        [ X[2],Y[2],Z[2],Origin[2] ],
                        [    0,   0,   0,       1.0  ]])
        
            count = 0
            while diffBeta > eps or count > 1000:
                if count > 1000:
                    logging.warning("count boubdary achieve")

                        	
                count = count + 1 
                idiff = diffBeta
                
                Salpha = Salpha + alphaincr
                Salpharad = Salpha * np.pi / 180.0
                Pplan = np.array([  [r*np.cos(Salpharad)],
                                    [ r*np.sin(Salpharad)],
                                     [0],
                                    [1]])
                P = np.dot(T,Pplan)
        
                P = P[0:3,0]
                nBone = A-P
        
                ProjC = np.cross(nBone,np.cross(C-P,nBone))
                ProjB = np.cross(nBone,np.cross(B-P,nBone))                
        
                
                sens = np.dot(np.cross(ProjC,ProjB).T,nBone)
            
        
                
                Betai = sens/np.linalg.norm(sens)*np.arccos((np.dot(ProjC.T,ProjB))/(np.linalg.norm(ProjC)*np.linalg.norm(ProjB)))*180.0/np.pi
                
                diffBeta = np.abs(beta - Betai)
                
                if (diffBeta - idiff) > 0:
                    if count == 1:
                        Salpha = Salpha - alphaincr
                        alphaincr = -alphaincr
                    else:
                        alphaincr = -alphaincr / 2;
                        
        
            return P  

    def displayStaticCoordinateSystem(self,aquiStatic,  segmentLabel, targetPointLabel, referential = "Anatomic" ):
        seg=self.getSegment(segmentLabel)
        if referential == "Anatomic":
            ref =seg.anatomicalFrame
        else:
            ref = seg.getReferential("TF") 
       
        val =  np.dot(ref.static.getRotation() , np.array([100.0,0.0,0.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_X",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc="")
        val =  np.dot(ref.static.getRotation() , np.array([0.0,100.0,0.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_Y",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc="")
        val =  np.dot(ref.static.getRotation() , np.array([0.0,0,100.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_Z",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc="")

    def displayMotionCoordinateSystem(self,acqui,  segmentLabel, targetPointLabel, referential = "Anatomic" ):
        seg=self.getSegment(segmentLabel)
        valX=np.zeros((acqui.GetPointFrameNumber(),3))
        valY=np.zeros((acqui.GetPointFrameNumber(),3))
        valZ=np.zeros((acqui.GetPointFrameNumber(),3))        
        
        
        if referential == "Anatomic":
            ref =seg.anatomicalFrame
        else:
            ref = seg.getReferential("TF")
            
        for i in range(0,acqui.GetPointFrameNumber()):    
            valX[i,:]= np.dot(ref.motion[i].getRotation() , np.array([100.0,0.0,0.0])) + ref.motion[i].getTranslation()
            valY[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,100.0,0.0])) + ref.motion[i].getTranslation()
            valZ[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,0.0,100.0])) + ref.motion[i].getTranslation()       
       
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_X",valX,desc="")
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_Y",valY,desc="")
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_Z",valZ,desc="")

class CGM1ModelInf(CGM):
    """ Pig is the vicon well-known name : Plugin gait ( i.e CGM1) 
   
    """

    nativeCgm1 = True        
    
    
    def __init__(self):
        """Constructor 
        
           - Run configuration internally
           - Initialize deviation data  

        """
        super(CGM1ModelInf, self).__init__()

        self.decoratedModel = False
        
#        self.__configure()
        
    
    
    @classmethod    
    def cleanAcquisition(cls, acq, subjetPrefix="",removelateralKnee=False, kadEnable= False, ankleMedEnable = False):
        markers = [subjetPrefix+"LASI",
                   subjetPrefix+"RASI",
                   subjetPrefix+"LPSI",
                   subjetPrefix+"RPSI",
                   subjetPrefix+"LTHI",
                   subjetPrefix+"LKNE",
                   subjetPrefix+"LTIB",
                   subjetPrefix+"LANK",
                   subjetPrefix+"LHEE",
                   subjetPrefix+"LTOE",
                   subjetPrefix+"RTHI",
                   subjetPrefix+"RKNE",
                   subjetPrefix+"RTIB",
                   subjetPrefix+"RANK",
                   subjetPrefix+"RHEE",
                   subjetPrefix+"RTOE",
                   subjetPrefix+"SACR"]

        if removelateralKnee:
            markers.append(subjetPrefix+"LKNE")                  
            markers.append(subjetPrefix+"RKNE")                  


        if kadEnable:
            markers.append(subjetPrefix+"LKAX")                  
            markers.append(subjetPrefix+"LKD1")                  
            markers.append(subjetPrefix+"LKD2")                                     
            markers.append(subjetPrefix+"RKAX")                  
            markers.append(subjetPrefix+"RKD1")                  
            markers.append(subjetPrefix+"RKD2")                                     

        if ankleMedEnable:
            markers.append(subjetPrefix+"LMED")   
            markers.append(subjetPrefix+"RMED")   

                   
        btkTools.clearPoints(acq,markers) 
        return acq

    def __repr__(self):
        return "cgm1"

    def configure(self):
        self.addSegment("Pelvis",0,pyCGM2Enums.SegmentSide.Central,["LASI","RASI","LPSI","RPSI"], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,pyCGM2Enums.SegmentSide.Left,["LKNE","LTHI"], tracking_markers = ["LKNE","LTHI"])
        self.addSegment("Right Thigh",4,pyCGM2Enums.SegmentSide.Right,["RKNE","RTHI"], tracking_markers = ["RKNE","RTHI"])
        self.addSegment("Left Shank",2,pyCGM2Enums.SegmentSide.Left,["LANK","LTIB"], tracking_markers = ["LANK","LTIB"])
        self.addSegment("Left Shank Proximal",7,pyCGM2Enums.SegmentSide.Left) # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Right Shank",5,pyCGM2Enums.SegmentSide.Right,["RANK","RTIB"], tracking_markers = ["RANK","RTIB"])
        self.addSegment("Right Shank Proximal",8,pyCGM2Enums.SegmentSide.Right)        # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Left Foot",3,pyCGM2Enums.SegmentSide.Left,["LAJC","LHEE","LTOE"], tracking_markers = ["LHEE","LTOE"] )
        self.addSegment("Right Foot",6,pyCGM2Enums.SegmentSide.Right,["RAJC","RHEE","RTOE"], tracking_markers = ["RHEE","RTOE"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])

        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ")
        self.addJoint("LKnee","Left Thigh", "Left Shank Proximal","YXZ")
        #self.addJoint("LKneeAngles_cgm","Left Thigh", "Left Shank","YXZ")
        self.addJoint("LAnkle","Left Shank", "Left Foot","YXZ")
        
        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ")
        self.addJoint("RKnee","Right Thigh", "Right Shank Proximal","YXZ")
        #self.addJoint("RKneeAngles_cgm","Right Thigh", "Right Shank","YXZ")
        self.addJoint("RAnkle","Right Shank", "Right Foot","YXZ")

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
        
        # ---- Pelvis-THIGH-SHANK CALIBRATION
        #-------------------------------------
        # calibration of technical Referentials
        logging.info(" --- Pelvis - TF calibration ---")
        logging.info(" -------------------------------")        
        self._pelvis_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        
        logging.info(" --- Left Thigh- TF calibration ---")
        logging.info(" ----------------------------------")
        self._left_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        logging.info(" --- Right Thigh - TF calibration ---")
        logging.info(" ------------------------------------")        
        self._right_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        
        logging.info(" --- Left Shank - TF calibration ---")
        logging.info(" -----------------------------------")        
        self._left_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        
        
        logging.info(" --- Richt Shank - TF calibration ---")
        logging.info(" ------------------------------------")        
        self._right_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        

        # calibration of anatomical Referentials
        logging.info(" --- Pelvis - AF calibration ---")
        logging.info(" -------------------------------") 
        self._pelvis_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Pelvis","Pelvis",referential = "Anatomic"  )

        logging.info(" --- Left Thigh - AF calibration ---")
        logging.info(" -----------------------------------") 
        self._left_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Left Thigh","LThigh",referential = "Anatomic"  )


        logging.info(" --- Right Thigh - AF calibration ---")
        logging.info(" ------------------------------------") 
        self._right_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Thigh","RThigh",referential = "Anatomic"  )  


        logging.info(" --- Thigh Offsets ---")
        logging.info(" --------------------") 
        self.getThighOffset(side="left")
        self.getThighOffset(side="right")


        logging.info(" --- Left Shank - AF calibration ---")
        logging.info(" -------------------------------")
        self._left_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self.displayStaticCoordinateSystem( aquiStatic, "Left Shank","LShank",referential = "Anatomic"  )
        
        
        logging.info(" --- Right Shank - AF calibration ---")
        logging.info(" -------------------------------")        
        self._right_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)    
        self.displayStaticCoordinateSystem( aquiStatic, "Right Shank","RShank",referential = "Anatomic"  )


        logging.info(" --- Shank Offsets ---")
        logging.info(" ---------------------") 
        self.getShankOffsets(side="both")# compute TibialRotation and Shank offset
        self.getAbdAddAnkleJointOffset(side="both")

        logging.info(" --- Left Shank Proximal- AF calibration ---")
        logging.info(" -------------------------------------------")
        #   shank Prox ( copy )
        self.updateSegmentFromCopy("Left Shank Proximal", self.getSegment("Left Shank")) # look out . I copied the shank instance and rename it 
        self._left_shankProximal_AnatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame
        self.displayStaticCoordinateSystem( aquiStatic, "Left Shank Proximal","LShankProx",referential = "Anatomic"  )        

        logging.info(" --- Right Shank Proximal- AF calibration ---")
        logging.info(" --------------------------------------------")
        self.updateSegmentFromCopy("Right Shank Proximal", self.getSegment("Right Shank"))        
        self._right_shankProximal_AnatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame
        self.displayStaticCoordinateSystem( aquiStatic, "Right Shank Proximal","RShankProx",referential = "Anatomic"  )
        
        
        # ---- FOOT CALIBRATION
        #-------------------------------------
        # foot ( need  Y-axis of the shank anatomic Frame)
        logging.info(" --- Left Foot - TF calibration (uncorrected) ---")
        logging.info(" -------------------------------------------------")
        self._left_unCorrectedFoot_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self.displayStaticCoordinateSystem( aquiStatic, "Left Foot","LFootUncorrected",referential = "technic"  )  

        logging.info(" --- Left Foot - AF calibration (corrected) ---")
        logging.info(" ----------------------------------------------")
        self._left_foot_corrected_calibrate(aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)
        self.displayStaticCoordinateSystem( aquiStatic, "Left Foot","LFoot",referential = "Anatomic"  )              


        logging.info(" --- Right Foot - TF calibration (uncorrected) ---")
        logging.info(" -------------------------------------------------")
        self._right_unCorrectedFoot_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Foot","RFootUncorrected",referential = "technic"  )              

        logging.info(" --- Right Foot - AF calibration (corrected) ---")
        logging.info(" -----------------------------------------------")
        self._right_foot_corrected_calibrate(aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)
        self.displayStaticCoordinateSystem( aquiStatic, "Right Foot","RFoot",referential = "Anatomic"  )      
 
        logging.info(" --- Foot Offsets ---")
        logging.info(" --------------------")
        self.getFootOffset(side = "both") 

  
    # ---- Technical Referential Calibration    
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
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0 






        seg=self.getSegment("Pelvis")

        # ---  additional markers and Update of the marker segment list

        # new markers
        valSACR=(aquiStatic.GetPoint("LPSI").GetValues() + aquiStatic.GetPoint("RPSI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aquiStatic,"SACR",valSACR,desc="")        

        valMidAsis=(aquiStatic.GetPoint("LASI").GetValues() + aquiStatic.GetPoint("RASI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aquiStatic,"midASIS",valMidAsis,desc="")        

        seg.addMarkerLabel("SACR")
        seg.addMarkerLabel("midASIS")
            
        # new mp    
        if self.mp.has_key("pelvisDepth"):
            self.mp_computed["pelvisDepth"] = self.mp["pelvisDepth"]
        else:
            logging.info("Pelvis Depth computed and added to model parameters")
            self.mp_computed["pelvisDepth"] = np.linalg.norm( valMidAsis.mean(axis=0)-valSACR.mean(axis=0)) - 2.0* (markerDiameter/2.0) -2.0* (basePlate/2.0)

        if self.mp.has_key("asisDistance"):
            self.mp_computed["asisDistance"] = self.mp["asisDistance"]
        else:
            logging.info("asisDistance computed and added to model parameters")            
            self.mp_computed["asisDistance"] = np.linalg.norm( aquiStatic.GetPoint("LASI").GetValues().mean(axis=0) - aquiStatic.GetPoint("RASI").GetValues().mean(axis=0))



        # --- Construction of the technical referential   
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

        # local Position of the hip joint centers
                
        LHJC_loc,RHJC_loc= CGM.hipJointCenters(self.mp,self.mp_computed,markerDiameter)

        # --- nodes manager
        # add HJC
        tf.static.addNode("LHJC_cgm1",LHJC_loc,positionType="Local")
        tf.static.addNode("RHJC_cgm1",RHJC_loc,positionType="Local")

        # add all point in the list
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")


        # --- Btk Points and decorator manager
        if not self.decoratedModel:
            # native : btkpoints LHJC and RHJC append with description cgm1-- "
            val = tf.static.getNode_byLabel("LHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
            btkTools.smartAppendPoint(aquiStatic,"LHJC",val, desc="cgm1") 

            val = tf.static.getNode_byLabel("RHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
            btkTools.smartAppendPoint(aquiStatic,"RHJC",val, desc="cgm1")
        else:
            # native : btkpoints LHJC_cgm1 and RHJC_cgm1 append with description cgm1-- "
            val = tf.static.getNode_byLabel("LHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LHJC_cgm1",val,desc="")        
                        
            val = tf.static.getNode_byLabel("RHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
            btkTools.smartAppendPoint(aquiStatic,"RHJC_cgm1",val,desc="")
            
            if "useLeftHJCnode" in options.keys():
                logging.info(" option (useLeftHJCnode) found ")

                nodeLabel = options["useLeftHJCnode"]
                desc = cmd.setDescription(nodeLabel)
                 
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(aquiStatic,"LHJC",val,desc=desc)
            else:
                logging.warning(" no option (useLeftHJCnode) found = > Left HJC comes from CGM1")
                val = tf.static.getNode_byLabel("LHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
                btkTools.smartAppendPoint(aquiStatic,"LHJC",val,desc="cgm1")                
                
            
            if "useRightHJCnode" in options.keys():
                logging.info(" option (useRightHJCnode) found ")                  

                nodeLabel = options["useRightHJCnode"]
                desc = cmd.setDescription(nodeLabel)

                # construction of the btkPoint label (RHJC)                
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(aquiStatic,"RHJC",val,desc=desc) 
            else:
                logging.warning(" no option (useLeftHJCnode) found = > Left HJC comes from CGM1")
                val = tf.static.getNode_byLabel("RHJC_cgm1").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
                btkTools.smartAppendPoint(aquiStatic,"RHJC",val,desc="cgm1")                  
        
        # ---- final HJCs and mid point
        final_LHJC = aquiStatic.GetPoint("LHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("LHJC",final_LHJC,positionType="Global")

        final_RHJC = aquiStatic.GetPoint("RHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("RHJC",final_RHJC,positionType="Global")

        seg.addMarkerLabel("LHJC")   
        seg.addMarkerLabel("RHJC")   

        val=(aquiStatic.GetPoint("LHJC").GetValues() + aquiStatic.GetPoint("RHJC").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aquiStatic,"midHJC",val,desc="")

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
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0             
            
        seg = self.getSegment("Left Thigh")


        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("LHJC")        


        
        # --- Construction of the technical referential 
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
        LKJC = CGM.chord( (self.mp["leftKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftThighOffset"] ) 
      
        # --- node manager
        tf.static.addNode("LKJC_chord",LKJC,positionType="Global")
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")


        # --- Btk Points and decorator manager
        if not self.decoratedModel:
            #Native: btkpoint LKJC append with description cgm1
            val = tf.static.getNode_byLabel("LKJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LKJC",val,desc="cgm1")
        else:
            val = LKJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LKJC_chord",val,desc="")

            if "useLeftKJCnode" in options.keys():
                logging.info(" option (useLeftKJCnode) found ")
                nodeLabel = options["useLeftKJCnode"]
                desc = cmd.setDescription(nodeLabel)
                
                # construction of the btkPoint label (LKJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"LKJC",val,desc=desc)
            else:
                logging.warning(" option (useLeftKJCnode) not found : KJC from chord ")
                btkTools.smartAppendPoint(aquiStatic,"LKJC",val,desc="chord")
           

        # --- final LKJC
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
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0 
            
        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0             
            
            
        seg = self.getSegment("Right Thigh")


        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("RHJC")        


        
        # --- Construction of the technical referential  
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
        RKJC = CGM.chord( (self.mp["rightKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3,beta=self.mp_computed["rightThighOffset"] ) # could consider decorqted LHJC

        # --- node manager
        tf.static.addNode("RKJC_chord",RKJC,positionType="Global")
      
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")



        # --- Btk Points and decorator manager
        if not self.decoratedModel:
            val = tf.static.getNode_byLabel("RKJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RKJC",val,desc="cgm1")
        else:
            val = RKJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RKJC_chord",val,desc="")

            if "useRightKJCnode" in options.keys():
                logging.info(" option (useRightKJCnode) found ")

                nodeLabel = options["useRightKJCnode"]
                desc = cmd.setDescription(nodeLabel)
                                                          
                # construction of the btkPoint label (LKJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"RKJC",val,desc=desc)
            else:
                logging.warning(" option (useRightKJCnode) not found : KJC from chord ")
                btkTools.smartAppendPoint(aquiStatic,"RKJC",val,desc="chord")
           

        # --- final KJC
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
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0
            
        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0             
            
        
        
        
        seg = self.getSegment("Left Shank")
        
        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("LKJC")

        # --- Construction of the technical referential  
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
        LAJC = CGM.chord( (self.mp["leftAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftShankOffset"] )

        # --- node manager
        tf.static.addNode("LAJC_chord",LAJC,positionType="Global")

        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")



        # --- Btk Points and decorator manager
        if not self.decoratedModel:
            #btkpoint LAJC append with description cgm1
            val = tf.static.getNode_byLabel("LAJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LAJC",val,desc="cgm1")
        else:
            val = LAJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"LAJC_chord",val,desc="")

            if "useLeftAJCnode" in options.keys():
                logging.info(" option (useLeftAJCnode) found ")
                
                nodeLabel = options["useLeftAJCnode"]
                desc = cmd.setDescription(nodeLabel)

                # construction of the btkPoint label (LAJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"LAJC",val,desc=desc)
            else:
                logging.warning(" option (useLeftAJCnode) not found : AJC from chord ")
                btkTools.smartAppendPoint(aquiStatic,"LAJC",val,desc="chord")
           

        # --- final AJC
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
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0             
            
        
        
        seg = self.getSegment("Right Shank")

        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("RKJC")

        # --- Construction of the technical Referential
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
        RAJC = CGM.chord( (self.mp["rightAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["rightShankOffset"] )

        # --- node manager
        tf.static.addNode("RAJC_chord",RAJC,positionType="Global")

        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")

       #Btk Points and decorator manager
        if not self.decoratedModel:
            val = tf.static.getNode_byLabel("RAJC_chord").m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RAJC",val,desc="cgm1")
        else:
            val = RAJC * np.ones((aquiStatic.GetPointFrameNumber(),3))
            btkTools.smartAppendPoint(aquiStatic,"RAJC_chord",val,desc="")

            if "useRightAJCnode" in options.keys():
                logging.info(" option (useRightAJCnode) found ")
                nodeLabel = options["useRightAJCnode"]
                desc = cmd.setDescription(nodeLabel)
     
                # construction of the btkPoint label (RAJC)
                val = tf.static.getNode_byLabel(nodeLabel).m_global * np.ones((aquiStatic.GetPointFrameNumber(),3))    
                btkTools.smartAppendPoint(aquiStatic,"RAJC",val,desc=desc)
            else:
                logging.warning( "option (useRightAJCnode) not found : AJC from chord" )
                btkTools.smartAppendPoint(aquiStatic,"RAJC",val,desc="chord")
           

        # --- Final AJC
        final_RAJC = aquiStatic.GetPoint("RAJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        tf.static.addNode("RAJC",final_RAJC,positionType="Global")
        seg.addMarkerLabel("RAJC")



    def _left_unCorrectedFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        """ 
        uncorrected Foot == technical Frame
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        ..Note:: Need shank anatomical Frame
            

        """ 

        seg = self.getSegment("Left Foot")
        
        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("LKJC")


        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            logging.warning("You use a Left uncorrected foot sequence different than native CGM1")
            dictRef["Left Foot"]={"TF" : {'sequence':"ZYX", 'labels':   ["LTOE","LAJC","LKJC","LAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis


        # --- Construction of the technical Referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#LTOE
        pt2=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#AJC
        
        if dictRef["Left Foot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
            v=self.getSegment("Left Shank").anatomicalFrame.static.m_axisY #(pt3-pt1) 

    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")


    def _right_unCorrectedFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):
        """ 
        uncorrected Foot == technical Frame
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        ..Note:: Need shank anatomical Frame
            

        """ 

        
        seg = self.getSegment("Right Foot")
        
        # ---  additional markers and Update of the marker segment list
        seg.addMarkerLabel("RKJC")

        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]: 
            logging.warning("You use a right uncorrected foot sequence different than native CGM1")
            dictRef["Right Foot"]={"TF" : {'sequence':"ZYX", 'labels':   ["RTOE","RAJC","RKJC","RAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis



        # --- Construction of the anatomical Referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        if dictRef["Right Foot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)         
        
        
    
        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence'])          
           
        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            tf.static.addNode(label,globalPosition,positionType="Global")
    
    # ---- Anatomical Referential Calibration -------

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
        
        
        # --- Construction of the anatomical Referential
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

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))
                
        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")
            
    def _left_thigh_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
           
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Left Thigh")
        
        # --- Construction of the anatomical Referential
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


        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        
        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")

        # --- compute length
        hjc = seg.anatomicalFrame.static.getNode_byLabel("LHJC").m_local
        kjc = seg.anatomicalFrame.static.getNode_byLabel("LKJC").m_local

        seg.setLength(np.linalg.norm(kjc-hjc))


    def _right_thigh_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Right Thigh")
        
        # --- Construction of the anatomical Referential
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

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")

        # --- compute lenght
        hjc = seg.anatomicalFrame.static.getNode_byLabel("RHJC").m_local
        kjc = seg.anatomicalFrame.static.getNode_byLabel("RKJC").m_local

        seg.setLength(np.linalg.norm(kjc-hjc))

    def _left_shank_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  anatomical calibration of the pelvis
        
        :synopsis: 
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Left Shank")
        
        # --- Construction of the anatomical Referential
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

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- Node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")

        # --- compute length
        kjc = seg.anatomicalFrame.static.getNode_byLabel("LKJC").m_local
        ajc = seg.anatomicalFrame.static.getNode_byLabel("LAJC").m_local

        seg.setLength(np.linalg.norm(ajc-kjc))

    def _left_shankProximal_AnatomicalCalibrate(self,aquiStatic,dictAnat,frameInit,frameEnd,options=None):

        if "useLeftTibialTorsion" in options.keys():
            logging.warning ("option (useLeftTibialTorsion) enable")
            #TODO : si   valeur de left tibial torsion dans mp, cad en data d entree. 
            tibialTorsion = np.deg2rad(self.mp_computed["leftTibialTorsion"])
        else:
            logging.warning ("option (useLeftTibialTorsion) disable - No tibial Torsion")
            tibialTorsion = 0.0 #np.deg2rad(self.mp_computed["leftTibialTorsion"])

        seg=self.getSegment("Left Shank Proximal")

        
        # --- set static anatomical Referential  
        # Rotation of the static anatomical Referential by the tibial Torsion angle 
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion) 
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)
          
        R = np.dot(seg.anatomicalFrame.static.getRotation(),rotZ_tibRot)
       
        # update frame   
        frame=cfr.Frame()  
        frame.update(R,seg.anatomicalFrame.static.getTranslation())
        seg.anatomicalFrame.setStaticFrame(frame)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")


    def _right_shank_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):    
        """  
        
        :synopsis: 
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Right Shank")
        
        # --- Construction of the anatomical Referential
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

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())) 

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")


        # --- compute length
        kjc = seg.anatomicalFrame.static.getNode_byLabel("RKJC").m_local
        ajc = seg.anatomicalFrame.static.getNode_byLabel("RAJC").m_local
        seg.setLength(np.linalg.norm(ajc-kjc))
 

    def _right_shankProximal_AnatomicalCalibrate(self,aquiStatic,dictAnat,frameInit,frameEnd,options=None):

        if "useRightTibialTorsion" in options.keys():
            logging.warning ("option (useRightTibialTorsion) enable")
            tibialTorsion = np.deg2rad(self.mp_computed["rightTibialTorsion"])
        else:
            logging.warning ("option (useRightTibialTorsion) disable - No Tibial Torsion")
            tibialTorsion = 0.0 #np.deg2rad(self.mp_computed["leftTibialTorsion"])

        seg=self.getSegment("Right Shank Proximal")

        # --- set static anatomical Referential  
        # Rotation of the static anatomical Referential by the tibial Torsion angle
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion) 
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)
          
        R = np.dot(seg.anatomicalFrame.static.getRotation(),rotZ_tibRot)
       
        frame=cfr.Frame()  
        frame.update(R,seg.anatomicalFrame.static.getTranslation() )
        seg.anatomicalFrame.setStaticFrame(frame)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")


    def _left_foot_corrected_calibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd,options = None):    
        """  corrected foot = anatomical Frame
        
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
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]        
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0             
            

        seg=self.getSegment("Left Foot")
        
        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            logging.warning("You use a Left corrected foot sequence different than native CGM1")
            dictAnatomic["Left Foot"]={'sequence':"ZYX", 'labels':  ["LTOE","LHEE","LKJC","LAJC"]}    # corrected foot      
            
        
        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # LTOE
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
              
        if ("leftFlatFoot" in options.keys() and options["leftFlatFoot"]):
            logging.warning ("option (leftFlatFoot) enable")            
            pt2[2] = pt1[2]    
    
        if dictAnatomic["Left Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
            v=self.getSegment("Left Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)         
    
        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                                                       
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Left Foot"]['sequence'])          
           
        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        # This section compute the actual Relative Rotation between anatomical and technical Referential  
        trueRelativeMatrixAnatomic = np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())
        y,x,z = ceuler.euler_yxz(trueRelativeMatrixAnatomic)

        # the native CGM relative rotation leaves out the rotation around Z  
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

    def _right_foot_corrected_calibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd,options = None):    
        """  corrected foot = anatomical Frame
        
        :synopsis: 
        
        
        :Parameters:
        
           - `aquiStatic` (btkAcquisition) - acquisition
           - `dictAnatomic` (dict) - 
           - `frameInit` (int) - starting frame
           - `frameEnd` (int) - starting frame           
           - `options` (kwargs) - use to pass option altering the standard construction

        """             
        
        seg=self.getSegment("Right Foot")

        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            logging.warning("You use a Right corrected foot sequence different than native CGM1")
            dictAnatomic["Right Foot"]={'sequence':"ZYX", 'labels':  ["RTOE","RHEE","RKJC","RAJC"]}    # corrected foot

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    
        if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
            logging.warning ("option (rightFlatFoot) enable")
            pt2[2] = pt1[2]    

        if dictAnatomic["Right Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
            v=self.getSegment("Right Shank").anatomicalFrame.static.m_axisY #(pt3-pt1)

    
        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        
        a1=(pt2-pt1)
        a1=a1/np.linalg.norm(a1)
                    
        v=v/np.linalg.norm(v)
                    
        a2=np.cross(a1,v)
        a2=a2/np.linalg.norm(a2)

        x,y,z,R=cfr.setFrameData(a1,a2,dictAnatomic["Right Foot"]['sequence'])          
           
        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        # actual Relative Rotation 
        trueRelativeMatrixAnatomic = np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())
        y,x,z = ceuler.euler_yxz(trueRelativeMatrixAnatomic)

        # native CGM relative rotation 
        rotX =np.array([[1,0,0],
                        [0,np.cos(x),-np.sin(x)],
                         [0,np.sin(x),np.cos(x)]]) 

        rotY =np.array([[np.cos(y),0,np.sin(y)],
                        [0,1,0],
                         [-np.sin(y),0,np.cos(y)]]) 

        relativeMatrixAnatomic = np.dot(rotY,rotX)
          
        tf.setRelativeMatrixAnatomic(relativeMatrixAnatomic) 

        # --- node manager
        for label in seg.m_markerLabels:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)       
            seg.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")

        # --- anthropo
        # length
        toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_local
        seg.setLength(np.linalg.norm(toe-hee))
        
        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_global
        com = (toe+hee)/2.0
        
        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")

    # ---- Offsets -------
    
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
            
            angle=np.rad2deg(geometry.angleFrom2Vectors(v_kneeFlexionAxis, v_thi, self.getSegment("Left Thigh").anatomicalFrame.static.m_axisZ))    
            self.mp_computed["leftThighOffset"]= -angle # angle needed : Thi toward knee flexion
            logging.info(" left Thigh Offset => %s " % str(self.mp_computed["leftThighOffset"]))
            


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
            
            angle=np.rad2deg(geometry.angleFrom2Vectors(v_kneeFlexionAxis_opp, v_thi,self.getSegment("Right Thigh").anatomicalFrame.static.m_axisZ))
            
            self.mp_computed["rightThighOffset"]=-angle # angle needed : Thi toward knee flexion
            logging.info(" right Thigh Offset => %s " % str(self.mp_computed["rightThighOffset"]))
        
        

    def getShankOffsets(self, side = "both"):
        
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
        
            if "leftTibialTorsion" in self.mp.keys():
                logging.warning("Left Tibial torsion defined from your mp file")
                self.mp_computed["leftTibialTorsion"] = - self.mp["leftTibialTorsion"] # sign - because vicon standard considers external offset as positive
            else:
                angle= np.rad2deg( geometry.angleFrom2Vectors(v_kneeFlexionAxis,v_ankleFlexionAxis,self.getSegment("Left Shank").anatomicalFrame.static.m_axisZ))
                self.mp_computed["leftTibialTorsion"] = angle
                logging.info(" left tibial torsion => %s " % str(self.mp_computed["leftTibialTorsion"]))
            

            #"****** left angle beetween tib and flexion axis **********"    
            tibLocal = self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LTIB").m_local
            proj_tib = np.array([ tibLocal[0],
                               tibLocal[1],
                                 0])                                 
            v_tib = proj_tib/np.linalg.norm(proj_tib)
        
            angle=np.rad2deg(geometry.angleFrom2Vectors(v_ankleFlexionAxis,v_tib,self.getSegment("Left Shank").anatomicalFrame.static.m_axisZ))
            self.mp_computed["leftShankOffset"]= -angle
            logging.info(" left shank offset => %s " % str(self.mp_computed["leftShankOffset"]))

            
            #"****** left angle beetween ank and flexion axis (not used by native pig)**********"        
            ANK =  self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LANK").m_local                    
            v_ank = ANK/np.linalg.norm(ANK)            
            angle = np.rad2deg(geometry.angleFrom2Vectors(v_ankleFlexionAxis,v_ank,self.getSegment("Left Shank").anatomicalFrame.static.m_axisZ))

            self.mp_computed["leftProjectionAngle_AnkleFlexion_LateralAnkle"] = angle
            logging.info(" left projection offset => %s " % str(self.mp_computed["leftProjectionAngle_AnkleFlexion_LateralAnkle"]))



        
        
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
        
            if "rightTibialTorsion" in self.mp.keys():
                logging.warning("Right  Tibial torsion defined from your mp file")
                self.mp_computed["rightTibialTorsion"] =  self.mp["rightTibialTorsion"]
            else:
                angle= np.rad2deg(geometry.angleFrom2Vectors(v_kneeFlexionAxis,v_ankleFlexionAxis,self.getSegment("Right Shank").anatomicalFrame.static.m_axisZ))
                self.mp_computed["rightTibialTorsion"] = angle
                logging.info(" Right tibial torsion => %s " % str(self.mp_computed["rightTibialTorsion"]))

            #"****** right angle beetween tib and flexion axis **********"    
            tibLocal = self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RTIB").m_local
            proj_tib = np.array([ tibLocal[0],
                               tibLocal[1],
                                 0])
                                 
            v_tib = proj_tib/np.linalg.norm(proj_tib)
            v_ankleFlexionAxis_opp = geometry.oppositeVector(v_ankleFlexionAxis) 

        
            angle = np.rad2deg(geometry.angleFrom2Vectors(v_ankleFlexionAxis_opp,v_tib,self.getSegment("Right Shank").anatomicalFrame.static.m_axisZ))
            self.mp_computed["rightShankOffset"]= -angle
            logging.info(" right shank offset => %s " % str(self.mp_computed["rightShankOffset"]))

            
            
            #"****** right angle beetween ank and flexion axis ( Not used by Native Pig)**********"        
            ANK =  self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RANK").m_local                    
            v_ank = ANK/np.linalg.norm(ANK)              
           
            angle = np.rad2deg(geometry.angleFrom2Vectors(v_ankleFlexionAxis_opp,v_ank,self.getSegment("Right Shank").anatomicalFrame.static.m_axisZ))

            self.mp_computed["rightProjectionAngle_AnkleFlexion_LateralAnkle"] = angle
            logging.info(" right projection offset => %s " % str(self.mp_computed["rightProjectionAngle_AnkleFlexion_LateralAnkle"]))
        

    def getAbdAddAnkleJointOffset(self,side="both"):

        if side == "both" or side == "left" :        
        
            ankleFlexionAxis=    np.dot(self.getSegment("Left Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Left Shank").anatomicalFrame.static.m_axisY)
        

                                 
            v_ankleFlexionAxis = ankleFlexionAxis/np.linalg.norm(ankleFlexionAxis)
            
            ANK =  self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LANK").m_local - \
                   self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LAJC").m_local              
            v_ank = ANK/np.linalg.norm(ANK)
            
            angle = np.rad2deg(geometry.angleFrom2Vectors(v_ankleFlexionAxis,v_ank,self.getSegment("Left Shank").anatomicalFrame.static.m_axisX))
            self.mp_computed["leftAJCAbAdOffset"] = angle
            logging.info(" leftAJCAbAdOffset => %s " % str(self.mp_computed["leftAJCAbAdOffset"]))
            

        if side == "both" or side == "right" : 
            ankleFlexionAxis=    np.dot(self.getSegment("Right Shank").anatomicalFrame.static.getRotation().T, 
                                       self.getSegment("Right Shank").anatomicalFrame.static.m_axisY)
        
                                 
            v_ankleFlexionAxis = ankleFlexionAxis/np.linalg.norm(ankleFlexionAxis)
            
            v_ankleFlexionAxis_opp = geometry.oppositeVector(v_ankleFlexionAxis)
            ANK =  self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RANK").m_local - \
                   self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RAJC").m_local                     
            v_ank = ANK/np.linalg.norm(ANK)
           
            angle = np.rad2deg(geometry.angleFrom2Vectors(v_ankleFlexionAxis_opp,v_ank,self.getSegment("Right Shank").anatomicalFrame.static.m_axisX))
            self.mp_computed["rightAJCAbAdOffset"] = angle
            logging.info(" rightAJCAbAdOffset => %s " % str(self.mp_computed["rightAJCAbAdOffset"]))


    def getFootOffset(self, side = "both"):
        
        if side == "both" or side == "left" :      
            R = self.getSegment("Left Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)
            
            self.mp_computed["leftStaticPlantarFlexion"] = np.rad2deg(y)
            logging.info(" leftStaticPlantarFlexion => %s " % str(self.mp_computed["leftStaticPlantarFlexion"])) 

            self.mp_computed["leftStaticRotOff"] = np.rad2deg(x)
            logging.info(" leftStaticRotOff => %s " % str(self.mp_computed["leftStaticRotOff"]))

        
        if side == "both" or side == "right" :      
            R = self.getSegment("Right Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = ceuler.euler_yxz(R)    
            
            self.mp_computed["rightStaticPlantarFlexion"] = np.rad2deg(y)
            logging.info(" rightStaticPlantarFlexion => %s " % str(self.mp_computed["rightStaticPlantarFlexion"])) 
            
            self.mp_computed["rightStaticRotOff"] = np.rad2deg(x)
            logging.info(" rightStaticPlantarFlexion => %s " % str(self.mp_computed["rightStaticPlantarFlexion"])) 


    def getViconFootOffset(self):
        """ 
        .. note: 
        standard CGM vicon : positive = dorsiflexion  and abduction
        """        
        
        spf_l = self.mp_computed["leftStaticPlantarFlexion"] * -1.0
        logging.info(" Left staticPlantarFlexion offset (Vicon compatible)  => %s " % str(spf_l))
        
        
        sro_l = self.mp_computed["leftStaticRotOff"] * -1.0
        logging.info("Left staticRotation offset (Vicon compatible)  => %s " % str(sro_l))

        
        spf_r = self.mp_computed["rightStaticPlantarFlexion"] * -1.0
        logging.info("Right staticRotation offset (Vicon compatible)  => %s " % str(spf_r))
       
        sro_r = self.mp_computed["rightStaticRotOff"] 
        logging.info("Right staticRotation offset (Vicon compatible)  => %s " % str(sro_r))        

        return spf_l,sro_l,spf_r,sro_r


    def getViconThighOffset(self, side):
        """ 
        .. note: 
        standard vicon : positive = internal rotation
        """        

        if side  == "Left":
            val = self.mp_computed["leftThighOffset"] * -1.0
            logging.info(" Left thigh offset (Vicon compatible)  => %s " % str(val))             
            return val
 
        if side  == "Right":
            val = self.mp_computed["rightThighOffset"]            
            logging.info(" Right thigh offset (Vicon compatible)  => %s " % str(val))             
            return val        
        
        
    def getViconShankOffset(self, side):
        """ 
        .. note: 
        standard vicon : positive = internal rotation
        """        

        if side  == "Left":
            val = self.mp_computed["leftShankOffset"] * -1.0
            logging.info(" Left shank offset (Vicon compatible)  => %s " % str(val)) 
            return val
 
        if side  == "Right":
            val = self.mp_computed["rightShankOffset"]
            logging.info(" Right shank offset (Vicon compatible)  => %s " % str(val))
            return val         

    
                

     
        

    # ----- Motion --------------
    def computeMotion(self,aqui, dictRef,dictAnat, motionMethod,options=None ):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `motionMethod` (Enum motionMethod) - method use to optimize pose         

        """         
        logging.info("=====================================================")         
        logging.info("===================  CGM MOTION   ===================")
        logging.info("=====================================================")
       
        if motionMethod == pyCGM2Enums.motionMethod.Native: #cmf.motionMethod.Native:
            logging.info("--- Native motion process ---")
            
            logging.info(" - Pelvis - motion -")
            logging.info(" -------------------")
            self._pelvis_motion(aqui, dictRef, dictAnat)
 
            logging.info(" - Left Thigh - motion -")
            logging.info(" -----------------------")            
            self._left_thigh_motion(aqui, dictRef, dictAnat,options=options)

            logging.info(" - Right Thigh - motion -")
            logging.info(" ------------------------")            
            self._right_thigh_motion(aqui, dictRef, dictAnat,options=options)


            logging.info(" - Left Shank - motion -")
            logging.info(" -----------------------") 
            self._left_shank_motion(aqui, dictRef, dictAnat,options=options)
 
            logging.info(" - Left Shank-proximal - motion -")
            logging.info(" --------------------------------") 
            self._left_shankProximal_motion(aqui,dictAnat,options=options)
 
            logging.info(" - Right Shank - motion -")
            logging.info(" ------------------------") 
            self._right_shank_motion(aqui, dictRef, dictAnat,options=options)

            logging.info(" - Right Shank-proximal - motion -")
            logging.info(" ---------------------------------") 
            self._right_shankProximal_motion(aqui,dictAnat,options=options)
            
            logging.info(" - Left foot - motion -")
            logging.info(" ----------------------")
            self._left_foot_motion(aqui, dictRef, dictAnat,options=options)

            logging.info(" - Right foot - motion -")
            logging.info(" ----------------------")
            self._right_foot_motion(aqui, dictRef, dictAnat,options=options)

            
        
        if motionMethod == pyCGM2Enums.motionMethod.Sodervisk:
            logging.info("--- Segmental Least-square motion process ---")
            self._pelvis_motion_optimize(aqui, dictRef,motionMethod)
            self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._left_shank_motion_optimize(aqui, dictRef,motionMethod)        
            self._right_shank_motion_optimize(aqui, dictRef,motionMethod)

        logging.info("--- Display Coordinate system ---")
        logging.info(" --------------------------------")
    
        self.displayMotionCoordinateSystem( aqui,  "Pelvis" , "Pelvis" )
        self.displayMotionCoordinateSystem( aqui,  "Left Thigh" , "LThigh" )
        self.displayMotionCoordinateSystem( aqui,  "Right Thigh" , "RThigh" )
        self.displayMotionCoordinateSystem( aqui,  "Left Shank" , "LShank" )
        self.displayMotionCoordinateSystem( aqui,  "Left Shank Proximal" , "LShankProx" )
        self.displayMotionCoordinateSystem( aqui,  "Right Shank" , "RShank" )
        self.displayMotionCoordinateSystem( aqui,  "Right Shank Proximal" , "RShankProx" )
        self.displayMotionCoordinateSystem( aqui,  "Left Foot" , "LFoot" )
        self.displayMotionCoordinateSystem( aqui,  "Right Foot" , "RFoot" )

        self.displayMotionCoordinateSystem( aqui,  "Left Foot" , "LFootUncorrected",referential="technical") 
        self.displayMotionCoordinateSystem( aqui,  "Right Foot" , "RFootUncorrected",referential="technical") 
    # ----- native motion ------

    

    def _pelvis_motion(self,aqui, dictRef,dictAnat):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `motionMethod` (Enum motionMethod) 

        """ 

        seg=self.getSegment("Pelvis")

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]        # reinit Technical Frame Motion (USEFUL if you work with several aquisitions)

        #  additional markers 
        val=(aqui.GetPoint("LPSI").GetValues() + aqui.GetPoint("RPSI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aqui,"SACR",val, desc="")
         
        val=(aqui.GetPoint("LASI").GetValues() + aqui.GetPoint("RASI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aqui,"midASIS",val, desc="")

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

        # --- HJCs 
        values_LHJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        values_RHJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")

        btkTools.smartAppendPoint(aqui,"LHJC",values_LHJCnode, desc="cgm1")
        btkTools.smartAppendPoint(aqui,"RHJC",values_RHJCnode, desc="cgm1")


        # --- motion of the anatomical referential

        seg.anatomicalFrame.motion=[]

        # additional markers         
        val=(aqui.GetPoint("LHJC").GetValues() + aqui.GetPoint("RHJC").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aqui,"midHJC",val,desc="")

        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Pelvis"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(frame)


    def _left_thigh_motion(self,aqui, dictRef,dictAnat,options=None):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """

        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]        
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0  
        
        seg=self.getSegment("Left Thigh")


        # --- motion of the technical referential                   
        seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

        # additional markers
        # NA

        # computation
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

            LKJCvalues[i,:] = CGM.chord( (self.mp["leftKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftThighOffset"] )
            
            
        # --- LKJC     
        btkTools.smartAppendPoint(aqui,"LKJC",LKJCvalues, desc="chord")

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]

        # additional markers
        # NA

        # computation        
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Left Thigh"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(frame)



     


    def _right_thigh_motion(self,aqui, dictRef,dictAnat,options=None):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]          
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0  
            
        seg=self.getSegment("Right Thigh")
        
        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]
        
        # additional markers
        # NA

        # computation           
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

            RKJCvalues[i,:] = CGM.chord( (self.mp["rightKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["rightThighOffset"] )
            
            
        # --- RKJC    
        btkTools.smartAppendPoint(aqui,"RKJC",RKJCvalues, desc="chord")

        # --- motion of the anatomical referential

        # additional markers
        # NA

        # computation
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Right Thigh"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(frame)        
        

    


    def _left_shank_motion(self,aqui, dictRef,dictAnat,options=None):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]        
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0  
        
        seg=self.getSegment("Left Shank")
                   
        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
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


            LAJCvalues[i,:] = CGM.chord( (self.mp["leftAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["leftShankOffset"] )
                
            # update of the AJC location with rotation around abdAddAxis 
            LAJCvalues[i,:] = self._rotateAjc(LAJCvalues[i,:],pt2,pt1,-self.mp_computed["leftAJCAbAdOffset"])
            

        # --- LAJC
        if self.mp_computed["leftAJCAbAdOffset"] > 0.01:
            desc="chord+AbAdRot"
        else:
            desc="chord"
            
        btkTools.smartAppendPoint(aqui,"LAJC",LAJCvalues, desc=desc)

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]

        # additional markers
        # NA

        # computation
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Left Shank"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(frame)        

    def _left_shankProximal_motion(self,aqui,dictAnat,options=None):

        seg=self.getSegment("Left Shank")
        segProx=self.getSegment("Left Shank Proximal")


        # --- managment of tibial torsion    
        if "useLeftTibialTorsion" in options.keys():
            logging.warning("option (useLeftTibialTorsion) enable")
            tibialTorsion = np.deg2rad(self.mp_computed["leftTibialTorsion"])
        else:
            logging.warning("option (useLeftTibialTorsion) disable")
            tibialTorsion = 0.0 #np.deg2rad(self.mp_computed["leftTibialTorsion"])


        # --- motion of both technical and anatomical referentials of the proximal shank
        segProx.getReferential("TF").motion =[]
        segProx.anatomicalFrame.motion=[]

        # additional markers
        # NA

        # computation
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion) 
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)
          
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3])).GetValues()[i,:] 

            segProx.getReferential("TF").addMotionFrame(seg.getReferential("TF").motion[i]) # copy technical shank

            R = np.dot(seg.anatomicalFrame.motion[i].getRotation(),rotZ_tibRot) # affect Tibial torsion to anatomical shank
            frame=cfr.Frame()  
            frame.update(R,ptOrigin)              
            segProx.anatomicalFrame.addMotionFrame(frame)

    
    def _right_shank_motion(self,aqui, dictRef,dictAnat,options=None):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        if "markerDiameter" in options.keys():
            logging.info(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]        
        else:
            markerDiameter=14.0 

        if "basePlate" in options.keys():
            logging.info(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0  
        
        seg=self.getSegment("Right Shank")

        
        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
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
            RAJCvalues[i,:] = CGM.chord( (self.mp["rightAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["rightShankOffset"] )

            # update of the AJC location with rotation around abdAddAxis 
            RAJCvalues[i,:] = self._rotateAjc(RAJCvalues[i,:],pt2,pt1,   self.mp_computed["rightAJCAbAdOffset"])

        # --- RAJC
        if self.mp_computed["rightAJCAbAdOffset"] >0.01:
            desc="chord+AbAdRot"
        else:
            desc="chord"
            
        btkTools.smartAppendPoint(aqui,"RAJC",RAJCvalues, desc=desc)
        

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]
        
        # additional markers
        # NA

        # computation
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
                     
            v=(pt3-pt1)
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictAnat["Right Shank"]['sequence']) 
            frame=cfr.Frame()                
                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(frame)

    def _right_shankProximal_motion(self,aqui,dictAnat,options=None):
        
        seg=self.getSegment("Right Shank")
        segProx=self.getSegment("Right Shank Proximal")        

       
        # --- management of the tibial torsion 
        if "useRightTibialTorsion" in options.keys():
            tibialTorsion = np.deg2rad(self.mp_computed["rightTibialTorsion"])
        else:
            tibialTorsion = 0.0 #np.deg2rad(self.mp_computed["leftTibialTorsion"])

        # --- motion of both technical and anatomical referentials of the proximal shank
        segProx.getReferential("TF").motion =[]
        segProx.anatomicalFrame.motion=[]

        # additional markers
        # NA

        # computation
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion) 
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)
          
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:] 

            segProx.getReferential("TF").addMotionFrame(seg.getReferential("TF").motion[i]) # copy technical shank

            R = np.dot(seg.anatomicalFrame.motion[i].getRotation(),rotZ_tibRot)
            frame=cfr.Frame()  
            frame.update(R,ptOrigin)              
            segProx.anatomicalFrame.addMotionFrame(frame)

    


    def _left_foot_motion(self,aqui, dictRef,dictAnat,options=None):
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
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][0])).GetValues()[i,:] #toe
            pt2=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc
            
            if dictRef["Left Foot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][2])).GetValues()[i,:]
                v=(pt3-pt1)
            else:
                if "viconCGM1compatible" in options.keys():
                    v=self.getSegment("Left Shank Proximal").anatomicalFrame.motion[i].m_axisY
                else:
                    v=self.getSegment("Left Shank").anatomicalFrame.motion[i].m_axisY           
            
            
            ptOrigin=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
            
            
            
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence']) 
            frame=cfr.Frame()                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)
            
            seg.getReferential("TF").addMotionFrame(frame)


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]
        
        # additional markers
        # NA

        # computation
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Foot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    def _right_foot_motion(self,aqui, dictRef,dictAnat,options=None):
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
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][0])).GetValues()[i,:] #toe
            pt2=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc
            
            if dictRef["Right Foot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][2])).GetValues()[i,:]
                v=(pt3-pt1)
            else:
                if "viconCGM1compatible" in options.keys():
                    v=self.getSegment("Right Shank Proximal").anatomicalFrame.motion[i].m_axisY
                else:
                    v=self.getSegment("Right Shank").anatomicalFrame.motion[i].m_axisY              
            
            ptOrigin=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][3])).GetValues()[i,:]             
         
         
            a1=(pt2-pt1)
            a1=a1/np.linalg.norm(a1)
            
            v=v/np.linalg.norm(v)
                    
            a2=np.cross(a1,v)
            a2=a2/np.linalg.norm(a2)

            x,y,z,R=cfr.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence']) 
            frame=cfr.Frame()                
            frame.m_axisX=x
            frame.m_axisY=y
            frame.m_axisZ=z
            frame.setRotation(R)
            frame.setTranslation(ptOrigin)
            
            seg.getReferential("TF").addMotionFrame(frame)

            
        # --- motion of the anatomical referential
        
        # additional markers
        # NA

        # computation
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Foot"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    # ----- least-square Segmental motion ------
    def _pelvis_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `motionMethod` (Enum motionMethod) 
        .. note :: 
           
        """
        seg=self.getSegment("Pelvis")

        #  --- check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            btkTools.isPointsExist(aqui,seg.m_tracking_markers)
            
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

        # --- HJC
        values_LHJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        values_RHJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")

        btkTools.smartAppendPoint(aqui,"LHJC",values_LHJCnode, desc="opt")
        btkTools.smartAppendPoint(aqui,"RHJC",values_RHJCnode, desc="opt")


        # --- Motion of the Anatomical frame
        seg.anatomicalFrame.motion=[]
        
        # additional markers
        val=(aqui.GetPoint("LASI").GetValues() + aqui.GetPoint("RASI").GetValues()) / 2.0        
        btkTools.smartAppendPoint(aqui,"midASIS",val, desc="")
        
        # computation         
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    def _left_thigh_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Left Thigh")

        #  --- add LHJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2: 
                if "LHJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LHJC")
                    logging.info("LHJC added to tracking marker list")
            
            btkTools.isPointsExist(aqui,seg.m_tracking_markers)

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get back static global position ( look out i use nodes)
                        
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

        # --- LKJC
        values_LKJCnode=seg.getReferential('TF').getNodeTrajectory("LKJC")
        btkTools.smartAppendPoint(aqui,"LKJC",values_LKJCnode, desc="opt")

        # --- Motion of the Anatomical frame

        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)
            
            
    def _right_thigh_motion_optimize(self,aqui, dictRef, dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Right Thigh")

        #  --- add RHJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2: 
                if "RHJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RHJC")
                    logging.info("RHJC added to tracking marker list")
            
        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

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

        # --- RKJC
        values_RKJCnode=seg.getReferential('TF').getNodeTrajectory("RKJC")
        btkTools.smartAppendPoint(aqui,"RKJC",values_RKJCnode, desc="opt")


        # --- Motion of the Anatomical frame

        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)


    def _left_shank_motion_optimize(self,aqui, dictRef,dictAnat,  motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Left Shank")
        
        #  --- add LKJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2: 
                if "LKJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LKJC")
                    logging.info("LKJC added to tracking marker list")
            
        # --- Motion of the Technical frame        
        seg.getReferential("TF").motion =[]

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

        # --- LAJC
        values_LAJCnode=seg.getReferential('TF').getNodeTrajectory("LAJC")
        btkTools.smartAppendPoint(aqui,"LAJC",values_LAJCnode, desc="opt")


        # --- motion of the anatomical Referential
        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3])).GetValues()[i,:]
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)

    def _right_shank_motion_optimize(self,aqui, dictRef,dictAnat, motionMethod):
        """ 
        :Parameters:
        
           - `aqui` (btkAcquisition) - acquisition
           - `dictRef` (dict) - instance of Model
           - `frameInit` (int) - starting frame
           - `motionMethod` (Enum motionMethod) - method use to optimize pose

        """
        seg=self.getSegment("Right Shank")

        #  --- add RKJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2: 
                if "RKJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RKJC")
                    logging.info("RKJC added to tracking marker list")
            
        # --- Motion of the Technical frame

        seg.getReferential("TF").motion =[]

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

        # RAJC 
        values_RAJCnode=seg.getReferential('TF').getNodeTrajectory("RAJC")
       
        btkTools.smartAppendPoint(aqui,"RAJC",values_RAJCnode, desc="opt")


        # --- Motion of anatomical Frame 

        seg.anatomicalFrame.motion=[]
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:]
            #R = np.dot(seg.getReferential("TF").motion[i].getRotation(),relativeSegTech )
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            frame=cfr.Frame() 
            frame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(frame)


    # ---- tools ----
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

        abAdangle = np.deg2rad(offset) 
       
        rotAbdAdd = np.array([[1, 0, 0],[0, np.cos(abAdangle), -1.0*np.sin(abAdangle)], [0, np.sin(abAdangle), np.cos(abAdangle) ]])

        finalRot= np.dot(R,rotAbdAdd)
        
        return  np.dot(finalRot,loc)+ank

    # ---- finalize methods ------
        
    def finalizeAbsoluteAngles(self,SegmentLabel,anglesValues):
        """
        Eigen::Matrix<double,Eigen::Dynamic,1> temp = angle->GetValues().col(1);
      angle->GetValues().col(1) = angle->GetValues().col(2);
      angle->GetValues().col(2) =  -1.0 * temp;
        """        
        
        values = np.zeros((anglesValues.shape))
        if SegmentLabel == "Left Foot" :      
            values[:,0] =  np.rad2deg(  anglesValues[:,0])     
            values[:,1] =  np.rad2deg(  anglesValues[:,2])
            values[:,2] = - np.rad2deg(  anglesValues[:,1])

        elif SegmentLabel == "Right Foot" :      
            values[:,0] =  np.rad2deg(  anglesValues[:,0])     
            values[:,1] =  - np.rad2deg(  anglesValues[:,2])
            values[:,2] =  np.rad2deg(  anglesValues[:,1])

        elif SegmentLabel == "RPelvis" :       
            values[:,0] =  np.rad2deg(  anglesValues[:,0])     
            values[:,1] =  - np.rad2deg(  anglesValues[:,1])
            values[:,2] =  np.rad2deg(  anglesValues[:,2])

        elif SegmentLabel == "LPelvis" :     
            values[:,0] =  np.rad2deg(  anglesValues[:,0])     
            values[:,1] =  np.rad2deg(  anglesValues[:,1])
            values[:,2] =  - np.rad2deg(  anglesValues[:,2])

        else:
            values[:,0] = np.rad2deg(  anglesValues[:,0])     
            values[:,1] = np.rad2deg(  anglesValues[:,1])
            values[:,2] = np.rad2deg(  anglesValues[:,2])
        
        return values
            


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

        else:
            values[:,0] = np.rad2deg(  jointValues[:,0])     
            values[:,1] = np.rad2deg(  jointValues[:,1])
            values[:,2] = np.rad2deg(  jointValues[:,2])
            
        return values

    def finalizeKinetics(self,jointLabel,forceValues,momentValues, projection):

        valuesF = np.zeros((forceValues.shape))
        valuesM = np.zeros((momentValues.shape))
        
        if jointLabel == "LAnkle" :
            if projection == pyCGM2Enums.MomentProjection.Distal:
                valuesF[:,0] = - forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] = - forceValues[:,2]            

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] =  - momentValues[:,2]
                valuesM[:,2] = momentValues[:,0]            

            elif projection == pyCGM2Enums.MomentProjection.Proximal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2]  

            elif projection == pyCGM2Enums.MomentProjection.Global:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2] 

        if jointLabel == "LKnee" :
            if projection == pyCGM2Enums.MomentProjection.Distal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = - momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2]            

            elif projection == pyCGM2Enums.MomentProjection.Proximal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2] 

                valuesM[:,0] = - momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2] 
               
            elif projection == pyCGM2Enums.MomentProjection.Global:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2] 

                valuesM[:,0] = - momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2]
                
        if jointLabel == "LHip" :
            if projection == pyCGM2Enums.MomentProjection.Distal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2]            

            elif projection == pyCGM2Enums.MomentProjection.Proximal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2]     
                
            elif projection == pyCGM2Enums.MomentProjection.Global:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2] 

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = momentValues[:,0]
                valuesM[:,2] = momentValues[:,2]     


        if jointLabel == "RAnkle" :
            if projection == pyCGM2Enums.MomentProjection.Distal:
                valuesF[:,0] = - forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] = - forceValues[:,2]            
                
                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = momentValues[:,2]
                valuesM[:,2] = - momentValues[:,0]            

            elif projection == pyCGM2Enums.MomentProjection.Proximal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2] 

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = - momentValues[:,0]
                valuesM[:,2] = - momentValues[:,2]                 
                
            elif projection == pyCGM2Enums.MomentProjection.Global:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]  

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = - momentValues[:,0]
                valuesM[:,2] = - momentValues[:,2]                 

        if jointLabel == "RKnee" :
            if projection == pyCGM2Enums.MomentProjection.Distal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = - momentValues[:,1]
                valuesM[:,1] = - momentValues[:,0]
                valuesM[:,2] = - momentValues[:,2]            

            elif projection == pyCGM2Enums.MomentProjection.Proximal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2] 

                valuesM[:,0] = -momentValues[:,1]
                valuesM[:,1] = -momentValues[:,0]
                valuesM[:,2] = -momentValues[:,2] 
               
            elif projection == pyCGM2Enums.MomentProjection.Global:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2] 

                valuesM[:,0] = -momentValues[:,1]
                valuesM[:,1] = -momentValues[:,0]
                valuesM[:,2] = -momentValues[:,2]
                
        if jointLabel == "RHip" :
            if projection == pyCGM2Enums.MomentProjection.Distal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = - momentValues[:,0]
                valuesM[:,2] = - momentValues[:,2]            

            elif projection == pyCGM2Enums.MomentProjection.Proximal:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2]            

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = -momentValues[:,0]
                valuesM[:,2] = -momentValues[:,2]     
                
            elif projection == pyCGM2Enums.MomentProjection.Global:
                valuesF[:,0] =  forceValues[:,0]
                valuesF[:,1] =  forceValues[:,1]
                valuesF[:,2] =  forceValues[:,2] 

                valuesM[:,0] = momentValues[:,1]
                valuesM[:,1] = -momentValues[:,0]
                valuesM[:,2] = -momentValues[:,2] 

        return valuesF,valuesM    
