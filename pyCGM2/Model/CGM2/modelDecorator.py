# -*- coding: utf-8 -*-

import numpy as np
import pdb
import logging
import matplotlib.pyplot as plt 
from scipy.optimize import least_squares

import cgm
import model

import pyCGM2
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Tools import  btkTools
from pyCGM2.Math import  numeric, geometry


import euler

def setDescription(nodeLabel):
    """
        return a node description
    """
    if "kad" in nodeLabel:
        return "kad"
    elif "sara" in nodeLabel:
        return "sara"
    elif "Hara" in nodeLabel:
        return "hara"        
    elif "Harrington" in nodeLabel:
        return "Harrington"
    elif "mid" in nodeLabel:
        return "mid"
    elif "us" in nodeLabel:
        return "us"
    elif "mri" in nodeLabel:
        return "mri"


    else:
        return "custom"




# ---- CONVENIENT FUNCTIONS ------
def calibration2Dof(proxMotionRef,distMotionRef,indexFirstFrame,indexLastFrame,sequence="YXZ",index=1):

    # onjective function : minimize variance of the knee varus valgus angle
    def objFun(x, proxMotionRef, distMotionRef,indexFirstFrame,indexLastFrame, sequence,index):
        nFrames= len(proxMotionRef) 
         
        frames0 = range(0,len(proxMotionRef))

        if indexFirstFrame and indexLastFrame:
            frames = frames0[indexFirstFrame:indexLastFrame+1]
    
        elif  not indexFirstFrame and indexLastFrame:
            frames = frames0[:indexLastFrame+1]
    
        elif  indexFirstFrame and not indexLastFrame:
            frames = frames0[indexFirstFrame:]
    
        nFrames = len(frames)
        
        
        angle=np.deg2rad(x)     
        rotZ = np.eye(3,3)
        rotZ[0,0] = np.cos(angle)
        rotZ[0,1] = - np.sin(angle)
        rotZ[1,0] = np.sin(angle)
        rotZ[1,1] = np.cos(angle)
    
        jointValues = np.zeros((nFrames,3))

        i=0
        for f in frames #range(0,nFrames): 
            Rprox = np.dot(proxMotionRef[f].getRotation(),rotZ) 
            Rdist = distMotionRef[f].getRotation() 
             
            Rrelative= np.dot(Rprox.T, Rdist)
    
            if sequence == "XYZ":
                Euler1,Euler2,Euler3 = euler.euler_xyz(Rrelative)
            elif sequence == "XZY":
                Euler1,Euler2,Euler3 = euler.euler_xzy(Rrelative)
            elif sequence == "YXZ":
                Euler1,Euler2,Euler3 = euler.euler_yxz(Rrelative)
            elif sequence == "YZX":
                Euler1,Euler2,Euler3 = euler.euler_yzx(Rrelative)
            elif sequence == "ZXY":
                Euler1,Euler2,Euler3 = euler.euler_zxy(Rrelative)
            elif sequence == "ZYX":
                Euler1,Euler2,Euler3 = euler.euler_zyx(Rrelative)
            else:
                raise Exception("[pyCGM2] joint sequence unknown ")

            jointValues[i,0] = Euler1
            jointValues[i,1] = Euler2
            jointValues[i,2] = Euler3
            i+=1
    
        variance = np.var(jointValues[:,index])
          
        return variance

    x0 = 0.0 # deg
    res = least_squares(objFun, x0, args=(proxMotionRef, distMotionRef,sequence,index), verbose=1)

    return res.x[0]




def saraCalibration(proxMotionRef,distMotionRef,indexFirstFrame,indexLastFrame, gap = 100, method = "1"):
    """ 
    
        Computation of the hip joint center position from Harrington's regressions.         
    
        :Parameters:
            - `proxMotionRef` (list of numpy.array(3,3)) - motion of the proximal referential             
            - `distMotionRef` (list of numpy.array(3,3)) - motion of the distal referential
            - `gap` (double) - distance in mm for positionning an axis limit
            - `method` (int) - affect the objective function (see Ehrig et al.). 

        :Returns:
            - `prox_origin` (np.array(3)) - position of the origin in the proximal referential             
            - `prox_axisLim` (np.array(3)) - position on a point on the axis in the proximal referential             
            - `dist_origin` (np.array(3)) - position of the origin in the distal referential             
            - `dist_axisLim` (np.array(3)) - position on a point on the axis in the distal referential
            - `prox_axisNorm` (np.array(3)) - axis in the proximal frame
            - `dist_axisNorm` (np.array(3)) - axis in the proximal frame       
            - `coeffDet`     (double) - See about it with morgan          


        .. warning :: 
            
            linalg.svd and matlab are different. V from scipy has to be transposed. 
            In addition, singular values are returned in a 1d array not a diagonal matrix

        **Reference**
        
        Ehrig, R., Taylor, W. R., Duda, G., & Heller, M. (2007). A survey of formal methods for determining functional joint axes. Journal of Biomechanics, 40(10), 2150–7.

    """ 
    frames0 = range(0,len(proxMotionRef))

    if indexFirstFrame and indexLastFrame:
        frames = frames0[indexFirstFrame:indexLastFrame+1]

    elif  not indexFirstFrame and indexLastFrame:
        frames = frames0[:indexLastFrame+1]

    elif  indexFirstFrame and not indexLastFrame:
        frames = frames0[indexFirstFrame:]

    nFrames = len(frames)    
    
    if method =="1": 
  
        A = np.zeros((nFrames*3,6))
        b = np.zeros((nFrames*3,1)) 
        
        i=0
        for f in frames:
            A[i*3:i*3+3,0:3] = proxMotionRef[f].getRotation()
            A[i*3:i*3+3,3:6] = -1.0 * distMotionRef[f].getRotation()
            b[i*3:i*3+3,:] = (distMotionRef[f].getTranslation() - proxMotionRef[f].getTranslation()).reshape(3,1)       
            i+=1
    
    
        
        
        U,s,V = np.linalg.svd(A,full_matrices=False)
        V = V.T # beware of V ( there is a difference between numpy and matlab)       
        invDiagS = np.identity(6) * (1/s) #s from sv is a line array not a matrix
    
        diagS=np.identity(6) * (s)
    
        CoR = V.dot(invDiagS).dot(U.T).dot(b)
        AoR = V[:,5]     
    
        
        
    elif method =="2": # idem programmation morgan
          
        SR = np.zeros((3,3))
        Sd = np.zeros((3,1))
        SRd = np.zeros((3,1))
    
        # For each frame compute the transformation matrix of the distal
        # segment in the proximal reference system

        for f in frames: 
            Rprox = proxMotionRef[f].getRotation() 
            tprox = proxMotionRef[f].getTranslation()
    
            Rdist = distMotionRef[f].getRotation() 
            tdist = distMotionRef[f].getTranslation() 
           
           
            P = np.concatenate((np.concatenate((Rprox,tprox.reshape(1,3).T),axis=1),np.array([[0,0,0,1]])),axis=0)
            D = np.concatenate((np.concatenate((Rdist,tdist.reshape(1,3).T),axis=1),np.array([[0,0,0,1]])),axis=0)

            T = np.dot(np.linalg.pinv(P),D)
            R = T[0:3,0:3 ]
            d= T[0:3,3].reshape(3,1) 
            

            SR = SR + R
            Sd = Sd + d
            SRd = SRd + np.dot(R.T,d)
            

         
        A0 = np.concatenate((nFrames*np.eye(3),-SR),axis=1)        
        A1 = np.concatenate((-SR.T,nFrames*np.eye(3)),axis=1)                
        
        A = np.concatenate((A0,A1))
        b = np.concatenate((Sd,-SRd))
        
        # CoR
        CoR = np.dot(np.linalg.pinv(A),b) 
   
        # AoR    
        U,s,V = np.linalg.svd(A,full_matrices=False)
        V = V.T # beware of V ( there is a difference between numpy and matlab)
        
        diagS = np.identity(6) * (s)   # s from sv is a line array not a matrix  
    
        AoR = V[:,5] 
    
    CoR_prox = CoR[0:3]
    CoR_dist = CoR[3:6]
    
    prox_axisNorm=AoR[0:3]/np.linalg.norm(AoR[0:3])
    dist_axisNorm=AoR[3:6]/np.linalg.norm(AoR[3:6])

    prox_origin = CoR_prox +  gap * prox_axisNorm.reshape(3,1)
    prox_axisLim = CoR_prox - gap * prox_axisNorm.reshape(3,1)

    dist_origin = CoR_dist + gap * dist_axisNorm.reshape(3,1)
    dist_axisLim = CoR_dist - gap * dist_axisNorm.reshape(3,1)


    S = diagS[3:6,3:6]
    coeffDet = S[2,2]/(np.trace(S)-S[2,2]) #TODO : explanation ? where does it come up

    return prox_origin.reshape(3),prox_axisLim.reshape(3),dist_origin.reshape(3),dist_axisLim.reshape(3),prox_axisNorm,dist_axisNorm,coeffDet
   
   
   
   
   
def haraRegression(mp_input,mp_computed,markerDiameter = 14.0,  basePlate = 2.0):   
    """
        Hip joint centre regression from Hara et al, 2016

        :Parameters:
            - `mp_input` (dict) - dictionnary of the measured anthropometric parameters
            - `mp_computed` (dict) - dictionnary of the cgm-computed anthropometric parameters
            - `markerDiameter` (double) - diameter of the marker
            - `basePlate` (double) - thickness of the base plate
        
        **Reference**
        
        Hara, R., Mcginley, J. L., C, B., Baker, R., & Sangeux, M. (2016). Generation of age and sex specific regression equations to locate the Hip Joint Centres. Gait & Posture
    
    """
    #TODO : remove mp_computed
   
   
    HJCx_L= 11.0 -0.063*mp_input["LeftLegLength"] - markerDiameter/2.0 - basePlate
    HJCy_L=8.0+0.086*mp_input["LeftLegLength"]
    HJCz_L=-9.0-0.078*mp_input["LeftLegLength"]
    
    logging.info("Left HJC position from Hara [ X = %s, Y = %s, Z = %s]" %(HJCx_L,HJCy_L,HJCz_L))    
    HJC_L_hara=np.array([HJCx_L,HJCy_L,HJCz_L])
    
    HJCx_R= 11.0 -0.063*mp_input["RightLegLength"]- markerDiameter/2.0 - basePlate
    HJCy_R=-1.0*(8.0+0.086*mp_input["RightLegLength"])
    HJCz_R=-9.0-0.078*mp_input["RightLegLength"]
        
    logging.info("Right HJC position from Hara [ X = %s, Y = %s, Z = %s]" %(HJCx_R,HJCy_R,HJCz_R))    
    HJC_R_hara=np.array([HJCx_R,HJCy_R,HJCz_R])
   
    HJC_L = HJC_L_hara       
    HJC_R = HJC_R_hara
    
    return HJC_L,HJC_R



def harringtonRegression(mp_input,mp_computed, predictors, markerDiameter = 14.0, basePlate = 2.0, cgmReferential=True):
    """
        Hip joint centre regression from Harrington et al, 2007

        :Parameters:
            - `mp_input` (dict) - dictionnary of the measured anthropometric parameters
            - `mp_computed` (dict) - dictionnary of the cgm-computed anthropometric parameters
            - `predictors` (str) - predictor choice of the regression (full,PWonly,LLonly)
            - `markerDiameter` (double) - diameter of the marker
            - `basePlate` (double) - thickness of the base plate
            - `cgmReferential` (bool) - flag indicating HJC position will be expressed in the CGM pelvis Coordinate system
    
        .. note:: Predictor choice allow using modified Harrington's regression from Sangeux 2015  
    
        '' warning:: this function requires pelvisDepth,asisDistance and meanlegLength which are automaticcly computed during CGM calibration

    
        **Reference**
       
          - Harrington, M., Zavatsky, A., Lawson, S., Yuan, Z., & Theologis, T. (2007). Prediction of the hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance imaging. Journal of Biomechanics, 40(3), 595–602
          - Sangeux, M. (2015). On the implementation of predictive methods to locate the hip joint centres. Gait and Posture, 42(3), 402–405.
    
    """
    #TODO : how to work without CGM calibration

    if predictors.value == "full":
        HJCx_L=-0.24*mp_computed["PelvisDepth"]-9.9  - markerDiameter/2.0 - basePlate # post/ant
        HJCy_L=-0.16*mp_computed["InterAsisDistance"]-0.04*mp_computed["MeanlegLength"]-7.1 
        HJCz_L=-1*(0.28*mp_computed["PelvisDepth"]+0.16*mp_computed["InterAsisDistance"]+7.9)
        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])

        HJCx_R=-0.24*mp_computed["PelvisDepth"]-9.9 - markerDiameter/2.0 - basePlate# post/ant
        HJCy_R=-0.16*mp_computed["InterAsisDistance"]-0.04*mp_computed["MeanlegLength"]-7.1 
        HJCz_R=1*(0.28*mp_computed["PelvisDepth"]+0.16*mp_computed["InterAsisDistance"]+7.9) 
        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R])    

    elif predictors.value=="PWonly":
        HJCx_L=-0.138*mp_computed["InterAsisDistance"]-10.4 - markerDiameter/2.0 - basePlate
        HJCy_L=-0.305*mp_computed["InterAsisDistance"]-10.9
        HJCz_L=-1*(0.33*mp_computed["InterAsisDistance"]+7.3)
        
        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])
    
        HJCx_R=-0.138*mp_computed["InterAsisDistance"]-10.4 - markerDiameter/2.0 - basePlate
        HJCy_R=-0.305*mp_computed["InterAsisDistance"]-10.9
        HJCz_R=1*(0.33*mp_computed["InterAsisDistance"]+7.3)
        
        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R]) 
    

    elif predictors.value=="LLonly":
        HJCx_L=-0.041*mp_computed["MeanlegLength"]-6.3 - markerDiameter/2.0 - basePlate
        HJCy_L=-0.083*mp_computed["MeanlegLength"]-7.9
        HJCz_L=-1*(0.0874*mp_computed["MeanlegLength"]+5.4)
        
        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])
    
        HJCx_R=-0.041*mp_computed["MeanlegLength"]-6.3 - markerDiameter/2.0 - basePlate
        HJCy_R=-0.083*mp_computed["MeanlegLength"]-7.9
        HJCz_R=1*(0.0874*mp_computed["MeanlegLength"]+5.4)
        
        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R])
        
    else:
        raise Exception("[pyCGM2] Predictor is unknown choixe possible : full, PWonly, LLonly")

    if cgmReferential :
        Rhar_cgm1=np.array([[1, 0, 0],[0, 0, -1], [0, 1, 0]])
        HJC_L = np.dot(Rhar_cgm1,HJC_L_har)       
        HJC_R = np.dot(Rhar_cgm1,HJC_R_har)
        logging.info("computation in cgm pelvis referential")
        logging.info("Left HJC position from Harrington [ X = %s, Y = %s, Z = %s]" %(HJC_L[0],HJC_L[1],HJC_L[2])) 
        logging.info("Right HJC position from Harrington [ X = %s, Y = %s, Z = %s]" %(HJC_L[0],HJC_L[1],HJC_L[2])) 
    else:
        HJC_L = HJC_L_har       
        HJC_R = HJC_R_har
    
    return HJC_L,HJC_R


# -------- ABSTRACT DECORATOR MODEL : INTERFACE ---------

class DecoratorModel(model.Model):
    # interface    

    def __init__(self, iModel):
        super(DecoratorModel,self).__init__()
        self.model = iModel

#-------- CONCRETE DECORATOR MODEL ---------
class Kad(DecoratorModel):
    """
         A concrete CGM decorator altering the knee joint centre from the Knee Aligment device    
    """ 
    def __init__(self, iModel,iAcq):
        """
            :Parameters:
                - `iModel` (pyCGM2.Model.CGM2.cgm.CGM) - a CGM instance   
                - `iAcq` (btkAcquisition) - btk aquisition inctance of a static c3d with the KAD          
        """
        
        super(Kad,self).__init__(iModel)
        self.acq = iAcq
        
    def compute(self,side="both",markerDiameter = 14):    
        """
             :Parameters:
                - `side` (str) - body side 
                - `markerDiameter` (double) - diameter of the marker
                - `displayMarkers` (bool) - display markers RKNE, RKJC and RAJC from KAD processing  

        """
        distSkin = 0           
        
        ff = self.acq.GetFirstFrame() 

        frameInit =  self.acq.GetFirstFrame()-ff  
        frameEnd = self.acq.GetLastFrame()-ff+1
                
        #self.model.nativeCgm1 = False
        self.model.decoratedModel = True

        LKJCvalues =  np.zeros((self.acq.GetPointFrameNumber(),3))
        LKNEvalues =  np.zeros((self.acq.GetPointFrameNumber(),3)) 
        LAJCvalues = np.zeros((self.acq.GetPointFrameNumber(),3)) 

        RKJCvalues = np.zeros((self.acq.GetPointFrameNumber(),3))
        RKNEvalues =  np.zeros((self.acq.GetPointFrameNumber(),3)) 
        RAJCvalues = np.zeros((self.acq.GetPointFrameNumber(),3)) 

        
        if side == "both" or side == "left":
            
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if self.model.mp.has_key("LeftThighRotation") : self.model.mp["LeftThighRotation"] =0 # look out, it's mp, not mp_computed.   
                if self.model.mp.has_key("LeftShankRotation") : self.model.mp["LeftShankRotation"] =0             
            
            for i in range(0,self.acq.GetPointFrameNumber()):
                #  compute points left and right lateral condyle
                LKAX = self.acq.GetPoint("LKAX").GetValues()[i,:]
                LKD1 = self.acq.GetPoint("LKD1").GetValues()[i,:]
                LKD2 = self.acq.GetPoint("LKD2").GetValues()[i,:]
                    
                dist = np.array([np.linalg.norm(LKAX-LKD1), np.linalg.norm(LKAX-LKD2),np.linalg.norm(LKD1-LKD2)] )
                dist =  dist / np.sqrt(2)
                res = np.array([np.mean(dist), np.var(dist)])
                n = np.cross(LKD2-LKD1 , LKAX-LKD1)
                n= n/np.linalg.norm(n)
                
                I = (LKD1+LKAX)/2
                PP1 = 2/3.0*(I-LKD2)+LKD2
                O = PP1 - n*np.sqrt(3)*res[0]/3.0   
                LKAXO = (O-LKAX)/np.linalg.norm(O-LKAX)
    
                LKNEvalues[i,:] = O + LKAXO * distSkin
            
                # locate KJC
    #            LKJC = LKNE + LKAXO * (self.model.mp["leftKneeWidth"]+markerDiameter )/2.0
                if btkTools.isPointExist(self.acq,"LHJC"):
                    LHJC = self.acq.GetPoint("LHJC").GetValues()[i,:]
                    LKJCvalues[i,:] = cgm.CGM1LowerLimbs.chord( (self.model.mp["LeftKneeWidth"]+markerDiameter )/2.0 ,LKNEvalues[i,:],LHJC,LKAX, beta= 0.0 )
                else:
                    LKJCvalues[i,:] = LKNEvalues[i,:] + LKAXO * (self.model.mp["LeftKneeWidth"]+markerDiameter )/2.0
                              
                # locate AJC
                LANK = self.acq.GetPoint("LANK").GetValues()[i,:]
                LAJCvalues[i,:] = cgm.CGM1LowerLimbs.chord( (self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0 ,LANK,LKJCvalues[i,:],LKAX,beta= 0.0 )
            

            # add nodes to referential 
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKNE_kad",LKNEvalues.mean(axis =0),positionType="Global")
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKJC_kad",LKJCvalues.mean(axis =0),positionType="Global")  
            
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKNE_kad",LKNEvalues.mean(axis =0),positionType="Global")
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKJC_kad",LKJCvalues.mean(axis =0),positionType="Global") 
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LAJC_kad",LAJCvalues.mean(axis =0),positionType="Global")

            
        if side == "both" or side == "right":

            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):

                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if self.model.mp.has_key("RightThighRotation") : self.model.mp["RightThighRotation"] =0 # look out, it's mp, not mp_computed.   
                if self.model.mp.has_key("RightShankRotation") : self.model.mp["RightShankRotation"] =0 


            for i in range(0,self.acq.GetPointFrameNumber()):
                #  compute points left and right lateral condyle                    
                RKAX = self.acq.GetPoint("RKAX").GetValues()[i,:]
                RKD1 = self.acq.GetPoint("RKD1").GetValues()[i,:]
                RKD2 = self.acq.GetPoint("RKD2").GetValues()[i,:]
                    
                dist = np.array([np.linalg.norm(RKAX-RKD1), np.linalg.norm(RKAX-RKD2),np.linalg.norm(RKD1-RKD2)] )
                dist =  dist / np.sqrt(2)
                res = np.array([np.mean(dist), np.var(dist)])
                n = np.cross(RKD2-RKD1 , RKAX-RKD1)
                n= n/np.linalg.norm(n)
                
                n=-n # look out the negative sign
                
                
                I = (RKD1+RKAX)/2
                PP1 = 2/3.0*(I-RKD2)+RKD2
                O = PP1 - n*np.sqrt(3)*res[0]/3.0   
                RKAXO = (O-RKAX)/np.linalg.norm(O-RKAX)
                RKNEvalues[i,:] = O + RKAXO * distSkin
            
                # locate KJC
    #            RKJC = RKNE + RKAXO * (self.model.mp["rightKneeWidth"]+markerDiameter )/2.0
                if btkTools.isPointExist(self.acq,"RHJC"):
                    RHJC = self.acq.GetPoint("RHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                    RKJCvalues[i,:] = cgm.CGM1LowerLimbs.chord( (self.model.mp["RightKneeWidth"]+markerDiameter )/2.0 ,RKNEvalues[i,:],RHJC,RKAX,beta= 0.0 )
                else:
                    RKJCvalues[i,:] = RKNEvalues[i,:] + RKAXO * (self.model.mp["RightKneeWidth"]+markerDiameter )/2.0
                    
                # locate AJC
                RANK = self.acq.GetPoint("RANK").GetValues()[i,:]
                RAJCvalues[i,:] = cgm.CGM1LowerLimbs.chord( (self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0 ,RANK,RKJCvalues[i,:],RKAX,beta= 0.0 )
            
            # add nodes to referential
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKNE_kad",RKNEvalues.mean(axis=0),positionType="Global")
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKJC_kad",RKJCvalues.mean(axis=0),positionType="Global")  
            
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKNE_kad",RKNEvalues.mean(axis=0),positionType="Global")
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKJC_kad",RKJCvalues.mean(axis=0),positionType="Global")
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RAJC_kad",RAJCvalues.mean(axis=0),positionType="Global")
    
        # add KNE markers to static c3d
        if side == "both" or side == "left":
            btkTools.smartAppendPoint(self.acq,"LKNE",LKNEvalues, desc="KAD")
            btkTools.smartAppendPoint(self.acq,"LKJC_KAD",LKJCvalues, desc="KAD")
            btkTools.smartAppendPoint(self.acq,"LAJC_KAD",LAJCvalues, desc="KAD")

        if side == "both" or side == "right":
            btkTools.smartAppendPoint(self.acq,"RKNE",RKNEvalues, desc="KAD") # KNE updated.     
            btkTools.smartAppendPoint(self.acq,"RKJC_KAD",RKJCvalues, desc="KAD")
            btkTools.smartAppendPoint(self.acq,"RAJC_KAD",RAJCvalues, desc="KAD")


        #btkTools.smartWriter(self.acq, "tmp-static-KAD.c3d")        

            
class Cgm1ManualOffsets(DecoratorModel):
    """

    """ 
    def __init__(self, iModel):
        """
            :Parameters:
              - `iModel` (pyCGM2.Model.CGM2.cgm.CGM) - a CGM instance 
        """
        super(Cgm1ManualOffsets,self).__init__(iModel)
        

    def compute(self,acq,side,thighoffset,markerDiameter,tibialTorsion,shankoffset):
        """
        Replicate behaviour of CGM1 in case of manual modification of offsets. 
        That means only a non zero value of thigh offset enable re computation of AJC           
        
         :Parameters:
            - `side` (str) - body side 
            - `thighoffset` (double) - thigh offset
            - `markerDiameter` (double) - diameter of marker
            - `shankoffset` (double) - shanl offset
            - `tibialTorsion` (double) - tinbial torsion value
        """
        
         
        self.model.decoratedModel = True
        
        ff = acq.GetFirstFrame()
        frameInit =  acq.GetFirstFrame()-ff  
        frameEnd = acq.GetLastFrame()-ff+1        
        
        if side == "left":
            
            # zeroing of shankRotation if non-zero
            if shankoffset!=0:
                if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                    if self.model.mp.has_key("LeftShankRotation") : 
                        self.model.mp["LeftShankRotation"] = 0
                        logging.warning("Special CGM1 case - shank offset cancelled")            
            
            # location of KJC and AJC depends on thighRotation and tibial torsion 
            HJC = acq.GetPoint("LHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            KNE = acq.GetPoint("LKNE").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            THI = acq.GetPoint("LTHI").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            
            KJC = cgm.CGM1LowerLimbs.chord((self.model.mp["LeftKneeWidth"]+markerDiameter )/2.0 ,KNE,HJC,THI, beta= -1*thighoffset )
            
            
            # locate AJC    
            ANK = acq.GetPoint("LANK").GetValues()[frameInit:frameEnd,:].mean(axis=0)

            if thighoffset !=0 :
                AJC = cgm.CGM1LowerLimbs.chord( (self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,KNE,beta= -1.*tibialTorsion )
 
            else:                                   
                TIB = acq.GetPoint("LTIB").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                AJC = cgm.CGM1LowerLimbs.chord( (self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,TIB,beta= 0 )

                
            # add nodes to referential 
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKJC_mo",KJC,positionType="Global")  


            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKJC_mo",KJC,positionType="Global") 
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LAJC_mo",AJC,positionType="Global")
            
            # enable tibialTorsion flag
            if thighoffset !=0 and tibialTorsion !=0:
                self.model.m_useLeftTibialTorsion=True
        
        
        if side == "right":
            
            # zeroing of shankRotation if non-zero
            if shankoffset!=0:
                if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                    if self.model.mp.has_key("RightShankRotation") : 
                        self.model.mp["RightShankRotation"] = 0
                        logging.warning("Special CGM1 case - shank offset cancelled")            

            # location of KJC and AJC depends on thighRotation and tibial torsion 
            HJC = acq.GetPoint("RHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            KNE = acq.GetPoint("RKNE").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            THI = acq.GetPoint("RTHI").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            
            KJC = cgm.CGM1LowerLimbs.chord((self.model.mp["RightKneeWidth"]+markerDiameter )/2.0 ,KNE,HJC,THI, beta= thighoffset )

            # locate AJC            
            ANK = acq.GetPoint("RANK").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            if thighoffset != 0 :
                AJC = cgm.CGM1LowerLimbs.chord( (self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,KNE,beta= tibialTorsion )
            else:                
                
                TIB = acq.GetPoint("RTIB").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                AJC = cgm.CGM1LowerLimbs.chord( (self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,TIB,beta= 0 )

                                        
            # create and add nodes to the technical referential 
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKJC_mo",KJC,positionType="Global")  
            
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKJC_mo",KJC,positionType="Global") 
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RAJC_mo",AJC,positionType="Global") 
            
            # enable tibialTorsion flag    
            if thighoffset !=0 and tibialTorsion!=0:
                self.model.m_useRightTibialTorsion=True





class HipJointCenterDecorator(DecoratorModel):
    """
        Concrete CGM decorators altering the hip joint centre     
    """ 
    def __init__(self, iModel):
        """
            :Parameters:
              - `iModel` (pyCGM2.Model.CGM2.cgm.CGM) - a CGM instance 
        """
        super(HipJointCenterDecorator,self).__init__(iModel)
        
    def custom(self,position_Left=0,position_Right=0,methodDesc="custom"):
        """ 
        
            Locate hip joint centres manually          
        
            :Parameters:
               - `position_Left` (np.array(3,)) - position of the left hip center in the Pelvis Referential           
               - `position_Right` (np.array(3,)) - position of the right hip center in the Pelvis Referential
               - `methodDesc` (str) - short description of the method 
           
           .. warning :: look out the Pelvis referential. It has to be similar with cgm1. 
    
        """

        self.model.decoratedModel = True

        if position_Left.shape ==(3,):
            nodeLabel= "LHJC_"+ methodDesc
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode(nodeLabel,position_Left, positionType="Local")
        if position_Left.shape ==(3,):
            nodeLabel= "RHJC_"+ methodDesc
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode(nodeLabel,position_Right, positionType="Local")

        
        

        
    def harrington(self,predictors= pyCGM2Enums.HarringtonPredictor.Native, side="both"):    
        """ 
            Use of the Harrington's regressions        
        
            :Parameters:
               - `predictors` (pyCGM2.enums) - enums specifying harrington's predictors to use
               - `side` (str) - body side 
        
        """ 

        self.model.decoratedModel = True

        LHJC_har,RHJC_har=harringtonRegression(self.model.mp,self.model.mp_computed,predictors)

        if side == "both":
            
            # add nodes to pelvis            
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("LHJC_Harrington",LHJC_har, positionType="Local")
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("RHJC_Harrington",RHJC_har, positionType="Local")


            # add nodes Thigh
            pos_L=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_Harrington").m_global
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LHJC_Harrington",pos_L, positionType="Global")

            pos_R=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_Harrington").m_global
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RHJC_Harrington",pos_R, positionType="Global")


        elif side == "left":
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("LHJC_Harrington",LHJC_har, positionType="Local")
            pos_L=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_Harrington").m_global
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LHJC_Harrington",pos_L, positionType="Global")


        elif side == "right":
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("RHJC_Harrington",RHJC_har, positionType="Local")
            pos_R=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_Harrington").m_global
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RHJC_Harrington",pos_R, positionType="Global")

    def hara(self, side="both"):    
        """ 
            Use of the Hara's regressions        
        
            :Parameters:
               - `side` (str) - body side 
        
        """ 
 
        self.model.decoratedModel = True

        LHJC_hara,RHJC_hara=haraRegression(self.model.mp,self.model.mp_computed)

        
        if side == "both":
            # add nodes to pelvis            
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("LHJC_Hara",LHJC_hara, positionType="Local")
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("RHJC_Hara",RHJC_hara, positionType="Local")
            
            # add nodes Thigh
            pos_L=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_Hara").m_global
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LHJC_Hara",pos_L, positionType="Global")

            pos_R=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_Hara").m_global
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RHJC_Hara",pos_R, positionType="Global")
    
    
    

class KneeCalibrationDecorator(DecoratorModel):
    """
        Concrete cgm decorator altering the knee joint      
    """ 
    def __init__(self, iModel):
        """
            :Parameters:
              - `iModel` (pyCGM2.Model.CGM2.cgm.CGM) - a CGM instance 
        """

        super(KneeCalibrationDecorator,self).__init__(iModel)
        
        
        
    def midCondyles(self,acq, side="both",
                    leftLateralKneeLabel="LKNE", leftMedialKneeLabel="LMEPI",rightLateralKneeLabel="RKNE", rightMedialKneeLabel="RMEPI", 
                    markerDiameter = 14, withNoModelParameter=False, cgm1Behaviour=False):   
        """ 
            Compute Knee joint centre from mid condyles.
            
            .. note:: AJC might be relocate, like KAD processing if cgm1Behaviour flag enable
        
            :Parameters:
                - `acq` (btkAcquisition) - a btk acquisition instance of a static c3d           
                - `side` (str) - body side
                - `leftLateralKneeLabel` (str) -  label of the left lateral knee marker
                - `leftMedialKneeLabel` (str) -  label of the left medial knee marker
                - `withNoModelParameter` (bool) -  use mid position directly instead of applying an offset along mediolateral axis
                - `cgm1Behaviour` (bool) -  relocate AJC
        
        """         
        # TODO : coding exception if label doesn t find.        
         
        ff = acq.GetFirstFrame()     
    
        frameInit =  acq.GetFirstFrame()-ff  
        frameEnd = acq.GetLastFrame()-ff+1
                
        #self.model.nativeCgm1 = False
        self.model.decoratedModel = True
        
        LKJCvalues = np.zeros((acq.GetPointFrameNumber(),3)) 
        RKJCvalues = np.zeros((acq.GetPointFrameNumber(),3))     
        
        LAJCvalues = np.zeros((acq.GetPointFrameNumber(),3)) 
        RAJCvalues = np.zeros((acq.GetPointFrameNumber(),3))         
        
        if side=="both" or side=="left":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if self.model.mp.has_key("LeftThighRotation") : self.model.mp["LeftThighRotation"] =0 

            for i in range(0,acq.GetPointFrameNumber()):
                LKNE = acq.GetPoint(leftLateralKneeLabel).GetValues()[i,:]
                LMEPI = acq.GetPoint(leftMedialKneeLabel).GetValues()[i,:]  
               
                v = LMEPI-LKNE
                v=v/np.linalg.norm(v)
               
                LKJCvalues[i,:] = LKNE + ((self.model.mp["LeftKneeWidth"]+markerDiameter )/2.0)*v
   
                 # locate AJC
                LANK = acq.GetPoint("LANK").GetValues()[i,:]
                LAJCvalues[i,:] = cgm.CGM1LowerLimbs.chord( (self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0 ,LANK,LKJCvalues[i,:],LKNE,beta= 0.0 )

        
            # nodes   
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKJC_mid",LKJCvalues.mean(axis=0), positionType="Global")
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKJC_mid",LKJCvalues.mean(axis=0), positionType="Global")

            # marker
            btkTools.smartAppendPoint(acq,"LKJC_MID",LKJCvalues, desc="MID")
            
   
            
            # add nodes to referential             
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LAJC_midKnee",LAJCvalues.mean(axis =0),positionType="Global")


            
        if side=="both" or side=="right":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if self.model.mp.has_key("RightThighRotation") : self.model.mp["RightThighRotation"] =0 

            for i in range(0,acq.GetPointFrameNumber()):

                RKNE = acq.GetPoint(rightLateralKneeLabel).GetValues()[i,:]
                RMEPI = acq.GetPoint(rightMedialKneeLabel).GetValues()[i,:] 
               
                v = RMEPI-RKNE
                v=v/np.linalg.norm(v)
               
                RKJCvalues[i,:] = RKNE + ((self.model.mp["RightKneeWidth"]+markerDiameter )/2.0)*v
                
                # locate AJC
                RANK = acq.GetPoint("RANK").GetValues()[i,:]
                RAJCvalues[i,:] = cgm.CGM1LowerLimbs.chord( (self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0 ,RANK,RKJCvalues[i,:],RKNE,beta= 0.0 )

            # nodes
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKJC_mid",RKJCvalues.mean(axis=0), positionType="Global")
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKJC_mid",RKJCvalues.mean(axis=0), positionType="Global")
            #marker
            btkTools.smartAppendPoint(acq,"RKJC_MID",RKJCvalues, desc="MID")                   
        
            
            
            # add nodes to referential             
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RAJC_midKnee",RAJCvalues.mean(axis =0),positionType="Global")
        
    
    def sara(self,side,**kwargs):    
        """ 
            Compute Knee flexion axis and relocate knee joint centre from SARA functional calibration         
        
            :Parameters:
                - `side` (str) - lower limb side

        """
        self.model.decoratedModel = True

        iff = kwargs["indexFirstFrame"] if kwargs.has_key("indexFirstFrame") else None
        ilf = kwargs["indexLastFrame"] if kwargs.has_key("indexLastFrame") else None


        

        if side == "Left":
            proxSegmentLabel = "Left Thigh"
            distSegmentlabel = "Left Shank"
            HJClabel = "LHJC"
            KJClabel = "LKJC"
        elif side == "Right":
            proxSegmentLabel = "Right Thigh"
            distSegmentlabel = "Right Shank"
            HJClabel = "RHJC"
            KJClabel = "RKJC"
        else:
            raise Exception("[pyCGM2] side doesn t recongnize")


    
        proxMotion = self.model.getSegment(proxSegmentLabel).getReferential("TF").motion
        distMotion = self.model.getSegment(distSegmentlabel).getReferential("TF").motion
        
        # -- main function -----
        prox_ori,prox_axisLim,dist_ori,dist_axisLim,axis_prox,axis_dist,quality = saraCalibration(proxMotion,distMotion,iff, ilf, method="2")
        # end function -----


        # add nodes in TF        
        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KneeFlexionOri",prox_ori,positionType="Local")
        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KneeFlexionAxis",prox_axisLim,positionType="Local")
    
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KneeFlexionOri",dist_ori,positionType="Local")
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KneeFlexionAxis",dist_axisLim,positionType="Local")     
   
        # compute error
        xp = self.model.getSegment(proxSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        xd = self.model.getSegment(distSegmentlabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")

        ferr = xp-xd
        Merr = numeric.rms(ferr)
        logging.info( " sara rms error : %s " % str(Merr))

        # --- registration of the Knee center ---
        
        # longitudinal axis of the femur 
        p1 = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel(HJClabel).m_global
        p2 = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel(KJClabel).m_global

        # middle of origin in Global
        p_proxOri = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel("KneeFlexionOri").m_global
        p_distOri = self.model.getSegment(distSegmentlabel).getReferential("TF").static.getNode_byLabel("KneeFlexionOri").m_global        
        center = np.mean((p_distOri,p_proxOri),axis= 0)
        
        # axis lim
        p_proxAxis = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel("KneeFlexionAxis").m_global
        p_distAxis = self.model.getSegment(distSegmentlabel).getReferential("TF").static.getNode_byLabel("KneeFlexionAxis").m_global
        meanAxis = np.mean((p_proxAxis,p_distAxis),axis= 0)

        # intersection beetween midcenter-axis and logitudinal axis
        proxIntersect,pb1 = geometry.LineLineIntersect(center,p_proxAxis,p1,p2)
        distIntersect,pb2 = geometry.LineLineIntersect(center,p_distAxis,p1,p2)# 


        # shortest distance                
        shortestDistance_prox =  np.linalg.norm(proxIntersect-pb1)
        logging.info(" 3d line intersect : shortest distance beetween logidudinal axis and flexion axis in Proximal  : %s  " % str(shortestDistance_prox))
        shortestDistance_dist =  np.linalg.norm(distIntersect-pb2)
        logging.info( " 3d line intersect : shortest distance beetween logidudinal axis and flexion axis in Distal  : %s  " % str(shortestDistance_dist))

        # mean of the intersection point        
        center = np.mean((proxIntersect,distIntersect), axis=0)

        # Node manager
        #  the node KJC_sara is added in all "Referentials. 

        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KJC_Sara",center, positionType="Global")        
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KJC_Sara",center, positionType="Global")       
 

        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KJC_SaraAxis",meanAxis, positionType="Global")        
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KJC_SaraAxis",meanAxis, positionType="Global") 

 
         # Comparison of local position of KJCs
        localKJC = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel(KJClabel).m_local 
        saraKJC = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel("KJC_Sara").m_local

        logging.info(" former KJC position in the proximal segment : [ %f, %f,%f]   " % (localKJC[0],localKJC[1],localKJC[2]))
        logging.info(" new KJC position in the proximal segment : [ %f, %f,%f]   " % (saraKJC[0],saraKJC[1],saraKJC[2]))
        
    def calibrate2dof(self,side,**kwargs):    
        """ 
            "2dof" knee calibration (similar to dynaKAD)
        
            :Parameters:
                - `side` (str) - lower limb side

        """
        self.model.decoratedModel = True

        iff = kwargs["indexFirstFrame"] if kwargs.has_key("indexFirstFrame") else None
        ilf = kwargs["indexLastFrame"] if kwargs.has_key("indexLastFrame") else None

        if side == "Left":
            proxSegmentLabel = "Left Thigh"
            distSegmentlabel = "Left Shank"
            offsetLabel ="LeftKnee2DofOffset"
        elif side == "Right":
            proxSegmentLabel = "Right Thigh"
            distSegmentlabel = "Right Shank"
            offsetLabel ="RightKnee2DofOffset"
        else:
            raise Exception("[pyCGM2] side doesn t recongnize")


    
        proxMotion = self.model.getSegment(proxSegmentLabel).getReferential("TF").motion
        distMotion = self.model.getSegment(distSegmentlabel).getReferential("TF").motion
        
        # -- main function -----
        longRot = calibration2Dof(proxMotion,distMotion)
        # end function -----
        self.model.mp_computed[offsetLabel] = longRot
        
        

        



class AnkleCalibrationDecorator(DecoratorModel):
    """
        Concrete cgm decorator altering the ankle joint     
    """
    def __init__(self, iModel):
        """
            :Parameters:
              - `iModel` (pyCGM2.Model.CGM2.cgm.CGM) - a CGM instance 
        """
        super(AnkleCalibrationDecorator,self).__init__(iModel)
        
    def midMaleolus(self,acq, side="both",
                    leftLateralAnkleLabel="LANK", leftMedialAnkleLabel="LMED",
                    rightLateralAnkleLabel="RANK", rightMedialAnkleLabel="RMED", markerDiameter= 14): 

        """ 
            Compute Ankle joint centre from mid malleoli         
        
            :Parameters:
                - `acq` (btkAcquisition) - a btk acquisition instance of a static c3d           
                - `side` (str) - body side
                - `leftLateralAnkleLabel` (str) -  label of the left lateral knee marker
                - `leftMedialAnkleLabel` (str) -  label of the left medial knee marker
                
        
        """                         
                        
                        
        ff = acq.GetFirstFrame()     
    
        frameInit =  acq.GetFirstFrame()-ff  
        frameEnd = acq.GetLastFrame()-ff+1
                
        #self.model.nativeCgm1 = False
        self.model.decoratedModel = True
        
        LAJCvalues = np.zeros((acq.GetPointFrameNumber(),3)) 
        RAJCvalues = np.zeros((acq.GetPointFrameNumber(),3)) 
        
        if side=="both" or side=="left":
            
            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                self.model.m_useLeftTibialTorsion=True
                if self.model.mp.has_key("LeftTibialTorsion") : self.model.mp["LeftTibialTorsion"] =0

            for i in range(0,acq.GetPointFrameNumber()):
                LANK = acq.GetPoint(leftLateralAnkleLabel).GetValues()[i,:]
                LMED = acq.GetPoint(leftMedialAnkleLabel).GetValues()[i,:]  
               
                v = LMED-LANK
                v=v/np.linalg.norm(v)
               
                LAJCvalues[i,:] = LANK + ((self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0)*v

            # add node            
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LAJC_mid",LAJCvalues.mean(axis=0), positionType="Global")

            if repr(self.model) == "LowerLimb CGM1":            
                self.model.getSegment("Left Foot").getReferential("TF").static.addNode("LAJC_mid",LAJCvalues.mean(axis=0), positionType="Global")
            # 
            btkTools.smartAppendPoint(acq,"LAJC_MID",LAJCvalues, desc="MID")             
             
         
            

        if side=="both" or side=="right":
            
            
            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                self.model.m_useRightTibialTorsion=True
                if self.model.mp.has_key("RightTibialTorsion") : self.model.mp["RightTibialTorsion"] =0
            
            for i in range(0,acq.GetPointFrameNumber()):
                RANK = acq.GetPoint(rightLateralAnkleLabel).GetValues()[i,:]
                RMED = acq.GetPoint(rightMedialAnkleLabel).GetValues()[i,:]  
               
                v = RMED-RANK
                v=v/np.linalg.norm(v)
               
                RAJCvalues[i,:] =  RANK + ((self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0)*v


            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RAJC_mid",RAJCvalues.mean(axis=0), positionType="Global")
            if repr(self.model) == "LowerLimb CGM1":            
                self.model.getSegment("Right Foot").getReferential("TF").static.addNode("RAJC_mid",RAJCvalues.mean(axis=0), positionType="Global")

            btkTools.smartAppendPoint(acq,"RAJC_MID",RAJCvalues, desc="MID")             

    def midMaleolusAxis(self,acq, side="both",
                    leftLateralAnkleLabel="LANK", leftMedialAnkleLabel="LMED",
                    rightLateralAnkleLabel="RANK", rightMedialAnkleLabel="RMED", markerDiameter= 14, withNoModelParameter=False):   
        
        
        ff = acq.GetFirstFrame()     
    
        frameInit =  acq.GetFirstFrame()-ff  
        frameEnd = acq.GetLastFrame()-ff+1
                
        #self.model.nativeCgm1 = False
        self.model.decoratedModel = True
        
        if side=="both" or side=="left":
            pass
            
            

        if side=="both" or side=="right":
            
            
            pt1=acq.GetPoint("RANK").GetValues().mean(axis=0)
            pt2=acq.GetPoint("RMED").GetValues().mean(axis=0)        
            a1=(pt2-pt1) #KJC-AJC
            midMaleolusAxis=a1/np.linalg.norm(a1)            
            
            loc1=    np.dot(self.model.getSegment("Right Shank").anatomicalFrame.static.getRotation().T, 
                        midMaleolusAxis)
                        
            proj1 = np.array([ loc1[0],
                               loc1[1],
                                 0]) 
                             
            v1 = proj1/np.linalg.norm(proj1)                             

            loc2=    np.dot(self.model.getSegment("Right Shank").anatomicalFrame.static.getRotation().T, 
                        self.model.getSegment("Right Thigh").anatomicalFrame.static.m_axisY)

            proj2 = np.array([ loc2[0],
                               loc2[1],
                                 0]) 
                                 
            v2 = proj2/np.linalg.norm(proj2) 
    
            angle= np.arccos(np.dot(v2,v1))
        
            print "****** right Tibial Main **********"
            print np.sign(np.cross(v2,v1)[2])*angle*360.0/(2.0*np.pi)        
        


