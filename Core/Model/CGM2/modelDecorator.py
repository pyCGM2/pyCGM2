# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:38:45 2015

@author: fleboeuf
"""

import numpy as np
import pdb


import cgm
import model

import pyCGM2.Core.enums as pyCGM2Enums
from pyCGM2.Core.Tools import  btkTools
from pyCGM2.Core.Math import  numeric, geometry



def setDescription(nodeLabel):
    if "kad" in nodeLabel:
        return "kad"
    elif "sara" in nodeLabel:
        return "sara"
    elif "hara" in nodeLabel:
        return "hara"        
    elif "har" in nodeLabel:
        return "har"
    elif "mid" in nodeLabel:
        return "mid"
    elif "us" in nodeLabel:
        return "us"
    elif "mri" in nodeLabel:
        return "mri"


    else:
        return "custom"



def saraCalibration(proxMotionRef,distMotionRef, gap = 100, method = "1"):
    """ Computation of the hip joint center position from Harrington's regressions.         
    
    :Parameters:
       - `proxMotionRef` (list of Frame) - motion of the proximal referential             
       - `distMotionRef` (list of Frame) - motion of the distal referential
       - `gap` (double) - distance in mm for positionning an axis limit
       - `method` (int) - affect the objective function ( see ehrig). 

    :Returns:
       - `prox_origin` (np.array(3)) - position of the origin in the proximal referential             
       - `prox_axisLim` (np.array(3)) - position on a point on the axis in the proximal referential             
       - `dist_origin` (np.array(3)) - position of the origin in the distal referential             
       - `dist_axisLim` (np.array(3)) - position on a point on the axis in the distal referential
       - `prox_axisNorm` (np.array(3)) - axis in the proximal frame
       - `dist_axisNorm` (np.array(3)) - axis in the proximal frame       
       - `coeffDet`     (double) - See about it with morgan          


    .. warning :: 
    
        linalg.svd and matlab are different. V from scipy has to be transpose. 
        Moreover, singular values output in a 1d array not a diagonal matrix

    .. todo :: 
    
        check the computation of the coefDet with Morgan
        
        is it the SARA method or the ATT ? ( morgan use another algorithm)

    """ 

    if method =="1": 
  
        nFrames= len(proxMotionRef)  
        
        A = np.zeros((nFrames*3,6))
        b = np.zeros((nFrames*3,1)) 
        
        
        for i in range(0,nFrames):
            A[i*3:i*3+3,0:3] = proxMotionRef[i].getRotation()
            A[i*3:i*3+3,3:6] = -1.0 * distMotionRef[i].getRotation()
            b[i*3:i*3+3,:] = (distMotionRef[i].getTranslation() - proxMotionRef[i].getTranslation()).reshape(3,1)       
    
    
        
        
        U,s,V = np.linalg.svd(A,full_matrices=False)
        V = V.T # beware of V ( there is a difference between numpy and matlab)       
        invDiagS = np.identity(6) * (1/s) #s from sv is a line array not a matrix
    
        diagS=np.identity(6) * (s)
    
        CoR = V.dot(invDiagS).dot(U.T).dot(b)
        AoR = V[:,5]     
    
        
        
    elif method =="2": # idem programmation morgan
        print "morgan"
        nFrames= len(proxMotionRef)  
    
    
        SR = np.zeros((3,3))
        Sd = np.zeros((3,1))
        SRd = np.zeros((3,1))
    
        # For each frame compute the transformation matrix of the distal
        # segment in the proximal reference system
        for i in range(0,nFrames): 
            Rprox = proxMotionRef[i].getRotation() 
            tprox = proxMotionRef[i].getTranslation()
    
            Rdist = distMotionRef[i].getRotation() 
            tdist = distMotionRef[i].getTranslation() 
           
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

    prox_origin = CoR_prox -  gap * prox_axisNorm.reshape(3,1)
    prox_axisLim = CoR_prox + gap * prox_axisNorm.reshape(3,1)

    dist_origin = CoR_dist - gap * dist_axisNorm.reshape(3,1)
    dist_axisLim = CoR_dist + gap * dist_axisNorm.reshape(3,1)


    S = diagS[3:6,3:6]
    coeffDet = S[2,2]/(np.trace(S)-S[2,2]) #TODO : explanation ? where is came up




    return prox_origin.reshape(3),prox_axisLim.reshape(3),dist_origin.reshape(3),dist_axisLim.reshape(3),prox_axisNorm,dist_axisNorm,coeffDet
   
   
   
   
   
def haraRegression(mp_input,mp_computed,markerDiameter = 14.0,  basePlate = 2.0):   
    # mp_computed not used.
   
    HJCx_L= 11.0 -0.063*mp_input["leftLegLength"] - markerDiameter/2.0 - basePlate
    HJCy_L=8.0+0.086*mp_input["leftLegLength"]
    HJCz_L=-9.0-0.078*mp_input["leftLegLength"]
        
    HJC_L_hara=np.array([HJCx_L,HJCy_L,HJCz_L])
    
    HJCx_R= 11.0 -0.063*mp_input["rightLegLength"]- markerDiameter/2.0 - basePlate
    HJCy_R=-1.0*(8.0+0.086*mp_input["rightLegLength"])
    HJCz_R=-9.0-0.078*mp_input["rightLegLength"]
        
    HJC_R_hara=np.array([HJCx_R,HJCy_R,HJCz_R])
   
    HJC_L = HJC_L_hara       
    HJC_R = HJC_R_hara
    
    return HJC_L,HJC_R



def harringtonRegression(mp_input,mp_computed, predictors, markerDiameter = 14.0, basePlate = 2.0, cgmReferential=True):
    """ Computation of the hip joint center position from Harrington's regressions.         
    
    :Parameters:
       - `mp_input` (dict) - a dictionnary of input anthropometric parameters           
       - `mp_computed` (dict) - a dictionnary of anthropometric parameters infered from input ones (meanlegLength for example)   

    .. todo :: not really clear : input and computed parameters

    """ 

    if predictors.value == "full":
        HJCx_L=-0.24*mp_computed["pelvisDepth"]-9.9  - markerDiameter/2.0 - basePlate # post/ant
        HJCy_L=-0.16*mp_computed["asisDistance"]-0.04*mp_computed["meanlegLength"]-7.1 
        HJCz_L=-1*(0.28*mp_computed["pelvisDepth"]+0.16*mp_computed["asisDistance"]+7.9)
        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])

        HJCx_R=-0.24*mp_computed["pelvisDepth"]-9.9 - markerDiameter/2.0 - basePlate# post/ant
        HJCy_R=-0.16*mp_computed["asisDistance"]-0.04*mp_computed["meanlegLength"]-7.1 
        HJCz_R=1*(0.28*mp_computed["pelvisDepth"]+0.16*mp_computed["asisDistance"]+7.9) 
        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R])    

    elif predictors.value=="PWonly":
        HJCx_L=-0.138*mp_computed["asisDistance"]-10.4 - markerDiameter/2.0 - basePlate
        HJCy_L=-0.305*mp_computed["asisDistance"]-10.9
        HJCz_L=-1*(0.33*mp_computed["asisDistance"]+7.3)
        
        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])
    
        HJCx_R=-0.138*mp_computed["asisDistance"]-10.4 - markerDiameter/2.0 - basePlate
        HJCy_R=-0.305*mp_computed["asisDistance"]-10.9
        HJCz_R=1*(0.33*mp_computed["asisDistance"]+7.3)
        
        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R]) 
    

    elif predictors.value=="LLonly":
        HJCx_L=-0.041*mp_computed["meanlegLength"]-6.3 - markerDiameter/2.0 - basePlate
        HJCy_L=-0.083*mp_computed["meanlegLength"]-7.9
        HJCz_L=-1*(0.0874*mp_computed["meanlegLength"]+5.4)
        
        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])
    
        HJCx_R=-0.041*mp_computed["meanlegLength"]-6.3 - markerDiameter/2.0 - basePlate
        HJCy_R=-0.083*mp_computed["meanlegLength"]-7.9
        HJCz_R=1*(0.0874*mp_computed["meanlegLength"]+5.4)
        
        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R])
        
    else:
        raise Exception("[pycga] Predictor is unknown choixe possible : full, PWonly, LLonly")

    if cgmReferential :
        Rhar_cgm1=np.array([[1, 0, 0],[0, 0, -1], [0, 1, 0]])
        HJC_L = np.dot(Rhar_cgm1,HJC_L_har)       
        HJC_R = np.dot(Rhar_cgm1,HJC_R_har)
    else:
        HJC_L = HJC_L_har       
        HJC_R = HJC_R_har
    
    return HJC_L,HJC_R


# -------- ABSTRACT DECORATOR MODEL ---------

class DecoratorModel(model.Model):
    """
    .. note ::     model decorator Interface   
    """

    def __init__(self, iModel):
        super(DecoratorModel,self).__init__()
        self.model = iModel

#-------- CONCRETE DECORATOR MODEL ---------
class Kad(DecoratorModel):
    """
    .. Note : a concrete decorator altering the hip joint centers with regression equations    
    """ 
    def __init__(self, iModel,iAcq):
        """Constructor

        :Parameters:
           - `iModel` (Model) - model instance   
           - `iAcq` (btkAcquisition) - btk aquisition   

        .. note:: if i want add a new regression, just add it as a method here       
        """
        super(Kad,self).__init__(iModel)
        self.acq = iAcq
        
    def compute(self,side="both",displayMarkers = False):    
        """
        1- calcul position de KNE
        2- ajout offset. 
        3- creer un node KJC_kad et AJC_kad
         
        """
        distSkin = 0 #%17-2-7; dist origin to plate minus markers plate thickness and half marker diameter        
        markerDiameter = 14        
        
        ff = self.acq.GetFirstFrame() 

        frameInit =  self.acq.GetFirstFrame()-ff  
        frameEnd = self.acq.GetLastFrame()-ff+1
                
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True

        if side == "both" or side == "left":
            
            #  compute points left and right lateral condyle
            LKAX = self.acq.GetPoint("LKAX").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            LKD1 = self.acq.GetPoint("LKD1").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            LKD2 = self.acq.GetPoint("LKD2").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                
            dist = np.array([np.linalg.norm(LKAX-LKD1), np.linalg.norm(LKAX-LKD2),np.linalg.norm(LKD1-LKD2)] )
            dist =  dist / np.sqrt(2)
            res = np.array([np.mean(dist), np.var(dist)])
            n = np.cross(LKD2-LKD1 , LKAX-LKD1)
            n= n/np.linalg.norm(n)
            
            I = (LKD1+LKAX)/2
            PP1 = 2/3.0*(I-LKD2)+LKD2
            O = PP1 - n*np.sqrt(3)*res[0]/3.0   
            LKAXO = (O-LKAX)/np.linalg.norm(O-LKAX)

            LKNE = O + LKAXO * distSkin
        
            # locate KJC
#            LKJC = LKNE + LKAXO * (self.model.mp["leftKneeWidth"]+markerDiameter )/2.0
            if btkTools.isPointExist(self.acq,"LHJC"):
                LHJC = self.acq.GetPoint("LHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                LKJC = cgm.CGM1ModelInf.chord( (self.model.mp["leftKneeWidth"]+markerDiameter )/2.0 ,LKNE,LHJC,LKAX, beta= 0.0 )
            else:
                LKJC = LKNE + LKAXO * (self.model.mp["leftKneeWidth"]+markerDiameter )/2.0
            
            
            
            # locate AJC
            LANK = self.acq.GetPoint("LANK").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            LAJC = cgm.CGM1ModelInf.chord( (self.model.mp["leftAnkleWidth"]+markerDiameter )/2.0 ,LANK,LKJC,LKAX,beta= 0.0 )
            
            
            # add nodes to referential 
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKNE_kad",LKNE,positionType="Global")
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKJC_kad",LKJC,positionType="Global")  
            
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKNE_kad",LKNE,positionType="Global")
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKJC_kad",LKJC,positionType="Global") 
            self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LAJC_kad",LAJC,positionType="Global")






            
        if side == "both" or side == "right":

            #  compute points left and right lateral condyle                    
            RKAX = self.acq.GetPoint("RKAX").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            RKD1 = self.acq.GetPoint("RKD1").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            RKD2 = self.acq.GetPoint("RKD2").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                
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
            RKNE = O + RKAXO * distSkin
        
            # locate KJC
#            RKJC = RKNE + RKAXO * (self.model.mp["rightKneeWidth"]+markerDiameter )/2.0
            if btkTools.isPointExist(self.acq,"RHJC"):
                RHJC = self.acq.GetPoint("RHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                RKJC = cgm.CGM1ModelInf.chord( (self.model.mp["rightKneeWidth"]+markerDiameter )/2.0 ,RKNE,RHJC,RKAX,beta= 0.0 )
            else:
                RKJC = RKNE + RKAXO * (self.model.mp["rightKneeWidth"]+markerDiameter )/2.0
                
            # locate AJC
            RANK = self.acq.GetPoint("RANK").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            RAJC = cgm.CGM1ModelInf.chord( (self.model.mp["rightAnkleWidth"]+markerDiameter )/2.0 ,RANK,RKJC,RKAX,beta= 0.0 )
            
            # add nodes to referential
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKNE_kad",RKNE,positionType="Global")
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKJC_kad",RKJC,positionType="Global")  
            
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKNE_kad",RKNE,positionType="Global")
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKJC_kad",RKJC,positionType="Global")
            self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RAJC_kad",RAJC,positionType="Global")
    
    

        if displayMarkers:
            if side == "both" or side == "right":
                val = RKNE * np.ones((self.acq.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(self.acq,"RKNE",val, desc="KAD") # KNE updated. 

                val = RKJC * np.ones((self.acq.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(self.acq,"RKJC-KAD",val, desc="KAD")
                
                val = RAJC * np.ones((self.acq.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(self.acq,"RAJC-KAD",val, desc="KAD")

            if side == "both" or side == "left":
                val = LKNE * np.ones((self.acq.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(self.acq,"LKNE",val, desc="KAD") 

                
                val = LKJC * np.ones((self.acq.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(self.acq,"LKJC-KAD",val, desc="KAD")
                val = LAJC * np.ones((self.acq.GetPointFrameNumber(),3))      
                btkTools.smartAppendPoint(self.acq,"LAJC-KAD",val, desc="KAD")

#            writer = btk.btkAcquisitionFileWriter() 
#            writer.SetInput(self.acq)
#            writer.SetFilename("C:\\Users\\AAA34169\\Documents\\Programming\\API\\pyCGA-DATA\\tmp-static-KAD.c3d")
#            writer.Update()

            


class HipJointCenterDecorator(DecoratorModel):
    """
    .. Note : a concrete decorator altering the hip joint centers with regression equations    
    """ 
    def __init__(self, iModel):
        """Constructor

        :Parameters:
           - `iModel` (Model ) - model instance   

        .. note:: if i want add a new regression, just add it as a method here       
        """
        super(HipJointCenterDecorator,self).__init__(iModel)
        
    def custom(self,position_Left=0,position_Right=0,methodDesc="custom"):
        """ add hip joint centres from golden device          
        
        :Parameters:
           - `position_Left` (np.array, 3) - position of the left hip center in the Pelvis Referential           
           - `position_Right` (np.array, 3) - position of the right hip center in the Pelvis Referential
           - `methodDesc` (str) - short description of the method 
           
        .. warning :: look out the Pevis Referential. it has to be similar with cgm1. 
    
        """
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True

        if position_Left.shape ==(3,):
            nodeLabel= "LHJC_"+ methodDesc
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode(nodeLabel,position_Left, positionType="Local")
        if position_Left.shape ==(3,):
            nodeLabel= "RHJC_"+ methodDesc
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode(nodeLabel,position_Right, positionType="Local")

        
        

        
    def harrington(self,predictors= pyCGM2Enums.HarringtonPredictor.Native, side="both"):    
        """ Use of the Harrington's regressions function        
        
        :Parameters:
           - `side` (str) - compute either *left* or *right* or *both* hip joint centres           
        
        """ 
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True

        LHJC_har,RHJC_har=harringtonRegression(self.model.mp,self.model.mp_computed,predictors)

        
        if side == "both":
            
            # add nodes to pelvis            
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("LHJC_har",LHJC_har, positionType="Local")
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("RHJC_har",RHJC_har, positionType="Local")

            # add nodes Thigh
            pos_L=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_har").m_global
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LHJC_har",pos_L, positionType="Global")

            pos_R=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_har").m_global
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RHJC_har",pos_R, positionType="Global")


        elif side == "left":
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("LHJC_har",LHJC_har, positionType="Local")
            pos_L=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_har").m_global
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LHJC_har",pos_L, positionType="Global")


        elif side == "right":
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("RHJC_har",RHJC_har, positionType="Local")
            pos_R=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_har").m_global
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RHJC_har",pos_R, positionType="Global")

    def hara(self, side="both"):    
        """ Use of the Hara's regressions function        
        
        :Parameters:
           - `side` (str) - compute either *left* or *right* or *both* hip joint centres           
        
        """ 
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True

        LHJC_hara,RHJC_hara=haraRegression(self.model.mp,self.model.mp_computed)

        
        if side == "both":
            # add nodes to pelvis            
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("LHJC_hara",LHJC_hara, positionType="Local")
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("RHJC_hara",RHJC_hara, positionType="Local")
            
            # add nodes Thigh
            pos_L=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_hara").m_global
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LHJC_hara",pos_L, positionType="Global")

            pos_R=self.model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_hara").m_global
            self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RHJC_hara",pos_R, positionType="Global")
    
    
    

class KneeCalibrationDecorator(DecoratorModel):
    """
    .. Note : a concrete decorator altering the knee joint axis from a functionnal method    
    """ 
    def __init__(self, iModel):
        """Constructor

        :Parameters:
           - `iModel` (Model ) - model instance   
       
        """
        super(KneeCalibrationDecorator,self).__init__(iModel)
        
    def midCondyles(self,acq, side="both",leftLateralKneeLabel="LKNE", leftMedialKneeLabel="LMEPI",rightLateralKneeLabel="RKNE", rightMedialKneeLabel="RMEPI", withNoModelParameter=False):   
        """ compute KJC as the mid-condyles point          
        
        :Parameters:
           - `acq` (btkAcquisition) - static acquisition           
           - `side` (str) - both, left or right
           - `leftLateralKneeLabel` (str) - point label of the left lateral knee marker
           - `leftMedialKneeLabel` (str) - point label of the left medial knee marker           
           
           
        .. todo :: coding exception if label doesn t find. 

        """
        markerDiameter = 14 # Todo : mettre a l exterieur 
        
        ff = acq.GetFirstFrame()     
    
        frameInit =  acq.GetFirstFrame()-ff  
        frameEnd = acq.GetLastFrame()-ff+1
                
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True
        
        if side=="both" or side=="left":

            mid_L = (acq.GetPoint(leftLateralKneeLabel).GetValues()[frameInit:frameEnd,:] + 
                     acq.GetPoint(leftMedialKneeLabel).GetValues()[frameInit:frameEnd,:])/2.0
            mid_L = mid_L.mean(axis =0)


            LKNE = acq.GetPoint(leftLateralKneeLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            LMEPI = acq.GetPoint(leftMedialKneeLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)  
           
            v = LMEPI-LKNE
            v=v/np.linalg.norm(v)
           
            LKJC = LKNE + ((self.model.mp["leftKneeWidth"]+markerDiameter )/2.0)*v
           
            if withNoModelParameter:
                self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKJC_mid",mid_L, positionType="Global")
                self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKJC_mid",mid_L, positionType="Global")
            else:
                self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKJC_mid",LKJC, positionType="Global")
                self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LKJC_mid",LKJC, positionType="Global")
            
        if side=="both" or side=="right":

            mid_R = (acq.GetPoint(rightLateralKneeLabel).GetValues()[frameInit:frameEnd,:] + 
                     acq.GetPoint(rightMedialKneeLabel).GetValues()[frameInit:frameEnd,:])/2.0
                     
            mid_R = mid_R.mean(axis =0)


            RKNE = acq.GetPoint(rightLateralKneeLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            RMEPI = acq.GetPoint(rightMedialKneeLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)  
           
            v = RMEPI-RKNE
            v=v/np.linalg.norm(v)
           
            RKJC = RKNE + ((self.model.mp["rightKneeWidth"]+markerDiameter )/2.0)*v

            if withNoModelParameter:
                self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKJC_mid",mid_R, positionType="Global")
                self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKJC_mid",mid_R, positionType="Global")
            else:
                self.model.getSegment("Right Thigh").getReferential("TF").static.addNode("RKJC_mid",RKJC, positionType="Global")
                self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RKJC_mid",RKJC, positionType="Global")             
                   
        
        
        
        

    
    def sara(self,side):    
        """
        """ 
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True

        proxMotion = self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").motion
        distMotion = self.model.getSegment("Left Shank").getReferential("TF_anaCalib").motion
        
        # -- main function -----
        prox_ori,prox_axisLim,dist_ori,dist_axisLim,axis_prox,axis_dist,quality = saraCalibration(proxMotion,distMotion,method="2")
        # end function -----


        # add nodes        
        self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.addNode("KneeFlexionOri",prox_ori,positionType="Local")
        self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.addNode("KneeFlexionAxis",prox_axisLim,positionType="Local")
    
        self.model.getSegment("Left Shank").getReferential("TF_anaCalib").static.addNode("KneeFlexionOri",dist_ori,positionType="Local")
        self.model.getSegment("Left Shank").getReferential("TF_anaCalib").static.addNode("KneeFlexionAxis",dist_axisLim,positionType="Local")     
   
        # compute error
        xp = self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").getNodeTrajectory("KneeFlexionOri")
        xd = self.model.getSegment("Left Shank").getReferential("TF_anaCalib").getNodeTrajectory("KneeFlexionOri")

        

        ferr = xp-xd
        Merr = numeric.rms(ferr)
        print " sara rms error : %s " % str(Merr)

        # --- registration of the Knee center ---
        
        # longitudinal axis of the femur 
        p1 = self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.getNode_byLabel("LHJC").m_global
        p2 = self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.getNode_byLabel("LKJC").m_global

        # middle of origin in Global
        p_proxOri = self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.getNode_byLabel("KneeFlexionOri").m_global
        p_distOri = self.model.getSegment("Left Shank").getReferential("TF_anaCalib").static.getNode_byLabel("KneeFlexionOri").m_global        
        center = np.mean((p_distOri,p_proxOri),axis= 0)
        
        # axis lim
        p_proxAxis = self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.getNode_byLabel("KneeFlexionAxis").m_global
        p_distAxis = self.model.getSegment("Left Shank").getReferential("TF_anaCalib").static.getNode_byLabel("KneeFlexionAxis").m_global

        # intersection beetween midcenter-axis and logitudinal axis
        proxIntersect,pb1 = geometry.LineLineIntersect(center,p_proxAxis,p1,p2)
        distIntersect,pb2 = geometry.LineLineIntersect(center,p_distAxis,p1,p2)# 


        # shortest distance                
        shortestDistance_prox =  np.linalg.norm(proxIntersect-pb1)
        print " 3d line intersect : shortest distance beetween logidudinal axis and flexion axis sur Proximal  : %s  " % str(shortestDistance_prox)
        shortestDistance_dist =  np.linalg.norm(distIntersect-pb2)
        print " 3d line intersect : shortest distance beetween logidudinal axis and flexion axis sur Distal  : %s  " % str(shortestDistance_dist)

        # mean of the intersection point        
        center = np.mean((proxIntersect,distIntersect), axis=0)

        # Node manager
        #  the node LKJC_sara is added in all "Referentials. 
        self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.addNode("LKJC_sara",center, positionType="Global")
        self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LKJC_sara",center, positionType="Global")        
       
        print " in  TF-------" 
        print " LKJC "
        print self.model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_local
        print " LKJC sara "
        print self.model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC_sara").m_local

        print " in  TF_anaCalib-------" 
        print " LKJC "
        print self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.getNode_byLabel("LKJC").m_local
        print " LKJC sara "
        print self.model.getSegment("Left Thigh").getReferential("TF_anaCalib").static.getNode_byLabel("LKJC_sara").m_local
        



class AnkleCalibrationDecorator(DecoratorModel):
    """
    .. Note : a concrete decorator altering the ankle joint axis from a functionnal method    
    """ 
    def __init__(self, iModel):
        """Constructor

        :Parameters:
           - `iModel` (Model ) - model instance   
       
        """
        super(AnkleCalibrationDecorator,self).__init__(iModel)
        
    def midMaleolus(self,acq, side="both",
                    leftLateralAnkleLabel="LANK", leftMedialAnkleLabel="LMED",
                    rightLateralAnkleLabel="RANK", rightMedialAnkleLabel="RMED", withNoModelParameter=False):   
        """ compute AJC as the mid-condyles point          
        
        :Parameters:
           - `acq` (btkAcquisition) - static acquisition           
           - `side` (str) - both, left or right
           - `leftLateralAnkleLabel` (str) - point label of the left lateral knee marker
           - `leftMedialAnkleLabel` (str) - point label of the left medial knee marker           
           
           
        .. todo :: coding exception if label doesn t find. 

        """    
        ff = acq.GetFirstFrame()     
    
        frameInit =  acq.GetFirstFrame()-ff  
        frameEnd = acq.GetLastFrame()-ff+1
                
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True
        
        markerDiameter = 14
        
        if side=="both" or side=="left":
            
            mid_L = (acq.GetPoint(leftLateralAnkleLabel).GetValues()[frameInit:frameEnd,:] + 
                     acq.GetPoint(leftMedialAnkleLabel).GetValues()[frameInit:frameEnd,:])/2.0
            mid_L = mid_L.mean(axis =0)            
            
            LANK = acq.GetPoint(leftLateralAnkleLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            LMED = acq.GetPoint(leftMedialAnkleLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)  
           
            v = LMED-LANK
            v=v/np.linalg.norm(v)
           
            LAJC = LANK + ((self.model.mp["leftAnkleWidth"]+markerDiameter )/2.0)*v
            
            if withNoModelParameter:
                self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LAJC_mid",mid_L, positionType="Global")
            else:
                self.model.getSegment("Left Shank").getReferential("TF").static.addNode("LAJC_mid",LAJC, positionType="Global")
            # add line below when foot referential will be built
            #self.model.getSegment("Left Foot").getReferential("TF").static.addNode("LAJC_mid",mid_L.mean(axis=0), positionType="Global")

        if side=="both" or side=="right":
            
            mid_R = (acq.GetPoint(rightLateralAnkleLabel).GetValues()[frameInit:frameEnd,:] + 
                     acq.GetPoint(rightMedialAnkleLabel).GetValues()[frameInit:frameEnd,:])/2.0
            mid_R = mid_R.mean(axis =0) 
            
            RANK = acq.GetPoint(rightLateralAnkleLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            RMED = acq.GetPoint(rightMedialAnkleLabel).GetValues()[frameInit:frameEnd,:].mean(axis=0)  
           
            v = RMED-RANK
            v=v/np.linalg.norm(v)
           
            RAJC =  RANK + ((self.model.mp["rightAnkleWidth"]+markerDiameter )/2.0)*v

            if withNoModelParameter:
                self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RAJC_mid",mid_R, positionType="Global")
                # TODO : foot
            else:
                self.model.getSegment("Right Shank").getReferential("TF").static.addNode("RAJC_mid",RAJC, positionType="Global")             

    def midMaleolusAxis(self,acq, side="both",
                    leftLateralAnkleLabel="LANK", leftMedialAnkleLabel="LMED",
                    rightLateralAnkleLabel="RANK", rightMedialAnkleLabel="RMED", withNoModelParameter=False):   
        
        """ compute AJC as the mid-condyles point          
        
        :Parameters:
           - `acq` (btkAcquisition) - static acquisition           
           - `side` (str) - both, left or right
           - `leftLateralAnkleLabel` (str) - point label of the left lateral knee marker
           - `leftMedialAnkleLabel` (str) - point label of the left medial knee marker           
           
           
        .. todo :: coding exception if label doesn t find. 

        """    
        ff = acq.GetFirstFrame()     
    
        frameInit =  acq.GetFirstFrame()-ff  
        frameEnd = acq.GetLastFrame()-ff+1
                
        self.model.nativeCgm1 = False
        self.model.decoratedModel = True
        markerDiameter = 14
        
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
        
        

