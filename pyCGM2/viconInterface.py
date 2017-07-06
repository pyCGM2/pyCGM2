# -*- coding: utf-8 -*-
import numpy as np

def updateNexusSubjectMp(NEXUS,model,subjectName):
    th_l = 0 if np.abs(model.getViconThighOffset("Left")) < 0.000001 else model.getViconThighOffset("Left")
    sh_l = 0 if np.abs(model.getViconShankOffset("Left"))< 0.000001 else model.getViconShankOffset("Left")
    tt_l = 0 if np.abs(model.getViconTibialTorsion("Left")) < 0.000001 else model.getViconTibialTorsion("Left")

    th_r = 0 if np.abs(model.getViconThighOffset("Right")) < 0.000001 else model.getViconThighOffset("Right")
    sh_r = 0 if np.abs(model.getViconShankOffset("Right")) < 0.000001 else model.getViconShankOffset("Right")
    tt_r = 0 if np.abs(model.getViconTibialTorsion("Right")) < 0.000001 else model.getViconTibialTorsion("Right")

    spf_l,sro_l = model.getViconFootOffset("Left")
    spf_r,sro_r = model.getViconFootOffset("Right")

    abdAdd_l = 0 if np.abs(model.getViconAnkleAbAddOffset("Left")) < 0.000001 else model.getViconAnkleAbAddOffset("Left") 
    abdAdd_r = 0 if np.abs(model.getViconAnkleAbAddOffset("Right")) < 0.000001 else model.getViconAnkleAbAddOffset("Right") 


    

    NEXUS.SetSubjectParam( subjectName, "InterAsisDistance",model.mp_computed["InterAsisDistance"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftAsisTrocanterDistance",model.mp_computed["LeftAsisTrocanterDistance"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftThighRotation",th_l,True)
    NEXUS.SetSubjectParam( subjectName, "LeftShankRotation",sh_l,True)
    NEXUS.SetSubjectParam( subjectName, "LeftTibialTorsion",tt_l,True)


    NEXUS.SetSubjectParam( subjectName, "RightAsisTrocanterDistance",model.mp_computed["RightAsisTrocanterDistance"],True)
    NEXUS.SetSubjectParam( subjectName, "RightThighRotation",th_r,True)
    NEXUS.SetSubjectParam( subjectName, "RightShankRotation",sh_r,True)
    NEXUS.SetSubjectParam( subjectName, "RightTibialTorsion",tt_r,True)


    NEXUS.SetSubjectParam( subjectName, "LeftStaticPlantFlex",spf_l,True)
    NEXUS.SetSubjectParam( subjectName, "LeftStaticRotOff",sro_l,True)
    NEXUS.SetSubjectParam( subjectName, "LeftAnkleAbAdd",abdAdd_l,True)

    NEXUS.SetSubjectParam( subjectName, "RightStaticPlantFlex",spf_r,True)
    NEXUS.SetSubjectParam( subjectName, "RightStaticRotOff",sro_r,True)
    NEXUS.SetSubjectParam( subjectName, "RightAnkleAbAdd",abdAdd_r,True)




class ViconInterface(object):
    def __init__(self,NEXUS,iModel, iAcq, vskName,pointSuffix, staticProcessing = False ):
        """
            Constructor
           
            :Parameters:
                - `NEXUS` () - Nexus environment 
                - `iModel` (pyCGM2.Model.CGM2.Model) - model instance
                - `vskName` (str) . subject name create in Nexus
                - `staticProcessingFlag` (bool`) : flag indicating only static model ouput will be export  
                
        """
        self.m_model = iModel
        self.m_acq = iAcq
        self.m_vskName = vskName
        self.NEXUS = NEXUS
        self.staticProcessing = staticProcessing
        self.m_pointSuffix = pointSuffix if pointSuffix=="" else str("_"+pointSuffix)

    def run(self):
        """
            method calling embedded-model method : viconExport  
        """
        self.m_model.viconExport(self.NEXUS,self.m_acq, self.m_vskName,self.m_pointSuffix,self.staticProcessing)
        
        
