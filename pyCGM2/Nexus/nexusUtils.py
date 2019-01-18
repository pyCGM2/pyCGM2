# -*- coding: utf-8 -*-
import numpy as np
from pyCGM2 import enums


def getNexusSubjectMp(NEXUS,subject, resetFlag=False):

    params = NEXUS.GetSubjectParamNames(subject)

    required_mp={
    'Bodymass'   : NEXUS.GetSubjectParamDetails( subject, "Bodymass")[0] if "Bodymass" in params else 0,#71.0,
    'LeftLegLength' : NEXUS.GetSubjectParamDetails( subject, "LeftLegLength")[0] if "LeftLegLength" in params else 0,#860.0,
    'RightLegLength' : NEXUS.GetSubjectParamDetails( subject, "RightLegLength")[0] if "RightLegLength" in params else 0,#865.0 ,
    'LeftKneeWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftKneeWidth")[0]if "LeftKneeWidth" in params else 0,#102.0,
    'RightKneeWidth' : NEXUS.GetSubjectParamDetails( subject, "RightKneeWidth")[0] if "RightKneeWidth" in params else 0,#103.4,
    'LeftAnkleWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftAnkleWidth")[0] if "LeftAnkleWidth" in params else 0,#75.3,
    'RightAnkleWidth' : NEXUS.GetSubjectParamDetails( subject, "RightAnkleWidth")[0] if "RightAnkleWidth" in params else 0,#72.9,
    'LeftSoleDelta' : NEXUS.GetSubjectParamDetails( subject, "LeftSoleDelta")[0] if "LeftSoleDelta" in params else 0,#75.3,
    'RightSoleDelta' : NEXUS.GetSubjectParamDetails( subject, "RightSoleDelta")[0] if "RightSoleDelta" in params else 0,#72.9,
    'LeftShoulderOffset' : NEXUS.GetSubjectParamDetails( subject, "LeftShoulderOffset")[0] if "LeftShoulderOffset" in params else 0,#72.9,
    'RightShoulderOffset' : NEXUS.GetSubjectParamDetails( subject, "RightShoulderOffset")[0] if "RightShoulderOffset" in params else 0,#72.9,
    'LeftElbowWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftElbowWidth")[0] if "LeftElbowWidth" in params else 0,#72.9,
    'LeftWristWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftWristWidth")[0] if "LeftWristWidth" in params else 0,#72.9,
    'LeftHandThickness' : NEXUS.GetSubjectParamDetails( subject, "LeftHandThickness")[0] if "LeftHandThickness" in params else 0,#72.9,
    'RightElbowWidth' : NEXUS.GetSubjectParamDetails( subject, "RightElbowWidth")[0] if "RightElbowWidth" in params else 0,#72.9,
    'RightWristWidth' : NEXUS.GetSubjectParamDetails( subject, "RightWristWidth")[0] if "RightWristWidth" in params else 0,#72.9,
    'RightHandThickness' : NEXUS.GetSubjectParamDetails( subject, "RightHandThickness")[0] if "RightHandThickness" in params else 0,#72.9,
    }

    if resetFlag:
        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftTibialTorsion' : 0 ,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightTibialTorsion' :0 ,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0
        }

    else:
        optional_mp={
        'InterAsisDistance'   : NEXUS.GetSubjectParamDetails( subject, "InterAsisDistance")[0] if "InterAsisDistance" in params else 0,#0,
        'LeftAsisTrocanterDistance' : NEXUS.GetSubjectParamDetails( subject, "LeftAsisTrocanterDistance")[0] if "LeftAsisTrocanterDistance" in params else 0,#0,
        'LeftTibialTorsion' : NEXUS.GetSubjectParamDetails( subject, "LeftTibialTorsion")[0] if "LeftTibialTorsion" in params else 0,#0 ,
        'LeftThighRotation' : NEXUS.GetSubjectParamDetails( subject, "LeftThighRotation")[0] if "LeftThighRotation" in params else 0,#0,
        'LeftShankRotation' : NEXUS.GetSubjectParamDetails( subject, "LeftShankRotation")[0] if "LeftShankRotation" in params else 0,#0,
        'RightAsisTrocanterDistance' : NEXUS.GetSubjectParamDetails( subject, "RightAsisTrocanterDistance")[0] if "RightAsisTrocanterDistance" in params else 0,#0,
        'RightTibialTorsion' : NEXUS.GetSubjectParamDetails( subject, "RightTibialTorsion")[0] if "RightTibialTorsion" in params else 0,#0 ,
        'RightThighRotation' : NEXUS.GetSubjectParamDetails( subject, "RightThighRotation")[0] if "RightThighRotation" in params else 0,#0,
        'RightShankRotation' : NEXUS.GetSubjectParamDetails( subject, "RightShankRotation")[0] if "RightShankRotation" in params else 0,#0,
        }
    return required_mp,optional_mp

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
