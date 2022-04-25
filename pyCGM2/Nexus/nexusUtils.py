# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Nexus
#APIDOC["Draft"]=False
#--end--


def getNexusSubjectMp(NEXUS,subject, resetFlag=False):
    """ return required and optional anthropometric parameters

    Args:
        NEXUS (): viconnexusapi handle.
        subject (str): subject ( eq. vsk) name.
        resetFlag (bool,Optional[False]):  reset optional mp.

    """

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

    optional_mp={
    'InterAsisDistance'   : NEXUS.GetSubjectParamDetails( subject, "InterAsisDistance")[0] if "InterAsisDistance" in params else 0,#0,
    'LeftAsisTrocanterDistance' : NEXUS.GetSubjectParamDetails( subject, "LeftAsisTrocanterDistance")[0] if "LeftAsisTrocanterDistance" in params else 0,#0,
    'RightAsisTrocanterDistance' : NEXUS.GetSubjectParamDetails( subject, "RightAsisTrocanterDistance")[0] if "RightAsisTrocanterDistance" in params else 0,#0,
    }


    if resetFlag:
        optional_mp.update({
        'LeftTibialTorsion' : 0 ,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0,
        'RightTibialTorsion' :0 ,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0
        })

    else:
        optional_mp.update({
        'LeftTibialTorsion' : NEXUS.GetSubjectParamDetails( subject, "LeftTibialTorsion")[0] if "LeftTibialTorsion" in params else 0,#0 ,
        'LeftThighRotation' : NEXUS.GetSubjectParamDetails( subject, "LeftThighRotation")[0] if "LeftThighRotation" in params else 0,#0,
        'LeftShankRotation' : NEXUS.GetSubjectParamDetails( subject, "LeftShankRotation")[0] if "LeftShankRotation" in params else 0,#0,
        'RightTibialTorsion' : NEXUS.GetSubjectParamDetails( subject, "RightTibialTorsion")[0] if "RightTibialTorsion" in params else 0,#0 ,
        'RightThighRotation' : NEXUS.GetSubjectParamDetails( subject, "RightThighRotation")[0] if "RightThighRotation" in params else 0,#0,
        'RightShankRotation' : NEXUS.GetSubjectParamDetails( subject, "RightShankRotation")[0] if "RightShankRotation" in params else 0,#0,
        })
    return required_mp,optional_mp

def updateNexusSubjectMp(NEXUS,model,subjectName):
    """ update anthropometric from  a pyCGM2.Model instance

    Args:
        NEXUS (): vicon nexus api handle.
        model (pyCGM2.Model.model.Model): a model instance.
        subjectName (str):  subject (ie vsk) name

    """

    NEXUS.SetSubjectParam( subjectName, "InterAsisDistance",model.mp_computed["InterAsisDistance"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftAsisTrocanterDistance",model.mp_computed["LeftAsisTrocanterDistance"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftThighRotation",model.mp_computed["LeftThighRotationOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftShankRotation",model.mp_computed["LeftShankRotationOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftTibialTorsion",model.mp_computed["LeftTibialTorsionOffset"],True)


    NEXUS.SetSubjectParam( subjectName, "RightAsisTrocanterDistance",model.mp_computed["RightAsisTrocanterDistance"],True)
    NEXUS.SetSubjectParam( subjectName, "RightThighRotation",model.mp_computed["RightThighRotationOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "RightShankRotation",model.mp_computed["RightShankRotationOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "RightTibialTorsion",model.mp_computed["RightTibialTorsionOffset"],True)


    NEXUS.SetSubjectParam( subjectName, "LeftStaticPlantFlex",model.mp_computed["LeftStaticPlantFlexOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftStaticRotOff",model.mp_computed["LeftStaticRotOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "LeftAnkleAbAdd",model.mp_computed["LeftAnkleAbAddOffset"],True)

    NEXUS.SetSubjectParam( subjectName, "RightStaticPlantFlex",model.mp_computed["RightStaticPlantFlexOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "RightStaticRotOff",model.mp_computed["RightStaticRotOffset"],True)
    NEXUS.SetSubjectParam( subjectName, "RightAnkleAbAdd",model.mp_computed["RightAnkleAbAddOffset"],True)
