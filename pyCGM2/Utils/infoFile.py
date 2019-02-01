# -*- coding: utf-8 -*-

def getFromInfoSubjectMp(infoSettings,resetFlag=True):

    required_mp={
    "Bodymass"   : infoSettings["MP"]["Required"]["Bodymass"],
    "LeftLegLength" :infoSettings["MP"]["Required"]["LeftLegLength"],
    "RightLegLength" : infoSettings["MP"]["Required"][ "RightLegLength"],
    "LeftKneeWidth" : infoSettings["MP"]["Required"][ "LeftKneeWidth"],
    "RightKneeWidth" : infoSettings["MP"]["Required"][ "RightKneeWidth"],
    "LeftAnkleWidth" : infoSettings["MP"]["Required"][ "LeftAnkleWidth"],
    "RightAnkleWidth" : infoSettings["MP"]["Required"][ "RightAnkleWidth"],
    "LeftSoleDelta" : infoSettings["MP"]["Required"][ "LeftSoleDelta"],
    "RightSoleDelta" : infoSettings["MP"]["Required"]["RightSoleDelta"],
    "LeftShoulderOffset" : mpInfo["MP"]["Required"]["LeftShoulderOffset"],
    "RightShoulderOffset" : mpInfo["MP"]["Required"]["RightShoulderOffset"],
    "LeftElbowWidth" : mpInfo["MP"]["Required"]["LeftElbowWidth"],
    "LeftWristWidth" : mpInfo["MP"]["Required"]["LeftWristWidth"],
    "LeftHandThickness" : mpInfo["MP"]["Required"]["LeftHandThickness"],
    "RightElbowWidth" : mpInfo["MP"]["Required"]["RightElbowWidth"],
    "RightWristWidth" : mpInfo["MP"]["Required"]["RightWristWidth"],
    "RightHandThickness" : mpInfo["MP"]["Required"]["RightHandThickness"]
    }

    if resetFlag:
        optional_mp={
        "InterAsisDistance"   : 0,
        "LeftAsisTrocanterDistance" : 0,
        "LeftTibialTorsion" : 0 ,
        "LeftThighRotation" : 0,
        "LeftShankRotation" : 0,
        "RightAsisTrocanterDistance" : 0,
        "RightTibialTorsion" :0 ,
        "RightThighRotation" : 0,
        "RightShankRotation" : 0
        }
    else:
        optional_mp={
        "InterAsisDistance"   : infoSettings["MP"]["Optional"][ "InterAsisDistance"],#0,
        "LeftAsisTrocanterDistance" : infoSettings["MP"]["Optional"][ "LeftAsisTrocanterDistance"],#0,
        "LeftTibialTorsion" : infoSettings["MP"]["Optional"][ "LeftTibialTorsion"],#0 ,
        "LeftThighRotation" : infoSettings["MP"]["Optional"][ "LeftThighRotation"],#0,
        "LeftShankRotation" : infoSettings["MP"]["Optional"][ "LeftShankRotation"],#0,
        "RightAsisTrocanterDistance" : infoSettings["MP"]["Optional"][ "RightAsisTrocanterDistance"],#0,
        "RightTibialTorsion" : infoSettings["MP"]["Optional"][ "RightTibialTorsion"],#0 ,
        "RightThighRotation" : infoSettings["MP"]["Optional"][ "RightThighRotation"],#0,
        "RightShankRotation" : infoSettings["MP"]["Optional"][ "RightShankRotation"],#0,
        }

    return required_mp,optional_mp
