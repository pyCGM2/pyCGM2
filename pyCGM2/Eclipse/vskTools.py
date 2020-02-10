# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from bs4 import BeautifulSoup
import string
import logging
import pyCGM2
from pyCGM2.Utils import files


def getVskFiles(path):

    path = path[:-1] if path[-1:]=="\\" else path
    vskFile = files.getFiles(path+"\\",".vsk")
    if len(vskFile)>1:
        logging.warning("Folder with several vsk. %s selected"%(vskFile[0]))

    return vskFile[0].encode(pyCGM2.ENCODER)


def checkSetReadOnly(vskfilename):
    file0 = open(vskfilename,'r')
    content = file0.read()

    flag=True  if content.find('READONLY="true"') !=-1 else False
    print flag

    file0.close()

    if flag:
        logging.warning("read Only found")
        content2 = string.replace(content, 'READONLY="true"', 'READONLY="false"')

        with open(vskfilename, "w") as text_file:
            text_file.write(content2)



class Vsk(object):
    """

    """

    def __init__(self,file):

        self.m_file=file

        infile = open(file,"r")
        contents = infile.read()
        soup = BeautifulSoup(contents,'xml')

        self.m_soup = soup



    def getStaticParameterValue(self, label):

        staticParameters = self.m_soup.find_all('StaticParameter')
        for sp in staticParameters:
            if sp.attrs["NAME"] == label:
                try:
                    val = sp.attrs["VALUE"]
                except KeyError:
                    logging.warning("static parameter (%s) has no value. Zero return"%(label))
                    val=0
                return val


def getFromVskSubjectMp(vskInstance, resetFlag=False):

    required_mp={
    'Bodymass'   : float(vskInstance.getStaticParameterValue("Bodymass")) if vskInstance.getStaticParameterValue("Bodymass") is not None else 0,
    'Height'   : float(vskInstance.getStaticParameterValue("Height")) if vskInstance.getStaticParameterValue("Height") is not None else 0,
    'LeftLegLength' :float(vskInstance.getStaticParameterValue("LeftLegLength")) if vskInstance.getStaticParameterValue("LeftLegLength") is not None else 0,
    'RightLegLength' : float(vskInstance.getStaticParameterValue("RightLegLength")) if vskInstance.getStaticParameterValue("RightLegLength") is not None else 0,
    'LeftKneeWidth' : float(vskInstance.getStaticParameterValue("LeftKneeWidth")) if vskInstance.getStaticParameterValue("LeftKneeWidth") is not None else 0,
    'RightKneeWidth' : float(vskInstance.getStaticParameterValue("RightKneeWidth")) if vskInstance.getStaticParameterValue("RightKneeWidth") is not None else 0,
    'LeftAnkleWidth' : float(vskInstance.getStaticParameterValue("LeftAnkleWidth")) if vskInstance.getStaticParameterValue("LeftAnkleWidth") is not None else 0,
    'RightAnkleWidth' : float(vskInstance.getStaticParameterValue("RightAnkleWidth")) if vskInstance.getStaticParameterValue("RightAnkleWidth") is not None else 0,
    'LeftSoleDelta' : float(vskInstance.getStaticParameterValue("LeftSoleDelta")) if vskInstance.getStaticParameterValue("LeftSoleDelta") is not None else 0,
    'RightSoleDelta' : float(vskInstance.getStaticParameterValue("RightSoleDelta")) if vskInstance.getStaticParameterValue("RightSoleDelta") is not None else 0,
    'LeftShoulderOffset' : float(vskInstance.getStaticParameterValue("LeftShoulderOffset")) if vskInstance.getStaticParameterValue("LeftShoulderOffset") is not None else 0,
    'RightShoulderOffset' : float(vskInstance.getStaticParameterValue("RightShoulderOffset")) if vskInstance.getStaticParameterValue("RightShoulderOffset") is not None else 0,
    'LeftElbowWidth' : float(vskInstance.getStaticParameterValue("LeftElbowWidth")) if vskInstance.getStaticParameterValue("LeftElbowWidth") is not None else 0,
    'LeftWristWidth' : float(vskInstance.getStaticParameterValue("LeftWristWidth")) if vskInstance.getStaticParameterValue("LeftWristWidth") is not None else 0,
    'LeftHandThickness' : float(vskInstance.getStaticParameterValue("LeftHandThickness")) if vskInstance.getStaticParameterValue("LeftHandThickness") is not None else 0,
    'RightElbowWidth' : float(vskInstance.getStaticParameterValue("RightElbowWidth")) if vskInstance.getStaticParameterValue("RightElbowWidth") is not None else 0,
    'RightWristWidth' : float(vskInstance.getStaticParameterValue("RightWristWidth")) if vskInstance.getStaticParameterValue("RightWristWidth") is not None else 0,
    'RightHandThickness' : float(vskInstance.getStaticParameterValue("RightHandThickness")) if vskInstance.getStaticParameterValue("RightHandThickness") is not None else 0
    }

    optional_mp={
    'InterAsisDistance'   : float(vskInstance.getStaticParameterValue("InterAsisDistance")) if vskInstance.getStaticParameterValue("InterAsisDistance") is not None else 0,#0,
    'LeftAsisTrocanterDistance' : float(vskInstance.getStaticParameterValue("LeftAsisTrocanterDistance")) if vskInstance.getStaticParameterValue("LeftAsisTrocanterDistance") is not None else 0,#0,
    'RightAsisTrocanterDistance' : float(vskInstance.getStaticParameterValue("RightAsisTrocanterDistance")) if vskInstance.getStaticParameterValue("RightAsisTrocanterDistance") is not None else 0
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
        'LeftTibialTorsion' : float(vskInstance.getStaticParameterValue("LeftTibialTorsion")),#0 ,
        'LeftThighRotation' : float(vskInstance.getStaticParameterValue("LeftThighRotation")),#0,
        'LeftShankRotation' : float(vskInstance.getStaticParameterValue("LeftShankRotation")),#0,
        'RightTibialTorsion' : float(vskInstance.getStaticParameterValue("RightTibialTorsion")),#0 ,
        'RightThighRotation' : float(vskInstance.getStaticParameterValue("RightThighRotation")),#0,
        'RightShankRotation' : float(vskInstance.getStaticParameterValue("RightShankRotation")),#0,
        })

    return required_mp,optional_mp
