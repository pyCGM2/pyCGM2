# -*- coding: utf-8 -*-
from pyCGM2.Utils.utils import *


def getFilename(measurement):
    return measurement.attrs["Filename"].replace("qtm","c3d")


def getForcePlateAssigment(measurement):
    mea = measurement

    mfpa = mea.find("ForcePlate1").text[0] + mea.find("ForcePlate2").text[0]+ \
    mea.find("ForcePlate3").text[0] + mea.find("ForcePlate4").text[0] + \
    mea.find("ForcePlate5").text[0]

    return mfpa


def isType(measurement,type):
    return measurement.attrs["Type"] == type



def findStatic(soup):
    qtmMeasurements = soup.find_all("Measurement")
    static=list()
    for measurement in qtmMeasurements:
        if measurement.attrs["Type"] == "Static" and toBool(measurement.Used.text):
            static.append(measurement)
        if len(static)>1:
            raise Exception("You can t have 2 activated static c3d within your session")
    return measurement #str(static[0].attrs["Filename"][:-4]+".c3d")

def findDynamic(soup):
    qtmMeasurements = soup.find_all("Measurement")

    measurements=list()
    for measurement in qtmMeasurements:
        if measurement.attrs["Type"] != "Static" and toBool(measurement.Used.text):
            measurements.append(measurement)

    return measurements


def detectMeasurementType(soup):
    measurements = soup.find_all("Measurement")

    types = list()
    for measurement in measurements:
        if measurement.attrs["Type"] != "Static" and toBool(measurement.Used.text):
            if measurement.attrs["Type"] not in types:
                types.append(measurement.attrs["Type"])

    return types



def SubjectMp(soup):

    required_mp={
    'Bodymass'   : float(soup.Subject.Bodymass.text),
    'LeftLegLength' : float(soup.Subject.LeftLegLength.text)*1000.0,
    'RightLegLength' : float(soup.Subject.RightLegLength.text)*1000.0,#865.0 ,
    'LeftKneeWidth' : float(soup.Subject.LeftKneeWidth.text)*1000.0,#102.0,
    'RightKneeWidth' : float(soup.Subject.RightKneeWidth.text)*1000.0,#103.4,
    'LeftAnkleWidth' : float(soup.Subject.LeftAnkleWidth.text)*1000.0,#75.3,
    'RightAnkleWidth' : float(soup.Subject.RightAnkleWidth.text)*1000.0,#72.9,
    'LeftSoleDelta' : float(soup.Subject.LeftSoleDelta.text)*1000.0,#75.3,
    'RightSoleDelta' : float(soup.Subject.RightSoleDelta.text)*1000.0,#72.9,
    'LeftShoulderOffset' : float(soup.Subject.LeftShoulderOffset.text)*1000.0,#72.9,
    'RightShoulderOffset' : float(soup.Subject.RightShoulderOffset.text)*1000.0,#72.9,
    'LeftElbowWidth' : float(soup.Subject.LeftElbowWidth.text)*1000.0,#72.9,
    'LeftWristWidth' : float(soup.Subject.LeftWristWidth.text)*1000.0,#72.9,
    'LeftHandThickness' : float(soup.Subject.LeftHandThickness.text)*1000.0,#72.9,
    'RightElbowWidth' : float(soup.Subject.RightElbowWidth.text)*1000.0,#72.9,
    'RightWristWidth' : float(soup.Subject.RightWristWidth.text)*1000.0,#72.9,
    'RightHandThickness' : float(soup.Subject.RightHandThickness.text)*1000.0#72.9,
    }

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

    return required_mp,optional_mp
