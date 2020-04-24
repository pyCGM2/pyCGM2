# -*- coding: utf-8 -*-
from pyCGM2.Utils.utils import toBool
from datetime import datetime

def getFilename(measurement):
    return measurement.attrs["Filename"].replace("qtm","c3d")


def getForcePlateAssigment(measurement):
    mea = measurement

    mfpa = mea.find("Forceplate1").text[0] + mea.find("Forceplate2").text[0]+ \
    mea.find("Forceplate3").text[0] + mea.find("Forceplate4").text[0] + \
    mea.find("Forceplate5").text[0]

    return mfpa


def isType(measurement,type):
    return measurement.attrs["Type"] == type



def findStatic(soup):
    qtmMeasurements = soup.find_all("Measurement")
    static=list()
    for measurement in qtmMeasurements:
        if measurement.attrs["Type"] == "Static - CGM2" and toBool(measurement.Used.text):
            static.append(measurement)
        if len(static)>1:
            raise Exception("You can t have 2 activated static c3d within your session")
    return measurement #str(static[0].attrs["Filename"][:-4]+".c3d")

def findDynamic(soup):
    qtmMeasurements = soup.find_all("Measurement")

    measurements=list()
    for measurement in qtmMeasurements:
        if measurement.attrs["Type"] != "Static - CGM2" and toBool(measurement.Used.text):
            measurements.append(measurement)

    return measurements


def detectMeasurementType(soup):
    measurements = soup.find_all("Measurement")

    types = list()
    for measurement in measurements:
        if measurement.attrs["Type"] != "Static - CGM2" and toBool(measurement.Used.text):
            if measurement.attrs["Type"] not in types:
                types.append(measurement.attrs["Type"])

    return types



def SubjectMp(soup):

    required_mp={
    'Bodymass'   : float(soup.Subject.Weight.text),
    'Height'   : float(soup.Subject.Height.text),
    'LeftLegLength' : float(soup.Subject.Leg_length_left.text)*1000.0,
    'RightLegLength' : float(soup.Subject.Leg_length_right.text)*1000.0,#865.0 ,
    'LeftKneeWidth' : float(soup.Subject.Knee_width_left.text)*1000.0,#102.0,
    'RightKneeWidth' : float(soup.Subject.Knee_width_right.text)*1000.0,#103.4,
    'LeftAnkleWidth' : float(soup.Subject.Ankle_width_left.text)*1000.0,#75.3,
    'RightAnkleWidth' : float(soup.Subject.Ankle_width_right.text)*1000.0,#72.9,
    'LeftSoleDelta' : float(soup.Subject.Sole_delta_left.text)*1000.0,#75.3,
    'RightSoleDelta' : float(soup.Subject.Sole_delta_right.text)*1000.0,#72.9,
    'LeftShoulderOffset' : float(soup.Subject.Shoulder_offset_left.text)*1000.0,#72.9,
    'RightShoulderOffset' : float(soup.Subject.Shoulder_offset_right.text)*1000.0,#72.9,
    'LeftElbowWidth' : float(soup.Subject.Elbow_width_left.text)*1000.0,#72.9,
    'LeftWristWidth' : float(soup.Subject.Wrist_width_left.text)*1000.0,#72.9,
    'LeftHandThickness' : float(soup.Subject.Hand_thickness_left.text)*1000.0,#72.9,
    'RightElbowWidth' : float(soup.Subject.Elbow_width_right.text)*1000.0,#72.9,
    'RightWristWidth' : float(soup.Subject.Wrist_width_right.text)*1000.0,#72.9,
    'RightHandThickness' : float(soup.Subject.Hand_thickness_right.text)*1000.0#72.9,
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


def get_modelled_trials(session_xml, measurement_type):
    modelled_trials = []
    dynamicMeasurements = findDynamic(session_xml)
    for dynamicMeasurement in dynamicMeasurements:
        if isType(dynamicMeasurement, measurement_type):
            filename = getFilename(dynamicMeasurement)
            modelled_trials.append(filename)
    return modelled_trials


def get_creation_date(session_xml):
    date_str = session_xml.Subject.Session.Creation_date.text
    year, month, day = date_str.split("-")
    time_str = session_xml.Subject.Session.Creation_time.text
    hour, minute, second = time_str.split(":")
    datetime_obj = datetime(
        int(year), int(month), int(day),
        int(hour), int(minute), int(second))
    return datetime_obj
