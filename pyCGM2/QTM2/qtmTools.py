# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/qtm
#APIDOC["Draft"]=False
#--end--

"""
Module contains convenient functions dealing with the session.xml file generated from QTM.

"""

from pyCGM2.Utils.utils import toBool
from datetime import datetime
import pyCGM2
LOGGER = pyCGM2.LOGGER


def getFilename(measurement):
    """return the filename of the measurement section

    Args:
        measurement (bs4.soup): a `measurement` section of the session.xml file.

    """
    return measurement.attrs["Filename"].replace("qtm", "c3d")


def getForcePlateAssigment(measurement):
    """return the force plate assigment from the measurement section

    Args:
        measurement (bs4.soup): a `measurement` section of the session.xml file.

    """
    mea = measurement

    mfpa = mea.find("Forceplate1").text[0] + mea.find("Forceplate2").text[0] + \
        mea.find("Forceplate3").text[0] + mea.find("Forceplate4").text[0] + \
        mea.find("Forceplate5").text[0]

    return mfpa


def isType(measurement, type):
    """check type of  measurement section

    Args:
        measurement (bs4.soup): a `measurement` section of the session.xml file.
        type (str): type

    """
    return measurement.attrs["Type"] == type


def findStatic(soup):
    """return the static file from the bs4.soup instance representing the content of the session.xml

    Args:
        soup (bs4.soup): content of the session.xml

    """
    qtmMeasurements = soup.find_all("Measurement")
    static = list()
    for measurement in qtmMeasurements:
        if "Static" in measurement.attrs["Type"] and toBool(measurement.Used.text):
            static.append(measurement)
        if len(static) > 1:
            raise Exception(
                "You can t have 2 activated static c3d within your session")
    return static[0]


def findDynamic(soup):
    """return the dynamic files from the bs4.soup instance representing the content of the session.xml

    Args:
        soup (bs4.soup): content of the session.xml

    """
    qtmMeasurements = soup.find_all("Measurement")

    measurements = list()
    for measurement in qtmMeasurements:
        if "Gait" in measurement.attrs["Type"] and toBool(measurement.Used.text):
            measurements.append(measurement)

    return measurements


def findKneeCalibration(soup, side):
    """return the knee functional calibration file from the bs4.soup instance representing the content of the session.xml

    Args:
        soup (bs4.soup): content of the session.xml
        side (str): lower limb side

    """
    qtmMeasurements = soup.find_all("Measurement")
    kneeCalib = list()

    for measurement in qtmMeasurements:
        if side + " Knee Calibration" in measurement.attrs["Type"] and toBool(measurement.Used.text):
            kneeCalib.append(measurement)
        if len(kneeCalib) > 1:
            raise Exception(
                "You can t have 2 activated %s functional knee Calib c3d within your session" % (side))
    return kneeCalib[0] if kneeCalib != [] else None


def getKneeFunctionCalibMethod(measurement):
    """return the method used for the knee calibration

    Args:
        measurement (bs4.soup): a `measurement` section of the session.xml file.

    """
    mea = measurement
    return mea.find("Knee_functional_calibration_method").text


def detectMeasurementType(soup):
    """return the type of each measurement section

    Args:
        soup (bs4.soup): content of the session.xml

    """

    measurements = soup.find_all("Measurement")

    types = list()
    for measurement in measurements:
        staticFlag = "Static" not in measurement.attrs["Type"]
        leftKneeFlag = "Left Knee Calibration - CGM2" not in measurement.attrs["Type"]
        rightKneeFlag = "Right Knee Calibration - CGM2" not in measurement.attrs["Type"]
        if staticFlag and leftKneeFlag and rightKneeFlag and toBool(measurement.Used.text):
            if measurement.attrs["Type"] not in types:
                types.append(measurement.attrs["Type"])

    return types


def SubjectMp(soup):
    """return the antropometric parameters

    Args:
        soup (bs4.soup): content of the session.xml

    """

    bodymass = float(soup.Subject.Weight.text)
    if bodymass == 0.0:
        LOGGER.logger.error(
            "[pyCGM2] Null Bodymass detected - Kinetics will be unnormalized")
        bodymass = 1.0

    required_mp = {
        'Bodymass': bodymass,
        'Height': float(soup.Subject.Height.text),
        'LeftLegLength': float(soup.Subject.Leg_length_left.text)*1000.0,
        # 865.0 ,
        'RightLegLength': float(soup.Subject.Leg_length_right.text)*1000.0,
        # 102.0,
        'LeftKneeWidth': float(soup.Subject.Knee_width_left.text)*1000.0,
        # 103.4,
        'RightKneeWidth': float(soup.Subject.Knee_width_right.text)*1000.0,
        # 75.3,
        'LeftAnkleWidth': float(soup.Subject.Ankle_width_left.text)*1000.0,
        # 72.9,
        'RightAnkleWidth': float(soup.Subject.Ankle_width_right.text)*1000.0,
        # 75.3,
        'LeftSoleDelta': float(soup.Subject.Sole_delta_left.text)*1000.0,
        # 72.9,
        'RightSoleDelta': float(soup.Subject.Sole_delta_right.text)*1000.0,
        # 72.9,
        'LeftShoulderOffset': float(soup.Subject.Shoulder_offset_left.text)*1000.0,
        # 72.9,
        'RightShoulderOffset': float(soup.Subject.Shoulder_offset_right.text)*1000.0,
        # 72.9,
        'LeftElbowWidth': float(soup.Subject.Elbow_width_left.text)*1000.0,
        # 72.9,
        'LeftWristWidth': float(soup.Subject.Wrist_width_left.text)*1000.0,
        # 72.9,
        'LeftHandThickness': float(soup.Subject.Hand_thickness_left.text)*1000.0,
        # 72.9,
        'RightElbowWidth': float(soup.Subject.Elbow_width_right.text)*1000.0,
        # 72.9,
        'RightWristWidth': float(soup.Subject.Wrist_width_right.text)*1000.0,
        # 72.9,
        'RightHandThickness': float(soup.Subject.Hand_thickness_right.text)*1000.0
    }

    optional_mp = {
        'InterAsisDistance': 0,
        'LeftAsisTrocanterDistance': 0,
        'LeftTibialTorsion': 0,
        'LeftThighRotation': 0,
        'LeftShankRotation': 0,
        'RightAsisTrocanterDistance': 0,
        'RightTibialTorsion': 0,
        'RightThighRotation': 0,
        'RightShankRotation': 0
        }

    return required_mp, optional_mp


def get_modelled_trials(session_xml, measurement_type):
    # Obsolete
    modelled_trials = []
    dynamicMeasurements = findDynamic(session_xml)
    for dynamicMeasurement in dynamicMeasurements:
        if isType(dynamicMeasurement, measurement_type):
            filename = getFilename(dynamicMeasurement)
            modelled_trials.append(filename)
    return modelled_trials


def get_creation_date(session_xml):
    # Obsolete
    date_str = session_xml.Subject.Session.Creation_date.text
    year, month, day = date_str.split("-")
    time_str = session_xml.Subject.Session.Creation_time.text
    hour, minute, second = time_str.split(":")
    datetime_obj = datetime(
        int(year), int(month), int(day),
        int(hour), int(minute), int(second))
    return datetime_obj
