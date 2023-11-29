"""
Module contains convenient functions dealing with the session.xml file generated from QTM.

"""

from pyCGM2.Utils.utils import toBool
from datetime import datetime
import pyCGM2
LOGGER = pyCGM2.LOGGER

import bs4


def getFilename(measurement:bs4.BeautifulSoup):
    """Returns the filename of the measurement section.

    Args:
        measurement (BeautifulSoup): A `measurement` section of the session.xml file.

    Returns:
        str: The filename of the measurement, with 'qtm' replaced by 'c3d'.
    """
    return measurement.attrs["Filename"].replace("qtm", "c3d")


def getForcePlateAssigment(measurement:bs4.BeautifulSoup):
    """Returns the force plate assignment from the measurement section.

    Args:
        measurement (BeautifulSoup): A `measurement` section of the session.xml file.

    Returns:
        str: The force plate assignment.
    """
    mea = measurement

    mfpa = mea.find("Forceplate1").text[0] + mea.find("Forceplate2").text[0] + \
        mea.find("Forceplate3").text[0] + mea.find("Forceplate4").text[0] + \
        mea.find("Forceplate5").text[0]

    return mfpa


def isType(measurement:bs4.BeautifulSoup, type:str):
    """Checks the type of measurement section.

    Args:
        measurement (BeautifulSoup): A `measurement` section of the session.xml file.
        type (str): The type to check against.

    Returns:
        bool: True if the measurement type matches, else False.
    """
    return measurement.attrs["Type"] == type


def findStatic(soup:bs4.BeautifulSoup):
    """Returns the static file from the BeautifulSoup instance representing the session.xml.

    Args:
        soup (BeautifulSoup): Content of the session.xml.

    Returns:
        BeautifulSoup: The static measurement section.
    """
    qtmMeasurements = soup.find_all("Measurement")
    static = []
    for measurement in qtmMeasurements:
        if "Static" in measurement.attrs["Type"] and toBool(measurement.Used.text):
            static.append(measurement)
        if len(static) > 1:
            raise Exception(
                "You can t have 2 activated static c3d within your session")
    return static[0]


def findDynamic(soup:bs4.BeautifulSoup):
    """Returns the dynamic files from the BeautifulSoup instance representing the session.xml.

    Args:
        soup (BeautifulSoup): Content of the session.xml.

    Returns:
        List[BeautifulSoup]: List of dynamic measurement sections.
    """
    qtmMeasurements = soup.find_all("Measurement")

    measurements = []
    for measurement in qtmMeasurements:
        if "Gait" in measurement.attrs["Type"] and toBool(measurement.Used.text):
            measurements.append(measurement)

    return measurements


def findKneeCalibration(soup:bs4.BeautifulSoup, side:str):
    """Returns the knee functional calibration file from the BeautifulSoup instance.

    Args:
        soup (BeautifulSoup): Content of the session.xml.
        side (str): Lower limb side ('Left' or 'Right').

    Returns:
        BeautifulSoup: The knee calibration measurement section for the specified side.
    """
    qtmMeasurements = soup.find_all("Measurement")
    kneeCalib = []

    for measurement in qtmMeasurements:
        if side + " Knee Calibration" in measurement.attrs["Type"] and toBool(measurement.Used.text):
            kneeCalib.append(measurement)
        if len(kneeCalib) > 1:
            raise Exception(
                "You can t have 2 activated %s functional knee Calib c3d within your session" % (side))
    return kneeCalib[0] if kneeCalib != [] else None


def getKneeFunctionCalibMethod(measurement:bs4.BeautifulSoup):
    """Returns the method used for the knee calibration.

    Args:
        measurement (BeautifulSoup): A `measurement` section of the session.xml file.

    Returns:
        str: The knee functional calibration method.
    """
    mea = measurement
    return mea.find("Knee_functional_calibration_method").text


def detectMeasurementType(soup:bs4.BeautifulSoup):
    """Returns the type of each measurement section.

    Args:
        soup (BeautifulSoup): Content of the session.xml.

    Returns:
        List[str]: List of types of each measurement section.
    """

    measurements = soup.find_all("Measurement")

    types = []
    for measurement in measurements:
        staticFlag = "Static" not in measurement.attrs["Type"]
        leftKneeFlag = "Left Knee Calibration - CGM2" not in measurement.attrs["Type"]
        rightKneeFlag = "Right Knee Calibration - CGM2" not in measurement.attrs["Type"]
        if staticFlag and leftKneeFlag and rightKneeFlag and toBool(measurement.Used.text):
            if measurement.attrs["Type"] not in types:
                types.append(measurement.attrs["Type"])

    return types


def SubjectMp(soup:bs4.BeautifulSoup):
    """Returns the anthropometric parameters.

    Args:
        soup (BeautifulSoup): Content of the session.xml.

    Returns:
        Tuple[Dict, Dict]: Required and optional anthropometric parameters.
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
    """(Obsolete) Returns modelled trials of a specified type from the session XML.

    Args:
        session_xml (BeautifulSoup): The parsed session XML.
        measurement_type (str): The measurement type to filter by.

    Returns:
        List[str]: List of filenames for modelled trials of the specified type.
    """
    modelled_trials = []
    dynamicMeasurements = findDynamic(session_xml)
    for dynamicMeasurement in dynamicMeasurements:
        if isType(dynamicMeasurement, measurement_type):
            filename = getFilename(dynamicMeasurement)
            modelled_trials.append(filename)
    return modelled_trials


def get_creation_date(session_xml):
    """(Obsolete)  Retrieves the creation date of the session from the session XML.

    Args:
        session_xml (BeautifulSoup): The parsed session XML.

    Returns:
        datetime: The datetime object representing the creation date and time of the session.
    """
    date_str = session_xml.Subject.Session.Creation_date.text
    year, month, day = date_str.split("-")
    time_str = session_xml.Subject.Session.Creation_time.text
    hour, minute, second = time_str.split(":")
    datetime_obj = datetime(
        int(year), int(month), int(day),
        int(hour), int(minute), int(second))
    return datetime_obj
