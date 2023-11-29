"""
The module contains convenient function for getting the normal emg activity of the muscles listed in the file
(``pyCGM2/Data/normativeData/emg/normalActivation.json``).

"""
from typing import Tuple
import pyCGM2
from pyCGM2.Utils import files

from typing import List, Tuple, Dict, Optional


def getNormalBurstActivity(muscle:str, fo:int):
    """
    Get onsets and offsets of a specific muscle's normal burst activity.

    This function retrieves the normal burst activity for a specified muscle based on the data in 'normalActivation.json'.
    It calculates the muscle's normal activation timings relative to the stance phase of a gait cycle.

    Args:
        muscle (str): Muscle label listed in 'normalActivation.json'.
        fo (int): Foot off frame number in the gait cycle.

    Returns:
        Tuple[List[int], List[int]]: Two lists containing the start frames and durations of the muscle's burst activities.
    """

    normalActivations = files.openJson(
        pyCGM2.NORMATIVE_DATABASE_PATH+"emg\\", "normalActivation.json")

    NORMAL_STANCE_PHASE = normalActivations["NORMAL_STANCE_PHASE"]
    TABLE = normalActivations["Activation"]

    beg = 0
    end = 100

    if muscle in TABLE.keys():
        list_beginBurst = []
        list_burstDuration = []

        for i, j in zip(range(0, 6, 2), range(1, 6, 2)):
            if TABLE[muscle][i] != "na" and TABLE[muscle][j] != "na":
                if TABLE[muscle][i] < NORMAL_STANCE_PHASE:
                    beginBurst = beg + \
                           (TABLE[muscle][i]/NORMAL_STANCE_PHASE)*(fo-beg)
                else:
                    beginBurst = fo + \
                           ((TABLE[muscle][i]-NORMAL_STANCE_PHASE)
                            / (100.0-NORMAL_STANCE_PHASE))*(end-fo)

                if TABLE[muscle][j] < NORMAL_STANCE_PHASE:
                    endBurst = beg + \
                           (TABLE[muscle][j]/NORMAL_STANCE_PHASE)*(fo-beg)
                else:
                    endBurst = fo + \
                           ((TABLE[muscle][j]-NORMAL_STANCE_PHASE)
                            / (100.0-NORMAL_STANCE_PHASE))*(end-fo)

                list_beginBurst.append(beginBurst)
                list_burstDuration.append(endBurst-beginBurst)

    else:
        list_beginBurst = [0]
        list_burstDuration = [0]

    return list_beginBurst, list_burstDuration


def getNormalBurstActivity_fromCycles(muscle:str, 
                                      ff:int, 
                                      begin:int, 
                                      fo:int, 
                                      end:int, 
                                      apf:int)-> Tuple[List, List]:
    
    """
    Get onsets and offsets of a specific muscle's normal burst activity for a given cycle.

    This function calculates the normal burst activity timings for a specified muscle during a specific gait cycle.
    The muscle's normal activation timings are adjusted based on the cycle's start, foot off, and end frame numbers.

    Args:
        muscle (str): Muscle label listed in 'normalActivation.json'.
        ff (int): First frame of the btk.Acquisition.
        begin (int): Initial frame of the gait cycle.
        fo (int): Foot off frame number within the cycle.
        end (int): Final frame of the gait cycle.
        apf (int): Number of analog samples per frame.

    Returns:
        Tuple[List[int], List[int]]: Two lists containing the start frames and durations of the muscle's burst activities within the cycle.
    """

    normalActivations = files.openJson(
        pyCGM2.NORMATIVE_DATABASE_PATH+"emg\\", "normalActivation.json")

    NORMAL_STANCE_PHASE = normalActivations["NORMAL_STANCE_PHASE"]
    TABLE = normalActivations["Activation"]

    beg = (begin-ff)*apf
    fo = (fo-ff)*apf
    end = (end-ff)*apf

    if muscle in TABLE.keys():
        list_beginBurst = []
        list_burstDuration = []

        for i, j in zip(range(0, 6, 2), range(1, 6, 2)):
            if TABLE[muscle][i] != "na" and TABLE[muscle][j] != "na":
                if TABLE[muscle][i] < NORMAL_STANCE_PHASE:
                    beginBurst = beg + \
                           (TABLE[muscle][i]/NORMAL_STANCE_PHASE)*(fo-beg)
                else:
                    beginBurst = fo + \
                           ((TABLE[muscle][i]-NORMAL_STANCE_PHASE)
                            / (100.0-NORMAL_STANCE_PHASE))*(end-fo)

                if TABLE[muscle][j] < NORMAL_STANCE_PHASE:
                    endBurst = beg + \
                           (TABLE[muscle][j]/NORMAL_STANCE_PHASE)*(fo-beg)
                else:
                    endBurst = fo + \
                           ((TABLE[muscle][j]-NORMAL_STANCE_PHASE)
                            / (100.0-NORMAL_STANCE_PHASE))*(end-fo)

                list_beginBurst.append(beginBurst)
                list_burstDuration.append(endBurst-beginBurst)

    else:
        list_beginBurst = [0]
        list_burstDuration = [0]

    return list_beginBurst, list_burstDuration
