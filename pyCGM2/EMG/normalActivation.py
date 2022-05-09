# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/EMG
#APIDOC["Draft"]=False
#--end--


"""
The module contains convenient function for getting the normal emg activity of the muscles listed in the file
(``pyCGM2/Data/normativeData/emg/normalActivation.json``).

"""
import pyCGM2
from pyCGM2.Utils import files


def getNormalBurstActivity(muscle, fo):
    """get onsets and offsets of a specific muscle.

    Args:
        muscle (str): muscle label listed in  normalActivation.json
        fo (int): foot off frame

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


def getNormalBurstActivity_fromCycles(muscle, ff, begin, fo, end, apf):
    """get onsets and offsets of a specific muscle from .

    Args:
        muscle (str): muscle label listed in  normalActivation.json
        ff (int): first frame of the btk.acquisition
        begin (int): initial frame of cycle
        fo (int): foot off frame
        end (int): final frame of the cycle
        apf (int): nNumber of analog sample per frame


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
