# -*- coding: utf-8 -*-
import os
from pyCGM2.Utils import files

import pyCGM2; LOGGER = pyCGM2.LOGGER


# -----emg------
def loadEmgConfiguration(DATA_PATH,emgSettingsFile ="emg.settings"):
    """This function returns emg details from the `emg.settings` file.
    Unless `emg.settings` file is found in the data folder, the code loads the `emg.settings`
    located in `pyCGM2/Settings`

    Args:
        DATA_PATH (str): path with double \ at the end.
        emgSettingsFile (str): name of the emg settings file ( default is `emg.settings`).

    Returns:
        list: channel labels
        list: names of the muscle
        list: context name of the muscle
        list : names of the muscle set in reference for displaying its normal activity during gait

    Examples

        >>> EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT,NORMAL_ACTIVITIES = configuration.loadEmgConfiguration(DATA_PATH)

    """



    if os.path.isfile(DATA_PATH + emgSettingsFile):
        LOGGER.logger.info("%s found in the data folder"%(emgSettingsFile))
        emgSettings = files.openFile(DATA_PATH,emgSettingsFile)

    else:
        if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "emg.settings"):
            emgSettings =  files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"emg.settings")
        else:
            emgSettings =  files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"emg.settings")

    labels = []
    contexts =[]
    normalActivities = []
    muscles =[]
    for emg in emgSettings["CHANNELS"].keys():
        if emg !="None":
            if emgSettings["CHANNELS"][emg]["Muscle"] != "None":
                labels.append((emg))
                muscles.append((emgSettings["CHANNELS"][emg]["Muscle"]))
                contexts.append((emgSettings["CHANNELS"][emg]["Context"])) if emgSettings["CHANNELS"][emg]["Context"] != "None" else contexts.append("NA")
                normalActivities.append((emgSettings["CHANNELS"][emg]["NormalActivity"])) if emgSettings["CHANNELS"][emg]["NormalActivity"] != "None" else normalActivities.append("NA")

    return labels,muscles,contexts,normalActivities
