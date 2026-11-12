"""Accessors of *flow* user settings (conditions, trials, EMG configuration)."""

import pandas as pd

def emg_ordered_dict_to_list(emg_odict):
    """Convert the EMG channels of an EMG settings dictionary into a list.

    Args:
        emg_odict (dict): EMG channels keyed by analog label (e.g. ``Voltage.EMG1``).

    Returns:
        List[dict]: Channels with ``name``, ``muscle``, ``context`` and ``normal_activity`` keys.
    """
    
    emg_list = []

    for key, subdict in emg_odict.items():
        # key = "Voltage.EMG1"
        name = key.replace("Voltage.", "")  # → EMG1

        emg_list.append({
            "name": name,
            "muscle": "" if subdict.get("Muscle") is None else subdict.get("Muscle"),
            "context": "" if subdict.get("Context") is None else subdict.get("Context") ,
            "normal_activity": "" if subdict.get("NormalActivity") is None else subdict.get("NormalActivity") 
        })

    return emg_list



def homogeneizeEmgSettings(settings, emgSettings=None):
    """    

    Args:
        settings (_type_): _description_
        emgSettings (_type_, optional): _description_. Defaults to None.
    """

    if emgSettings is None:
        emgSettings = settings["Protocol"]["Conditions"][0]["EmgSettings"]

        for i in range(1,len(settings["Protocol"]["Conditions"])):
            settings["Protocol"]["Conditions"][i]["EmgSettings"].update(emgSettings)
    else:
        for i in range(0,len(settings["Protocol"]["Conditions"])):
            settings["Protocol"]["Conditions"][i]["EmgSettings"].update(emgSettings)

def homogeneizeTranslators(settings, translators=None):
    """Apply the same marker translators to all calibration and fitting trials.

    Args:
        settings (dict): flow user settings.
        translators (dict, optional): Translators. Defaults to the translators of the first calibration.
    """
    
    for it in settings["Calibration"]:
        id = it["ID"]
        if translators is  None:
            translators = it["Translators"]
        else:
            it["Translators"].update(translators)

    
        for fittingIt in settings["Fitting"]["Trials"]:
            if fittingIt["CalibrationID"] == id:
                fittingIt["Translators"].update (translators)

    

def get_condition(settings,condition_id):
    """Return a condition of the protocol.

    Args:
        settings (dict): flow user settings.
        condition_id (str): Condition identifier (e.g. ``Condition1``).

    Returns:
        dict: The condition.

    Raises:
        Exception: If the condition is not found.
    """

    out=None
    for conditionIt in settings["Protocol"]["Conditions"]:
        if conditionIt["ConditionID"] == condition_id:
            out = conditionIt

    if out is None:
        raise Exception("[pyCGM2]  Condition Id not found")

    return out



def get_emg_configuration(settings,condition_id,outputType = "list"):
    """Return the EMG configuration of a condition (channels with an assigned muscle).

    Args:
        settings (dict): flow user settings.
        condition_id (str): Condition identifier (e.g. ``Condition1``).
        outputType (str, optional): ``"list"`` returns a tuple of lists, otherwise a dictionary.

    Returns:
        Labels, muscles, contexts and normal activities, as a tuple of lists or as a dictionary
        (``Labels``, ``Muscles``, ``Contexts``, ``NormalActivity``).
    """

    conditionItem = get_condition(settings,condition_id)

    labels = []
    contexts =[]
    normalActivities = []
    muscles =[]
    for emg in  conditionItem["EmgSettings"]["CHANNELS"].keys():
        if emg !="None":
            if conditionItem["EmgSettings"]["CHANNELS"][emg]["Muscle"] != "None":
                labels.append((emg))
                muscles.append((conditionItem["EmgSettings"]["CHANNELS"][emg]["Muscle"]))
                contexts.append((conditionItem["EmgSettings"]["CHANNELS"][emg]["Context"])) if conditionItem["EmgSettings"]["CHANNELS"][emg] != "None" else contexts.append("NA")
                normalActivities.append((conditionItem["EmgSettings"]["CHANNELS"][emg]["NormalActivity"])) if conditionItem["EmgSettings"]["CHANNELS"][emg]["NormalActivity"] != "None" else normalActivities.append("NA")

    if outputType == "list":
        return labels,muscles,contexts,normalActivities
    else:
        return {"Labels": labels, "Muscles": muscles, "Contexts": contexts, "NormalActivity": normalActivities}


def get_emg_processing(settings,condition_id,outputType = "list"):
    """Return the EMG processing settings of a condition.

    Args:
        settings (dict): flow user settings.
        condition_id (str): Condition identifier (e.g. ``Condition1``).
        outputType (str, optional): Not used.

    Returns:
        dict: EMG processing settings.
    """

    conditionItem = get_condition(settings,condition_id)

    return conditionItem["EmgSettings"]["Processing"]




def list_conditions(data):
    """Return the available condition identifiers.

    Args:
        data (dict): flow user settings.

    Returns:
        list: ConditionIDs.
    """
    return [cond['ConditionID'] for cond in data['Protocol']['Conditions']]

def get_trials_by_condition(data, condition_id):
    """Return the fitting trial files of a condition.

    Args:
        data (dict): flow user settings.
        condition_id (str): Condition identifier (e.g. ``Condition1``).

    Returns:
        list: Trial filenames.
    """
    trials = data.get("Fitting", {}).get("Trials", [])
    return [trial["File"] for trial in trials if trial.get("ConditionID") == condition_id]

def get_condition_details(data, condition_id):
    """Return the metadata of a condition.

    Args:
        data (dict): flow user settings.
        condition_id (str): Condition identifier (e.g. ``Condition1``).

    Returns:
        Optional[dict]: The condition, or None if not found.
    """
    for cond in data['Protocol']['Conditions']:
        if cond["ConditionID"] == condition_id:
            return cond
    return None

def get_emg_trials_by_condition(data, condition_id):
    """Return the trial files with EMG of a condition.

    Both ``Emg['Trials']`` (if defined) and ``Fitting['Trials']`` with ``Emg == True`` are explored.

    Args:
        data (dict): flow user settings (loaded YAML).
        condition_id (str): Condition identifier (e.g. ``Condition1``).

    Returns:
        list: Trial filenames with EMG.
    """
    emg_files = []

    # Vérifie dans Emg['Trials']
    emg_trials = data.get("Emg", {}).get("Trials", [])
    if emg_trials:
        emg_files += [trial["File"] for trial in emg_trials if trial.get("ConditionID") == condition_id]

    
    # Vérifie dans Fitting['Trials'] avec Emg == True
    fitting_trials = data.get("Fitting", {}).get("Trials", [])
    emg_files += [trial["File"] for trial in fitting_trials
                  if trial.get("ConditionID") == condition_id and trial.get("Emg") is True]

    return emg_files

# def get_all_trials(data):
#     conditions = list_conditions(data)

def build_session_conditions_dataframe(settings):
    """Build a dataframe of the conditions of a session.

    Args:
        settings (dict): flow user settings.

    Returns:
        pd.DataFrame: One row per condition (ConditionID, Context, ContextComments, NerveBlock,
        Task, Shoes, Orthosis, ExternalAid, PersonalAid).
    """

    rows = []

    conditions = list_conditions(settings)

    for condition in conditions:
        conditionDetails = get_condition_details(settings, condition)

        rows.append({
            "ConditionID": conditionDetails.get("ConditionID"),
            "Context": conditionDetails.get("Context"),
            "ContextComments": conditionDetails.get("ContextComments"),
            "NerveBlock": conditionDetails.get("NerveBlock"),
            "Task": conditionDetails.get("Task"),
            "Shoes": conditionDetails.get("Shoes"),
            "Orthosis": conditionDetails.get("Orthosis"),
            "ExternalAid": conditionDetails.get("ExternalAid"),
            "PersonalAid": conditionDetails.get("PersonalAid"),
        })


    df = pd.DataFrame(rows)

    return df

def get_EmgRepresentative(settings,condition_id):
    """Return the EMG representative trial of a condition.

    Args:
        settings (dict): flow user settings.
        condition_id (str): Condition identifier (e.g. ``Condition1``).

    Returns:
        str: Trial filename.

    Raises:
        Exception: If the condition is not found.
    """

    out=None
    for conditionIt in settings["Protocol"]["Conditions"]:
        if conditionIt["ConditionID"] == condition_id:
            out = conditionIt["EmgRepresentativeTrial"]
           

    if out is None:
        raise Exception("[pyCGM2]  Condition Id not found")

    return out


def get_FittingTrials(settings):
    """Return the fitting trial files.

    Args:
        settings (dict): flow user settings.

    Returns:
        list: Trial filenames.
    """
    trials = []
    for it in settings["Fitting"]["Trials"]:
        trials.append(it["File"])
    return trials

def get_CalibrationTrials(settings):
    """Return the static (calibration) trial files.

    Args:
        settings (dict): flow user settings.

    Returns:
        list: Trial filenames.
    """
    trials = []
    for it in settings["Calibration"]:
        trials.append(it["StaticTrial"])
    return trials