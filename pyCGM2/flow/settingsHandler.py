

def emg_ordered_dict_to_list(emg_odict):
    
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

    out=None
    for conditionIt in settings["Protocol"]["Conditions"]:
        if conditionIt["ConditionID"] == condition_id:
            out = conditionIt

    if out is None:
        raise Exception("[pyCGM2]  Condition Id not found")

    return out



def get_emg_configuration(settings,condition_id,outputType = "list"):

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

    conditionItem = get_condition(settings,condition_id)

    return conditionItem["EmgSettings"]["Processing"]




def list_conditions(data):
    """Retourne la liste des ConditionIDs disponibles"""
    return [cond['ConditionID'] for cond in data['Protocol']['Conditions']]

def get_trials_by_condition(data, condition_id):
    """Retourne la liste des fichiers de trial associés à une condition"""
    trials = data.get("Fitting", {}).get("Trials", [])
    return [trial["File"] for trial in trials if trial.get("ConditionID") == condition_id]

def get_condition_details(data, condition_id):
    """Retourne les métadonnées associées à une condition"""
    for cond in data['Protocol']['Conditions']:
        if cond["ConditionID"] == condition_id:
            return cond
    return None

def get_emg_trials_by_condition(data, condition_id):
    """
    Retourne la liste des fichiers contenant de l'EMG associés à une condition donnée.
    Explore à la fois :
      - Emg['Trials'] s'ils sont définis
      - Fitting['Trials'] avec Emg == True

    Paramètres :
        data (dict) : Données YAML chargées.
        condition_id (str) : L'identifiant de la condition (e.g., 'Condition1').

    Retour :
        list : Liste des fichiers de trials contenant de l'EMG.
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
#     import ipdb; ipdb.set_trace()

