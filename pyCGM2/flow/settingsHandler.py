


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