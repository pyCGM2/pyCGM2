def getConditionItemIndex(settings,conditionID):

        flag = False
        index = 0
        for conditionIt in settings["Protocol"]["Conditions"]:
            if conditionIt["ConditionID"] == conditionID:
                flag = True
                break
            index+=1

        if flag:
            return index
        else:
            raise Exception("[pyCGM2]  Condition Id not  found")



def getConditionItem(settings,conditionID):

    out=None
    for conditionIt in settings["Protocol"]["Conditions"]:
        if conditionIt["ConditionID"] == conditionID:
            out = conditionIt

    if out is None:
        raise Exception("[pyCGM2]  Condition Id not found")

    return out



def getEmgFiles(settings,conditionID):

        out = []
        for trial in settings["Fitting"]["Trials"]:
            if trial["ConditionID"] == conditionID and trial["Emg"]:
                out.append(trial["File"])
        return out

def getFittingFiles(settings,conditionID):

        out = []
        for trial in settings["Fitting"]["Trials"]:
            if trial["ConditionID"] == conditionID :
                out.append(trial["File"])
        return out


def getEmgConfiguration(settings,conditionID,outputType = "list"):

    
    conditionItem = getConditionItem(settings,conditionID)

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