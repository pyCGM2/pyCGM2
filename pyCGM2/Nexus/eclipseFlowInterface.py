# coding: utf-8
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os

import pyCGM2; LOGGER = pyCGM2.LOGGER

from pyCGM2.Utils import files
from pyCGM2.Nexus import eclipse
from pyCGM2.Nexus import vskTools

from pyCGM2 import enums
import re




def findStaticTrials(path):
    enfs = eclipse.getEnfFiles(path,enums.EclipseType.Trial)

    detected = list()
    for enf in enfs:
        try:
            enfTrial = eclipse.TrialEnfReader(path,enf)
            if enfTrial.get("TrialType") is not None:
                if enfTrial.get("TrialType") == "Static":

                    detected.append(enf)
        except UnicodeDecodeError:
            LOGGER.logger.error( "UnicodeError in file: %s"%(enf))


    if detected ==[] : raise Exception("No static file detected")
    return detected

def findMotionTrials(path):
    enfs = eclipse.getEnfFiles(path,enums.EclipseType.Trial)

    detected = list()
    for enf in enfs:
        try:
            enfTrial = eclipse.TrialEnfReader(path,enf)
            if enfTrial.get("TrialType") is not None:
                if enfTrial.get("TrialType") == "Motion":
                    detected.append(enf)
        except UnicodeDecodeError:
            LOGGER.logger.error( "UnicodeError in file: %s"%(enf))


    if detected ==[] : raise Exception("No Motion file detected")
    return detected

def findEmgTrials(path):
    enfs = eclipse.getEnfFiles(path,enums.EclipseType.Trial)

    detected = list()
    for enf in enfs:
        try:
            enfTrial = eclipse.TrialEnfReader(path,enf)
            if enfTrial.get("TrialType") is not None:
                if enfTrial.get("TrialType") == "EMG":
                    detected.append(enf)
        except UnicodeDecodeError:
            LOGGER.logger.error( "UnicodeError in file: %s"%(enf))

    if detected ==[] : LOGGER.logger.info("No emg file detected")
    return detected

def findMVCTrials(path):
    enfs = eclipse.getEnfFiles(path,enums.EclipseType.Trial)

    detected = list()
    for enf in enfs:
        try:
            enfTrial = eclipse.TrialEnfReader(path,enf)
            if enfTrial.get("TrialType") is not None:
                if enfTrial.get("TrialType") == "MVC":
                    detected.append(enf)
        except UnicodeDecodeError:
            LOGGER.logger.error( "UnicodeError in file: %s"%(enf))
    if detected ==[] : LOGGER.logger.info("No MVC file detected")
    return detected



def staticDetails(path,enfs):

    id_checking = None
    details = list()

    for enf in enfs:

        enfTrial = eclipse.TrialEnfReader(path,enf)
        lff = enfTrial.get("LeftFlatFoot")
        rff = enfTrial.get("RightFlatFoot")
        id = enfTrial.get("CalibID")
        if id == id_checking:
            raise Exception("Calibration ID is not unique")
        else:
            id_checking = id

        file = enf[0:enf.find(".")] +".c3d"
        staticDetails =[id,file,lff,rff,]


        details.append(staticDetails)


    return details


def FittingDetails(path,enfs):

    details = list()

    for enf in enfs:

        enfTrial = eclipse.TrialEnfReader(path,enf)
        file = enf[0:enf.find(".")] +".c3d"
        conditionID = enfTrial.get("ConditionID")
        calibID = enfTrial.get("CalibID")
        fpa = enfTrial.getForcePlateAssigment()

        fittingDetails =[file,calibID,fpa,conditionID]

        details.append(fittingDetails)


    return details


def EmgDetails(path,enfs):

    details = list()

    for enf in enfs:

        enfTrial = eclipse.TrialEnfReader(path,enf)
        file = enf[0:enf.find(".")] +".c3d"
        conditionID = enfTrial.get("ConditionID")

        emgDetails =[file,conditionID]

        details.append(emgDetails)


    return details

def getConditions(path,motionEnfs):

    conditions = list()


    for enf in motionEnfs:
        enfTrial = eclipse.TrialEnfReader(path,enf)

        if enfTrial.get("Share"):
            condition=dict()

            condition["ConditionID"] = enfTrial.get("ConditionID")
            condition["Context"] = enfTrial.get("Context")
            condition["Block"] = enfTrial.get("Block")
            condition["Task"] = enfTrial.get("Task")
            condition["Shoes"] = enfTrial.get("Shoes")
            condition["ProthesisOrthosis"] = enfTrial.get("ProthesisOrthosis")
            condition["ExternalAid"] = enfTrial.get("ExternalAid")
            condition["PersonalAid"] = enfTrial.get("PersonalAid")
            condition["EmgRepresentativeTrial"] = enf[0:enf.find(".")] +".c3d"
            condition["EmgReferenceConditionID"] = ""

            conditions.append(condition)


    conditionsNumbers = list()
    for conditionIt in conditions:
        if conditionIt["ConditionID"] is None:
            raise Exception("[pyCGM2f] - ConditionID cannot be None")

        pos = int(re.findall("\d+",conditionIt["ConditionID"])[0])
        conditionsNumbers.append(pos)
    conditionsNumbers.sort()

    sortConditions = list()
    for number in conditionsNumbers:
        for conditionIt in conditions:
            if conditionIt["ConditionID"] == "Condition"+str(number):
                if  number != 1: conditionIt["EmgReferenceConditionID"] = "Condition1"
                sortConditions.append(conditionIt)
    return sortConditions

def MvcDetails(path,mvcEnfs):

    mvcs = list()


    for enf in mvcEnfs:
        enfTrial = eclipse.TrialEnfReader(path,enf)
        file = enf[0:enf.find(".")] +".c3d"

        mvc=dict()
        mvc["MvcID"] = enfTrial.get("mvcID")
        mvc["Channels"]  = enfTrial.get("mvcChannels")
        mvc["assignConditions"]  = enfTrial.get("mvcAssignConditions")
        mvc["trial"]  = file

        mvcs.append(mvc)

    mvcIDs = list()
    for mvcIt in mvcs:
        mvcIDs = mvcIDs +[mvcIt["MvcID"]]

    if None in mvcIDs:
        raise Exception("[pycgm2-flow] there is at leat one mvcID wint no label")
    else:
        mvcIDs2=[ii for n,ii in enumerate(mvcIDs) if ii not in mvcIDs[:n]] # remove duplicates

        mvcs_groups = [None]*len(mvcIDs2)

        i = 0
        for mvcIdIt in mvcIDs2:
            group= list()
            for mvcIt in mvcs:
                if mvcIt["MvcID"] == mvcIdIt:
                    group.append(mvcIt)
            mvcs_groups[i] =  group
            i+=1


        for groupIt in mvcs_groups:
            conditions =  list()
            channels = list()
            for mvcIt in groupIt:
                if mvcIt is not None:  channels =  channels + [mvcIt["Channels"]]
                if mvcIt["assignConditions"] is not None: conditions = conditions + [mvcIt["assignConditions"]]
            channels2 = [i for i in channels if i] # remove None
            conditions2 = [i for i in conditions if i]
            channels3=[ii for n,ii in enumerate(channels2) if ii not in channels2[:n]] # remove duplicates
            conditions3=[ii for n,ii in enumerate(conditions2) if ii not in conditions2[:n]]

            if len(channels3)>1:
                raise Exception ("MVC channels from the same mvc ID targets different channels")

            if len(conditions3)>1:
                raise Exception ("MVC assigned conditions from the same mvc ID targets different conditions")

            # assign similar channels and condition to all item of the group
            for mvcIt in groupIt:
                mvcIt["Channels"] = channels3
                mvcIt["assignConditions"] = conditions3


        mvc_final_groups = list()
        for groupIt in mvcs_groups:

            mvc_final= dict()
            mvc_final["MvcID"] = groupIt[0]["MvcID"]
            mvc_final["Channels"] = list()

            channels_list= ["Voltage.EMG"+textIt for textIt in groupIt[0]["Channels"][0].split("-")]
            conditions_list= ["Condition"+textIt for textIt in groupIt[0]["assignConditions"][0].split("-")]

            trial_list = list()
            for mvcIt in groupIt:
                trial_list.append(mvcIt["trial"])

            mvc_final["ConditionIDs"] = conditions_list
            mvc_final["Channels"] = channels_list
            mvc_final["Trials"] = trial_list

            mvc_final_groups.append(mvc_final)

    return mvc_final_groups


def list_duplicates(seq):
  seen = set()
  seen_add = seen.add
  # adds all elements it doesn't know yet to seen and all other to seen_twice
  seen_twice = set( x for x in seq if x in seen or seen_add(x) )
  # turn the set into a list (as requested)
  return list( seen_twice )



def repareEnf(DATA_PATH):

    enfs = files.getFiles(DATA_PATH,"enf")

    for enf in enfs:
        sep0 = enf.find(".")
        if enf[sep0:].find("Trial")!=-1:
            if  enf[sep0:]!=".Trial.enf":
                newName = enf[0:sep0]
                if newName+".Trial.enf" in files.getFiles(DATA_PATH,".Trial.enf"):
                    os.remove(DATA_PATH+newName+".Trial.enf")
                os.rename(DATA_PATH+enf,DATA_PATH+newName+".Trial.enf")

        if enf[sep0:].find("Session")!=-1:
            if enf[sep0:]!=".Session.enf":
                newName = enf[0:sep0]
                os.rename(DATA_PATH+enf,DATA_PATH+newName+".Session.enf")
